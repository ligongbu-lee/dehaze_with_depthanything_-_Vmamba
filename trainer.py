import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader

import torch.backends.cudnn as cudnn
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#import encoding
from torchvision import transforms
import argparse
import pytorch_ssim
import dataset
import utils
import options as option

from data import create_dataloader, create_dataset


def Pre_train(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    #torch.cuda.set_device(1)
    
    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    utils.check_path(save_folder)
    utils.check_path(sample_folder)

    # Loss functions
    if opt.no_gpu == False:
        criterion_L1 = torch.nn.L1Loss().cuda()
        criterion_L2 = torch.nn.MSELoss().cuda()
        #criterion_rainypred = torch.nn.L1Loss().cuda()
        criterion_ssim = pytorch_ssim.SSIM().cuda()
    else: 
        criterion_L1 = torch.nn.L1Loss()
        criterion_L2 = torch.nn.MSELoss()
        #criterion_rainypred = torch.nn.L1Loss().cuda()
        criterion_ssim = pytorch_ssim.SSIM()

    # Initialize Generator
    generator = utils.create_generator(opt)

    # To device
    if opt.no_gpu == False:
        if opt.multi_gpu:
            # 修改：使用DistributedDataParallel替代DataParallel
            generator = nn.parallel.DistributedDataParallel(generator.cuda(), find_unused_parameters=True)
        else:
            generator = generator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    print("pretrained models loaded")

    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        save_model_path = os.path.join(opt.save_path, 'model_epoch%d_iter%d.pth' % (epoch, iteration))
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    # 修改：只在主进程保存模型
                    if torch.distributed.get_rank() == 0:
                        torch.save(generator.module.state_dict(), save_model_path)
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    # 修改：只在主进程保存模型
                    if torch.distributed.get_rank() == 0:
                        torch.save(generator.module.state_dict(), save_model_path)
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Handle multiple GPUs
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""    
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)

    opt_gan = option.parse(opt.yaml_path, is_train=False)

    opt_gan = option.dict_to_nonedict(opt_gan)
    # 添加：设置分布式训练参数
    opt_gan["dist"] = opt.multi_gpu
    im = 0

    #### Create train gan dataset and dataloader
    train_gan_loaders = []
    for phase, dataset_opt in sorted(opt_gan["datasets"].items()):
        train_gan_set = create_dataset(dataset_opt)
        # 修改：添加分布式训练支持
        if opt.multi_gpu:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_gan_set)
            train_gan_loader = create_dataloader(train_gan_set, dataset_opt, opt_gan, sampler=train_sampler)
        else:
            train_gan_loader = create_dataloader(train_gan_set, dataset_opt, opt_gan)
        train_gan_loaders.append(train_gan_loader)

    train_gan_set = create_dataset(dataset_opt)
    # 修改：添加分布式训练支持
    if opt.multi_gpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_gan_set)
        train_gan_loader = create_dataloader(train_gan_set, dataset_opt, opt_gan, sampler=train_sampler)
    else:
        train_gan_loader = create_dataloader(train_gan_set, dataset_opt, opt_gan)
    print(len(train_gan_loader))

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        # 修改：设置采样器的epoch
        if opt.multi_gpu:
            train_sampler.set_epoch(epoch)
        for i, train_data in enumerate(train_gan_loader):
            # 修改：直接使用数据集返回的深度图
            true_input, true_target, LQdepth, GTdepth = train_data["LQ"], train_data["GT"], train_data['LQdepth'], train_data['GTdepth']

            if opt.no_gpu == False:
                # To device
                true_input = true_input.cuda()
                true_target = true_target.cuda()
                LQdepth = LQdepth.cuda()
                GTdepth = GTdepth.cuda()

            # Train Generator
            optimizer_G.zero_grad()
            fake_target = generator(true_input, true_input, LQdepth)
            
            ssim_loss = -criterion_ssim(true_target, fake_target)
            Pixellevel_L1_Loss = criterion_L1(fake_target, true_target)
            loss = Pixellevel_L1_Loss + 0.2 * ssim_loss
            loss.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(train_gan_loader) + i
            iters_left = opt.epochs * len(train_gan_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # 修改：只在主进程打印日志
            if not opt.multi_gpu or torch.distributed.get_rank() == 0:
                print("\r[Epoch %d/%d] [Batch %d/%d] [Loss: %.4f %.4f] Time_left: %s" %
                    ((epoch + 1), opt.epochs, i, len(train_gan_loader), Pixellevel_L1_Loss.item(), ssim_loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(train_gan_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)

        ### Sample data every epoch
        # 修改：只在主进程保存样本
        if (epoch + 1) % 1 == 0 and (not opt.multi_gpu or torch.distributed.get_rank() == 0):
            img_list = [true_input, fake_target, true_target]
            name_list = ['in', 'pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'train_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)







