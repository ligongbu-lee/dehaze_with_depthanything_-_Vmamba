import argparse
import os
import sys
import subprocess
import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity

import utils
import dataset

# ----------------------------------------
#        Depth-Anything 深度图生成配置
# ----------------------------------------
DEPTH_ANYTHING_PATH = '/data/skc/projects/Depth-Anything'
DEPTH_ANYTHING_ENCODER = 'vitb'  # vits, vitb, vitl

def count_images(directory):
    """统计目录中的图像数量"""
    if not os.path.exists(directory):
        return 0
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    return sum(1 for f in os.listdir(directory) 
               if os.path.splitext(f.lower())[1] in valid_extensions)

def generate_depth_with_depth_anything(input_dir, output_dir, encoder='vitb'):
    """使用 Depth-Anything 生成深度图"""
    sys.path.insert(0, DEPTH_ANYTHING_PATH)
    
    # 切换到 Depth-Anything 目录加载模型
    original_cwd = os.getcwd()
    os.chdir(DEPTH_ANYTHING_PATH)
    
    try:
        from depth_anything.dpt import DepthAnything
        from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
        from torchvision.transforms import Compose
        import torch.nn.functional as F
        from tqdm import tqdm
        
        # 模型配置
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建模型
        print(f"加载 Depth-Anything ({encoder}) 模型...")
        depth_anything = DepthAnything(model_configs[encoder])
        checkpoint = os.path.join(DEPTH_ANYTHING_PATH, 'checkpoints', f'depth_anything_{encoder}14.pth')
        depth_anything.load_state_dict(torch.load(checkpoint, map_location=device))
        depth_anything = depth_anything.to(device).eval()
        
        # Transform
        transform = Compose([
            Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True,
                   ensure_multiple_of=14, resize_method='lower_bound',
                   image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取图像文件
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
        filenames = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in valid_ext]
        filenames.sort()
        
        print(f"为 {len(filenames)} 张图像生成深度图...")
        
        for filename in tqdm(filenames, desc="生成深度图"):
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}.png")
            
            if os.path.exists(output_path):
                continue
            
            raw_image = cv2.imread(input_path)
            if raw_image is None:
                continue
            
            image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
            h, w = image.shape[:2]
            
            image = transform({'image': image})['image']
            image = torch.from_numpy(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                depth = depth_anything(image)
            
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.cpu().numpy().astype(np.uint8)
            
            # 保存为 3 通道 (与原代码兼容)
            depth_3ch = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(output_path, depth_3ch)
        
        print(f"深度图已保存到: {output_dir}")
        return True
        
    except Exception as e:
        print(f"深度图生成失败: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def check_and_generate_depth_maps(baseroot):
    """检查并生成测试集深度图"""
    image_dir = os.path.join(baseroot, "image")
    depth_dir = os.path.join(baseroot, "depth_img")
    
    if not os.path.exists(image_dir):
        print(f"警告: 图像目录不存在: {image_dir}")
        return
    
    image_count = count_images(image_dir)
    depth_count = count_images(depth_dir)
    
    print("\n" + "=" * 60)
    print("检查 Depth-Anything 深度图 (测试集)...")
    print("=" * 60)
    print(f"图像目录: {image_dir}")
    print(f"深度图目录: {depth_dir}")
    print(f"源图像: {image_count} 张, 深度图: {depth_count} 张")
    
    if depth_count >= image_count and image_count > 0:
        print("深度图已就绪")
    else:
        print("需要生成深度图...")
        generate_depth_with_depth_anything(image_dir, depth_dir, DEPTH_ANYTHING_ENCODER)
    
    print("=" * 60 + "\n")

def str2bool(v):
    #print(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    #GPU parameters
    parser.add_argument('--no_gpu', default = False, help = 'True for CPU')
    # Saving, and loading parameters
    parser.add_argument('--save_name', type = str, default = './results_tmp', help = 'save the generated with certain epoch')
    parser.add_argument('--load_name', type = str, default = './models_k9_loss1/KPN_rainy_image_epoch170_bs16.pth', help = 'load the pre-trained model with certain epoch')
    #parser.add_argument('--load_name', type = str, default = './models/KPN_single_image_epoch120_bs16_mu0_sigma30.pth', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'number of workers')

    # Initialization parameters
    parser.add_argument('--color', type = str2bool, default = True, help = 'input type')
    parser.add_argument('--burst_length', type = int, default = 1, help = 'number of photos used in burst setting')
    parser.add_argument('--blind_est', type = str2bool, default = True, help = 'variance map')
    parser.add_argument('--kernel_size', type = list, default = [3], help = 'kernel size')
    parser.add_argument('--sep_conv', type = str2bool, default = False, help = 'simple output type')
    parser.add_argument('--channel_att', type = str2bool, default = False, help = 'channel wise attention')
    parser.add_argument('--spatial_att', type = str2bool, default = False, help = 'spatial wise attention')
    parser.add_argument('--upMode', type = str, default = 'bilinear', help = 'upMode')
    parser.add_argument('--core_bias', type = str2bool, default = False, help = 'core_bias')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = 'rainy_image_dataset/testing', help = 'images baseroot')
    parser.add_argument('--crop', type = str2bool, default = False, help = 'whether to crop input images')
    parser.add_argument('--crop_size', type = int, default = 128, help = 'single patch size')
    parser.add_argument('--geometry_aug', type = str2bool, default = False, help = 'geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type = str2bool, default = False, help = 'geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type = float, default = 1, help = 'min scaling factor')
    parser.add_argument('--scale_max', type = float, default = 1, help = 'max scaling factor')
    parser.add_argument('--add_noise', type = str2bool, default = False, help = 'whether to add noise to input images')
    parser.add_argument('--mu', type = int, default = 0, help = 'Gaussian noise mean')
    parser.add_argument('--sigma', type = int, default = 30, help = 'Gaussian noise variance: 30 | 50 | 70')
    opt = parser.parse_args()
    # print(opt)
    
#    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # ----------------------------------------
    #     自动检查并生成深度图 (Depth-Anything)
    # ----------------------------------------
    check_and_generate_depth_maps(opt.baseroot)

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    if opt.no_gpu:
        generator = utils.create_generator(opt)
    else:
        generator = utils.create_generator(opt).cuda()

    '''
    parm={}
    for name,parameters in generator.named_parameters():
        print(name,':',parameters.size())
        parm[name]=parameters.detach().cpu().numpy()
    print(parm['conv_final.weight'])
    print(parm['conv_final.bias'])
    '''

    # ============================ 导入测试图片=============================
    test_dataset = dataset.DenoisingValDataset(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    sample_folder = opt.save_name
    utils.check_path(sample_folder)
    psnr_sum, psnr_ave, ssim_sum, ssim_ave, eval_cnt = 0, 0, 0, 0, 0.1

    print(len(test_loader))
    # forward
    for i, (true_input, true_target, LQdepth, height_origin, width_origin) in enumerate(test_loader):
        # print("11111111111111111111111")

        # To device
        if opt.no_gpu:
            true_input = true_input
            true_target = true_target
            LQdepth = LQdepth   # new added==============================================
        else:
            true_input = true_input.cuda()
            true_target = true_target.cuda()
            LQdepth = LQdepth.cuda() # new added==============================================

        """
        训练过程中 网络模型生成图像需要三个参数
        fake_target = generator(true_input, true_input, LQdepth)  训练时的生成器
        
        """
        # Forward propagation
        with torch.no_grad():
            print("=======================================================================================")
            print("输入图像shape",true_input.size())

            """
             # fake_target = generator(true_input, true_input)
            # LQdepth = torch.randn(true_input.shape).cuda()     # 发现使用其作为输入桐乡可以具有很好的效果=================================未来可以进行进一步研究。。。。
            # print(LQdepth.shape)
            """
            fake_target = generator(true_input, true_input, LQdepth)



        # print(fake_target.shape, true_input.shape)

        # Save
        print('The %d-th iteration' % (i))
        img_list = [true_input, fake_target, true_target]
        name_list = ['in', 'pred', 'gt']
        sample_name = '%d' % (i+1)
        utils.save_sample_png(sample_folder = sample_folder, sample_name = '%d' % (i + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255, height = height_origin, width = width_origin)
        
        # Evaluation
        #psnr_sum = psnr_sum + utils.psnr(cv2.imread(sample_folder + '/' + sample_name + '_' + name_list[1] + '.png').astype(np.float32), cv2.imread(sample_folder + '/' + sample_name + '_' + name_list[2] + '.png').astype(np.float32))
        img_pred_recover = utils.recover_process(fake_target, height = height_origin, width = width_origin)
        img_gt_recover = utils.recover_process(true_target, height = height_origin, width = width_origin)
        #psnr_sum = psnr_sum + utils.psnr(utils.recover_process(fake_target, height = height_origin, width = width_origin), utils.recover_process(true_target, height = height_origin, width = width_origin))
        psnr_sum = psnr_sum + utils.psnr(img_pred_recover, img_gt_recover)
        # ssim_sum = ssim_sum + compare_ssim(img_gt_recover, img_pred_recover, multichannel = True, data_range = 255)
        eval_cnt = eval_cnt + 1
        
    psnr_ave = psnr_sum / eval_cnt
    ssim_ave = ssim_sum / eval_cnt
    psnr_file = "./data/psnr_data.txt"
    ssim_file = "./data/ssim_data.txt"
    psnr_content = opt.load_name + ": " + str(psnr_ave) + "\n"
    ssim_content = opt.load_name + ": " + str(ssim_ave) + "\n"
    utils.text_save(content = psnr_content, filename = psnr_file)
    utils.text_save(content = ssim_content, filename = ssim_file)
    
    
