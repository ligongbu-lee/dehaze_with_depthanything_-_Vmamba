import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import subprocess
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import trainer

# ----------------------------------------
#        深度图自动生成配置
# ----------------------------------------
DEPTH_CONFIG = {
    'base_path': '/data/skc/projects/efficientdefog_with_depth_+_biffusion/datasets/datasets_train',
    'encoder': 'vitb',  # Depth-Anything 编码器: vits, vitb, vitl
    'auto_generate': True,  # 是否自动生成缺失的深度图
}

def count_images(directory):
    """统计目录中的图像数量"""
    if not os.path.exists(directory):
        return 0
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return sum(1 for f in os.listdir(directory) 
               if os.path.splitext(f.lower())[1] in valid_extensions)

def check_and_generate_depth_maps():
    """检查深度图是否完整，不完整则自动生成"""
    if not DEPTH_CONFIG['auto_generate']:
        return
    
    base = DEPTH_CONFIG['base_path']
    paths = {
        'fog': {
            'source': os.path.join(base, 'fog', 'fog'),
            'depth': os.path.join(base, 'fog', 'depthanything'),
        },
        'gt': {
            'source': os.path.join(base, 'GT', 'gt'),
            'depth': os.path.join(base, 'GT', 'depthanything'),
        }
    }
    
    print("\n" + "=" * 60)
    print("检查 Depth-Anything 深度图...")
    print("=" * 60)
    
    need_generate = False
    for key, p in paths.items():
        name = "雾图" if key == "fog" else "原图(GT)"
        src_count = count_images(p['source'])
        depth_count = count_images(p['depth'])
        complete = depth_count >= src_count and src_count > 0
        
        status = "完成" if complete else "需要生成"
        print(f"{name}: 源图 {src_count} 张, 深度图 {depth_count} 张 [{status}]")
        
        if not complete:
            need_generate = True
    
    if need_generate:
        print("\n检测到深度图不完整，开始自动生成...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        generate_script = os.path.join(script_dir, 'generate_depth_anything.py')
        
        if os.path.exists(generate_script):
            cmd = [sys.executable, generate_script, '--encoder', DEPTH_CONFIG['encoder']]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print("警告: 深度图生成可能未完全成功")
        else:
            print(f"警告: 找不到生成脚本 {generate_script}")
            print("请先手动运行: python generate_depth_anything.py")
    else:
        print("深度图已就绪")
    
    print("=" * 60 + "\n")

def str2bool(v):
    #print(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
        
# 添加：分布式训练初始化函数
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 添加：分布式训练清理函数
def cleanup():
    dist.destroy_process_group()

# 添加：主训练函数
def main(rank, world_size, opt):
    if opt.multi_gpu:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
    
    # 开始训练
    trainer.Pre_train(opt)
    
    if opt.multi_gpu:
        cleanup()

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--yaml_path', type = str, default = 'ir-sde_train_gan.yml', help = 'saving path that is a folder')
    parser.add_argument('--save_path', type = str, default = './models_k9_loss14_ft', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 10, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 10000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    # GPU parameters
    parser.add_argument('--no_gpu', type = str2bool, default = False, help = 'True for CPU')
    parser.add_argument('--multi_gpu', type = str2bool, default = False, help = 'True for more than 1 GPU')
    #parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = str2bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 100, help = 'number of epochs of training')
    parser.add_argument('--train_batch_size', type = int, default = 16, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 0.0002, help = 'Adam: learning rate for G / D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 20, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'number of cpu threads to use during batch generation')
    # Initialization parameters
    parser.add_argument('--color', type = str2bool, default = True, help = 'input type')
    parser.add_argument('--burst_length', type = int, default = 1, help = 'number of photos used in burst setting')
    parser.add_argument('--blind_est', type = str2bool, default = True, help = 'variance map')
    parser.add_argument('--kernel_size', type = str2bool, default = [3], help = 'kernel size')
    parser.add_argument('--sep_conv', type = str2bool, default = False, help = 'simple output type')
    parser.add_argument('--channel_att', type = str2bool, default = False, help = 'channel wise attention')
    parser.add_argument('--spatial_att', type = str2bool, default = False, help = 'spatial wise attention')
    parser.add_argument('--upMode', type = str, default = 'bilinear', help = 'upMode')
    parser.add_argument('--core_bias', type = str2bool, default = False, help = 'core_bias')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = './rainy_image_dataset/training', help = 'images baseroot')
    parser.add_argument('--rainaug', type = str2bool, default = False, help = 'true for using rainaug')
    parser.add_argument('--crop_size', type = int, default = 128, help = 'single patch size')
    parser.add_argument('--geometry_aug', type = str2bool, default = False, help = 'geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type = str2bool, default = False, help = 'geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type = float, default = 1, help = 'min scaling factor')
    parser.add_argument('--scale_max', type = float, default = 1, help = 'max scaling factor')
    parser.add_argument('--mu', type = int, default = 0, help = 'Gaussian noise mean')
    parser.add_argument('--sigma', type = int, default = 30, help = 'Gaussian noise variance: 30 | 50 | 70')
    opt = parser.parse_args()
    print(opt)
    
    # 自动检查并生成深度图
    check_and_generate_depth_maps()
    
    '''
    print(opt.no_gpu)
    print(opt.no_gpu==False)
    '''
    
    ''' 
    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (opt.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('Single-GPU mode')
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    '''


    # ----------------------------------------
    #       Choose pre / continue train
    # ----------------------------------------
    # 修改：根据是否使用多GPU选择不同的训练方式
    if opt.multi_gpu:
        # 设置可见的GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        world_size = torch.cuda.device_count()
        print(f'Multi-GPU mode, using {world_size} GPUs')
        mp.spawn(main, args=(world_size, opt), nprocs=world_size, join=True)
    else:
        # 单GPU模式
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids.split(',')[0]
        print('Single-GPU mode')
        main(0, 1, opt)
    
