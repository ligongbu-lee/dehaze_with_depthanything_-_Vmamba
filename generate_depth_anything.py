"""
使用 Depth-Anything 模型为训练集批量生成深度图
生成的深度图保存为 PNG 格式，文件名与原图保持一致
"""
import argparse
import cv2
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

# 添加 Depth-Anything 路径
DEPTH_ANYTHING_PATH = '/data/skc/projects/Depth-Anything'
sys.path.insert(0, DEPTH_ANYTHING_PATH)

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# 模型配置
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# 预训练权重路径
CHECKPOINT_PATH = os.path.join(DEPTH_ANYTHING_PATH, 'checkpoints')


def create_depth_anything_model(encoder='vitl', device='cuda'):
    """创建 Depth-Anything 模型（本地加载）"""
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"不支持的编码器类型: {encoder}，可选: {list(MODEL_CONFIGS.keys())}")
    
    # 保存当前工作目录
    original_cwd = os.getcwd()
    
    try:
        # 切换到 Depth-Anything 目录（因为 torchhub 路径是相对路径）
        os.chdir(DEPTH_ANYTHING_PATH)
        
        # 创建模型
        config = MODEL_CONFIGS[encoder]
        depth_anything = DepthAnything(config)
        
        # 加载预训练权重
        checkpoint_file = os.path.join(CHECKPOINT_PATH, f'depth_anything_{encoder}14.pth')
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"找不到预训练权重: {checkpoint_file}")
        
        print(f"加载预训练权重: {checkpoint_file}")
        depth_anything.load_state_dict(torch.load(checkpoint_file, map_location=device))
        depth_anything = depth_anything.to(device).eval()
        
        total_params = sum(param.numel() for param in depth_anything.parameters())
        print(f'Depth-Anything 模型参数量: {total_params / 1e6:.2f}M')
        
        return depth_anything
    finally:
        # 恢复原工作目录
        os.chdir(original_cwd)


def create_transform():
    """创建图像预处理 transform"""
    return Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])


def generate_depth_for_directory(model, transform, input_dir, output_dir, device='cuda'):
    """
    为一个目录中的所有图像生成深度图
    
    Args:
        model: Depth-Anything 模型
        transform: 图像预处理 transform
        input_dir: 输入图像目录
        output_dir: 输出深度图目录
        device: 运行设备
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    filenames = [
        f for f in os.listdir(input_dir) 
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]
    filenames.sort()
    
    print(f"\n处理目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"共 {len(filenames)} 张图像")
    
    # 统计已存在的文件
    existing_count = 0
    
    for filename in tqdm(filenames, desc="生成深度图"):
        input_path = os.path.join(input_dir, filename)
        
        # 输出文件名：保持原文件名，扩展名改为 .png
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")
        
        # 跳过已存在的文件
        if os.path.exists(output_path):
            existing_count += 1
            continue
        
        try:
            # 读取图像
            raw_image = cv2.imread(input_path)
            if raw_image is None:
                print(f"警告: 无法读取图像 {input_path}")
                continue
                
            image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
            h, w = image.shape[:2]
            
            # 预处理
            image = transform({'image': image})['image']
            image = torch.from_numpy(image).unsqueeze(0).to(device)
            
            # 推理
            with torch.no_grad():
                depth = model(image)
            
            # 后处理：恢复原始尺寸并归一化到 0-255
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.cpu().numpy().astype(np.uint8)
            
            # 保存为灰度 PNG（单通道，节省空间）
            cv2.imwrite(output_path, depth)
            
        except Exception as e:
            print(f"错误: 处理 {filename} 时出错: {e}")
            continue
    
    if existing_count > 0:
        print(f"跳过 {existing_count} 张已存在的深度图")
    print(f"完成！深度图保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='使用 Depth-Anything 批量生成深度图')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'],
                        help='编码器类型: vits(小), vitb(中), vitl(大)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备 (cuda/cpu)')
    parser.add_argument('--fog-only', action='store_true',
                        help='只处理雾图')
    parser.add_argument('--gt-only', action='store_true',
                        help='只处理原图(GT)')
    args = parser.parse_args()
    
    # 数据集路径配置
    BASE_PATH = '/data/skc/projects/efficientdefog_with_depth_+_biffusion/datasets/datasets_train'
    
    # 输入输出路径
    paths = {
        'fog': {
            'input': os.path.join(BASE_PATH, 'fog', 'fog'),
            'output': os.path.join(BASE_PATH, 'fog', 'depthanything'),
        },
        'gt': {
            'input': os.path.join(BASE_PATH, 'GT', 'gt'),
            'output': os.path.join(BASE_PATH, 'GT', 'depthanything'),
        }
    }
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("Depth-Anything 深度图批量生成工具")
    print("=" * 60)
    print(f"编码器: {args.encoder}")
    print(f"设备: {args.device}")
    
    # 创建模型和 transform
    print("\n正在加载 Depth-Anything 模型...")
    model = create_depth_anything_model(encoder=args.encoder, device=args.device)
    transform = create_transform()
    
    # 生成深度图
    if not args.gt_only:
        print("\n" + "=" * 60)
        print("处理雾图 (fog)")
        print("=" * 60)
        generate_depth_for_directory(
            model, transform,
            paths['fog']['input'],
            paths['fog']['output'],
            device=args.device
        )
    
    if not args.fog_only:
        print("\n" + "=" * 60)
        print("处理原图 (GT)")
        print("=" * 60)
        generate_depth_for_directory(
            model, transform,
            paths['gt']['input'],
            paths['gt']['output'],
            device=args.device
        )
    
    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)
    print("\n生成的深度图路径:")
    print(f"  雾图深度: {paths['fog']['output']}")
    print(f"  原图深度: {paths['gt']['output']}")
    print("\n请修改 ir-sde_train_gan.yml 中的深度图路径为 'depthanything' 目录")


if __name__ == '__main__':
    main()

