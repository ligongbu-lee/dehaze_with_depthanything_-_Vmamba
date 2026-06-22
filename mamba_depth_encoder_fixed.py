import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial

# 从WDmamba导入Mamba相关组件
import sys
import os

# 添加WDmamba路径（现在在同一个文件夹中）
wdmamba_path = os.path.join(os.path.dirname(__file__), 'WDMamba')
if wdmamba_path not in sys.path:
    sys.path.append(wdmamba_path)

try:
    from basicsr.archs.wavemamba_arch import MambaBlock, LFSSBlock, SS2D
    print("Successfully imported Mamba components from WDmamba")
except ImportError:
    print("Warning: Could not import from wavemamba_arch. Using CNN fallback.")
    # 定义基本的MambaBlock作为fallback
    class MambaBlock(nn.Module):
        def __init__(self, dim, n_l_blocks=1, expand=2):
            super().__init__()
            self.l_blk = nn.Sequential(*[nn.Conv2d(dim, dim, 3, 1, 1) for _ in range(n_l_blocks)])
        
        def forward(self, x):
            return self.l_blk(x)

class SimpleMambaDepthEncoder(nn.Module):
    """简化版Mamba深度编码器，使用repeat操作处理通道问题"""
    def __init__(self, in_channel=1, dims=[64, 128, 256, 512, 512]):
        super().__init__()
        
        # 初始特征提取（接受1通道输入）
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channel, dims[0], 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dims[0], dims[0], 3, 1, 1),
            nn.ReLU()
        )
        
        # 简化的编码层
        self.encoder_layers = nn.ModuleList()
        for i in range(len(dims)-1):
            layer = nn.Sequential(
                # 下采样
                nn.Conv2d(dims[i], dims[i+1], 2, 2),
                nn.BatchNorm2d(dims[i+1]),  # 添加BatchNorm
                nn.ReLU(),
                # Mamba处理
                MambaBlock(dims[i+1], n_l_blocks=1, expand=2),
                nn.BatchNorm2d(dims[i+1]),  # 添加BatchNorm
                nn.ReLU()
            )
            self.encoder_layers.append(layer)
        
    def forward(self, depth_map):
        """
        深度图编码
        Args:
            depth_map: [B, 1, H, W] 或 [B, 3, H, W] 深度图
        Returns:
            features: [B, 512, H//16, W//16] 编码后的特征
        """
        # 处理通道数：如果是3通道，取第一个通道；如果是1通道，直接使用
        if depth_map.shape[1] == 3:
            depth_map = depth_map[:, 0:1, :, :]  # 取第一个通道 [B, 1, H, W]
        
        # 检查输入是否包含nan或inf
        if torch.isnan(depth_map).any() or torch.isinf(depth_map).any():
            print("Warning: Input depth_map contains nan or inf values")
            depth_map = torch.clamp(depth_map, -10.0, 10.0)
        
        # 初始特征提取
        x = self.initial_conv(depth_map)  # [B, 64, H, W]
        
        # 逐层编码
        for layer in self.encoder_layers:
            x = layer(x)
            # 检查中间输出是否包含nan或inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("Warning: Intermediate output contains nan or inf values")
                x = torch.clamp(x, -10.0, 10.0)
        
        return x  # [B, 512, H//16, W//16]

class MambaDepthBlock(nn.Module):
    """基于Mamba的深度图处理块"""
    def __init__(self, in_dim, out_dim, n_l_blocks=1, expand=2):
        super().__init__()
        
        # 特征投影
        self.proj = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        
        # Mamba块
        self.mamba_block = MambaBlock(out_dim, n_l_blocks=n_l_blocks, expand=expand)
        
        # 残差缩放
        self.residual_scale = nn.Parameter(torch.ones(out_dim))
        
    def forward(self, x):
        residual = self.proj(x)
        
        # Mamba全局建模
        x_mamba = self.mamba_block(residual)
        
        # 残差连接（确保维度匹配）
        if residual.shape == x_mamba.shape:
            output = residual * self.residual_scale + x_mamba
        else:
            # 如果维度不匹配，只使用Mamba输出
            output = x_mamba
        
        return output

class MambaDepthEncoder(nn.Module):
    """基于Mamba的深度图编码器"""
    def __init__(self, in_channel=1, dims=[64, 128, 256, 512, 512]):
        super().__init__()
        
        # 初始特征提取
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channel, dims[0], 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dims[0], dims[0], 3, 1, 1),
            nn.ReLU()
        )
        
        # Mamba编码层
        self.mamba_layers = nn.ModuleList([
            MambaDepthBlock(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])
        
        # 下采样层
        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(dims[i+1], dims[i+1], 2, 2) for i in range(len(dims)-1)
        ])
        
    def forward(self, depth_map):
        """
        深度图编码
        Args:
            depth_map: [B, 3, H, W] 深度图（3通道，由数据集repeat处理）
        Returns:
            features: [B, 512, H//16, W//16] 编码后的特征
        """
        # 初始特征提取
        x = self.initial_conv(depth_map)  # [B, 64, H, W]
        
        # 逐层Mamba编码
        for i, (mamba_layer, downsample) in enumerate(zip(self.mamba_layers, self.downsample_layers)):
            # Mamba全局建模
            x = mamba_layer(x)
            
            # 下采样
            if i < len(self.downsample_layers) - 1:  # 最后一层不下采样
                x = downsample(x)
        
        return x  # [B, 512, H//16, W//16]

class EnhancedMambaDepthEncoder(nn.Module):
    """增强版Mamba深度编码器，包含更多优化"""
    def __init__(self, in_channel=1, dims=[64, 128, 256, 512, 512], n_l_blocks=[1, 1, 2, 2, 4]):
        super().__init__()
        
        # 初始特征提取
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channel, dims[0], 3, 1, 1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(),
            nn.Conv2d(dims[0], dims[0], 3, 1, 1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU()
        )
        
        # Mamba编码层（不同层使用不同的Mamba块数量）
        self.mamba_layers = nn.ModuleList([
            MambaDepthBlock(dims[i], dims[i+1], n_l_blocks=n_l_blocks[i]) 
            for i in range(len(dims)-1)
        ])
        
        # 下采样层
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dims[i], dims[i+1], 2, 2),
                nn.BatchNorm2d(dims[i+1]),
                nn.ReLU()
            ) for i in range(len(dims)-1)
        ])
        
        # 特征融合层
        self.feature_fusion = nn.ModuleList([
            nn.Conv2d(dims[i], dims[i], 1) for i in range(len(dims))
        ])
        
    def forward(self, depth_map):
        """
        深度图编码
        Args:
            depth_map: [B, 3, H, W] 深度图（3通道，由数据集repeat处理）
        Returns:
            features: [B, 512, H//16, W//16] 编码后的特征
        """
        # 初始特征提取
        x = self.initial_conv(depth_map)  # [B, 64, H, W]
        
        # 逐层Mamba编码
        for i, (mamba_layer, downsample) in enumerate(zip(self.mamba_layers, self.downsample_layers)):
            # Mamba全局建模
            x = mamba_layer(x)
            
            # 特征融合
            x = self.feature_fusion[i](x)
            
            # 下采样
            if i < len(self.downsample_layers) - 1:  # 最后一层不下采样
                x = downsample(x)
        
        return x  # [B, 512, H//16, W//16]

if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    encoder = SimpleMambaDepthEncoder(in_channel=1, dims=[64, 128, 256, 512, 512])
    encoder = encoder.to(device)
    
    # 测试输入（1通道和3通道都测试）
    depth_map_1ch = torch.randn(2, 1, 256, 256).to(device)
    depth_map_3ch = torch.randn(2, 3, 256, 256).to(device)
    
    # 前向传播
    with torch.no_grad():
        # 测试1通道输入
        output_1ch = encoder(depth_map_1ch)
        print(f"1通道输入形状: {depth_map_1ch.shape}")
        print(f"1通道输出形状: {output_1ch.shape}")
        
        # 测试3通道输入
        output_3ch = encoder(depth_map_3ch)
        print(f"3通道输入形状: {depth_map_3ch.shape}")
        print(f"3通道输出形状: {output_3ch.shape}")
        
        print(f"模型参数量: {sum(p.numel() for p in encoder.parameters()):,}")
        print("✓ 支持1通道和3通道输入的Mamba深度编码器测试通过")








