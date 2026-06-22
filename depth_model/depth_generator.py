import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthGenerator(nn.Module):
    def __init__(self, pretrained_path=None):
        super(DepthGenerator, self).__init__()
        # 这里定义您的深度图生成网络结构
        # 示例结构，您需要根据实际的预训练模型结构进行修改
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        if pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def load_pretrained(self, pretrained_path):
        """加载预训练模型"""
        state_dict = torch.load(pretrained_path)
        self.load_state_dict(state_dict)
        print(f"Successfully loaded pretrained model from {pretrained_path}")
    
    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth

def create_depth_generator(pretrained_path=None):
    """创建深度图生成器实例"""
    model = DepthGenerator(pretrained_path)
    return model 