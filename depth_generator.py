import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from depth_model.networks import hrnet18, DepthDecoder_MSF

class DepthGenerator(nn.Module):
    def __init__(self, model_path):
        super(DepthGenerator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载编码器
        self.encoder = hrnet18(False)
        encoder_path = os.path.join(model_path, "encoder.pth")
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()
        # 加载解码器
        self.depth_decoder = DepthDecoder_MSF(num_ch_enc=self.encoder.num_ch_enc, scales=range(1))
        depth_decoder_path = os.path.join(model_path, "depth.pth")
        loaded_dict = torch.load(depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)
        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

    def forward(self, x):
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                # 支持PIL或numpy输入
                if isinstance(x, np.ndarray):
                    x = Image.fromarray(x)
                original_width, original_height = x.size
                x = x.resize((self.feed_width, self.feed_height), Image.LANCZOS)
                x = transforms.ToTensor()(x).unsqueeze(0).to(self.device)
            else:
                original_height, original_width = x.shape[2:]
                x = F.interpolate(x, (self.feed_height, self.feed_width), mode='bilinear', align_corners=False)
            features = self.encoder(x)
            outputs = self.depth_decoder(features)
            disp = outputs[("disp", 0)]
            # 恢复原图尺寸
            disp_resized = F.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
            # 视差转深度
            min_depth, max_depth = 0.1, 100
            min_disp = 1 / max_depth
            max_disp = 1 / min_depth
            scaled_disp = min_disp + (max_disp - min_disp) * disp_resized
            depth = 1 / scaled_disp
            return depth

def create_depth_generator(model_path, device='cuda'):
    """
    创建深度图生成器实例
    Args:
        model_path: 深度图模型路径
        device: 运行设备
    Returns:
        depth_generator: 深度图生成器实例
    """
    depth_generator = DepthGenerator(model_path)
    depth_generator = depth_generator.to(device)
    return depth_generator