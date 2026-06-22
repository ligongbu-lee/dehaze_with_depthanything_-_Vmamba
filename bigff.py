import torch
import torch.nn as nn


class BiGFF(nn.Module):
    '''Bi-directional Gated Feature Fusion.'''
    
    def __init__(self, in_channels, out_channels):
        super(BiGFF, self).__init__()

        self.structure_gate = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.texture_gate = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.structure_gamma = nn.Parameter(torch.zeros(1))
        self.texture_gamma = nn.Parameter(torch.zeros(1))
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(in_channels + in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, texture_feature, structure_feature):
        # 检查输入特征
        if torch.isnan(texture_feature).any() or torch.isinf(texture_feature).any():
            print("Warning: Texture feature contains nan or inf values")
            texture_feature = torch.clamp(texture_feature, -10.0, 10.0)
        
        if torch.isnan(structure_feature).any() or torch.isinf(structure_feature).any():
            print("Warning: Structure feature contains nan or inf values")
            structure_feature = torch.clamp(structure_feature, -10.0, 10.0)

        energy = torch.cat((texture_feature, structure_feature), dim=1)
        
        # 检查拼接后的能量特征
        if torch.isnan(energy).any() or torch.isinf(energy).any():
            print("Warning: Energy feature contains nan or inf values")
            energy = torch.clamp(energy, -10.0, 10.0)

        gate_structure_to_texture = self.structure_gate(energy)
        gate_texture_to_structure = self.texture_gate(energy)
        
        # 检查门控值
        if torch.isnan(gate_structure_to_texture).any() or torch.isinf(gate_structure_to_texture).any():
            print("Warning: Structure gate contains nan or inf values")
            gate_structure_to_texture = torch.clamp(gate_structure_to_texture, 0.0, 1.0)
            
        if torch.isnan(gate_texture_to_structure).any() or torch.isinf(gate_texture_to_structure).any():
            print("Warning: Texture gate contains nan or inf values")
            gate_texture_to_structure = torch.clamp(gate_texture_to_structure, 0.0, 1.0)

        # 限制gamma参数的范围
        texture_gamma_clamped = torch.clamp(self.texture_gamma, -1.0, 1.0)
        structure_gamma_clamped = torch.clamp(self.structure_gamma, -1.0, 1.0)
        
        texture_feature = texture_feature + texture_gamma_clamped * (gate_structure_to_texture * structure_feature)
        structure_feature = structure_feature + structure_gamma_clamped * (gate_texture_to_structure * texture_feature)
        
        # 检查融合前的特征
        if torch.isnan(texture_feature).any() or torch.isinf(texture_feature).any():
            print("Warning: Fused texture feature contains nan or inf values")
            texture_feature = torch.clamp(texture_feature, -10.0, 10.0)
            
        if torch.isnan(structure_feature).any() or torch.isinf(structure_feature).any():
            print("Warning: Fused structure feature contains nan or inf values")
            structure_feature = torch.clamp(structure_feature, -10.0, 10.0)
        
        output = self.fusion_layer(torch.cat((texture_feature, structure_feature), dim=1))
        
        # 检查最终输出
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: BiGFF output contains nan or inf values")
            output = torch.clamp(output, -10.0, 10.0)

        return output




if __name__ == "__main__":
    big = BiGFF(3,3)
    x1 = torch.randn(2,3,64,64)
    x2 = torch.randn(2,3,64,64)

    print("x1_shape:",x1.shape)
    y = big(x1,x2)
    print("outshape:",y.shape) #2x3x64x64

