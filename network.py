import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dwt_conv_model import Dwt_Conv
from bigff import BiGFF
from mamba_depth_encoder_fixed import SimpleMambaDepthEncoder

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#      Kernel Prediction Network (KPN)
# ----------------------------------------
class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2*out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )
        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv1(data)
        if self.channel_att:
            # fm_pool = F.adaptive_avg_pool2d(fm, (1, 1)) + F.adaptive_max_pool2d(fm, (1, 1))
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)
            fm = fm * att
        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool)
            fm = fm * att
        return fm

class KPN(nn.Module):
    def __init__(self, color=True, burst_length=1, blind_est=True, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(KPN, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length+1)
        out_channel = (3 if color else 1) * (2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length
        self.ooooo = out_channel
        if core_bias:
            out_channel += (3 if color else 1) * burst_length
        # 各个卷积层定义
        # 2~5层都是均值池化+3层卷积
        self.dwtcon = Dwt_Conv(in_channel)


        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)

        # 新的Mamba深度编码器（使用repeat方式处理通道问题）
        self.mamba_depth_encoder = SimpleMambaDepthEncoder(
            in_channel=1,  # 接受1通道输入，内部处理3通道情况
            dims=[64, 128, 256, 512, 512]
        )

        # 6~8层要先上采样再卷积
        self.conv6 = Basic(512+512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256+512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256+128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = nn.Conv2d(out_channel, out_channel, 1, 1, 0)

        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)  #4 3 224 224
        
        self.conv_final = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.bigff = BiGFF(512,512)

    # 前向传播函数
    def forward(self, data_with_est, data, LQdepth, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        data_with_est = self.dwtcon(data_with_est)

        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        # print("fogggshape",conv5.shape)  # 4x512x16x16


        # 使用Mamba深度编码器（替换原来的CNN深度编码）
        deepconv5 = self.mamba_depth_encoder(LQdepth)

        #特征融合
        # 使用 bigff进行融合
        feature_fusion = self.bigff(conv5,deepconv5)
        # print("feature_fusion_shape",feature_fusion.shape) #4x512x16x16

        # 开始上采样  同时要进行 skip connection
        # 添加数值稳定性检查
        feature_fusion_clamped = torch.clamp(feature_fusion, -10.0, 10.0)
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(feature_fusion_clamped, scale_factor=2, mode=self.upMode)], dim=1))
        
        # 检查并限制conv6的值
        if torch.isnan(conv6).any() or torch.isinf(conv6).any():
            print("Warning: Intermediate output contains nan or inf values")
            conv6 = torch.clamp(conv6, -10.0, 10.0)
            
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.upMode)], dim=1))
        
        # 检查并限制conv7的值
        if torch.isnan(conv7).any() or torch.isinf(conv7).any():
            print("Warning: Intermediate output contains nan or inf values")
            conv7 = torch.clamp(conv7, -10.0, 10.0)
            
        #print(conv7.size())
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
        # return channel K*K*N
        core = self.outc(F.interpolate(conv8, scale_factor=2, mode=self.upMode))


        pred1 = self.kernel_pred(data, core, white_level, rate=1)
        pred2 = self.kernel_pred(data, core, white_level, rate=2)
        pred3 = self.kernel_pred(data, core, white_level, rate=3)
        pred4 = self.kernel_pred(data, core, white_level, rate=4)

        pred_cat = torch.cat([torch.cat([torch.cat([pred1, pred2], dim=1), pred3], dim=1), pred4], dim=1)
        
        pred = self.conv_final(pred_cat)
        
        # pred = self.kernel_pred(data, core, white_level, rate=1)
        
        return pred



"""
这段代码定义了一个名为 KernelConv 的 PyTorch 模型类，用于执行核卷积操作，
通常用于图像处理中。以下是代码的主要作用和功能：

初始化函数：在初始化时，你可以指定以下参数：

kernel_size：一个列表，表示要使用的卷积核大小。例如，kernel_size=[3, 5] 表示使用卷积核大小为 3 和 5。
sep_conv：一个布尔值，表示是否进行分离卷积（separable convolution）。
分离卷积是一种卷积操作，它首先分别在水平和垂直方向上进行卷积，然后将结果组合起来。
core_bias：一个布尔值，表示是否包含偏差（bias）项。
_sep_conv_core 函数：这个函数将分离卷积的核转换为标准的卷积核。它将分离卷积的核分割成水平和垂直方向的卷积核，并将它们组合成标准的卷积核。
函数返回一个包含卷积核的字典和（如果启用 core_bias）一个包含偏差项的张量。

_convert_dict 函数：这个函数用于将卷积核的张量转换为字典格式。通常情况下，只有一个卷积核大小适用于该函数。
它返回一个包含卷积核的字典和（如果启用 core_bias）一个包含偏差项的张量。

forward 函数：这是前向传播函数，用于计算预测图像。它接受输入 frames（图像序列）和 core（卷积核），以及可选的 white_level 和 rate 参数。具体操作包括：

将输入的 frames 调整为适当的形状，以确保输入符合预期的尺寸和通道。
将卷积核按指定的核大小转换为字典格式。
为每个核大小执行卷积操作，生成预测图像。
将预测图像叠加起来，取平均值，并根据 core_bias 和 white_level 进行偏差校正和白平衡。
总的来说，KernelConv 类用于执行卷积操作，支持多个卷积核大小，并可以选择是否进行分离卷积和是否包含偏差项。
此类通常用于图像处理任务中，例如超分辨率重建等。
"""
class KernelConv(nn.Module):
    """
    the class of computing prediction
    """
    def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width
        :return:
        """
        kernel_total = sum(self.kernel_size)
        core = core.view(batch_size, N, -1, color, height, width)
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)
        # output core
        core_out = {}
        cur = 0
        for K in self.kernel_size:
            t1 = core_1[:, :, cur:cur + K, ...].view(batch_size, N, K, 1, 3, height, width)
            t2 = core_2[:, :, cur:cur + K, ...].view(batch_size, N, 1, K, 3, height, width)
            core_out[K] = torch.einsum('ijklno,ijlmno->ijkmno', [t1, t2]).view(batch_size, N, K * K, color, height, width)
            cur += K
        # it is a dict
        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
        core = core.view(batch_size, N, -1, color, height, width)
        core_out[self.kernel_size[0]] = core[:, :, 0:self.kernel_size[0]**2, ...]
        bias = None if not self.core_bias else core[:, :, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0, rate=1):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, 3, height, width]
        :param core: [batch_size, N, dict(kernel), 3, height, width]
        :return:
        """
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)
        img_stack = []
        pred_img = []
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):
            if not img_stack:
                padding_num = (K//2) * rate
                frame_pad = F.pad(frames, [padding_num, padding_num, padding_num, padding_num])
                for i in range(0, K):
                    for j in range(0, K):
                        img_stack.append(frame_pad[..., i*rate:i*rate + height, j*rate:j*rate + width])
                img_stack = torch.stack(img_stack, dim=2)
            else:
                k_diff = (kernel[index - 1] - kernel[index]) // 2
                img_stack = img_stack[:, :, k_diff:-k_diff, ...]
            # print('img_stack:', img_stack.size())
            pred_img.append(torch.sum(
                core[K].mul(img_stack), dim=2, keepdim=False
            ))
        pred_img = torch.stack(pred_img, dim=0)
        #print('pred_stack:', pred_img.size())
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)
        #print("pred_img_i", pred_img_i.size())
        # N = 1
        pred_img_i = pred_img_i.squeeze(2)
        #print("pred_img_i", pred_img_i.size())
        # if bias is permitted
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i += bias
        # print('white_level', white_level.size())
        pred_img_i = pred_img_i / white_level
        #pred_img = torch.mean(pred_img_i, dim=1, keepdim=True)
        # print('pred_img:', pred_img.size())
        # print('pred_img_i:', pred_img_i.size())
        return pred_img_i

class LossFunc(nn.Module):
    """
    loss function of KPN
    """
    def __init__(self, coeff_basic=1.0, coeff_anneal=1.0, gradient_L1=True, alpha=0.9998, beta=100):
        super(LossFunc, self).__init__()
        self.coeff_basic = coeff_basic
        self.coeff_anneal = coeff_anneal
        self.loss_basic = LossBasic(gradient_L1)
        self.loss_anneal = LossAnneal(alpha, beta)

    def forward(self, pred_img_i, pred_img, ground_truth, global_step):
        """
        forward function of loss_func
        :param frames: frame_1 ~ frame_N, shape: [batch, N, 3, height, width]
        :param core: a dict coverted by ......
        :param ground_truth: shape [batch, 3, height, width]
        :param global_step: int
        :return: loss
        """
        return self.coeff_basic * self.loss_basic(pred_img, ground_truth), self.coeff_anneal * self.loss_anneal(global_step, pred_img_i, ground_truth)

class LossBasic(nn.Module):
    """
    Basic loss function.
    """
    def __init__(self, gradient_L1=True):
        super(LossBasic, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gradient = TensorGradient(gradient_L1)

    def forward(self, pred, ground_truth):
        return self.l2_loss(pred, ground_truth) + \
               self.l1_loss(self.gradient(pred), self.gradient(ground_truth))

class LossAnneal(nn.Module):
    """
    anneal loss function
    """
    def __init__(self, alpha=0.9998, beta=100):
        super(LossAnneal, self).__init__()
        self.global_step = 0
        self.loss_func = LossBasic(gradient_L1=True)
        self.alpha = alpha
        self.beta = beta

    def forward(self, global_step, pred_i, ground_truth):
        """
        :param global_step: int
        :param pred_i: [batch_size, N, 3, height, width]
        :param ground_truth: [batch_size, 3, height, width]
        :return:
        """
        loss = 0
        for i in range(pred_i.size(1)):
            loss += self.loss_func(pred_i[:, i, ...], ground_truth)
        loss /= pred_i.size(1)
        return self.beta * self.alpha ** global_step * loss


"""
这段代码定义了一个名为 TensorGradient 的 PyTorch 模块，它用于计算输入张量的梯度或梯度幅值。梯度通常用于图像处理任务，
可以用于检测图像中的边缘和纹理等特征。
具体来说，这个模块的作用如下：
初始化函数 (__init__):
L1: 这个参数是一个布尔值，用于控制计算梯度幅值时是否使用 L1 范数。如果 L1 为 True，则计算 L1 范数，如果 False，则计算 L2 范数。
前向传播函数 (forward):
img: 这是输入张量，通常是一个表示图像的多维张量。
在前向传播函数中，该模块执行以下步骤：
首先，它获取输入张量的宽度 (w) 和高度 (h)。
然后，它使用 F.pad 函数对输入张量 img 进行填充，分别在左侧、右侧、上方和下方添加一行或一列的零填充，以便计算图像的边界梯度。
填充后的张量命名为 l（左侧填充）、r（右侧填充）、u（上方填充）和 d（下方填充）。
接下来，根据 L1 参数的值，它分别计算水平方向和垂直方向的梯度。如果 L1 为 True，则计算 L1 范数，即绝对值之和，否则计算 L2 范数，即平方和的平方根。
最后，返回计算出的梯度张量或梯度幅值张量。
总之，TensorGradient 模块可以用于计算图像的梯度信息，它提供了在图像处理任务中常用的梯度计算功能，可用于检测图像的边缘等特征。
根据 L1 参数的设置，它可以计算梯度的 L1 范数或 L2 范数。
"""

class TensorGradient(nn.Module):
    """
    the gradient of tensor
    """
    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if self.L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2)
            )

if __name__ == '__main__':
    
    kpn = KPN().cuda()
    a = torch.randn(4, 3, 256, 256).cuda()
    b = kpn(a, a, a)
    print(b.shape)
