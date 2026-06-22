import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_patches(x, kernel_size=3, stride=1):
    if kernel_size != 1:
        x = nn.ZeroPad2d(1)(x)  # nn.zeropad2d(1) 在四个方向分别填充1行后1列0
    x = x.permute(0, 2, 3, 1)
    x = x.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
    return x.contiguous()

"""
这段代码定义了一个名为 RAL（Region Affinity Learning）的神经网络模块，用于学习前景和背景之间的区域关联关系。以下是代码的详细解释：

构造函数 __init__：

初始化了 RAL 模块的各种参数，包括卷积核大小 kernel_size、步长 stride、扩张率 rate 以及 softmax 缩放因子 softmax_scale。
前向传播函数 forward：

输入 background 和 foreground 是背景图像和前景图像。
首先，通过插值将前景图像 foreground 缩小（按 rate 缩小），以便后续计算。
通过一系列操作，将背景图像和前景图像切分成不同的区域块，并对这些区域块进行处理。这包括将图像块规范化为单位长度（L2 范数），计算前景图像中的区域块与前景图像中每个像素的相似度（得分图），应用 softmax 函数以获取注意力图（关注前景的区域），并通过卷积操作将这些区域块合并到背景图像中。
最后，通过将所有区域块的输出拼接在一起，将得到一个整个背景图像上的区域关联映射。
这个模块的主要作用是根据前景和背景之间的区域关联，生成一个具有注意力机制的输出图像。
这种机制可以用于图像分割、目标定位等计算机视觉任务，以帮助模型更好地理解和处理前景与背景之间的关系。
"""
class RAL(nn.Module):
    '''Region affinity learning.'''

    def __init__(self, kernel_size=3, stride=1, rate=2, softmax_scale=10.):
        super(RAL, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.softmax_scale = softmax_scale

    def forward(self, background, foreground):

        # accelerated calculation
        foreground = F.interpolate(foreground, scale_factor=1. / self.rate, mode='bilinear', align_corners=True)

        foreground_size, background_size = list(foreground.size()), list(background.size())

        background_kernel_size = 2 * self.rate
        background_patches = extract_patches(background, kernel_size=background_kernel_size, stride=self.stride * self.rate)
        background_patches = background_patches.view(background_size[0], -1,
            background_size[1], background_kernel_size, background_kernel_size)
        background_patches_list = torch.split(background_patches, 1, dim=0)

        foreground_list = torch.split(foreground, 1, dim=0)
        foreground_patches = extract_patches(foreground, kernel_size=self.kernel_size, stride=self.stride)
        foreground_patches = foreground_patches.view(foreground_size[0], -1,
            foreground_size[1], self.kernel_size, self.kernel_size)
        foreground_patches_list = torch.split(foreground_patches, 1, dim=0)

        output_list = []
        padding = 0 if self.kernel_size == 1 else 1
        escape_NaN = torch.FloatTensor([1e-4])
        if torch.cuda.is_available():
            #2025-5-10：删除 device= torch.device("cuda:0")
            #2025-5-10：删除 escape_NaN = escape_NaN.to(device)
            escape_NaN = escape_NaN.to(background.device)#2025-5-10：增加

        # print("len:::::",len(foreground_list), len(foreground_patches_list), len(background_patches_list))
        # print("len:::::",foreground_list[0].shape, len(foreground_patches_list), len(background_patches_list))

        for foreground_item, foreground_patches_item, background_patches_item in zip(
            foreground_list, foreground_patches_list, background_patches_list
        ):

            foreground_patches_item = foreground_patches_item[0]
            foreground_patches_item_normed = foreground_patches_item / torch.max(
                torch.sqrt((foreground_patches_item * foreground_patches_item).sum([1, 2, 3], keepdim=True)), escape_NaN)

            score_map = F.conv2d(foreground_item, foreground_patches_item_normed, stride=1, padding=padding)
            # print(score_map.shape,"ssssssssssssssssssssssspppppppppppppppp")

            # print("stride",self.stride)
            # print("size: 2, 3 ::::",foreground_size[2],foreground_size[3])
            # print("value:  ",foreground_size[2] // self.stride * foreground_size[3] // self.stride, foreground_size[2], foreground_size[3])
            score_map = score_map.view(1, foreground_size[2] // self.stride * foreground_size[3] // self.stride,
                foreground_size[2], foreground_size[3])
            attention_map = F.softmax(score_map * self.softmax_scale, dim=1)
            attention_map = attention_map.clamp(min=1e-8)

            background_patches_item = background_patches_item[0]
            output_item = F.conv_transpose2d(attention_map, background_patches_item, stride=self.rate, padding=1) / 4.
            output_list.append(output_item)

        output = torch.cat(output_list, dim=0)
        output = output.view(background_size)
        return output


"""
这段代码定义了一个名为 MSFA（Multi-scale Feature Aggregation）的神经网络模块，用于多尺度特征的聚合。以下是代码的详细解释：

构造函数 __init__：

初始化了 MSFA 模块的各种参数和子模块。
in_channels 表示输入特征的通道数，通常为 64。
out_channels 表示输出特征的通道数，通常也为 64。
dilation_rate_list 是一个列表，包含了不同尺度的空洞卷积的扩张率（dilation rate）。
前向传播函数 forward：

输入 x 是输入特征。
MSFA 模块首先计算一个权重图 weight_map，用于确定如何聚合不同尺度的特征。这是通过一系列卷积和激活函数来实现的。
接下来，MSFA 模块遍历不同的扩张率（不同尺度的特征），并对每个扩张率应用相应的卷积操作。这些卷积操作被命名为 dilated_conv_0、dilated_conv_1、dilated_conv_2 等。
最后，根据权重图 weight_map 对不同尺度的特征进行加权求和，以产生聚合后的多尺度特征。
这个模块的主要作用是从输入特征中提取多尺度的信息，通过空洞卷积和权重映射来聚合这些尺度，以获得更丰富的特征表示。这种多尺度特征的聚合通常用于图像分割、目标检测和其他计算机视觉任务，以提高模型性能。
"""
class MSFA(nn.Module):
    '''Multi-scale feature aggregation.'''

    def __init__(self, in_channels=64, out_channels=64, dilation_rate_list=[1, 2, 4, 8]):
        super(MSFA, self).__init__()

        self.dilation_rate_list = dilation_rate_list

        for _, dilation_rate in enumerate(dilation_rate_list):

            self.__setattr__('dilated_conv_{:d}'.format(_), nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rate, padding=dilation_rate),
                nn.ReLU(inplace=True))
            )

        self.weight_calc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, len(dilation_rate_list), 1),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        weight_map = self.weight_calc(x)

        x_feature_list =[]
        for _, dilation_rate in enumerate(self.dilation_rate_list):
            x_feature_list.append(
                self.__getattr__('dilated_conv_{:d}'.format(_))(x)
            )

        output = weight_map[:, 0:1, :, :] * x_feature_list[0] + \
                 weight_map[:, 1:2, :, :] * x_feature_list[1] + \
                 weight_map[:, 2:3, :, :] * x_feature_list[2] + \
                 weight_map[:, 3:4, :, :] * x_feature_list[3]

        return output


"""
这段代码定义了一个名为 CFA（Contextual Feature Aggregation）的神经网络模块，用于将背景和前景特征进行上下文特征聚合。以下是代码的详细解释：

构造函数 __init__：

初始化了 CFA 模块的各种参数和子模块。
RAL 是上下文感知层（Contextual Aware Layer）的缩写，用于融合背景和前景特征。
MSFA 是多尺度自注意力层（Multi-Scale Self-Attention Layer）的缩写，用于引入多尺度的自注意力机制。
前向传播函数 forward：

接受两个输入，background 和 foreground，它们分别表示背景和前景特征。
首先，将这两个输入传递给 RAL 模块，以获得上下文感知的特征。RAL 模块会融合背景和前景特征以捕捉上下文信息。
输出的特征经过 RAL 后会被传递给 MSFA 模块进行多尺度自注意力操作，以更好地聚合特征信息。
最终，CFA 模块返回经过上下文特征聚合的输出特征。
这个模块的作用是将背景和前景特征进行上下文感知的聚合，以提高特征表示的能力。上下文感知有助于模型更好地理解图像内容，
并在图像分割、目标检测等任务中提高性能。整个 CFA 模块包括上下文感知层和多尺度自注意力层，用于有效融合和处理输入特征。
"""
class CFA(nn.Module):
    '''Contextual Feature Aggregation.'''

    def __init__(self,
        kernel_size=3, stride=1, rate=2, softmax_scale=10.,
        in_channels=64, out_channels=64, dilation_rate_list=[1, 2, 4, 8]):
        super(CFA, self).__init__()

        self.ral = RAL(kernel_size=kernel_size, stride=stride, rate=rate, softmax_scale=softmax_scale)
        self.msfa = MSFA(in_channels=in_channels, out_channels=out_channels, dilation_rate_list=dilation_rate_list)


    def forward(self, background, foreground):

        output = self.ral(background, foreground)
        # print("ral_output_size:   ",output.shape)
        output = self.msfa(output)
        # print("msfa_output_size:   ", output.shape)

        return output


if __name__ == "__main__":
    cfa = CFA(in_channels=3, out_channels=3).cuda()
    x1 = torch.randn(3,3,64,64).cuda()
    x2 = torch.randn(3,3,64,64).cuda()
    y = cfa(x1,x1)
    print(y.shape)