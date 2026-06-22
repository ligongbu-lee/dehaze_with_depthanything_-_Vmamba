import math

import torch
import torch.nn as nn


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    #2025-5-10：删除 h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h = torch.zeros([out_batch, out_channel, out_height, out_width], device=x.device).float()#2025-5-10：增加

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)



if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms
    from matplotlib import pyplot as plt
    # 分解
    dwt_module = DWT()
    # x = Image.open('20_gt.png')
    # x=Image.open('./mountain.png')
    x = torch.randn(3,3,128,128)
    # x = transforms.ToTensor()(x)
    # x = torch.unsqueeze(x, 0)
    x = transforms.Resize(size=(128, 128))(x)
    subbands = dwt_module(x)
    print("000000000000",subbands.shape)


    # title = ['LL', 'HL', 'LH', 'HH']
    #
    # plt.figure()
    # for i in range(4):
    #     plt.subplot(2, 2, i + 1)
    #     temp = torch.permute(subbands[0, 3 * i:3 * (i + 1), :, :], dims=[1, 2, 0])
    #     plt.imshow(temp)
    #     plt.title(title[i])
    #     plt.axis('off')
    # plt.show()
    #


    # 重构

    # title = ['Original Image', 'Reconstruction Image']
    reconstruction_img = IWT()(subbands).cpu()
    print("111111111111111",reconstruction_img.shape)
    # # ssim_value = ssim(x, reconstruction_img)  # 计算原图与重构图之间的结构相似度
    # # print("SSIM Value:", ssim_value)  # tensor(1.)
    # show_list = [torch.permute(x[0], dims=[1, 2, 0]), torch.permute(reconstruction_img[0], dims=[1, 2, 0])]

    # plt.figure()
    # for i in range(2):
    #     plt.subplot(1, 2, i + 1)
    #     plt.imshow(show_list[i])
    #     plt.title(title[i])
    #     plt.axis('off')
    # plt.show()
