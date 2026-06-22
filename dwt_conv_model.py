import torch
from torch import nn
from DWT import DWT
from DWT import IWT
from cfa import CFA
from self_attention_model import Self_Attn
# 二级小波变换
# ----------------------------------------
# 根据g Multi-Attention Network with Wavelet Transform 中的处理方式对det进行改进
class Dwt_Conv(nn.Module):
    def __init__(self, in_ch):
        super(Dwt_Conv, self).__init__()
        self.dwt = DWT()
        self.idwt = IWT()
        self.cfa = CFA(in_channels=4*4 * in_ch, out_channels=4*4 * in_ch)

        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=4*in_ch, out_channels=8*in_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8*in_ch),
                nn.ReLU(),
                # nn.Conv2d(in_channels=8*in_ch, out_channels=8*in_ch, kernel_size=3, stride=1, padding=1),
                # # nn.BatchNorm2d(out_ch),
                # nn.ReLU(),
                nn.Conv2d(in_channels=8*in_ch, out_channels=4*in_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4*4 * in_ch, out_channels=4*8 * in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*8 * in_ch),
            nn.ReLU(),
            # nn.Conv2d(in_channels=8*in_ch, out_channels=8*in_ch, kernel_size=3, stride=1, padding=1),
            # # nn.BatchNorm2d(out_ch),
            # nn.ReLU(),
            nn.Conv2d(in_channels=4*8 * in_ch, out_channels=4*4 * in_ch, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        self.selfatten = Self_Attn(4*4 * in_ch)

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=4*4 * in_ch, out_channels=4*8 * in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*8 * in_ch),
            nn.ReLU(),
            # nn.Conv2d(in_channels=8*in_ch, out_channels=8*in_ch, kernel_size=3, stride=1, padding=1),
            # # nn.BatchNorm2d(out_ch),
            # nn.ReLU(),
            nn.Conv2d(in_channels=4*8 * in_ch, out_channels=4*4 * in_ch, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )


        self.conv3 =  self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=4*in_ch, out_channels=8*in_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8*in_ch),
                nn.ReLU(),
                # nn.Conv2d(in_channels=8*in_ch, out_channels=8*in_ch, kernel_size=3, stride=1, padding=1),
                # # nn.BatchNorm2d(out_ch),
                # nn.ReLU(),
                nn.Conv2d(in_channels=8*in_ch, out_channels=4*in_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )



    def forward(self, x):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        x = self.dwt(x)
        x_conv1 = self.conv1(x)
        x2 = self.dwt(x_conv1)

        x_conv2 = self.conv2(x2)

        atten_out = self.selfatten(x_conv2)

        x_conv2_2 = self.conv2_2(atten_out)
        cfa_out = self.cfa(x_conv2_2,x_conv2_2)


        # print("llllllllllllll",x_conv2.shape)


        idwt_1 = self.idwt(cfa_out)
        x_conv3 = self.conv3(idwt_1)
        # idwt_2 = self.idwt(x_conv3)

        # print(" idwt_2 ",  idwt_2.shape)

        out = self.idwt(x_conv3)

        return out


if __name__ =="__main__":
    import time
    a = torch.randn(2,3,256,256).cuda()
    t1 = time.time()
    print(a.shape)
    DC = Dwt_Conv(3).cuda()

    a2 = DC(a)

    print(a.shape)



    t2  = time.time()
    print("time: ",t2-t1)
    print("mean: ",torch.mean(torch.abs(a2-a)))