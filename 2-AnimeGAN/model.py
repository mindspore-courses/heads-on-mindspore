'''model'''
# coding:utf8
import mindspore.nn as nn


class NetG(nn.Cell):
    """
    生成器定义
    """

    def __init__(self, opt):
        super().__init__()
        ngf = opt.ngf  # 生成器feature map数

        self.main = nn.SequentialCell(
            # 输入是一个nz维度的噪声，我们可以认为它是一个1*1*nz的feature map
            nn.Conv2dTranspose(opt.nz, ngf * 8, kernel_size=4,
                               stride=1, pad_mode="pad", padding=0, has_bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),
            # 上一步的输出形状：(ngf*8) x 4 x 4

            nn.Conv2dTranspose(ngf * 8, ngf * 4, 4, 2, padding=1,
                               pad_mode="pad", has_bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            # 上一步的输出形状： (ngf*4) x 8 x 8

            nn.Conv2dTranspose(ngf * 4, ngf * 2, 4, 2, padding=1,
                               pad_mode="pad", has_bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            # 上一步的输出形状： (ngf*2) x 16 x 16

            nn.Conv2dTranspose(ngf * 2, ngf, 4, 2, padding=1,
                               pad_mode="pad", has_bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            # 上一步的输出形状：(ngf) x 32 x 32

            nn.Conv2dTranspose(
                ngf, 3, 5, 3, padding=1, pad_mode="pad", has_bias=False),
            nn.Tanh()  # 输出范围 -1~1 故而采用Tanh
            # 输出形状：3 x 96 x 96
        )

    def construct(self, x):
        """input: x"""
        return self.main(x)


class NetD(nn.Cell):
    """
    判别器定义
    """

    def __init__(self, opt):
        super().__init__()
        ndf = opt.ndf
        self.main = nn.SequentialCell(
            # 输入 3 x 96 x 96
            nn.Conv2d(3, ndf, 5, 3, padding=1, pad_mode="pad", has_bias=False),
            nn.LeakyReLU(0.2),
            # 输出 (ndf) x 32 x 32

            nn.Conv2d(ndf, ndf * 2, 4, 2, padding=1,
                      pad_mode="pad", has_bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            # 输出 (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, padding=1,
                      pad_mode="pad", has_bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
            # 输出 (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, padding=1,
                      pad_mode="pad", has_bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),
            # 输出 (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 8, 1, 4, 1, padding=0,
                      pad_mode="pad", has_bias=False),
            nn.Sigmoid()  # 输出一个数(概率)
        )

    def construct(self, x):
        """input: x"""
        return self.main(x).view(-1)
