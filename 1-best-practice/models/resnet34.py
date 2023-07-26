'''resnet34'''
# pylint disable = E0401
import mindspore.ops as F
from mindspore import nn
from .basic_module import BasicModule


class ResidualBlock(nn.Cell):
    """
    实现子module: Residual Block
    """
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super().__init__()
        self.left = nn.SequentialCell(
            nn.Conv2d(inchannel, outchannel, kernel_size=3,
                      stride=stride, padding=1, has_bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3,
                      stride=1, padding=1, has_bias=False),
            nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def construct(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(BasicModule):
    """
    实现主module：ResNet34
    ResNet34包含多个layer，每个layer又包含多个Residual block
    用子module来实现Residual block，用_make_layer函数来实现layer
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.model_name = 'resnet34'

        # 前几层: 图像转换
        self.pre = nn.SequentialCell(
            nn.Conv2d(3, 64, 7, 2, 3, has_bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="pad", padding=1))

        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self.make_layer(64, 128, 3)
        self.layer2 = self.make_layer(128, 256, 4, stride=2)
        self.layer3 = self.make_layer(256, 512, 6, stride=2)
        self.layer4 = self.make_layer(512, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Dense(512, num_classes)

    def make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        构建layer,包含多个residual block
        """
        shortcut = nn.SequentialCell(
            nn.Conv2d(inchannel, outchannel, kernel_size=1,
                      stride=stride, has_bias=False),
            nn.BatchNorm2d(outchannel))

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for _ in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.SequentialCell(*layers)

    def construct(self, x):
        """input: x"""
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)
