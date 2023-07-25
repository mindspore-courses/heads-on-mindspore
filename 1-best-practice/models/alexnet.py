'''alexnet'''
# coding:utf8
import mindspore.nn as nn
from .basic_module import BasicModule


class AlexNet(BasicModule):
    """
    code from torchvision/models/alexnet.py
    结构参考 <https://arxiv.org/abs/1404.5997>
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.model_name = 'alexnet'

        self.features = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=11, stride=4,
                      padding=2, has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2, has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.SequentialCell(
            nn.Dropout(),
            nn.Dense(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dense(4096, num_classes),
        )

    def construct(self, x):
        """input: x"""
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
