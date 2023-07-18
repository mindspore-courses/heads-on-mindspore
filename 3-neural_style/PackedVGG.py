# coding:utf8
# pylint: disable=C0103
from collections import namedtuple
import mindspore.nn as nn
from mindcv.models import create_model

class Vgg16(nn.Cell):
    '''获取每个激活函数后的输出值'''
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(create_model(model_name='vgg16', pretrained=True).features)[:23]#前23
        # the 3rd, 8th, 15th and 22nd layer of self.features are: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.CellList(features).eval()

    def construct(self, x):
        """input: x"""
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22}:
                results.append(x)
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(*results)
