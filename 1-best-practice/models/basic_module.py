'''
base modle
'''
# coding:utf8
import time
import mindspore
import mindspore.nn as nn


class BasicModule(nn.Cell):
    """
    封装了nn.Cell,主要是提供了save和load两个方法
    """

    def __init__(self):
        super().__init__()
        self.model_name = str(type(self))  # 默认名字

    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(mindspore.load_checkpoint(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            # 将pth改为ckpt
            name = time.strftime(prefix + '%m%d_%H:%M:%S.ckpt')
        mindspore.save_checkpoint(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        """设置优化器"""
        return nn.Adam(self.trainable_params(), lr=lr, weight_decay=weight_decay)


class Flat(nn.Cell):
    """
    把输入reshape成（batch_size,dim_length）
    """

    def __init__(self):
        super().__init__()
        # self.size = size

    def construct(self, x):
        """input: x"""
        return x.view(x.size(0), -1)
