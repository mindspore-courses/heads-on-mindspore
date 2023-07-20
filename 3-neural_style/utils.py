'''utils'''
# coding:utf8
# pylint: disable=C0103
# from itertools import chain
import time
import numpy as np
import mindspore.dataset as dt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def gram_matrix(y):
    """
    Input shape: b,c,h,w
    Output shape: b,c,c
    """
    (b, ch, h, w) = y.shape()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def get_style_data(path):
    """
    load style image，
    Return： tensor shape 1*c*h*w, normalized
    """
    style_transform = dt.transforms.Compose([
        dt.vision.ToTensor(),
        dt.vision.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    style_image = dt.vision.read_image(path)
    style_tensor = style_image.map(operations=style_transform)
    return style_tensor.unsqueeze(0)


def normalize_batch(batch):
    """
    Input: b,ch,h,w  0~255
    Output: b,ch,h,w  -2~2
    """
    mean = batch.data.new(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = batch.data.new(IMAGENET_STD).view(1, -1, 1, 1)
    mean = (mean.expand_as(batch.data))
    std = (std.expand_as(batch.data))
    return (batch / 255.0 - mean) / std
