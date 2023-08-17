'''主程序'''
# coding:utf8
import os
import tqdm
import ipdb

import mindspore.dataset as ds
from mindspore import context, load_checkpoint, value_and_grad, save_checkpoint
from mindspore.train import Loss
from mindspore.train.summary import SummaryRecord
import mindspore.nn as nn
import mindcv

from model import CaptionModel
from config import Config
from data import get_dataloader
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def generate(**kwargs):
    '''生成结果'''
    opt = Config()
    for k, v in kwargs.items():
        setattr(opt, k, v)
    if opt.use_gpu:
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    # 数据预处理
    data = load_checkpoint(opt.caption_data_path, choice_func=lambda s, l:s)
    word2ix, ix2word = data['word2ix'], data['ix2word']

    normalize = ds.vision.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    transforms = ds.transforms.Compose([
        ds.vision.Decode(),
        ds.vision.Resize(opt.scale_size),
        ds.vision.CenterCrop(opt.img_size),
        ds.vision.ToTensor(),
        normalize
    ])
    img = Image.open(opt.test_img)
    img = transforms(img).unsqueeze(0)

    # 用resnet50来提取图片特征
    resnet50 = mindcv.models.resnet50(True).set_train(False)
    del resnet50.fc
    resnet50.fc = lambda x: x
    img_feats = resnet50(img).detach()

    # Caption模型
    model = CaptionModel(opt, word2ix, ix2word)
    model = model.load(opt.model_ckpt).set_train(False)

    results = model.generate(img_feats.data[0])
    print('\r\n'.join(results))


def train(**kwargs):
    '''训练'''
    opt = Config()
    for k, v in kwargs.items():
        setattr(opt, k, v)
    if opt.use_gpu:
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    opt.caption_data_path = 'caption.pth'  # 原始数据
    opt.test_img = ''  # 输入图片
    # opt.model_ckpt='caption_0914_1947' # 预训练的模型

    # 数据
    dataloader = get_dataloader(opt)
    _data = dataloader.dataset._data
    word2ix, ix2word = _data['word2ix'], _data['ix2word']

    # 模型
    model = CaptionModel(opt, word2ix, ix2word)
    if opt.model_ckpt:
        model.load(opt.model_ckpt)
    optimizer = model.get_optimizer(opt.lr)
    criterion = nn.CrossEntropyLoss()

    # 统计
    loss_meter = Loss()

    # 前向传播
    def forward_fn(imgs, captions, lengths):
        input_captions = captions[:-1]
        target_captions = captions[0]#pack_padded_sequence(captions, lengths)[0]
        score, _ = model(imgs, input_captions, lengths)
        loss = criterion(score, target_captions)
        return loss, score

    # 梯度函数
    grad_fn = value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True)

    # 更新，训练
    def train_step(imgs, captions, lengths):
        (loss, y), grads = grad_fn(imgs, captions, lengths)
        optimizer(grads)
        return loss, y

    with SummaryRecord(log_dir="./summary_dir", network=model) as summary_record:
        for epoch in range(opt.epoch):
            loss_meter.clear()
            for ii, (imgs, (captions, lengths), indexes) in tqdm.tqdm(enumerate(dataloader)):
                # 训练
                loss, _ = train_step(imgs, captions, lengths)

                loss_meter.update(loss.item())

                # 可视化
                if (ii + 1) % opt.plot_every == 0:
                    if os.path.exists(opt.debug_file):
                        ipdb.set_trace()

                    summary_record.add_value('scalar', 'loss', loss_meter.eval())

                    # 可视化原始图片 + 可视化人工的描述语句
                    raw_img = _data['ix2id'][indexes[0]]
                    img_path = opt.img_path + raw_img
                    raw_img = Image.open(img_path).convert('RGB')
                    raw_img = ds.vision.ToTensor()(raw_img)

                    raw_caption = captions.data[:, 0]
                    raw_caption = ''.join([_data['ix2word'][ii] for ii in raw_caption])
                    with open('raw_caption','a') as f:
                        f.write(raw_caption)
                    summary_record.add_value('image', 'raw', raw_img)
                    # vis.img('raw', raw_img, caption=raw_caption)

                    # 可视化网络生成的描述语句
                    results = model.generate(imgs.data[0])
                    with open('caption','a') as f:
                        test = '</br>'.join(results)
                        f.write(test)
            save_checkpoint(model, "./" + str(epoch) + ".ckpt")


if __name__ == '__main__':
    import fire

    fire.Fire()
