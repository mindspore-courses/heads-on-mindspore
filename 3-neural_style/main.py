'''main'''
# coding:utf8
# pylint: disable=W0612
import os
import tqdm
import ipdb

import mindspore
from mindspore import load_checkpoint, load_param_into_net, context
import mindspore.dataset as dt
from mindspore import nn
import mindspore.ops as ops
from mindspore.train.summary import SummaryRecord
from mindspore.train import Loss

from transformer_net import TransformerNet
import utils
from PackedVGG import Vgg16

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Config():
    '''setting'''
    # General Args
    use_gpu = True
    model_path = None # pretrain model path (for resume training or test)

    # Train Args
    image_size = 256 # image crop_size for training
    batch_size = 8
    data_root = 'data/' # dataset root：$data_root/coco/a.jpg
    num_workers = 4 # dataloader num of workers

    lr = 1e-3
    epoches = 2 # total epoch to train
    content_weight = 1e5 # weight of content_loss
    style_weight = 1e10 # weight of style_loss

    style_path= 'style.jpg' # style image path
    env = 'neural-style' # visdom env
    plot_every = 10 # visualize in visdom for every 10 batch

    debug_file = '/tmp/debugnn' # touch $debug_fie to interrupt and enter ipdb

    # Test Args
    content_path = 'input.png' # input file to do style transfer [for test]
    result_path = 'output.png' # style transfer result [for test]


def train(**kwargs):
    '''model traing'''
    opt = Config()
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    if opt.use_gpu:
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    # vis = utils.Visualizer(opt.env)可视化

    # Data loading
    transfroms = dt.transforms.Compose([
        dt.vision.Resize(opt.image_size),
        dt.vision.CenterCrop(opt.image_size),
        dt.vision.ToTensor(),
    ])

    #读取图片文件
    dataset = dt.ImageFolderDataset(opt.data_root)
    dataset = dataset.map(operations=transfroms, input_columns=[
                          "image"])
    # dataset = dataset.apply(lambda x: x * 255)
    dataset = dataset.batch(batch_size=opt.batch_size)
    dataloader = dataset.create_dict_iterator()

    # style transformer network
    transformer = TransformerNet()
    if opt.model_path:  # 加载模型
        # 将模型参数存入parameter的字典中，这里加载的是上面训练过程中保存的模型参数
        param_dict = load_checkpoint(opt.model_path)
        # 将参数加载到网络中
        load_param_into_net(transformer, param_dict)

    # Vgg16 for Perceptual Loss
    vgg = Vgg16().eval()
    for param in vgg.get_parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = nn.Adam(transformer.trainable_params(), opt.lr)

    with SummaryRecord(log_dir="./summary_dir", network=transformer) as summary_record:
        # Get style image
        style = utils.get_style_data(opt.style_path)
        # 使用summary保存数值
        summary_record.add_value('image', 'style', (style.data[0] * 0.225 + 0.45).clamp(min=0, max=1))

        # gram matrix for style image
        features_style = vgg(style)
        gram_style = [utils.gram_matrix(y) for y in features_style]

        # Loss meter
        style_meter = Loss()  # 所有数的平均值和标准差
        content_meter = Loss()  # 所有数的平均值和标准差

        # 前向传播
        def forward_fn(x):
            y = transformer(x)
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)
            features_y = vgg(y)
            features_x = vgg(x)
            # content loss
            content_loss = opt.content_weight * ops.mse_loss(features_y.relu2_2, features_x.relu2_2)
            # style loss
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gram_y = utils.gram_matrix(ft_y)
                style_loss += ops.mse_loss(gram_y, gm_s.expand_as(gram_y))
            style_loss *= opt.style_weight

            total_loss = content_loss + style_loss
            return total_loss, [content_loss, style_loss, y]

        # 梯度函数
        grad_fn = mindspore.value_and_grad(
            forward_fn, None, optimizer.parameters, has_aux=True)

        # 更新，训练
        def train_step(data):
            (loss, y), grads = grad_fn(data)
            optimizer(grads)
            return loss, y

        for epoch in range(opt.epoches):
            content_meter.clear()
            style_meter.clear()

            for ii, (x, _) in tqdm.tqdm(enumerate(dataloader)):

                # Train
                transformer.set_train()
                total_loss, L = train_step(x)
                content_loss, style_loss, y = L[0], L[1], L[2]

                # Loss smooth for visualization
                content_meter.update(content_loss.item())
                style_meter.update(style_loss.item())

                if (ii + 1) % opt.plot_every == 0:
                    if os.path.exists(opt.debug_file):
                        ipdb.set_trace()

                    # visualization
                    summary_record.add_value('scalar', 'content_loss', content_meter.eval())
                    summary_record.add_value('scalar', 'style_loss', style_meter.eval())
                    # denorm input/output, since we have applied (utils.normalize_batch)
                    summary_record.add_value('image', 'output', (y.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                    summary_record.add_value('image', 'input', (x.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))

        # save checkpoint
        mindspore.save_checkpoint(transformer, 'checkpoints/%s_style.ckpt' % epoch)

def stylize(**kwargs):
    """
    perform style transfer
    """
    opt = Config()

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    # input image preprocess
    content_image = dt.vision.read_image(opt.content_path)
    content_transform = dt.transforms.Compose([
        dt.vision.ToTensor()
    ])
    content_image = content_image.map(operations=content_transform)
    content_image = content_image.apply(lambda x: x.mul(255))
    content_image = content_image.unsqueeze(0).detach()

    # model setup
    style_model = TransformerNet().set_train(False)
    # 将模型参数存入parameter的字典中，这里加载的是上面训练过程中保存的模型参数
    param_dict = load_checkpoint(opt.model_path)
    # 将参数加载到网络中
    load_param_into_net(style_model, param_dict)

    # style transfer and save output
    output = style_model(content_image)
    output_data = output.cpu().data[0]
    img = dt.vision.ToPIL()(((output_data / 255)).clamp(min=0, max=1))
    img.save(opt.result_path)


if __name__ == '__main__':
    import fire

    fire.Fire()
