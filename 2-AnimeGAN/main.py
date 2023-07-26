'''主程序'''
import os
import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.train.summary import SummaryRecord
import mindspore.dataset as ds
import mindspore.dataset.vision as v
from mindspore import load_checkpoint, load_param_into_net, context
from mindspore.train import Loss
from tqdm import tqdm
from model import NetG, NetD
import ipdb  # 断点调试


class Config():
    '''配置类'''
    data_path = 'data/'  # 数据集存放路径
    num_workers = 4  # 多进程加载数据所用的进程数
    image_size = 96  # 图片尺寸
    batch_size = 256
    max_epoch = 200
    lr1 = 2e-4  # 生成器的学习率
    lr2 = 2e-4  # 判别器的学习率
    beta1 = 0.5  # Adam优化器的beta1参数
    gpu = True  # 是否使用GPU
    nz = 100  # 噪声维度
    ngf = 64  # 生成器feature map数
    ndf = 64  # 判别器feature map数

    save_path = 'imgs/'  # 生成图片保存路径

    vis = True  # 是否使用visdom可视化
    env = 'GAN'  # visdom的env
    plot_every = 20  # 每间隔20 batch，visdom画图一次

    debug_file = '/tmp/debuggan'  # 存在该文件则进入debug模式
    d_every = 1  # 每1个batch训练一次判别器
    g_every = 5  # 每5个batch训练一次生成器
    save_every = 10  # 没10个epoch保存一次模型
    netd_path = None  # 'checkpoints/netd_.pth' #预训练模型
    netg_path = None  # 'checkpoints/netg_211.pth'

    # 只测试不训练
    gen_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 噪声的均值
    gen_std = 1  # 噪声的方差


opt = Config()


def train(**kwargs):
    '''模型训练，参数对应配置类'''
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device_target = "GPU" if opt.gpu else "CPU"
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=device_target)

    # 数据
    trans = [
        v.Decode(),
        v.Resize(opt.image_size),
        v.CenterCrop(opt.image_size),
        v.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        v.HWC2CHW()
    ]

    dataset = ds.ImageFolderDataset(opt.data_path, shuffle=False,
                                    num_parallel_workers=opt.num_workers)
    dataset = dataset.map(operations=trans,input_columns="image",
                          num_parallel_workers=opt.num_workers)
    dataset = dataset.batch(batch_size=opt.batch_size, drop_remainder=True)
    dataloader = dataset.create_dict_iterator()

    # 网络
    netg, netd = NetG(opt), NetD(opt)
    if opt.netd_path:
        # 将模型参数存入parameter的字典中，这里加载的是上面训练过程中保存的模型参数
        param_dict = load_checkpoint(opt.netd_path)
        # 将参数加载到网络中
        load_param_into_net(netd, param_dict)
    if opt.netg_path:
        # 将模型参数存入parameter的字典中，这里加载的是上面训练过程中保存的模型参数
        param_dict = load_checkpoint(opt.netg_path)
        # 将参数加载到网络中
        load_param_into_net(netg, param_dict)

    # 定义优化器和损失
    optimizer_g = nn.Adam(netg.trainable_params(), learning_rate=opt.lr1,
                          beta1=opt.beta1, beta2=0.999)
    optimizer_d = nn.Adam(netd.trainable_params(), learning_rate=opt.lr2,
                          beta1=opt.beta1, beta2=0.999)
    criterion = nn.BCELoss()

    # 真图片label为1，假图片label为0
    # noises为生成网络的输入
    true_labels = ops.ones(opt.batch_size)
    fake_labels = ops.zeros(opt.batch_size)
    fix_noises = ops.randn(opt.batch_size, opt.nz, 1, 1)
    noises = ops.randn(opt.batch_size, opt.nz, 1, 1)

    errord_meter = Loss()
    errorg_meter = Loss()

    epochs = range(opt.max_epoch)

    # 前向传播
    def forward_fn_g(data, label):
        fake_img = netg(data)
        output = netd(fake_img)
        loss = criterion(output, label)
        return loss, output
    # 梯度函数
    grad_fn_g = mindspore.value_and_grad(
        forward_fn_g, None, optimizer_g.parameters, has_aux=True)
    # 更新，训练

    def train_step_g(data, label):
        (loss, logits), grads = grad_fn_g(data, label)
        optimizer_g(grads)
        return loss, logits

    # 前向传播
    def forward_fn_d(data, label):
        logits = netd(data)
        loss = criterion(logits, label)
        return loss, logits
    # 梯度函数
    grad_fn_d = mindspore.value_and_grad(
        forward_fn_d, None, optimizer_d.parameters, has_aux=True)
    # 更新，训练

    def train_step_d(data, label):
        (loss, logits), grads = grad_fn_d(data, label)
        optimizer_d(grads)
        return loss, logits

    # 前向传播
    def forward_fn_d1(data, label):
        fake_img = netg(data)  # 根据噪声生成假图
        output = netd(fake_img)
        loss = criterion(output, label)
        return loss, output
    # 梯度函数
    grad_fn_d1 = mindspore.value_and_grad(
        forward_fn_d1, None, optimizer_d.parameters, has_aux=True)
    # 把假图片判别为错误

    def train_step_d1(data, label):
        (loss, logits), grads = grad_fn_d1(data, label)
        optimizer_d(grads)
        return loss, logits

    # 训练
    with SummaryRecord(log_dir="./summary_dir") as summary_record:
        epoch = 0
        for epoch in iter(epochs):
            errord_meter.clear()
            errorg_meter.clear()
            netg.set_train()
            netd.set_train()
            for i, (img, _) in tqdm(enumerate(dataloader)):
                real_img = img
                if i % opt.d_every == 0:
                    # 训练判别器
                    # 尽可能的把真图片判别为正确
                    error_d_real, _ = train_step_d(real_img, true_labels)

                # 尽可能把假图片判别为错误
                noises = ops.randn(opt.batch_size, opt.nz, 1, 1).copy()
                error_d_fake = train_step_d1(noises, fake_labels)
                _ = netg(noises)  # 根据噪声生成假图

                error_d = error_d_fake + error_d_real

                errord_meter.update(error_d)

                if i % opt.g_every == 0:
                    # 训练生成器
                    noises = noises.copy()
                    noises = ops.randn(opt.batch_size, opt.nz, 1, 1).copy()
                    error_g, output = train_step_g(noises, true_labels)

                    errorg_meter.update(error_g)

                if opt.vis and i % opt.plot_every == opt.plot_every - 1:
                    # 可视化
                    if os.path.exists(opt.debug_file):
                        ipdb.set_trace()
                    fix_fake_imgs = netg(fix_noises)

                    summary_record.add_value('image', 'fake_img', fix_fake_imgs)
                    summary_record.add_value('image', 'real_img', real_img)
                    summary_record.add_value(
                        'scalar', 'errord', errord_meter.eval())
                    summary_record.add_value(
                        'scalar', 'errorg', errorg_meter.eval())
                    summary_record.record(i + 1, train_network=netg)
                    summary_record.record(i + 1, train_network=netd)

        if (epoch+1) % opt.save_every == 0:
            # 保存模型、图片
            mindspore.save_checkpoint(netd, f"./netd{epoch}.ckpt")
            mindspore.save_checkpoint(netg, f"./netg{epoch}.ckpt")
            errord_meter.clear()
            errorg_meter.clear()


if __name__ == '__main__':
    import fire
    fire.Fire()
