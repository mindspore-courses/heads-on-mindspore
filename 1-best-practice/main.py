'''The main file for the project'''
# coding:utf8
# pylint: disable=W0612
import os
import logging
import csv
from inspect import getsource
from config import opt
import models
from data.dataset import DogCat
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.summary import SummaryRecord
import mindspore.dataset as ds
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import  ConfusionMatrix, Loss
from tqdm import tqdm
import ipdb

def write_csv(results, file_name):
    '''write results in csv file'''
    with open(file_name, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)

def train(**kwargs):
    '''model traing'''
    opt._parse(kwargs)
    os.system(
        f"mindinsight start --summary-base-dir  {opt.summary_base_dir} --port=8080")

    # step1: 模型配置
    model = getattr(models, opt.model)()
    if opt.load_model_path:  # 加载模型
        # 将模型参数存入parameter的字典中，这里加载的是上面训练过程中保存的模型参数
        param_dict = load_checkpoint(opt.load_model_path)
        # 将参数加载到网络中
        load_param_into_net(model, param_dict)

    # step2: 数据集制作
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)

    train_dataloader = ds.GeneratorDataset(
        train_data, ["data", "label"], num_parallel_workers=opt.num_workers, shuffle=True)
    train_dataloader = train_dataloader.batch(batch_size=opt.batch_size)
    val_dataloader = ds.GeneratorDataset(
        val_data, ["data", "label"], num_parallel_workers=opt.num_workers, shuffle=False)
    val_dataloader = val_dataloader.batch(batch_size=opt.batch_size)

    # step3: 损失函数以及优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = model.get_optimizer(opt.lr, opt.weight_decay)  # 使用adam优化器

    # 前向传播
    def forward_fn(data, label):
        logits = model(data)
        loss = criterion(logits, label)
        return loss, logits

    # 梯度函数
    grad_fn = mindspore.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True)

    # 更新，训练
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        optimizer(grads)
        return loss, logits

    # step4: 计算损失函数的均值以及混淆矩阵
    loss_meter = Loss()  # 所有数的平均值和标准差
    confusion_matrix = ConfusionMatrix(
        num_classes=2, normalize='no_norm', threshold=0.5)
    previous_loss = 1e10

    # train
    with SummaryRecord(log_dir="./summary_dir", network=model) as summary_record:

        for epoch in range(opt.max_epoch):
            loss_meter.clear()
            confusion_matrix.clear()
            model.set_train()

            for ii, (data, label) in tqdm(enumerate(train_dataloader)):
                # 损失值以及预测值
                loss, logits = train_step(data, label)

                # 使用summary保存数值
                summary_record.add_value('image', 'image', data)

                loss_meter.update(loss.item())
                # detach 一下更安全保险
                confusion_matrix.update(logits, label)

                if (ii + 1) % opt.print_freq == 0:
                    summary_record.add_value(
                        'scalar', 'train_loss', loss)

                    summary_record.record(ii + 1, train_network=model)

                    # 进入debug模式
                    if os.path.exists(opt.debug_file):
                        # 断点调试
                        ipdb.set_trace()

            mindspore.save_checkpoint(model, "./" + opt.model + ".ckpt")

            # 验证集
            val_cm, val_accuracy = val(model, val_dataloader)

            summary_record.add_value(
                'scalar', 'eval_acc', val_accuracy)

            logging.basicConfig(f"epoch:{epoch},lr:{lr},loss:{loss_meter.eval()},train_cm:{str(confusion_matrix.eval())},val_cm:{str(val_cm.eval())}",
                level=logging.DEBUG,
                filename=opt.log_path,
                filemode='a')

            # 更新学习率
            if loss_meter.eval() > previous_loss:
                lr = lr * opt.lr_decay
                # 第二种降低学习率的方法:不会有moment等信息的丢失，动态学习率
                ###############################
                optimizer.learning_rate = lr

            previous_loss = loss_meter.eval()



def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.set_train(False)
    confusion_matrix = ConfusionMatrix(
        num_classes=2, normalize='no_norm', threshold=0.5)
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        score = model(val_input)
        confusion_matrix.update(score.squeeze(),
                                label.long())

    cm_value = confusion_matrix.eval()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def m_help():
    """
    打印帮助的信息： python file.py m_help
    """

    print(f"""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {__file__} train --env='env0701' --lr=0.01
            python {__file__} test --dataset='path/to/dataset/root/'
            python {__file__} help
    avaiable args:""")

    print(getsource(opt.__class__))


def test(**kwargs):
    '''model test'''
    opt._parse(kwargs)
    os.system(
        f"mindinsight start --summary-base-dir  {opt.summary_base_dir} --port=8080")

    # step1: 模型配置
    model = getattr(models, opt.model)()
    if opt.load_model_path:  # 加载模型
        # 将模型参数存入parameter的字典中，这里加载的是上面训练过程中保存的模型参数
        param_dict = load_checkpoint(opt.load_model_path)
        # 将参数加载到网络中
        load_param_into_net(model, param_dict)

    # step2: 数据集制作
    train_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = ds.GeneratorDataset(
        train_data, ["data", "label"], num_parallel_workers=opt.num_workers, shuffle=False)
    test_dataloader = test_dataloader.batch(batch_size=opt.batch_size)
    results = []

    model.set_train(False)
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):

        score = model(data)
        probability = ops.softmax(score, 1)[:, 0].tolist()

        batch_results = [(path_.item(), probability_)
                         for path_, probability_ in zip(path, probability)]

        results += batch_results
    write_csv(results, opt.result_file)

    return results


if __name__ == '__main__':
    # import fire
    # fire.Fire()
    train()
