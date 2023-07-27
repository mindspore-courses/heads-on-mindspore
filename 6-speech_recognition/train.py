'''本文件为训练训练声学模型文件'''
#encoding=utf-8
#pylint: disable = C0325, W0401, W0614

import os
import sys
import time
import copy
import logging
from logging.handlers import RotatingFileHandler
import argparse
from random import randint
import configparser as ConfigParser

import mindspore
import mindspore.nn as nn

from model import *
from decoder import GreedyDecoder
from data import int2char, SpeechDataset, SpeechDataLoader

#支持的rnn类型
RNN = {'lstm': nn.LSTM, 'rnn':nn.RNN, 'gru':nn.GRU }

parser = argparse.ArgumentParser(description='lstm_ctc')
parser.add_argument('--conf', default='./conf/ctc_model_setting.conf' ,
                    help='conf file with Argument of LSTM and training')

def train(model, train_loader, loss_fn, optimizer, logger, print_every=20, USE_CUDA=True):
    """训练一个epoch，即将整个训练集跑一次
    Args:
        model         :  定义的网络模型
        train_loader  :  加载训练集的类对象
        loss_fn       :  损失函数，此处为CTCLoss
        optimizer     :  优化器类对象
        logger        :  日志类对象
        print_every   :  每20个batch打印一次loss
        USE_CUDA      :  是否使用GPU
    Returns:
        average_loss  :  一个epoch的平均loss
    """
    model.set_train()

    total_loss = 0
    print_loss = 0
    i = 0

    # 前向传播
    def forward_fn(inputs, input_sizes, targets, target_sizes):
        out = model(inputs)
        loss = loss_fn(out, targets, input_sizes, target_sizes)
        return loss, out

    # 梯度函数
    grad_fn = mindspore.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True)

    # 更新，训练
    def train_step(inputs, input_sizes, targets, target_sizes):
        (loss, logits), grads = grad_fn(inputs, input_sizes, targets, target_sizes)
        optimizer(grads)
        return loss, logits
    if USE_CUDA:
        mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE, device_target="GPU")
    else:
        mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE, device_target="CPU")

    for data in train_loader:
        inputs, targets, input_sizes, target_sizes, _ = data
        batch_size = inputs.size(0)
        inputs = inputs.transpose(0, 1)

        inputs = mindspore.Tensor(inputs)
        input_sizes = mindspore.Tensor(input_sizes)
        targets = mindspore.Tensor(targets)
        target_sizes = mindspore.Tensor(target_sizes)

        loss, _ = train_step(inputs, input_sizes, targets, target_sizes)
        loss /= batch_size
        print_loss += loss

        if (i + 1) % print_every == 0:
            print('batch = %d, loss = %.4f', i+1, (print_loss / print_every))
            logger.debug('batch = %d, loss = %.4f', i+1, (print_loss / print_every))
            print_loss = 0

        total_loss += loss
        i += 1
    average_loss = total_loss / i
    print("Epoch done, average loss: %.4f", average_loss)
    logger.info("Epoch done, average loss: %.4f", average_loss)
    return average_loss

def dev(model, dev_loader, loss_fn, decoder, USE_CUDA=True):
    """验证集的计算过程，与train()不同的是不需要反向传播过程，并且需要计算字符正确率
    Args:
        model       :   模型
        dev_loader  :   加载验证集的类对象
        loss_fn     :   损失函数
        decoder     :   解码类对象，即将网络的输出解码成文本
        logger      :   日志类对象
        USE_CUDA    :   是否使用GPU
    Returns:
        acc * 100    :   字符正确率，如果space不是一个标签的话，则为词正确率
        average_loss :   验证集的平均loss
    """
    model.set_train(False)
    total_cer = 0
    total_tokens = 0
    total_loss = 0
    i = 0

    if USE_CUDA:
        mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE, device_target="GPU")
    else:
        mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE, device_target="CPU")

    for data in dev_loader:
        inputs, targets, input_sizes, target_sizes, input_sizes_list = data
        batch_size = inputs.size(0)
        inputs = inputs.transpose(0, 1)

        inputs = mindspore.Tensor(inputs)
        input_sizes = mindspore.Tensor(input_sizes)
        targets = mindspore.Tensor(targets)
        target_sizes = mindspore.Tensor(target_sizes)

        out, probs = model(inputs, input_sizes, dev=True)

        loss = loss_fn(out, targets, input_sizes, target_sizes)
        loss /= batch_size
        total_loss += loss

        if decoder.space_idx == -1:
            total_cer += decoder.phone_word_error(probs, input_sizes_list, targets, target_sizes)[1]
        else:
            total_cer += decoder.phone_word_error(probs, input_sizes_list, targets, target_sizes)[0]
        total_tokens += sum(target_sizes)
        i += 1
    acc = 1 - float(total_cer) / total_tokens
    average_loss = total_loss / i
    return acc * 100, average_loss

def init_logger(log_file):
    """得到一个日志的类对象
    Args:
        log_file   :  日志文件名
    Returns:
        logger     :  日志类对象
    """

    logger = logging.getLogger()
    hdl = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=10)
    formatter=logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    hdl.setFormatter(formatter)
    logger.addHandler(hdl)
    logger.setLevel(logging.DEBUG)
    return logger

def main():
    '''主程序'''
    args = parser.parse_args()
    cf = ConfigParser.ConfigParser()
    try:
        cf.read(args.conf)
    except ValueError:
        print("conf file not exists")
        sys.exit(1)
    USE_CUDA = cf.getboolean('Training', 'use_cuda')
    try:
        seed = int(cf.get('Training', 'seed'))
    except ValueError:
        seed=randint(1,5000)
        cf.set('Training', 'seed', seed)
        cf.write(open(args.conf, 'w', encoding='utf-8'))

    mindspore.set_seed(seed)
    if USE_CUDA:
        mindspore.set_seed(seed)

    log_dir = cf.get('Data', 'log_dir')
    log_file = os.path.join(log_dir, cf.get('Data', 'log_file'))
    logger = init_logger(log_file)

    #Define Model
    rnn_input_size = cf.getint('Model', 'rnn_input_size')
    rnn_hidden_size = cf.getint('Model', 'rnn_hidden_size')
    rnn_layers = cf.getint('Model', 'rnn_layers')
    rnn_type = RNN[cf.get('Model', 'rnn_type')]
    bidirectional = cf.getboolean('Model', 'bidirectional')
    batch_norm = cf.getboolean('Model', 'batch_norm')
    rnn_param = {"rnn_input_size":rnn_input_size, "rnn_hidden_size":rnn_hidden_size, "rnn_layers":rnn_layers,
                    "rnn_type":rnn_type, "bidirectional":bidirectional, "batch_norm":batch_norm}
    num_class = cf.getint('Model', 'num_class')
    drop_out = cf.getfloat('Model', 'drop_out')

    model = CTC_Model(rnn_param=rnn_param, num_class=num_class, drop_out=drop_out)
    print("Model Structure:")
    logger.info("Model Structure:")
    for idx, m in enumerate(model.cells()):
        print(idx, m)
        logger.info("%s -> %s", idx, m)

    data_dir = cf.get('Data', 'data_dir')
    batch_size = cf.getint("Training", 'batch_size')

    #Data Loader
    train_dataset = SpeechDataset(data_dir, data_set='train')
    dev_dataset = SpeechDataset(data_dir, data_set="dev")
    train_loader = SpeechDataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                        num_workers=4, pin_memory=False)
    dev_loader = SpeechDataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=4, pin_memory=False)

    #ensure the feats is equal to the rnn_input_Size
    assert train_dataset.n_feats == rnn_input_size

    #decoder for dev set
    decoder = GreedyDecoder(int2char, space_idx=len(int2char) - 1, blank_index=0)

    #Training
    init_lr = cf.getfloat('Training', 'init_lr')
    num_epoches = cf.getint('Training', 'num_epoches')
    end_adjust_acc = cf.getfloat('Training', 'end_adjust_acc')
    decay = cf.getfloat("Training", 'lr_decay')
    weight_decay = cf.getfloat("Training", 'weight_decay')

    params = { 'num_epoches':num_epoches, 'end_adjust_acc':end_adjust_acc, 'seed':seed,
            'decay':decay, 'learning_rate':init_lr, 'weight_decay':weight_decay, 'batch_size':batch_size, 'n_feats':train_dataset.n_feats }
    print(params)


    loss_fn = nn.CTCLoss()
    optimizer = nn.Adam(model.trainable_params(), lr=init_lr, weight_decay=weight_decay)

    #visualization for training
    title = 'TIMIT LSTM_CTC Acoustic Model'

    count = 0
    learning_rate = init_lr
    loss_best = 1000
    loss_best_true = 1000
    adjust_rate_flag = False
    stop_train = False
    adjust_time = 0
    acc_best = 0
    start_time = time.time()
    loss_results = []
    dev_loss_results = []
    dev_cer_results = []

    while not stop_train:
        if count >= num_epoches:
            break
        count += 1

        if adjust_rate_flag:
            learning_rate *= decay
            adjust_rate_flag = False
            for param in optimizer.param_groups:
                param['lr'] *= decay

        print("Start training epoch: %d, learning_rate: %.5f", count, learning_rate)
        logger.info("Start training epoch: %d, learning_rate: %.5f", count, learning_rate)

        loss = train(model, train_loader, loss_fn, optimizer, logger, print_every=20, USE_CUDA=USE_CUDA)
        loss_results.append(loss)
        acc, dev_loss = dev(model, dev_loader, loss_fn, decoder, USE_CUDA=USE_CUDA)
        print("loss on dev set is %.4f", dev_loss)
        logger.info("loss on dev set is %.4f", dev_loss)
        dev_loss_results.append(dev_loss)
        dev_cer_results.append(acc)

        #adjust learning rate by dev_loss
        #adjust_rate_count  :  表示连续超过count个epoch的loss在end_adjust_acc区间内认为稳定
        if dev_loss < (loss_best - end_adjust_acc):
            loss_best = dev_loss
            loss_best_true = dev_loss
            adjust_rate_count = 0
            acc_best = acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_op_state = copy.deepcopy(optimizer.state_dict())
        elif (dev_loss < loss_best + end_adjust_acc):
            adjust_rate_count += 1
            if dev_loss < loss_best and dev_loss < loss_best_true:
                loss_best_true = dev_loss
                acc_best = acc
                best_model_state = copy.deepcopy(model.state_dict())
                best_op_state = copy.deepcopy(optimizer.state_dict())
        else:
            adjust_rate_count = 10

        print("adjust_rate_count: %d", adjust_rate_count)
        print('adjust_time: %d', adjust_time)
        logger.info("adjust_rate_count: %d", adjust_rate_count)
        logger.info('adjust_time: %d', adjust_time)

        if adjust_rate_count == 10:
            adjust_rate_flag = True
            adjust_time += 1
            adjust_rate_count = 0
            if loss_best > loss_best_true:
                loss_best = loss_best_true
            model.load_state_dict(best_model_state)
            optimizer.load_state_dict(best_op_state)

        if adjust_time == 8:
            stop_train = True

        time_used = (time.time() - start_time) / 60
        print("epoch %d done, dev acc is: %.4f, time_used: %.4f minutes",
              count, acc, time_used)
        logger.info("epoch %d done, dev acc is: %.4f, time_used: %.4f minutes",
                    count, acc, time_used)

    print("End training, best dev loss is: %.4f, acc is: %.4f", loss_best_true, acc_best)
    logger.info("End training, best dev loss acc is: %.4f, acc is: %.4f" , loss_best_true, acc_best)
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_op_state)
    best_path = os.path.join(log_dir, 'best_model'+'_dev'+str(acc_best)+'.pkl')
    cf.set('Model', 'model_file', best_path)
    cf.write(open(args.conf, 'w', encoding='utf-8'))
    params['epoch']=count

    mindspore.save_checkpoint([CTC_Model.save_package(model, optimizer=optimizer,
                                                      epoch=params, loss_results=loss_results,
                                                      dev_loss_results=dev_loss_results, dev_cer_results=dev_cer_results)], best_path)

if __name__ == '__main__':
    main()
