'''The configuration file for the project'''
# coding:utf8
import warnings
from mindspore import context

class DefaultConfig(object):
    '''The configuration class'''
    save_model = './checkpoint'  # 模型保存路径
    summary_base_dir = "./summary_dir"
    model = 'SqueezeNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    log_path = 'train./log'  # 日志文件

    train_data_root = '1-best-practice/data/train/'  # 训练集存放路径
    test_data_root = '1-best-practice/data/test1'  # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        device_target = "GPU" if opt.use_gpu else "CPU"
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=device_target)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()
