'''处理音频和标签文件，转化为网络可输入的格式'''
#encoding=utf-8
#pylint: disable = W0612
import os
import mindspore
from mindspore import dtype as mstype
import mindspore.ops as ops
import scipy.signal
from utils import parse_audio, process_label_file

windows = {'hamming':scipy.signal.hamming, 'hann':scipy.signal.hann, 'blackman':scipy.signal.blackman,
            'bartlett':scipy.signal.bartlett}
audio_conf = {"sample_rate":16000, 'window_size':0.025, 'window_stride':0.01, 'window': 'hamming'}
int2char = ["_", "'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
            "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " "]

class SpeechDataset():
    '''自定义数据集'''
    def __init__(self, data_dir, data_set='train', normalize=True):
        self.data_set = data_set
        self.normalize = normalize
        self.char2int = {}
        self.n_feats = int(audio_conf['sample_rate']*audio_conf['window_size']/2+1)
        for i, _ in enumerate(int2char):
            self.char2int[int2char[i]] = i

        wav_path = os.path.join(data_dir, data_set+'_wav.scp')
        label_file = os.path.join(data_dir, data_set+'.text')
        self.process_audio(wav_path, label_file)

    def process_audio(self, wav_path, label_file):
        '''read the label file'''
        self.label = process_label_file(label_file, self.char2int)

        #read the path file
        self.path  = []
        with open(wav_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                utt, path = line.strip().split()
                self.path.append(path)

        #ensure the same samples of input and label
        assert len(self.label) == len(self.path)

    def __getitem__(self, idx):
        return parse_audio(self.path[idx], audio_conf, windows, normalize=self.normalize), self.label[idx]

    def __len__(self):
        return len(self.path)

def collate_fn(batch):
    '''
    将输入和标签转化为可输入网络的batch
    batch :     batch_size * (seq_len * nfeats, target_length)
    '''
    def func(p):
        return p[0].size(0)

    #sort batch according to the frame nums
    batch = sorted(batch, reverse=True, key=func)
    longest_sample = batch[0][0]
    feat_size = longest_sample.size(1)
    max_length = longest_sample.size(0)
    batch_size = len(batch)

    inputs = ops.zeros((batch_size, max_length, feat_size))   #网络输入,相当于长度不等的补0
    input_sizes = mindspore.Tensor(batch_size, dtype=mstype.int32)               #输入每个样本的序列长度，即帧数
    target_sizes = mindspore.Tensor(batch_size, dtype=mstype.int32)                #每句标签的长度
    targets = []
    input_size_list = []

    for x in range(batch_size):
        sample = batch[x]
        feature = sample[0]
        label = sample[1]
        seq_length = feature.size(0)
        inputs[x].narrow(0, 0, seq_length).copy_(feature)
        input_sizes[x] = seq_length
        input_size_list.append(seq_length)
        target_sizes[x] = len(label)
        targets.extend(label)
    targets = mindspore.Tensor(targets, dtype=mstype.int32)
    return inputs, targets, input_sizes, target_sizes, input_size_list

class SpeechDataLoader():
    '''数据加载'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = collate_fn
