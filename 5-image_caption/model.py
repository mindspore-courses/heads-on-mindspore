'''搭建模型'''
# coding:utf8
import time
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore
from utils.beam_search import CaptionGenerator


class CaptionModel(nn.Cell):
    '''模型'''
    def __init__(self, opt, word2ix, ix2word):
        super().__init__()
        self.ix2word = ix2word
        self.word2ix = word2ix
        self.opt = opt
        self.fc = nn.Dense(2048, opt.rnn_hidden)

        self.rnn = nn.LSTM(opt.embedding_dim, opt.rnn_hidden, num_layers=opt.num_layers)
        self.classifier = nn.Dense(opt.rnn_hidden, len(word2ix))
        self.embedding = nn.Embedding(len(word2ix), opt.embedding_dim)
        # if opt.share_embedding_weights:
        #     # rnn_hidden=embedding_dim的时候才可以
        #     self.embedding.weight

    def construct(self, img_feats, captions, lengths):
        embeddings = self.embedding(captions)
        # img_feats是2048维的向量,通过全连接层转为256维的向量,和词向量一样
        img_feats = self.fc(img_feats).unsqueeze(0)
        # 将img_feats看成第一个词的词向量 
        embeddings = ops.cat([img_feats, embeddings], 0)
        # PackedSequence
        packed_embeddings = embeddings
        outputs, state = self.rnn(packed_embeddings, seq_length = lengths)
        # lstm的输出作为特征用来分类预测下一个词的序号
        # 因为输入是PackedSequence,所以输出的output也是PackedSequence
        # PackedSequence第一个元素是Variable,第二个元素是batch_sizes,
        # 即batch中每个样本的长度
        pred = self.classifier(outputs[0])
        return pred, state

    def generate(self, img, eos_token='</EOS>',
                 beam_size=3,
                 max_caption_length=30,
                 length_normalization_factor=0.0):
        """
        根据图片生成描述,主要是使用beam search算法以得到更好的描述
        """
        cap_gen = CaptionGenerator(embedder=self.embedding,
                                   rnn=self.rnn,
                                   classifier=self.classifier,
                                   eos_id=self.word2ix[eos_token],
                                   beam_size=beam_size,
                                   max_caption_length=max_caption_length,
                                   length_normalization_factor=length_normalization_factor)
        if next(self.parameters()).is_cuda:
            img = img.cuda()
        img =img.unsqueeze(0)
        img = self.fc(img).unsqueeze(0)
        sentences, _ = cap_gen.beam_search(img)
        sentences = [' '.join([self.ix2word[idx] for idx in sent])
                     for sent in sentences]
        return sentences

    def states(self):
        '''获取数据'''
        opt_state_dict = {attr: getattr(self.opt, attr)
                          for attr in dir(self.opt)
                          if not attr.startswith('__')}
        return {
            'state_dict': self.state_dict(),
            'opt': opt_state_dict
        }

    def save(self, path=None, **kwargs):
        '''保存参数'''
        if path is None:
            path = '{prefix}_{time}'.format(prefix=self.opt.prefix,
                                            time=time.strftime('%m%d_%H%M'))
        states = self.states()
        states.update(kwargs)
        mindspore.save_checkpoint(states, path)
        return path

    def load(self, path, load_opt=False):
        '''加载对应数据集'''
        data = mindspore.load_checkpoint(path, choice_func=lambda s, l: s)
        state_dict = data['state_dict']
        mindspore.load_param_into_net(self, state_dict)

        if load_opt:
            for k, v in data['opt'].items():
                setattr(self.opt, k, v)

        return self

    def get_optimizer(self, lr):
        '''返回优化器'''
        return nn.Adam(self.trainable_params(), lr=lr)
