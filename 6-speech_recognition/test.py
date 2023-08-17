'''本文件为数据集测试文件，解码类型在run.sh中定义'''
#encoding=utf-8
#pylint: disable = E1133, W0614, W0401
import time
import argparse
import configparser as ConfigParser
import mindspore

from model import *
from decoder import GreedyDecoder, BeamDecoder
from data  import int2char, SpeechDataset, SpeechDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--conf', help='conf file for training')
def test():
    '''模型测试'''
    args = parser.parse_args()
    cf = ConfigParser.ConfigParser()
    cf.read(args.conf)
    USE_CUDA = cf.getboolean('Training', 'USE_CUDA')
    model_path = cf.get('Model', 'model_file')
    data_dir = cf.get('Data', 'data_dir')
    beam_width = cf.getint('Decode', 'beam_width')
    package = mindspore.load_checkpoint(model_path)

    rnn_param = package["rnn_param"]
    num_class = package["num_class"]
    drop_out = package['_drop_out']

    decoder_type =  cf.get('Decode', 'decoder_type')
    data_set = cf.get('Decode', 'eval_dataset')

    test_dataset = SpeechDataset(data_dir, data_set=data_set)

    model = CTC_Model(rnn_param=rnn_param, num_class=num_class, drop_out=drop_out)

    test_loader = SpeechDataLoader(test_dataset, batch_size=8,
                                   shuffle=False, num_workers=4, pin_memory=False)

    model.load_state_dict(package['state_dict'])
    model.set_train()

    if decoder_type == 'Greedy':
        decoder  = GreedyDecoder(int2char, space_idx=len(int2char) - 1, blank_index = 0)
    else:
        decoder = BeamDecoder(int2char, beam_width=beam_width, blank_index = 0, space_idx = len(int2char) - 1)

    total_wer = 0
    total_cer = 0
    start = time.time()

    if USE_CUDA:
        mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE, device_target="GPU")
    else:
        mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE, device_target="CPU")
    for data in test_loader:
        inputs, target, input_sizes, target_sizes, input_size_list = data
        inputs = inputs.transpose(0,1)
        inputs = mindspore.Tensor(inputs)

        probs = model(inputs, input_sizes)

        decoded = decoder.decode(probs, input_size_list)
        targets = decoder._unflatten_targets(target, target_sizes)
        labels = decoder._process_strings(decoder._convert_to_strings(targets))

        for x, _ in enumerate(labels):
            print("origin : " + labels[x])
            print("decoded: " + decoded[x])
        cer = 0
        wer = 0
        for x, _ in enumerate(labels):
            cer += decoder.cer(decoded[x], labels[x])
            wer += decoder.wer(decoded[x], labels[x])
            decoder.num_word += len(labels[x].split())
            decoder.num_char += len(labels[x])
        total_cer += cer
        total_wer += wer
    CER = (1 - float(total_cer) / decoder.num_char)*100
    WER = (1 - float(total_wer) / decoder.num_word)*100
    print("Character error rate on test set: %.4f", CER)
    print("Word error rate on test set: %.4f", WER)
    end = time.time()
    time_used = (end - start) / 60.0
    print("time used for decode %d sentences: %.4f minutes.", len(test_dataset), time_used)

if __name__ == "__main__":
    test()
