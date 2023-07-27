# coding:utf8
"""
利用resnet50提取图片的语义信息
并保存层results.pth
"""
import os
from config import Config
import tqdm
from PIL import Image
import mindspore
import mindspore.dataset as ds
import mindspore.ops as ops
import mindcv

opt = Config()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
normalize = ds.vision.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


class CaptionDataset():
    '''自定义数据类'''
    def __init__(self, caption_data_path):
        self.transforms = ds.transforms.Compose([
            ds.vision.Decode(),
            ds.vision.Resize(256),
            ds.vision.CenterCrop(256),
            ds.vision.ToTensor(),
            normalize
        ])

        data = mindspore.load_checkpoint(caption_data_path)
        self.ix2id = data['ix2id']
        # 所有图片的路径
        self.imgs = [os.path.join(opt.img_path, self.ix2id[_]) \
                     for _ in range(len(self.ix2id))]

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.transforms(img)
        return img, index

    def __len__(self):
        return len(self.imgs)


def get_dataloader(opt):
    '''生成数据集'''
    dataset = CaptionDataset(opt.caption_data_path)
    dataloader = ds.GeneratorDataset(dataset, num_parallel_workers=opt.num_workers, shuffle=False)
    dataloader = dataloader.batch(batch_size=opt.batch_size)
    return dataloader


# 数据
opt.batch_size = 256
dataloader = get_dataloader(opt)
cap_tensor = ops.fill(type=mindspore.float32,shape=(len(dataloader), 2048), value=0)
batch_size = opt.batch_size
results = {}

# 模型
resnet50 = mindcv.models.resnet50(pretrained=True)
del resnet50.fc
resnet50.fc = lambda x: x

# 前向传播，计算分数
for ii, (imgs, indexs) in tqdm.tqdm(enumerate(dataloader)):
    # 确保序号没有对应错
    assert indexs[0] == batch_size * ii
    imgs = imgs.cuda()
    features = resnet50(imgs)
    results[ii * batch_size:(ii + 1) * batch_size] = features.data.cpu()

# 200000*2048 20万张图片，每张图片2048维的feature
mindspore.save_checkpoint(results, 'results.ckpt')
