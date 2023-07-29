# Heads On Mindspore
    Refer to the[pytorch-book](https://github.com/chenyuntc/pytorch-book) content and [mindspore documentation](https://www.mindspore.cn/docs/zh-CN/r2.0/index.html) for code writing.

## 1-best-practice
    The project data uses the cat and dog dataset, which can be downloaded from the[Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).
    If you want to run the project, use the main file for training and testing. At the same time, the project comes with three preset models - resnet34, alexnet, squeezenet, and you can change the training and testing models by modifying the config file

## 2-AnimeGAN
    Project usage data is the avatar data crawled down from the web, and users can crawl from the web by themselves, or use the[data](https://pan.baidu.com/s/1eSifHcA?pwd=g5qa) that has been organized., which contains about 50,000 pictures, and its format is as follows:

data/
└── faces/
    ├── 0000fdee4208b8b7e12074c920bc6166-0.jpg
    ├── 0001a0fca4e9d2193afea712421693be-0.jpg
    ├── 0001d9ed32d932d298e1ff9cc5b7a2ab-0.jpg
    ├── 0001d9ed32d932d298e1ff9cc5b7a2ab-1.jpg
    ├── 00028d3882ec183e0f55ff29827527d3-0.jpg
    ├── 00028d3882ec183e0f55ff29827527d3-1.jpg
    ├── 000333906d04217408bb0d501f298448-0.jpg
    ├── 0005027ac1dcc32835a37be806f226cb-0.jpg

    If you want to run the project, use the main file for training and testing. Due to the large amount of image data, try to use the GPU for training.

## 3-neural_style
    The data used in this trial were from[COCO](http://images.cocodataset.org/zips/train2014.zip). You can also use other data, such as ImageNet. Please try to ensure the diversity of data, **It is not recommended** to use a single kind of data set, such as LSUN or face recognition data set.
    If you want to run the project, use the main file for training and testing.

## 4-neural_poet_RNN
    The data for this experiment came from[Chinese-poetry](https://github.com/chinese-poetry/chinese-poetry). But the author has processed it into a binary file 'tang.npz', which can be used directly. Readers can download 'tang.npz' [here](https://yun.sfo2.digitaloceanspaces.com/pytorch_book/pytorch_book/tang.npz).
    If you want to run the project, use the main file for training and testing. When the training is complete, you can use the gen function in the main file to generate the corresponding verse.

## 5-image_caption
    The data for this experiment comes from the[AI Challenger image description](https://challenger.ai/competition/caption/). Download the corresponding training data (ai_challenger_caption_train_20170902.zip). If you just want to test the effect, you can skip this step. Readers can also download data from [MEGA](https://mega.nz/#!fP4TSJ6I!mgG_HSXqi1Kgg5gvwYArUnuRNgcDqpd8qoj09e0Yg10).
    In the process of running the project, it is necessary to go through the two stages of data preprocessing and image feature extraction, in which data preprocessing can run data_preprocess files, and in extracting image features, resnet50 is used for feature extraction, and a CKPT file is generated in the current folder, which holds a tensor array with 210,000 image information. Fianlly, if you want to run the project, use the main file for training and testing.

## 6-speech_recognition
    In this experiment, an LSTM-CTC speech recognition acoustic model was built through mindspore. The data for the experiment is the TIMIT dataset (download the dataset by clicking[academictorrents](http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3/tech) or [luojie1987/TIMIT](http://luojie1987.com/index.php/post/110.html)). There are many other publicly available speech-related databases available for download from [Open Speech and Language Resources](http://www.openslr.org/resources.php) here.
    Corresponding usage methods:
* Open the top-level script run.sh and modify the corresponding file path (TIMIT_dir, CONF_FILE).
* Open the ctc_model_setting.conf in the conf directory to set the network structure and other settings.
* Run the top-level script, followed by a parameter stage, 0 means to start from data, 1 means to start from training, and 2 means to test directly.

## Operating environment

* python 3.8
* mindspore 2.0.0
* mindcv
* mindinsight

Note: To run the project, you need to install the dependent library first, in addition, the training visualization can refer to the [mindspore](https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.0/index.html) official website tutorial.
