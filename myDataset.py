import torchtext
from torchtext.legacy import data
from torchtext.legacy.data import *
from torchtext.vocab import Vectors
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import random
import os


class Shop_Dataset(torchtext.legacy.data.Dataset):
    def __init__(self, path, text_field, label_field, test=False, aug=False, **kwargs):
        fields = [("review", text_field), ("cat_id", label_field)]

        examples = []
        csv_data = pd.read_csv(path).astype(str)
        print('read data from {}'.format(path))

        if test:
            # 如果为测试集，则不加载label
            for text in tqdm(csv_data['review']):
                examples.append(data.Example.fromlist([text, None], fields))
        else:
            for text, label in tqdm(zip(csv_data['review'], csv_data['cat_id'])):
                if aug:
                    # do augmentation
                    rate = random.random()
                    if rate > 0.5:
                        text = self.dropout(text)
                    else:
                        text = self.shuffle(text)
                examples.append(data.Example.fromlist([text, int(label)], fields))
        # 之前是一些预处理操作，此处调用super初始化父类，构造自定义的Dataset类。
        super(Shop_Dataset, self).__init__(examples, fields, **kwargs)

    def shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    def dropout(self, text, p=0.5):
        # random delete some text
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)


