import time
from sched import scheduler

import torch
from torchtext.legacy import data
from torchtext.legacy.data import BucketIterator, Iterator
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from myDataset import Shop_Dataset


train_path = 'data/train.csv'
valid_path = 'data/valid.csv'
test_path = "data/test.csv"

tokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, fix_length=200)
LABEL = data.Field(sequential=False, use_vocab=False)


def data_iter():
    # 1. load data
    # step1:自定义dataset

    train = Shop_Dataset(train_path, text_field=TEXT, label_field=LABEL, test=False, aug=1)
    valid = Shop_Dataset(valid_path, text_field=TEXT, label_field=LABEL, test=False, aug=1)
    test = Shop_Dataset(test_path, text_field=TEXT, label_field=None, test=True, aug=1)

    # step2:迭代器
    TEXT.build_vocab(train)
    weight_matrix = TEXT.vocab.vectors
    # 若只针对训练集构造迭代器
    # train_iter = data.BucketIterator(dataset=train, batch_size=8, shuffle=True, sort_within_batch=False, repeat=False)
    train_iter, val_iter = BucketIterator.splits(
        (train, valid),
        batch_sizes=(8, 8),
        device=0,
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_key=lambda x: len(x.comment_text),
        sort_within_batch=False,
        repeat=False
    )
    test_iter = Iterator(test, batch_size=8, device=-1, sort=False, sort_within_batch=False, repeat=False)
    return train_iter, val_iter, test_iter, weight_matrix


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.word_embedding = nn.Embedding(len(TEXT.vocab), 300)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1)
        self.decoder = nn.Linear(128, 10)

    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        # print(embeds.shape)
        lstm_out = self.lstm(embeds)[0]
        # print(lstm_out.shape)
        final = lstm_out[-1]
        y = self.decoder(final)
        return y


def fit(epoch, model, trainloader, testloder):

    # 训练
    correct = 0
    total = 0
    running_loss = 0
    model.train()
    # 注意这里与前几次图片分类的不一样
    # 返回的是一个批次成对数据
    for b in trainloader:
        x, y = b.review, b.cat_id
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)

        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

    epoch_loss = running_loss/len(trainloader.dataset)
    epoch_acc = correct/total

    # 测试
    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        # 这里也是同样变化
        for b in testloder:
            x, y = b.review, b.cat_id
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloder.dataset)
    epoch_test_acc = test_correct / test_total

    print('epoch: ',epoch,
          'train_loss: ',round(epoch_loss,3),
          'train_accuracy: ',round(epoch_acc,3),
          'test_loss: ',round(epoch_test_loss,3),
          'test_accuracy: ',round(epoch_test_acc,3)
              )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc


train_iter, val_iter, test_iter, weight_matrix = data_iter()

# 训练
device = torch.device("cpu")
model = LSTM()
epochs = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
best_valid_acc = float('-inf')
model_save_path = './LSTM_model.pkl'

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch, model, train_iter, val_iter)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)




