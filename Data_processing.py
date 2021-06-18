from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import jieba as jb
import re


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 去除停用词
def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line


def stop_words_list(filepath):
    stop_words = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stop_words


def Shopping_data():
    # 1.读入数据, 数据清洗，处理空值
    df = pd.read_csv('./data/online_shopping_10_cats.csv')
    df = df[['cat', 'review']]
    print("数据总量: %d ." % len(df))
    df.sample(10)
    print("在 cat 列中总共有 %d 个空值." % df['cat'].isnull().sum())
    print("在 review 列中总共有 %d 个空值." % df['review'].isnull().sum())
    df[df.isnull().values == True]
    df = df[pd.notnull(df['review'])]
    d = {'cat': df['cat'].value_counts().index, 'count': df['cat'].value_counts()}
    df_cat = pd.DataFrame(data=d).reset_index(drop=True)
    print(df_cat)
    df_cat.plot(x='cat', y='count', kind='bar', legend=False, figsize=(8, 5))
    plt.ylabel('数量', fontsize=18)
    plt.xlabel('类别', fontsize=18)
    plt.show()

    # 2.数据预处理
    # step1:id表示类别
    df['cat_id'] = df['cat'].factorize()[0]
    cat_id_df = df[['cat', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
    cat_to_id = dict(cat_id_df.values)
    id_to_cat = dict(cat_id_df[['cat_id', 'cat']].values)

    # step2:清除停用词
    stopwords = stop_words_list("./data/哈工大停用词表.txt")
    df['clean_review'] = df['review'].apply(remove_punctuation)

    # step3:分词
    df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
    df.head()
    df.sample(10)
    print(df)

    # 3.划分训练集，测试集，写csv
    X_train0, X_test, y_train0, y_test = train_test_split(df['cut_review'], df['cat_id'], test_size=0.2, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train0, y_train0, test_size=0.2, random_state=0)

    dataframe1 = pd.DataFrame({'review': X_train0, 'cat_id': y_train0})
    dataframe1.to_csv("data/train.csv", index=False, sep=',')

    dataframe2 = pd.DataFrame({'review': X_test, 'cat_id': y_test})
    dataframe2.to_csv("data/test.csv", index=False, sep=',')

    dataframe3 = pd.DataFrame({'review': X_valid, 'cat_id': y_valid})
    dataframe3.to_csv("data/valid.csv", index=False, sep=',')
    return X_train0, X_valid, X_test, y_train0, y_valid, y_test


Shopping_data()
