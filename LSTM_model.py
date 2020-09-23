# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/8 14:58
# @Author : Zoe
# @File : LSTM_model.py
# @Software : PyCharm

import multiprocessing
import jieba
import pandas as pd
import re
import numpy as np
from gensim.models.word2vec import Word2Vec
import xlwt
import yaml
import json
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation, Bidirectional, SpatialDropout1D
import keras.utils
from keras.preprocessing import sequence
from  sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.models import load_model

from gensim.models.word2vec import Word2Vec

# 将csv表中不规则的数据化为规则的xls表，方便下一步操作
# def getLabel(x):
#     line = x.strip()
#     line = re.sub(".*\\t",' ',line)
#     # print(line)
#     return line
#
# def getData(x):
#     line = x.strip()
#     line = re.sub("\\t.*",' ',line)
#     # print(line)
#     return line
#
# data = pd.read_csv('data3/data.csv',names=['data'])
# # print(data['data'])
# data['labels'] = data['data'].apply(lambda x: getLabel(x))
# data['datas'] = data['data'].apply(lambda x: getData(x))
#
# print(data['datas'][0])
# excel = xlwt.Workbook(encoding="utf-8")
# sheet = excel.add_sheet("sheet1")
# sheet.write(0, 0, "data")
# sheet.write(0, 1, "label")
# for i in range(len(data['datas'])):
#     print(i)
#     sheet.write(i + 1, 0, data['datas'][i])
#     sheet.write(i + 1, 1, data['labels'][i])
# excel.save('data3/ddd.xls')
#

#***********************************************数据预处理***************************************************************
#*********************************************清洗/分词/去停用词*********************************************************

def getClear(x):        # 去非中文符号
    line = x.strip()
    # 去除文本中的英文和数字
    line = re.sub("[a-zA-Z0-9]", "", line)
    # 去除文本中的中文符号和英文符号
    line = re.sub("[^\u4e00-\u9fff]", "", line)
    return line

def afterStopwords(x,stopword):     # 去停用词
    for i in x:
        if i in stopword or i == ' ':
            x.remove(i)
    return x

def getGoodData(file_name,stopword):
    data = pd.read_excel('eight_classes_data/'+file_name+'.xls',names=['data'])
    data['clearwords'] = data['data'].apply(lambda x: getClear(x))
    data['words'] = data['clearwords'].apply(lambda x: jieba.lcut(x))
    data['final_words'] = data['words'].apply(lambda x: afterStopwords(x, stopword))

    np.save('eight_classes_data/'+file_name+'.npy', data['final_words'])

def getData():
    anger = np.load('eight_classes_data/anger.npy', allow_pickle=True)
    anger2 = np.load('eight_classes_data/2000-fennu.npy', allow_pickle=True)
    disgust = np.load('eight_classes_data/disgust.npy', allow_pickle=True)
    disgust2 = np.load('eight_classes_data/2000-yanwu.npy', allow_pickle=True)
    fear = np.load('eight_classes_data/fear.npy', allow_pickle=True)
    happiness = np.load('eight_classes_data/happiness.npy', allow_pickle=True)
    happiness2 = np.load('eight_classes_data/2000-xiyue.npy', allow_pickle=True)
    like = np.load('eight_classes_data/like.npy', allow_pickle=True)
    none = np.load('eight_classes_data/none_less.npy', allow_pickle=True)
    sadness = np.load('eight_classes_data/sadness.npy', allow_pickle=True)
    sadness2 = np.load('eight_classes_data/2000-yanwu.npy', allow_pickle=True)
    surprise = np.load('eight_classes_data/surprise.npy', allow_pickle=True)

    X_Vec = np.concatenate((anger,anger2,disgust,disgust2,fear,happiness,happiness2,like,none,sadness,sadness2,surprise))
    # X_Vec = np.concatenate(
    #     (anger, anger2, disgust, disgust2, happiness, happiness2, like, none, sadness, sadness2))
    # y = np.concatenate((np.zeros(len(anger) + len(anger2), dtype=int),
    #                     np.ones(len(disgust) + len(disgust2), dtype=int),
    #                     2 * np.ones(len(happiness) + len(happiness2), dtype=int),
    #                     3 * np.ones(len(like), dtype=int),
    #                     4 * np.ones(len(none), dtype=int),
    #                     5 * np.ones(len(sadness) + len(sadness2), dtype=int),))

    y = np.concatenate((np.zeros(len(anger) + len(anger2), dtype=int),
                        np.ones(len(disgust) + len(disgust2), dtype=int),
                        2 * np.ones(len(fear), dtype=int),
                        3 * np.ones(len(happiness)+len(happiness2), dtype=int),
                        4 * np.ones(len(like), dtype=int),
                        5 * np.ones(len(none), dtype=int),
                        6 * np.ones(len(sadness) + len(sadness2), dtype=int),
                        7 * np.ones(len(surprise), dtype=int),))
    print(len(anger) + len(anger2))
    print(len(disgust) + len(disgust2))
    print(len(fear))
    print(len(happiness)+len(happiness2))
    print(len(like))
    print(len(none))
    print(len(sadness) + len(sadness2))
    print(len(surprise))
    np.random.seed(169)
    np.random.shuffle(X_Vec)
    np.random.seed(169)
    np.random.shuffle(y)
    print(y)
    return X_Vec,y

#****************************************************word2vec***********************************************************
#**************************************************训练词向量模型********************************************************

def word2vec_train(X_Vec):
    # model = Word2Vec(size=150, # 维度
    #                       min_count=5,  #单词出现频率数
    #                       window=7, #窗口数
    #                       workers=multiprocessing.cpu_count(),
    #                       iter=100)
    # model.build_vocab(X_Vec)
    # model.train(X_Vec, total_examples=model.corpus_count, epochs=model.iter)
    # model.save('word2vec/eight_classes_Word2Vec_model-more.pkl')
    model = Word2Vec.load('word2vec/eight_classes_Word2Vec_model-more.pkl')

    print(len(model.wv.vocab.keys()))   #词向量中一共有10751个词
    input_dim = len(model.wv.vocab.keys()) + 1  # 频数小于阈值的词语统统放一起，编码为0

    #词嵌入有两种方式：一是单词嵌入方法（如Word2Vec）、二是神经网络中的embedding层（如keras的embedding层）
    # 其实二者的目标是一样的，都是我们为了学到词的稠密的嵌入表示。只不过学习的方式不一样。
    # word2vec是无监督的学习方式，利用上下文环境来学习词的嵌入表示，因此可以学到相关词；
    # 而在keras的embedding层中，权重的更新是基于标签的信息进行学习，为了达到较高的监督学习的效果，它本身也可能会学到相关词。

    #使用word2vec模型初始化embedding层的权重  这里没用初始化，直接设置为0了
    embedding_weights = np.zeros((input_dim, 150))      #这里的数字是维度  形成了一个9669*150维的填充值为0 的矩阵
    w2dic = {}          #字典，词向量中所有词组成了一个字典，与数字对应
    for i in range(len(model.wv.vocab.keys())):
        embedding_weights[i + 1, :] = model[list(model.wv.vocab.keys())[i]]
        w2dic[list(model.wv.vocab.keys())[i]] = i + 1
    with open("word2vec/ccc-8-word2vec-more.json", "w", encoding="utf-8") as f:
        json.dump(w2dic, f)
    return input_dim, embedding_weights, w2dic


def data2inx(w2indx,X_Vec):  # 将一句话变为数字的组合，如[286,86,6,2205,0]
    data = []
    for sentence in X_Vec:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)

        data.append(new_txt)
    return data

#**********************************************  L  S  T  M  ***********************************************************
#***********************************************************************************************************************

lstm_input = 150#lstm输入维度
voc_dim = 150 #word的向量维度
epoch_time = 20#epoch   epoch越大，迭代次数越多
batch_size = 128 #batch Batch Size就是每个batch中训练样本的数量

def bi_lstm(input_dim, embedding_weights):
    model = Sequential()
    model.add(Embedding(output_dim=voc_dim,
                        input_dim=input_dim,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=lstm_input))
    model.add(Dropout(0.3))

    model.add(Bidirectional(LSTM(100,activation='tanh',kernel_regularizer=keras.regularizers.l2(0.001))))
    model.add(Dropout(0.3))

    model.add(Dense(8))
    model.add(Activation('softmax'))

    print(model.summary())
    return model

def train_lstm(model, x_train, y_train, x_test, y_test):
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='acc',
            patience=epoch_time,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='model/bilstm_model/ccc-bilstm_total-more.h5',
            monitor='val_acc',  # 如果val_acc不改善，则不需要覆盖模型文件
            save_best_only=True,
            mode='max'
        ),
        keras.callbacks.TensorBoard(
            log_dir='mylog',
            histogram_freq=1  # 每一轮之后记录直方图
        )
    ]
    print('Compiling the Model...')
    # 损失函数: 二分类交叉熵 binary_crossentropy  categorical_crossentropy
    model.compile(loss='categorical_crossentropy',#hinge
                  optimizer='adam', metrics=['mae', 'acc'])

    print("Train..." )
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_time, validation_data=(x_test,y_test),
                          callbacks=callbacks_list)

    print("Evaluate...")
    print(model.predict(x_test))
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size)

    print('Test score:', score)



def main():
    # # 1-anger   2-disgust   3-fear   4-happiness   5-like   6-none   7-sadness   8-surprise
    # print("开始清洗数据................")
    # stopword = [w.strip() for w in open('stopwords\\stopWord.txt', 'r', encoding='utf-8').readlines()]
    #
    # getGoodData('anger',stopword)
    # getGoodData('disgust', stopword)
    # getGoodData('fear', stopword)
    # getGoodData('happiness', stopword)
    # getGoodData('like', stopword)
    # getGoodData('none_less', stopword)
    # getGoodData('sadness', stopword)
    # getGoodData('surprise', stopword)
    # getGoodData('2000-fennu', stopword)
    # getGoodData('2000-xiyue', stopword)
    # getGoodData('2000-diluo', stopword)
    # getGoodData('2000-yanwu', stopword)
    # print("清洗数据完成................")

    print("开始加载数据................")
    X_Vec,y = getData()
    print("加载数据完成................")
    # print(X_Vec)

    print("开始构建词向量................")
    input_dim, embedding_weights, w2dic = word2vec_train(X_Vec)
    print("构建词向量完成................")

    index = data2inx(w2dic, X_Vec)
    # pad_sequences将index转换为150维的矩阵，缺少的用0填充
    # padding: 在句子前端（pre）或后端(post)填充
    index2 = sequence.pad_sequences(index, maxlen=voc_dim,padding='pre')
    print(index2)

    # 划分训练集。测试集
    x_train, x_test, y_train, y_test = train_test_split(index2, y, test_size=0.2)
    print(len(x_train))
    print(len(y_test))

    # 将整型标签转为onehot
    y_train = keras.utils.to_categorical(y_train, num_classes=8)
    y_test = keras.utils.to_categorical(y_test, num_classes=8)


    # metrics = Metrics()
    model = lstm(input_dim, embedding_weights)
    train_lstm(model, x_train, y_train, x_test, y_test)



    # model_word = Word2Vec.load('word2vec/eight_classes_Word2Vec_model-more.pkl')
    #
    # with open("word2vec/ccc-8-word2vec-more.json", "r", encoding="utf-8") as f:
    #     word2idx = json.load(f)
    # w2dic = {}
    # for i in range(len(model_word.wv.vocab.keys())):
    #     w2dic[list(model_word.wv.vocab.keys())[i]] = i + 1
    #
    # model = load_model('model/bilstm_model/ccc-bilstm_total-more.h5')
    #
    # pchinese = re.compile('([\u4e00-\u9fff]+)+?')
    # # 1-anger   2-disgust   3-fear   4-happiness   5-like   6-none   7-sadness   8-surprise
    # # label = {0: "愤怒", 1: "恶心", 2: "害怕", 3: "开心", 4: "喜欢", 5: "中性", 6: "伤心", 7: "惊喜"}
    # label = {0: "愤怒", 1: "恶心", 2: "开心", 3: "喜欢", 4: "中性", 5: "伤心"}
    #
    # # in_stc=["明天","就要","考试","我","特别","紧张","一点","都","没有","复习"]
    # in_str = "害怕"
    # # 太过分了，他怎么可以这样   害怕   这件事太令人恶心了，想吐
    # # 他竟然是男的    太伤心了   这盆花真好看，好想买一盆回家
    # # 我喜欢花
    #
    # in_stc = ''.join(pchinese.findall(in_str))
    #
    # in_stc = list(jieba.lcut(in_stc))
    #
    # new_txt = []
    #
    # data = []
    # for word in in_stc:
    #     try:
    #         new_txt.append(w2dic[word])
    #         print(word2idx[word])
    #         print(w2dic[word])
    #         print('*******************')
    #     except:
    #         new_txt.append(0)
    # data.append(new_txt)
    #
    # data = sequence.pad_sequences(data, maxlen=voc_dim)
    #
    # pre = model.predict(data)[0].tolist()
    #
    # print(pre)
    # print("输入：")
    # print("  ", in_str)
    # print("        ")
    # print("输出:")
    # print("  ", label[pre.index(max(pre))])


if __name__ == '__main__':
    main()