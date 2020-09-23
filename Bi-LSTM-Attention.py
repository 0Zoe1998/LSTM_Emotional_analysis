# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/22 16:50
# @Author : Zoe
# @File : Bi-LSTM-Attention.py
# @Software : PyCharm


import json

import warnings
import keras
import re
import numpy as np
from gensim.models import Word2Vec
from keras.optimizers import Adam
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import *
from keras.models import load_model
from matplotlib import pyplot
import jieba

warnings.filterwarnings("ignore")


# 配置参数

class TrainingConfig(object):
    epoches = 20

class ModelConfig(object):
    embeddingSize = 150
    hiddenSizes = [128,64] # 神经元个数
    dropoutKeepProb = 0.3

class path(object):
    # word2vec_model_path = "wiki_data\\wiki_150.model"  # 加载word2vec模型
    word2vec_model_path = "word2vec/eight_classes_Word2Vec_model-less.pkl"   #加载word2vec模型
    word2vec_dict_path = "word2vec/8-bilstm_word2idx-final.json"                #保存字典
    vocab_path = 'model/word_and_embedding/8-vocab-final.npy'
    wordEmbedding_path = 'model/word_and_embedding/8-wordEmbedding-final.npy'   #词-词向量一一对应保存

class Config(object):
    sequenceLength = 100  # 取了所有序列长度的均值
    batchSize = 64

    numClasses = 8  # 二分类设置为1，多分类设置为类别的数目
    rate = 0.8  # 训练集的比例
    training = TrainingConfig()
    model = ModelConfig()
    path = path()


# 数据预处理的类，生成训练集和测试集
class Dataset(object):
    def __init__(self, config):
        self.config = config                                    # config类
        self._sequenceLength = config.sequenceLength            # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize        # embeddingsize 嵌入层维度
        self._batchSize = config.batchSize                      # 一次处理128个神经元
        self._rate = config.rate                                # 训练集比例
        self.label2idx = {}                                     # 标签字典
        self._numclasses = config.numClasses                    # 分类
        self.word2vec_model_path = config.path.word2vec_model_path
        self.word2vec_dict_path = config.path.word2vec_dict_path
        self.vocab_path = config.path.vocab_path
        self.wordEmbedding_path = config.path.wordEmbedding_path

        self.x_train = []                                  # 训练集评论
        self.y_train = []                                   # 训练集类别

        self.x_test = []                                   # 测试集评论
        self.y_test = []                                    # 测试集类别

        self.wordEmbedding = None                               # 词嵌入
        self.labelList = []                                     # 标签类别

    def _readData(self):
        """
        从npy文件中读取数据集
        """

        anger = np.load('eight_classes_data/anger.npy', allow_pickle=True)
        # anger2 = np.load('eight_classes_data/2000-fennu .npy', allow_pickle=True)
        disgust = np.load('eight_classes_data/disgust.npy', allow_pickle=True)
        fear = np.load('eight_classes_data/fear.npy', allow_pickle=True)
        happiness = np.load('eight_classes_data/happiness.npy', allow_pickle=True)
        # happiness2 = np.load('eight_classes_data/2000-xiyue.npy', allow_pickle=True)
        like = np.load('eight_classes_data/like.npy', allow_pickle=True)
        none = np.load('eight_classes_data/none.npy', allow_pickle=True)
        sadness = np.load('eight_classes_data/sadness.npy', allow_pickle=True)
        surprise = np.load('eight_classes_data/surprise.npy', allow_pickle=True)

        reviews = np.concatenate((anger, disgust, fear, happiness, like, none, sadness, surprise))
        # reviews = np.concatenate((anger,anger2, disgust,fear, happiness,happiness2, like, none, sadness, surprise))

        labelIds = np.concatenate((np.zeros(len(anger) , dtype=int),
                            np.ones(len(disgust), dtype=int),
                            2 * np.ones(len(fear), dtype=int),
                            3 * np.ones(len(happiness) , dtype=int),
                            4 * np.ones(len(like), dtype=int),
                            5 * np.ones(len(none), dtype=int),
                            6 * np.ones(len(sadness), dtype=int),
                            7 * np.ones(len(surprise), dtype=int),))



        np.random.seed(114)
        np.random.shuffle(reviews)
        np.random.shuffle(labelIds)


        print(str(reviews))
        print(str(labelIds))

        return reviews, labelIds

    def _wordToIndex(self, reviews, word2idx):
        """
        将词转换成索引
        """
        reviewIds = []
        for sentence in reviews:
            new_txt = []
            for word in sentence:
                try:
                    new_txt.append(word2idx[word])
                except:
                    new_txt.append(0)

            reviewIds.append(new_txt)
        return reviewIds

    def _genTrainEvalData(self, x, y, rate):
        """
        生成训练集和验证集
        """
        reviews = sequence.pad_sequences(x, maxlen=self._sequenceLength, padding='pre')
        x_train, x_test, y_train, y_test = train_test_split(reviews, y, train_size=rate)
        print(y_test)
        y_train = keras.utils.to_categorical(y_train, num_classes=self._numclasses)
        y_test = keras.utils.to_categorical(y_test,num_classes=self._numclasses)

        return x_train,  y_train,x_test, y_test

    def _genVocabulary(self, reviews, labels):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """

        allWords = [word for review in reviews for word in review]

        # wordCount = Counter(allWords)  # 统计词频
        # sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
        #
        # # 去除低频词
        # words = [item[0] for item in sortWordCount if item[1] >= 5]

        vocab, wordEmbedding = self._getWordEmbedding(allWords)
        self.wordEmbedding = wordEmbedding

        word2idx = dict(zip(vocab, list(range(len(vocab)))))

        uniqueLabel = list(set(labels))
        label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
        self.labelList = list(range(len(uniqueLabel)))

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据

        with open(self.word2vec_dict_path, "w", encoding="utf-8") as f:
            json.dump(word2idx, f)
        return word2idx, label2idx

    def _getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """
        wordVec = Word2Vec.load(self.word2vec_model_path)

        # vocab和wordembedding一一对应   将所有数据（训练+测试）的词-词向量一一对应分别放在vocab和wordembedding数组中
        vocab = []
        wordEmbedding = []

        # 添加 "pad" 和 "UNK",
        # PAD意思为0
        vocab.append("PAD")
        # 装载不认识的词
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        # randn函数返回一个或一组样本，具有标准正态分布
        wordEmbedding.append(np.random.randn(self._embeddingSize))

        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                # print(word + "不存在于词向量中")
                continue
        np.save(self.vocab_path, vocab)
        np.save(self.wordEmbedding_path, wordEmbedding)
        return vocab, np.array(wordEmbedding)


    def dataGen(self):
        """
        初始化训练集和验证集
        """
        # 初始化数据集
        reviews, labelIds = self._readData()

        # 初始化词汇-索引映射表和词向量矩阵
        word2idx, label2idx = self._genVocabulary(reviews, labelIds)
        # 将标签和句子数值化
        reviewIds = self._wordToIndex(reviews, word2idx)
        # 初始化训练集和测试集
        x_train, y_train, x_test, y_test = self._genTrainEvalData(reviewIds, labelIds,self._rate)

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test
        self.label2idx = label2idx


# def getloss():
#     losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)
#
#     self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss


class Bi_LSTM(object):
    def __init__(self,data):
        self.data = data
        self.wordEmbedding = data.wordEmbedding     # 测试集
        self._sequenceLength = data._sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize =data._embeddingSize  # embeddingsize 嵌入层维度
        self._batchSize = data._batchSize  # 一次处理128个神经元
        self._epoches = data.config.training.epoches
        self._drop = data.config.model.dropoutKeepProb
        self.hiddensizes = data.config.model.hiddenSizes
        self._numclasses = data._numclasses
        self.model = None
        self.x_train = data.x_train
        self.y_train = data.y_train

        self.x_test = data.x_test
        self.y_test = data.y_test

    def create_model(self):
        model=Sequential()
        model.add(Embedding(output_dim=self._embeddingSize,
                            input_dim=len(self.wordEmbedding),
                            # mask_zero=True,
                            weights=[self.wordEmbedding],
                            input_length=self._sequenceLength))
        model.add(Dropout(self._drop))

        # concat是将待合并层输出沿着最后一个维度进行拼接，因此要求待合并层输出只有最后一个维度不同
        model.add(Bidirectional(LSTM(self.hiddensizes[1],return_sequences=True,
                                     kernel_regularizer=keras.regularizers.l2(0.001),
                                     activation="softsign"),merge_mode='concat'))

        # model.add(LSTM(units=self.hiddensizes[1], return_sequences=False))

        model.add(Dropout(self._drop))
        # model.add(AttentionLayer())
        model.add(Flatten())

        # model.add(Dense(128,activation='softsign',kernel_regularizer=keras.regularizers.l2(0.001)))
        # model.add(Dropout(self._drop))
        model.add(Dense(self._numclasses,activation='softmax'))
        # model.add(Activation('tanh'))


        model.summary()
        # plot_model(model,to_file='txtemotion_model.png',show_shapes=True)
        # print(wordEmbedding)
        self.model = model

    def train_model(self):
        callbacks_list=[
            keras.callbacks.EarlyStopping(
                monitor='acc',
                patience=self._epoches,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='model/bilstm_model/8_classes_txtemotion_model_checkpoint-final.h5',
                monitor='val_acc',#如果val_acc不改善，则不需要覆盖模型文件
                save_best_only=True,
                mode = 'max'
            ),
            keras.callbacks.TensorBoard(
                log_dir='mylog',
                histogram_freq=1#每一轮之后记录直方图
            )
        ]
        optimizer = Adam(lr=1e-3)
        # 整体来讲 adam 是最好的选择
        # 在多标签分类中，大多使用binary_crossentropy损失而不是通常在多类分类中使用的categorical_crossentropy损失函数。这可能看起来不合理，但因为每个输出节点都是独立的
        self.model.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=['acc','mae'])
        # validation_data用来在每个epoch之后，或者每几个epoch，验证一次验证集，用来及早发现问题，比如过拟合，或者超参数设置有问题。
        # 这样可以方便我们及时调整参数
        history = self.model.fit(self.x_train,self.y_train,
                      epochs=self._epoches,
                      batch_size=self._batchSize,
                      validation_data=(self.x_test,self.y_test),
                      callbacks=callbacks_list,
                      shuffle=True
                         )

        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.title('model train vs validation loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.show()
        # print("loss  " + history.history['loss'])
        # print(history.history['val_loss'])
        # print("acc  " + history.history['acc'])
        # print(history.history['val_acc'])

        # save_weights 它只保存了模型的参数，但并没有保存模型的图结构
        self.model.save('model/bilstm_model/8_classes_text_emotion-final.h5')
        print(self.model.evaluate(self.x_test,self.y_test,batch_size=self._batchSize))
        print(self.x_test)
        print(self.y_test)


# class AttentionLayer(Layer):
#
#     def __init__(self, **kwargs):
#         self.init = initializers.get('glorot_uniform')
#         super(AttentionLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         assert len(input_shape) == 3
#
#         self.W = self.add_weight(name='Attention_Weight',
#                                  shape=(input_shape[-1], input_shape[-1]),
#                                  initializer=self.init,
#                                  trainable=True)
#         self.b = self.add_weight(name='Attention_Bias',
#                                  shape=(input_shape[-1],),
#                                  initializer=self.init,
#                                  trainable=True)
#         self.u = self.add_weight(name='Attention_Context_Vector',
#                                  shape=(input_shape[-1], 1),
#                                  initializer=self.init,
#                                  trainable=True)
#         super(AttentionLayer, self).build(input_shape)
#
#     def compute_mask(self, input, input_mask=None):
#         # do not pass the mask to the next layers
#         return None
#
#     def call(self, x):
#         # refer to the original paper
#         u_it = K.tanh(K.dot(x, self.W) + self.b)
#
#         a_it = K.dot(u_it, self.u)
#         a_it = K.squeeze(a_it, -1)
#         a_it = K.softmax(a_it)
#
#         return a_it
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[1])
#
# class Attention(Layer):
#     def __init__(self, step_dim,
#                  W_regularizer=None, b_regularizer=None,
#                  W_constraint=None, b_constraint=None,
#                  bias=True, **kwargs):
#         self.supports_masking = True
#         self.init = initializers.get('glorot_uniform')
#
#         self.W_regularizer = regularizers.get(W_regularizer)
#         self.b_regularizer = regularizers.get(b_regularizer)
#
#         self.W_constraint = constraints.get(W_constraint)
#         self.b_constraint = constraints.get(b_constraint)
#
#         self.bias = bias
#         self.step_dim = step_dim
#         self.features_dim = 0
#         super(Attention, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         assert len(input_shape) == 3
#
#         self.W = self.add_weight((input_shape[-1],),
#                                  initializer=self.init,
#                                  name='{}_W'.format(self.name),
#                                  regularizer=self.W_regularizer,
#                                  constraint=self.W_constraint)
#         self.features_dim = input_shape[-1]
#
#         if self.bias:
#             self.b = self.add_weight((input_shape[1],),
#                                      initializer='zero',
#                                      name='{}_b'.format(self.name),
#                                      regularizer=self.b_regularizer,
#                                      constraint=self.b_constraint)
#         else:
#             self.b = None
#
#         self.built = True
#
#     def compute_mask(self, input, input_mask=None):
#     # 后面的层不需要mask了，所以这里可以直接返回none
#         return None
#
#     def call(self, x, mask=None):
#         features_dim = self.features_dim
#         # 这里应该是 step_dim是我们指定的参数，它等于input_shape[1],也就是rnn的timesteps
#         step_dim = self.step_dim
#         eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
#                         K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
#
#         if self.bias:
#             eij += self.b
#         # RNN一般默认激活函数为tanh, 对attention来说激活函数差别不打，因为要做softmax
#         eij = K.tanh(eij)
#
#         a = K.exp(eij)
#
#         if mask is not None:
#             # 如果前面的层有mask，那么后面这些被mask掉的timestep肯定是不能参与计算输出的，也就是将他们的attention权重设为0
#             a *= K.cast(mask, K.floatx())
#         # cast是做类型转换，keras计算时会检查类型，可能是因为用gpu的原因
#         a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
#
#         # a = K.expand_dims(a, axis=-1) , axis默认为-1， 表示在最后扩充一个维度。
#         # 比如shape = (3,)变成 (3, 1)
#         a = K.expand_dims(a)
#         # 此时a.shape = (batch_size, timesteps, 1), x.shape = (batch_size, timesteps, units)
#         weighted_input = x * a
#
#         # weighted_input的shape为 (batch_size, timesteps, units), 每个timestep的输出向量已经乘上了该timestep的权重
#         # weighted_input在axis=1上取和，返回值的shape为 (batch_size, 1, units)
#         return K.sum(weighted_input, axis=1)
#
#     def compute_output_shape(self, input_shape):
#         # 返回的结果是c，其shape为 (batch_size, units)
#         return input_shape[0],  self.features_dim




def getClear(x):        # 去非中文符号
    line = x.strip()
    # 去除文本中的英文和数字
    line = re.sub("[a-zA-Z0-9]", "", line)
    # 去除文本中的中文符号和英文符号
    line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", line)
    line = re.sub("【", "", line)
    return line

def afterStopwords(x,stopword):     # 去停用词
    for i in x:
        if i in stopword or i == ' ':
            x.remove(i)
    return x



# train loss 不断下降，val loss 不断下降——网络仍在学习；
# train loss 不断下降，val loss 不断上升——网络过拟合；
# train loss 不断下降，val loss 趋于不变——网络欠拟合；
# train loss 趋于不变，val loss 趋于不变——网络陷入瓶颈；
# train loss 不断上升，val loss 不断上升——网络结构问题；
# train loss 不断上升，val loss 不断下降——数据集有问题；

def predict(in_str,stopword):
    with open("word2vec/8-bilstm_word2idx-final.json", "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    model = load_model('model/bilstm_model/8_classes_txtemotion_model_checkpoint-final.h5')


    # 1-anger   2-disgust   3-fear   4-happiness   5-like   6-none   7-sadness   8-surprise
    label = {0: "愤怒", 1: "恶心", 2: "害怕", 3: "开心", 4: "喜欢", 5: "中性", 6: "伤心", 7: "惊喜"}
    # in_stc=["明天","就要","考试","我","特别","紧张","一点","都","没有","复习"]

    in_stc = getClear(in_str)
    in_stc = jieba.lcut(in_stc)
    print(in_stc)
    in_stc = afterStopwords(in_stc, stopword)
    new_txt = []

    data = []
    for word in in_stc:
        try:
            new_txt.append(word2idx[word])
        except:
            new_txt.append(0)
    data.append(new_txt)
    print(data)

    data = sequence.pad_sequences(data, maxlen=100,padding='pre')

    pre = model.predict(data)[0].tolist()

    print(pre)
    print("输入：")
    print("  ", in_str)
    print("        ")
    print("输出:")
    print("  ", label[pre.index(max(pre))])


def main():
    # 实例化配置参数对象
    config = Config()
    data = Dataset(config)
    data.dataGen()

    model = Bi_LSTM(data)
    model.create_model()
    model.train_model()
    # stopword = [w.strip() for w in open('stopwords\\stopWord.txt', 'r', encoding='utf-8').readlines()]
    #
    # in_str = "没想到这次考的还挺好，我还以为会很差呢"
    # predict(in_str,stopword)
    # 太过分了，他怎么可以这样   这里好可怕，我要回家找妈妈   这件事太令人恶心了，想吐
    # 没想到这次考的还挺好，我还以为会很差呢    好难受啊，这次没考好   这盆花真好看，好想买一盆回家


if __name__ == '__main__':
    main()


# loss: 0.2911 - acc: 0.9137 - mean_absolute_error: 0.1319 - val_loss: 0.2192 - val_acc: 0.9224 - val_mean_absolute_error: 0.1204
# loss: 0.2200 - acc: 0.9205 - mean_absolute_error: 0.1204 - val_loss: 0.2147 - val_acc: 0.9234 - val_mean_absolute_error: 0.1174
# loss: 0.2108 - acc: 0.9242 - mean_absolute_error: 0.1153 - val_loss: 0.2252 - val_acc: 0.9189 - val_mean_absolute_error: 0.1031
# loss: 0.2032 - acc: 0.9270 - mean_absolute_error: 0.1109 - val_loss: 0.2121 - val_acc: 0.9247 - val_mean_absolute_error: 0.1167
# loss: 0.1966 - acc: 0.9300 - mean_absolute_error: 0.1066 - val_loss: 0.2143 - val_acc: 0.9242 - val_mean_absolute_error: 0.1083
# loss: 0.1898 - acc: 0.9323 - mean_absolute_error: 0.1023 - val_loss: 0.2141 - val_acc: 0.9260 - val_mean_absolute_error: 0.1081
# loss: 0.1828 - acc: 0.9356 - mean_absolute_error: 0.0980 - val_loss: 0.2188 - val_acc: 0.9247 - val_mean_absolute_error: 0.1026
# loss: 0.1772 - acc: 0.9379 - mean_absolute_error: 0.0942 - val_loss: 0.2192 - val_acc: 0.9245 - val_mean_absolute_error: 0.1034
# loss: 0.1711 - acc: 0.9406 - mean_absolute_error: 0.0903 - val_loss: 0.2317 - val_acc: 0.9235 - val_mean_absolute_error: 0.1038
# loss: 0.1667 - acc: 0.9428 - mean_absolute_error: 0.0871 - val_loss: 0.2308 - val_acc: 0.9246 - val_mean_absolute_error: 0.1022
# loss: 0.1610 - acc: 0.9449 - mean_absolute_error: 0.0836 - val_loss: 0.2317 - val_acc: 0.9222 - val_mean_absolute_error: 0.1069
# loss: 0.1565 - acc: 0.9470 - mean_absolute_error: 0.0809 - val_loss: 0.2413 - val_acc: 0.9227 - val_mean_absolute_error: 0.1028
# loss: 0.1525 - acc: 0.9483 - mean_absolute_error: 0.0781 - val_loss: 0.2487 - val_acc: 0.9240 - val_mean_absolute_error: 0.0974
# loss: 0.1474 - acc: 0.9516 - mean_absolute_error: 0.0747 - val_loss: 0.2460 - val_acc: 0.9215 - val_mean_absolute_error: 0.1028
# loss: 0.1439 - acc: 0.9528 - mean_absolute_error: 0.0726 - val_loss: 0.2520 - val_acc: 0.9226 - val_mean_absolute_error: 0.0995
# loss: 0.1394 - acc: 0.9540 - mean_absolute_error: 0.0700 - val_loss: 0.2667 - val_acc: 0.9211 - val_mean_absolute_error: 0.0994
# loss: 0.1367 - acc: 0.9555 - mean_absolute_error: 0.0679 - val_loss: 0.2689 - val_acc: 0.9209 - val_mean_absolute_error: 0.0982
# loss: 0.1326 - acc: 0.9575 - mean_absolute_error: 0.0651 - val_loss: 0.2761 - val_acc: 0.9207 - val_mean_absolute_error: 0.0960
# loss: 0.1290 - acc: 0.9592 - mean_absolute_error: 0.0631 - val_loss: 0.2799 - val_acc: 0.9210 - val_mean_absolute_error: 0.0969
# loss: 0.1253 - acc: 0.9607 - mean_absolute_error: 0.0605 - val_loss: 0.2906 - val_acc: 0.9175 - val_mean_absolute_error: 0.0994















