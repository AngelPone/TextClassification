#! /usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
@description:
    使用爬取自ChinaDaily的新闻数据，借鉴keras实现CNN文本分类的例子，对ChinaDaily文本进行分类
@author:pone
"""

from __future__ import print_function

import os
import sys
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.initializers import Constant
from keras.constraints import maxnorm

BASE_DIR = '/Users/Pone/OneDrive/Research/Text-Mining/Project/textClassificationwithCNN'
GLOVE_FILE = '/Users/Pone/OneDrive/Research/Text-Mining/resource/glove.twitter.27B/glove.twitter.27B.200d.txt'

MAX_SEQUENCE_LENGTH = 500
MAX_NUM_WORDS = 30000

EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

# 取出GLove词向量

print('Indexing word vectors.')

embeddings_index = {}
with open(GLOVE_FILE, 'r', encoding = 'utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

# 处理文本数据

print('Processing text dataset')

texts = []
labels_index = {'culture':1,'life':2,'business':3,'opinion':0}
labels = []

with open(os.path.join(BASE_DIR,'data/ChinaDaily.txt'),'r', encoding = 'utf-8') as f:
    for line in f.readlines():
        label = line.split('\001')[0]
        text = line.split('\001')[2].strip()
        labels.append(labels_index[label])
        texts.append(text)

print('Found %s texts.' % len(texts))

# 将文本向量化
## 创建分词器并分词
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
## 文本转化为词序列，序列为词的id
sequences = tokenizer.texts_to_sequences(texts)
## 建立词典向词ID的映射
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#将文本序列补齐成相同的长度，并转化为ndarray，维数为(sample_numbers, text_len)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# 将标签转化为分类变量，onehot型 (0,0,0,1)
labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

#把数据集切分成训练集和验证集

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


# 将文本转化为 CNN 的输入

## 获取当前文本的词典的词向量
print('Preparing embedding matrix.')
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

#将词向量嵌入到Embedding层
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu',W_constraint=maxnorm(3))(x)
x = Dropout(0.3)(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# 设置Log文件以便用于Tensorboard可视化
log_filepath = '/tmp/keras_log' 
tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)  
# 设置log的存储位置，将网络权值以图片格式保持在tensorboard中显示，设置每一个周期计算一次网络的权值，每层输出值的分布直方图 

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          callbacks = [tb_cb],
          validation_data=(x_val, y_val))

val = model.evaluate(x_val, y_val)
print("accuracy:%s"%val[1])


model.summary()

model.save(os.path.join(BASE_DIR,'outputent/V3-trainableVec+Drop0.4-epoch30'))
