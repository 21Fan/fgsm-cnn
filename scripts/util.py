# -*- coding:utf-8 -*-
# name: util.py

import numpy as np
from captcha_gen import gen_captcha_text_and_image
from captcha_gen import CAPTCHA_LIST, CAPTCHA_LEN, CAPTCHA_HEIGHT, CAPTCHA_WIDTH,CAPTCHA_CHANNEL
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D


def text2vec(text, captcha_len=CAPTCHA_LEN, captcha_list=CAPTCHA_LIST):
    """
    验证码文本转为向量
    :param text:
    :param captcha_len:
    :param captcha_list:
    :return: vector 文本对应的向量形式
    """
    text_len = len(text)    # 欲生成验证码的字符长度
    if text_len > captcha_len:
        raise ValueError('验证码最长4个字符')
    vector = np.zeros(captcha_len * len(captcha_list))      # 生成一个一维向量 验证码长度*字符列表长度
    for i in range(text_len):
        vector[captcha_list.index(text[i])+i*len(captcha_list)] = 1     # 找到字符对应在字符列表中的下标值+字符列表长度*i 的 一维向量 赋值为 1
    return vector


def get_data(batch_count=100,test_count=100, height=CAPTCHA_HEIGHT,width=CAPTCHA_WIDTH ):
    """
    获取训练图片组
    :param batch_count: default 60
    :param width: 验证码宽度
    :param height: 验证码高度
    :return: batch_x, batch_yc
    """

    batch_x = np.zeros([batch_count, height , width,CAPTCHA_CHANNEL])
    batch_y = np.zeros([batch_count, CAPTCHA_LEN * len(CAPTCHA_LIST)])#60*40

    test_x = np.zeros([test_count, height, width, CAPTCHA_CHANNEL])
    test_y = np.zeros([test_count, CAPTCHA_LEN * len(CAPTCHA_LIST)])#60*40
    for i in range(batch_count):    # 生成对应的训练集
        image, text = gen_captcha_text_and_image()

        batch_x[i, :] = image

        batch_y[i, :] = text2vec(text)  # 验证码文本的向量形式
    for i in range(test_count):    # 生成对应的训练集
        image, text = gen_captcha_text_and_image()

        test_x[i, :] = image
        test_y[i, :] = text2vec(text)  # 验证码文本的向量形式

    return batch_x, batch_y,test_x,test_y


def get_model(dataset='captcha'):
    """
    Takes in a parameter indicating which model type to use ('mnist',
    'cifar' or 'svhn') and returns the appropriate Keras model.
    :param dataset: A string indicating which dataset we are building
                    a model for.
    :return: The model; a Keras 'Sequential' instance.
    """
    # assert dataset in ['captcha', 'cifar', 'svhn'], \
    #     "dataset parameter must be either 'mnist' 'cifar' or 'svhn'"
    if dataset == 'captcha':
        # captcha model
        dropout=0.5
        layers = [
            Conv2D(32, (3, 3), padding='same', input_shape=(CAPTCHA_HEIGHT, CAPTCHA_WIDTH, CAPTCHA_CHANNEL)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout),
            #卷积(60,160,1)->(60,160,32),池化->(30,80,32)
            Conv2D(64, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout),
            # 卷积(30,80,32)->(30,80,64),池化->(15,40,64)
            Conv2D(128, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), padding='same'),
            Dropout(dropout),
            # 卷积(15,40,64)->(15,40,64),池化->(8,20,64)same向上取整
            Flatten(),
            Dense(1024),
            Activation('relu'),
            Dropout(dropout),
            #(8,20,64)->(1024,1)
            Dense(CAPTCHA_LEN * len(CAPTCHA_LIST)),
            # (1024,1)->(248,1)
            Activation('sigmoid')
        ]

    model = Sequential()
    for layer in layers:
        model.add(layer)

    return model

