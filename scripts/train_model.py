from __future__ import division, absolute_import, print_function
import tensorflow.compat.v1 as tf
import argparse
import numpy as np
from util import get_data,get_model
import keras.backend as K
import keras
import datetime

from keras.models import load_model

from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
#回调函数，准确率，自定义accuracy
# def recall(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
#
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
# def my_acc(y_true, y_pred):
#     acc = y_true * y_pred/4
#     return acc

def main(epochs,batch_size):
    print('Data set: captcha')
    X_train, Y_train, X_test, Y_test = get_data(10000,1000)
    # print(X_train)
    # print(Y_train)

    def fn(correct, predicted):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)
    model_path = '../ALPHABET5_best.h5py'
    model = load_model(model_path,custom_objects={'fn': fn})
    # 定义温度

    predicted=model.predict(X_train)
    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    temperature = 10
    soft_label=sigmoid(predicted/temperature)
    save_path = '../data/ALPHABET5_best.h5py'
    #stu_model = get_model('captcha')
    stu_model = load_model(save_path, custom_objects={'fn': fn})
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    stu_model.compile(
        loss=fn,
        optimizer=sgd,
        metrics=['accuracy']
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, mode='min', epsilon=0.0001,
                                  cooldown=0, min_lr=0)

    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)

    stu_model.fit(


        X_train,soft_label ,
        callbacks=[reduce_lr, checkpoint],
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        validation_data=(X_test, Y_test),
    )
    stu_model.save('../data/ALPHABET5_stu.h5py')


def train(epochs,batch_size):
    print('Data set: captcha')
    X_train, Y_train, X_test, Y_test = get_data(10000,1000)
    # print(X_train)
    # print(Y_train)

    model = get_model('captcha')
    filepath = '../data/ALPHABET5_stu.h5py'
    #model = load_model(filepath)
    def fn(correct, predicted):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)
    model.compile(
        loss=fn,
        optimizer='adam',
        metrics=['accuracy']
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5, verbose=1,patience=2, mode='min',epsilon=0.0001, cooldown=0, min_lr=0)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
    model.fit(

        X_train, Y_train,
        callbacks=[reduce_lr,checkpoint],
        # callbacks=[early_stopping],
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        validation_data=(X_test, Y_test),
    )

    model.save('../data/ALPHABET5.h5py')



if __name__ == "__main__":
    tf.enable_eager_execution()#numpy() is only available when eager execution is enabled


    time_before=datetime.datetime.now()
    main(epochs=50,batch_size=32)
    #train(epochs=25, batch_size=32)
    time_after = datetime.datetime.now()
    print(time_before)
    print(time_after)
    print(time_after-time_before)
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-d', '--dataset',
    #     help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
    #     required=False, type=str
    # )
    # parser.add_argument(
    #     '-e', '--epochs',
    #     help="The number of epochs to train for.",
    #     required=False, type=int
    # )
    # parser.add_argument(
    #     '-b', '--batch_size',
    #     help="The batch size to use for training.",
    #     required=False, type=int
    # )
    # parser.set_defaults(epochs=100)
    # parser.set_defaults(batch_size=128)
    # args = parser.parse_args()
    # main(args)
