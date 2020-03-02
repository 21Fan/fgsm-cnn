from __future__ import division, absolute_import, print_function
import tensorflow.compat.v1 as tf
import argparse
import numpy as np
from util import get_data,get_model
import keras.backend as K
import keras
import datetime

from keras.callbacks import ReduceLROnPlateau
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
    X_train, Y_train, X_test, Y_test = get_data(10000,100)
    #print(Y_train)

    model = get_model('captcha')
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[precision]
    )
    #提前停止
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
    #                                                patience=0, verbose=0, mode='auto',
    #                                                baseline=None, restore_best_weights=False)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=3, mode='auto')
    model.fit(

        X_train, Y_train,
        callbacks=[reduce_lr],
        # callbacks=[early_stopping],
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        validation_data=(X_test, Y_test)
    )

    model.save('../data/model_captcha6_100epochs.h5py')



if __name__ == "__main__":
    tf.enable_eager_execution()#numpy() is only available when eager execution is enabled
    time_before=datetime.datetime.now()
    main(epochs=200,batch_size=15)
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
