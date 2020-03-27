from __future__ import division, absolute_import, print_function
import tensorflow.compat.v1 as tf
import argparse
import numpy as np
from util import get_data,get_model,CAPTCHA
import keras.backend as K
import keras
import datetime
from keras.optimizers import SGD
from keras.models import load_model
from captcha_gen import LEN
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from captcha_gen import CAPTCHA_LIST, CAPTCHA_LEN, CAPTCHA_HEIGHT, CAPTCHA_WIDTH,CAPTCHA_CHANNEL,LEN
from keras.models import *#自定义
from keras.layers import *
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

def distillation(epochs,batch_size):
    print('Data set: captcha')
    X_train, Y_train, X_test, Y_test = get_data(10000,1000)
    temperature = 1000
    def fn(correct, predicted):
        loss = 0
        for i in range(CAPTCHA_LEN):
            loss += tf.nn.softmax_cross_entropy_with_logits(labels=correct[:, LEN * i:LEN * (i + 1)],
                                                            logits=predicted[:, LEN * i:LEN * (i + 1)]/temperature)
        return loss
    model_path = '../data/Captcha.h5py'
    model = load_model(model_path,custom_objects={'fn': fn})
    # 定义温度

    predicted=model.predict(X_train)/temperature
    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s
    def softmax(X,i):  # softmax函数
        return np.exp(X[LEN*i:LEN*(i+1)]) / np.sum(np.exp(X[LEN*i:LEN*(i+1)]))


    #print(np.concatenate((softmax(predicted,0), softmax(predicted,1), softmax(predicted,2),softmax(predicted,3)), axis=1))
    def soften(predicted):
        soft=predicted
        for i in range(10000):
            soft[i] = np.concatenate(
                (softmax(predicted[i], 0), softmax(predicted[i], 1), softmax(predicted[i], 2), softmax(predicted[i], 3)),
                axis=0)
        return soft

    np.set_printoptions(threshold=9999)
    print(softmax(predicted[0],0))
    soft_label=soften(predicted)
    print(soft_label[0],soft_label[1],soft_label[2],soft_label[5000],soft_label[9999],Y_train[0])

    save_path = '../data/Captcha_stu_best.h5py'
    stu_model = get_model('captcha')
    #stu_model = load_model(save_path, custom_objects={'fn': fn})
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    stu_model.compile(
        #loss='binary_crossentropy',
        loss=fn,
        optimizer='adam',
        metrics=['accuracy']
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, mode='min', epsilon=0.0001,
                                  cooldown=0, min_lr=0)

    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)

    stu_model.fit(


        X_train,soft_label ,
        callbacks=[ checkpoint],
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        validation_data=(X_test, Y_test),
    )
    stu_model.save( '../data/Captcha_stu.h5py')

# def train_softmax(epochs,batch_size):
#     input_tensor = Input(shape=(CAPTCHA_HEIGHT,CAPTCHA_WIDTH,CAPTCHA_CHANNEL))
#     x = input_tensor
#     for i in range(4):
#         x = Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)
#         x = Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)
#         x = MaxPooling2D((2, 2))(x)
#
#     x = Flatten()(x)
#     x = Dropout(0.25)(x)
#     x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x)
#          for i in range(4)]
#     model = Model(input=input_tensor, output=x)
#
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adadelta',
#                   metrics=['accuracy'])
#     X_train, Y_train, X_test, Y_test = get_data(10000, 1000)
#     save_path = '../data/Captcha_best.h5py'
#     checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
#     model.fit(
#
#         X_train, Y_train,
#         callbacks=[ checkpoint],
#         epochs=epochs,
#         batch_size=batch_size,
#         shuffle=True,
#         verbose=1,
#         validation_data=(X_test, Y_test),
#     )
#     model.save('../data/Captcha.h5py')

def train_model(epochs,batch_size):
    print('Data set: captcha')
    X_train, Y_train, X_test, Y_test = get_data(20000,2000)

    def fn(correct, predicted):
        loss=0
        for i in range(CAPTCHA_LEN):
            loss+=tf.nn.softmax_cross_entropy_with_logits(labels=correct[:,LEN*i:LEN*(i+1)],
                                                       logits=predicted[:,LEN*i:LEN*(i+1)])
        return loss

    filepath = '../data/Captcha_best.h5py'
    model = get_model('captcha')
    #model = load_model(filepath, custom_objects={'fn': fn})


    model.compile(
        loss=fn,
        optimizer='adam',
        metrics=['accuracy']
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5, verbose=1,patience=2, mode='min',epsilon=0.0001, cooldown=0, min_lr=0)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
    model.fit(

        X_train, Y_train,
        callbacks=[checkpoint],
        # callbacks=[early_stopping],
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        validation_data=(X_test, Y_test),
    )

    #model.save(filepath)
    model.save('../data/Captcha.h5py')


# def train(data, file_name, num_epochs=50, batch_size=128, train_temp=1, init=None):
#     """
#     Standard neural network training procedure.
#     """
#     model = Sequential()
#
#     print(data.train_data.shape)
#     model = get_model('captcha')
#     if init != None:
#         model.load_weights(init)
#     def fn(correct, predicted):
#         loss = 0
#         for i in range(CAPTCHA_LEN):
#             loss += tf.nn.softmax_cross_entropy_with_logits(labels=correct[:, LEN * i:LEN * (i + 1)],
#                                                             logits=predicted[:, LEN * i:LEN * (i + 1)]/train_temp)
#         return loss
#
#     sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#
#     model.compile(loss=fn,
#                   optimizer=sgd,
#                   metrics=['accuracy'])
#
#     model.fit(data.train_data, data.train_labels,
#               batch_size=batch_size,
#               validation_data=(data.validation_data, data.validation_labels),
#               nb_epoch=num_epochs,
#               shuffle=True)
#
#     if file_name != None:
#         model.save(file_name)
#
#     return model
#
#
# def train_distillation(data, file_name, num_epochs=50, batch_size=128, train_temp=1):
#     """
#     Train a network using defensive distillation.
#
#     Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks
#     Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami
#     IEEE S&P, 2016.
#     """
#     if not os.path.exists(file_name + "_init"):
#         # Train for one epoch to get a good starting point.
#         train(data, file_name + "_init", 1, batch_size)
#
#     # now train the teacher at the given temperature
#     teacher = train(data, file_name + "_teacher", num_epochs, batch_size, train_temp,
#                     init=file_name + "_init")
#
#     # evaluate the labels at temperature t
#     predicted = teacher.predict(data.train_data)
#     with tf.Session() as sess:
#         y = sess.run(tf.nn.softmax(predicted / train_temp))
#         print(y)
#         data.train_labels = y
#
#     # train the student model at temperature t
#     student = train(data, file_name, num_epochs, batch_size, train_temp,
#                     init=file_name + "_init")
#
#     # and finally we predict at temperature 1
#     predicted = student.predict(data.train_data)
#
#     print(predicted)

if __name__ == "__main__":
    tf.enable_eager_execution()#numpy() is only available when eager execution is enabled


    time_before=datetime.datetime.now()
    distillation(epochs=30,batch_size=32)
    #train_softmax(epochs=50, batch_size=32)
    #train_model(epochs=40, batch_size=32)
    # train_distillation(CAPTCHA(), "../data/distilled",
    #                    num_epochs=50, train_temp=100)
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
