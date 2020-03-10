from __future__ import division, absolute_import, print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras.backend as K
from keras.models import load_model
from captcha_gen import CAPTCHA_LIST,LEN
from attacks import (fast_gradient_sign_method, basic_iterative_method,
                            saliency_map_method)
from util import get_data
from cleverhans.utils_tf import batch_eval
# attack parameters
ATTACK_PARAMS = {

    # 'fgsm': {'eps': 0.18, 'eps_iter': 0.020},#纯数字
    'fgsm': {'eps': 0.1},
    'bim': {'eps': 0.16, 'eps_iter': 0.16},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010}
}
from PIL import Image
import matplotlib.pyplot as plt

def get_logits(predictions):

    logits, = predictions.op.inputs

    return logits

def aget_logits(sess, model, X, Y, eps=0.1, clip_min=None,
                              clip_max=None, batch_size=256):
    """
    TODO
    :param sess:
    :param model:
    :param X:
    :param Y:
    :param eps:
    :param clip_min:
    :param clip_max:
    :param batch_size:
    :return:
    """
    # Define TF placeholders for the input and output
    x = tf.placeholder(tf.float32, shape=(None,) + X.shape[1:])
    y = tf.placeholder(tf.float32, shape=(None,) + Y.shape[1:])
    adv_x = get_logits(
        model(x)
    )
    # X_adv, = batch_eval(
    #     sess, [x, y], [adv_x],
    #     [X, Y], args={'batch_size': batch_size}
    # )

    return adv_x

def craft_one_type(sess, model, X, Y, dataset, attack, batch_size):
    """
    TODO
    :param sess:
    :param model:
    :param X:
    :param Y:
    :param dataset:
    :param attack:
    :param batch_size:
    :return:
    """

    if attack == 'fgsm':
        # FGSM attack
        print('Crafting fgsm adversarial samples...')
        X_adv = fast_gradient_sign_method(
            sess, model, X, Y, eps=ATTACK_PARAMS['fgsm']['eps'], clip_min=0.0,
            clip_max=1.0, batch_size=batch_size
        )
    elif attack in ['bim-a', 'bim-b']:
        # BIM attack
        print('Crafting %s adversarial samples...' % attack)
        its, results = basic_iterative_method(
            sess, model, X, Y, eps=ATTACK_PARAMS['bim']['eps'],
            eps_iter=ATTACK_PARAMS['bim']['eps_iter'], clip_min=0.,
            clip_max=1., batch_size=batch_size
        )
        if attack == 'bim-a':
            # BIM-A
            # For each sample, select the time step where that sample first
            # became misclassified
            X_adv = np.asarray([results[its[i], i] for i in range(len(Y))])
        else:
            # BIM-B
            # For each sample, select the very last time step
            X_adv = results[-1]
    elif attack == 'jsma':
        # JSMA attack
        print('Crafting jsma adversarial samples. This may take a while...')
        X_adv = saliency_map_method(
            sess, model, X, Y, theta=1, gamma=0.1, clip_min=0., clip_max=1.
        )
    else:
        # TODO: CW attack
        raise NotImplementedError('CW attack not yet implemented.')
    _, acc = model.evaluate(X_adv, Y, batch_size=batch_size,
                            verbose=0)
    np.set_printoptions(threshold=9999)  # 打印全部


    # print(model(inputs=X[0], outputs=model.layers[30].input))

    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    pred_adv=sigmoid(model.predict(X_adv))

    pred_x=sigmoid(model.predict(X))


    # print(pred_x)
    # print(sigmoid(pred_x))

    # x=tf.convert_to_tensor(X.astype(np.float32))
    # logits,=model(x).op.input
    # print(aget_logits(sess,model,X,Y,))

    succeed=0
    t_all=0
    for j in range (1):
        t = 0
        for i in range(4):
            if np.argmax(pred_x[j, LEN*i:LEN*(i+1)])==np.argmax(Y[j, LEN*i:LEN*(i+1)]):
                t=t+1
        t_all+=t
        if t==4:
            succeed+=1
    print (succeed,' ',t_all)
    print("真实值")
    for i in range(4):
        print(CAPTCHA_LIST[np.argmax(Y[0, LEN*i:LEN*(i+1)])]," ", end="")
    print("   ")
    print("攻击前预测值")
    for i in range(4):
        print(CAPTCHA_LIST[np.argmax(pred_x[0, LEN*i:LEN*(i+1)])]," ", end="")
    print("   ")
    print("攻击后预测值")
    for i in range(4):
        print(CAPTCHA_LIST[np.argmax(pred_adv[0, LEN * i:LEN * (i + 1)])]," ", end="")

    print("   ")
    #np.argmax(pred_adv)

    acc_x=pred_x[0]*(Y[0]/4)#每个数字权重是1/4
    acc_adv=pred_adv[0]*(Y[0]/4)
    print("攻击前准确度",sum(acc_x),"攻击后准确度",sum(acc_adv))

    print("攻击前模型输出")
    print(pred_x)
    print("攻击后模型输出")
    print(pred_adv)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(X[0,:,:,0])
    plt.subplot(2, 1, 2)
    plt.imshow(X_adv[0,:,:,0])
    plt.show()

    print("Model accuracy on the adversarial test set: %0.2f%%" % (100 * acc))
    np.save('../data/captcha6.npy' , X_adv)

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def main(args):
    # assert args.dataset in ['captcha', 'cifar', 'svhn'], \
    #     "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    # assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw', 'all'], \
    #     "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
    #     "'jsma' or 'cw'"
    # assert os.path.isfile('../data/model_%s.h5' % args.dataset), \
    #     'model file not found... must first train model using train_model.py.'
    print('Dataset: %s. Attack: %s' % (args.dataset, args.attack))
    # Create TF session, set it as Keras backend
    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(0)
    #model = load_model('../data/model_captcha8_100epochs_ALPHABET.h5py',custom_objects={'precision': precision} )
    #ALPHABET_100epochs_best
    def fn(correct, predicted):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)
    model = load_model('../data/ALPHABET5_best.h5py',custom_objects={'fn': fn})
    print(model.summary())
    X_train, Y_train, X_test, Y_test = get_data(1,1)
    #y = model.predict(X_train)

    #print('Predicted:', y)
    _, acc = model.evaluate(X_train, Y_train, batch_size=args.batch_size,
                            verbose=0)

    print("Accuracy on the test set: %0.2f%%" % (100*acc))
    if args.attack == 'all':
        # Cycle through all attacks
        for attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw']:
            craft_one_type(sess, model, X_test, Y_test, args.dataset, attack,
                           args.batch_size)
    else:
        # Craft one specific attack type
        craft_one_type(sess, model, X_test, Y_test, args.dataset, args.attack,
                       args.batch_size)

    print('Adversarial samples crafted and saved to data/ subfolder.')
    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma', 'cw' "
             "or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=128)
    args = parser.parse_args()
    main(args)
