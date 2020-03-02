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

from attacks import (fast_gradient_sign_method, basic_iterative_method,
                            saliency_map_method)
from util import get_data

# attack parameters
ATTACK_PARAMS = {
    'fgsm': {'eps': 0.18, 'eps_iter': 0.020},
    'bim': {'eps': 0.16, 'eps_iter': 0.16},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010}
}
from PIL import Image
import matplotlib.pyplot as plt

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

    pred_adv=model.predict(X_adv)

    pred_x=model.predict(X)
    print(np.argmax(Y[0, 0:10]), np.argmax(Y[0, 10:20]), np.argmax(Y[0, 20:30]), np.argmax(Y[0, 30:40]))
    print(np.argmax(pred_x[0, 0:10]), np.argmax(pred_x[0, 10:20]), np.argmax(pred_x[0, 20:30]), np.argmax(pred_x[0, 30:40]))
    print(np.argmax(pred_adv[0, 0:10]), np.argmax(pred_adv[0, 10:20]), np.argmax(pred_adv[0, 20:30]), np.argmax(pred_adv[0, 30:40]))


    #np.argmax(pred_adv)
    print(pred_x)
    print(pred_adv)
    acc_x=pred_x[0]*(Y[0]/4)#每个数字权重是1/4
    acc_adv=pred_adv[0]*(Y[0]/4)
    print(sum(acc_x),sum(acc_adv))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(X[0,:,:,0])
    plt.subplot(2, 1, 2)
    plt.imshow(X_adv[0,:,:,0])
    plt.show()

    print("Model accuracy on the adversarial test set: %0.2f%%" % (100 * acc))
    np.save('../data/captcha6.npy' , X_adv)


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
    model = load_model('../data/model_captcha6_100epochs.h5py' )
    X_train, Y_train, X_test, Y_test = get_data(100,1)
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
