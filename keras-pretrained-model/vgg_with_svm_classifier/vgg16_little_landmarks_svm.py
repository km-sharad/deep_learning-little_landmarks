# -*- coding: utf-8 -*-
'''VGG16 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function

import numpy as np
import json
import warnings
import glob
import os
import operator
import h5py

from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.optimizers import SGD
from itertools import izip

#from imagenet_utils import decode_predictions, preprocess_input

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def preprocess_input(x, dim_ordering='default'):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        dim_ordering: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
        # Zero-center by mean pixel
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
    return x

def decode_predictions(preds, top=5):
    """Decodes the prediction of an ImageNet model.

    # Arguments
        preds: Numpy tensor encoding a batch of predictions.
        top: integer, how many top-guesses to return.

    # Returns
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.

    # Raises
        ValueError: in case of invalid shape of the `pred` array
            (must be 2D).
    """
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results    


def VGG16(include_top=True, weights='imagenet',
          input_tensor=None):
    '''Instantiate the VGG16 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', trainable=False, name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', trainable=False, name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), trainable=False, name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', trainable=False, name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', trainable=False, name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), trainable=False, name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', trainable=False, name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', trainable=False, name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', trainable=False, name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), trainable=False, name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', trainable=False, name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', trainable=False, name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', trainable=False, name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), trainable=False, name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', trainable=False, name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', trainable=False, name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', trainable=False, name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), trainable=False, name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, name='fc1', trainable=False)(x)
        x = Dense(4096, name='fc2', trainable=False)(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    if weights == 'imagenet':

        if include_top:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')

        model.load_weights(weights_path)

        # Gets rid of the softmax classification layer
        
        model.layers.pop() 
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        
        lastCnnLayer = model.output
        newSVMClassificationLayer = Dense(2, activation='linear', name='svm-classifier')(lastCnnLayer)
        model_with_new_svm_layer = Model(input=model.input, output=newSVMClassificationLayer)
        
    return model_with_new_svm_layer


if __name__ == '__main__':

    class_dict = dict()
    model = VGG16(include_top=True, weights='imagenet')
    #model.layers.add(Dense(2), activation='linear')
    #model.add(Dense(200, activation='softmax', name='fc8'))

    train_x = []
    train_y = []

    for single_train_pos_img, single_train_neg_img in zip (glob.glob('data/train/positive/*.jpg'), glob.glob('data/train/negative/*.jpg')):    
        train_img_path = single_train_pos_img
        train_img = image.load_img(train_img_path, target_size=(224, 224)) #appears to resize the image correctly: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
        train_x.append(np.asarray(train_img))        
        train_y.append([1,-1])       #holding leash = 1

        train_img_path = single_train_neg_img
        train_img = image.load_img(train_img_path, target_size=(224, 224)) #appears to resize the image correctly: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
        train_x.append(np.asarray(train_img))        
        train_y.append([-1,1])        #not holding leash = 0

    print('x_', np.array(train_x).shape)
    print('y_', np.array(train_y).shape)
    #print('x_', np.array(train_x))
    #print('y_', np.array(train_y))    

    
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])            
    #model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])            
    model.compile(loss='hinge', optimizer='sgd', metrics=['accuracy'])            
    
    history = model.fit(np.array(train_x),np.array(train_y),nb_epoch=8, batch_size=20)
    print('training stats: ', history.history)

    test_x = []
    test_y = []

    for single_test_pos_img, single_test_neg_img in zip (glob.glob('data/test/positive/*.jpg'), glob.glob('data/test/negative/*.jpg')):    
        test_img_path = single_test_pos_img
        test_img = image.load_img(test_img_path, target_size=(224, 224)) #appears to resize the image correctly: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
        test_x.append(np.asarray(test_img))        
        test_y.append([1,-1])       #holding leash = 1

        test_img_path = single_test_neg_img
        test_img = image.load_img(test_img_path, target_size=(224, 224)) #appears to resize the image correctly: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
        test_x.append(np.asarray(test_img))        
        test_y.append([-1,1])        #not holding leash = 0

    score = model.evaluate(np.array(test_x),np.array(test_y))
    print('score: ', score)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

