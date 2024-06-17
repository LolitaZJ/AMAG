#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:32:02 2023

@author: zhangj2
"""

# In[] Libs
import os
import numpy as np
os.getcwd()
import argparse
import matplotlib
# matplotlib.use('agg')

import datetime
import keras
from keras import losses
from keras import optimizers
from keras.models import Sequential
from keras.models import Model,load_model
from keras.layers import Input, Dense, Dropout, Flatten,Embedding, LSTM,GRU,Bidirectional
from keras.layers import Conv1D,Conv2D,MaxPooling1D,MaxPooling2D,BatchNormalization,Reshape
from keras.layers import UpSampling1D,AveragePooling1D,AveragePooling2D,TimeDistributed 
from keras.layers import UpSampling2D
try:
    from tensorflow.keras.layers import LeakyReLU
except:
    from keras.layers.advanced_activations import LeakyReLU

from keras.layers import Lambda,concatenate,add,Conv2DTranspose,Concatenate
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint
from keras.layers import Reshape

from keras_self_attention import SeqSelfAttention

from keras import backend as K
import tensorflow as tf

import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import random
import math
import pandas as pd 
import h5py

import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

# In[] suit for any input

def Conv2d_BN1(x, nb_filter, kernel_size, strides=(1,1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
# def Conv2d_BN2(x, nb_filter, kernel_size, strides=(4,1), padding='same'):
def Conv2d_BN2(x, nb_filter, kernel_size, strides=(2,1), padding='same'):    
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
# def Conv2dT_BN1(x, filters, kernel_size, strides=(4,1), padding='same'):
def Conv2dT_BN1(x, filters, kernel_size, strides=(2,1), padding='same'):    
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
def Conv2dT_BN2(x, filters, kernel_size, strides=(1,1), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
def Conv2dT_BN3(x, filters, kernel_size, strides=(1,1), padding='same'):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    # x = UpSampling2D(size=(4,1))(x) #1
    x = UpSampling2D(size=(2,1))(x) #1
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x    
    
def crop_and_cut(net):
    net1,net2=net
    net1_shape = net1.get_shape().as_list()
    # net2_shape = net2.get_shape().as_list()
    offsets = [0, 0, 0, 0]
    size = [-1, net1_shape[1], net1_shape[2], -1]
    net2_resize = tf.slice(net2, offsets, size)
    return net2_resize 

# def my_reshape(x,a,b):
#     return K.reshape(x,(-1,a,b)) 

# x2=Lambda(my_reshape,arguments={'a':750*2,'b':4*3})(inpt)

    
def mag_model(time_input,num=1,nb_filter=8, kernel_size=(7,1),depths=5):
    
    inpt = Input(shape=time_input,name='input')
    # Down/Encode
    convs=[None]*depths
    net = Conv2d_BN1(inpt, nb_filter, kernel_size)
    for depth in range(depths):
        filters=int(2**depth*nb_filter)
        
        net = Conv2d_BN1(net, filters, kernel_size)
        convs[depth] = net
    
        if depth < depths - 1:
            net = Conv2d_BN2(net, filters, kernel_size)
    # Reshape        
    net_shape = net.get_shape().as_list()    
    
    net=Reshape((net_shape[2]*net_shape[1],net_shape[3]))(net)
    # LSTM
    net = keras.layers.LSTM(units=filters, return_sequences=True)(net)
    # Attention
    at_x,wt = SeqSelfAttention(return_attention=True, attention_width= None,  
                            attention_activation='relu',name='Atten')(net)         
    net=Reshape((net_shape[1],net_shape[2],net_shape[3]))(at_x)    
             
    # Up/Decode
    net1=net
    for depth in range(depths-2,-1,-1):
        filters = int(2**(depth) * nb_filter)  
        net1 = Conv2dT_BN3(net1, filters, kernel_size)
        # skip and concat
        net1 =Lambda(crop_and_cut)([convs[depth], net1])
                
    outenv = Conv2D(num, kernel_size=(3,1),padding='same',name='mag')(net1)
    
    model = Model(inpt, [outenv],name='pk_model')
    return model 

# In[]
def pm_model(time_input,num=1,nb_filter=8, kernel_size=(7,1),depths=5):
    inpt = Input(shape=time_input,name='input')
    # Down/Encode
    convs=[None]*depths
    net = Conv2d_BN1(inpt, nb_filter, kernel_size)
    for depth in range(depths):
        filters=int(2**depth*nb_filter)
        
        net = Conv2d_BN1(net, filters, kernel_size)
        convs[depth] = net
        
        if depth < depths - 1:
            net = Conv2d_BN2(net, filters, kernel_size)
    # Reshape        
    net_shape = net.get_shape().as_list()    
    
    net=Reshape((net_shape[2]*net_shape[1],net_shape[3]))(net)
    # LSTM
    net = keras.layers.LSTM(units=filters, return_sequences=True)(net)
    # Attention
    at_x,wt = SeqSelfAttention(return_attention=True, attention_width= None,  
                            attention_activation='relu',name='Atten')(net)         
    net=Reshape((net_shape[1],net_shape[2],net_shape[3]))(at_x)    
    
    # Up/Decode/mag
    net1=net
    for depth in range(depths-2,-1,-1):
        filters = int(2**(depth) * nb_filter)  
        net1 = Conv2dT_BN3(net1, filters, kernel_size)
        # skip and concat
        net1 =Lambda(crop_and_cut)([convs[depth], net1])
    
    outm = Conv2D(num, kernel_size=(3,1),padding='same',name='mag')(net1)
    
    # Up/Decode/pick
    net2=net
    for depth in range(depths-2,-1,-1):
        filters = int(2**(depth) * nb_filter)  
        net2 = Conv2dT_BN3(net2, filters, kernel_size)
        # skip and concat
        net2 =Lambda(crop_and_cut)([convs[depth], net2])
    
    outp = Conv2D(num, kernel_size=(3,1),padding='same',name='pick')(net2)    

    model = Model(inpt, [outm,outp],name='pm_model')
    return model


