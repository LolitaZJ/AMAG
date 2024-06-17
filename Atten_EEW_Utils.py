#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:15:08 2023

Utils the Atten_EEW

@author: zhangj
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
import time
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

# In[]
def we(y_true):

    tmp=y_true[:,-1,0,0]
    c=[]
    for i in tmp:
        if i<=0.5:
            c.append(-i+2.5)
        elif i<=2:
            c.append(1)
        else:
           c.append((i-2)**2+2)
    return tf.convert_to_tensor(c)

def mse_mae(y_true,y_pred):
    l2 = K.mean(K.square(y_pred - y_true), axis=-1)
    l1 = K.mean(K.abs(y_pred - y_true), axis=-1)

    return l1+l2

def wmse(y_true,y_pred):
    tmp=y_true
    b=tf.ones_like(y_true)
    c=tf.ones_like(y_true)
    c=tf.where(tf.logical_and(tmp<=0.5,tmp >0),b*2,c)
    c=tf.where(tf.logical_and(tmp>2,tmp<=3),b*4,c)
    c=tf.where(tf.logical_and(tmp>3,tmp<=4),b*8,c)
    c=tf.where(tf.logical_and(tmp>4,tmp<=5),b*16,c)
    c=tf.where(tmp>5,b*32,c)
    wl2 = tf.reduce_mean(tf.multiply(c, K.square(y_pred-y_true)))
    return wl2

def wmae(y_true,y_pred):
    tmp=y_true
    b=tf.ones_like(y_true)
    c=tf.ones_like(y_true)
    c=tf.where(tf.logical_and(tmp<=0.5,tmp >0),b*2,c)
    c=tf.where(tf.logical_and(tmp>2,tmp<=3),b*4,c)
    c=tf.where(tf.logical_and(tmp>3,tmp<=4),b*8,c)
    c=tf.where(tf.logical_and(tmp>4,tmp<=5),b*16,c)
    c=tf.where(tmp>5,b*32,c)
    wl1 = tf.reduce_mean(tf.multiply(c, K.abs(y_pred-y_true)))
    return wl1

def wmse_wmae(y_true,y_pred):
    tmp=y_true
    b=tf.ones_like(y_true)
    c=tf.ones_like(y_true)
    c=tf.where(tf.logical_and(tmp<=0.5,tmp >0),b*2,c)
    c=tf.where(tf.logical_and(tmp>2,tmp<=3),b*4,c)
    c=tf.where(tf.logical_and(tmp>3,tmp<=4),b*8,c)
    c=tf.where(tf.logical_and(tmp>4,tmp<=5),b*16,c)
    c=tf.where(tmp>5,b*32,c)
    wl12 = tf.reduce_mean(tf.multiply(c, K.square(y_pred-y_true)+K.abs(y_pred-y_true) ))
    return wl12

def mae_std(y_true,y_pred):
    l1 = K.mean(K.abs(y_pred - y_true), axis=-1)

    a = tf.ones_like(y_true)
    b = tf.zeros_like(y_true)

    mask1 = tf.where(y_true>0.0,a,b )
    mask2 = K.abs(1-mask1)    
    s1 = K.std((y_true-y_pred)*mask1)+K.std((y_true-y_pred)*mask2) 
    
    return l1+s1

def mse_std(y_true,y_pred):
    l2 = K.mean(K.square(y_pred - y_true), axis=-1)

    a = tf.ones_like(y_true)
    b = tf.zeros_like(y_true)

    mask1 = tf.where(y_true>0.0,a,b )
    mask2 = K.abs(1-mask1)    
    s1 = K.std((y_true-y_pred)*mask1)+K.std((y_true-y_pred)*mask2) 
    
    return l2+s1

def msemae_std(y_true,y_pred):
    l1 = K.mean(K.abs(y_pred - y_true), axis=-1)
    l2 = K.mean(K.square(y_pred - y_true), axis=-1)
    a = tf.ones_like(y_true)
    b = tf.zeros_like(y_true)

    mask1 = tf.where(y_true>0.0,a,b )
    mask2 = K.abs(1-mask1)    
    s1 = K.std((y_true-y_pred)*mask1)+K.std((y_true-y_pred)*mask2) 
    
    return l1+l2+s1


def loss_std(y_true,y_pred):
    a = tf.ones_like(y_true)
    b = tf.zeros_like(y_true)

    mask1 = tf.where(y_true>0.0,a,b )
    mask2 = K.abs(1-mask1)    
    s1 = K.std((y_true-y_pred)*mask1)+K.std((y_true-y_pred)*mask2) 
    
    return s1


class myErr(keras.losses.Loss):
    def call(self, y_true, y_pred):
        tmp=y_true
        a=tf.ones_like(y_true)
        b=tf.ones_like(y_true)
        c=tf.ones_like(y_true)
        '''
        tmp=y_true[:,-1,0,0]
        c=tf.ones_like(y_true[:,-1,0,0])
        for i in range(len(tmp)):
            if tmp[i]<=0.5:
                c.append(-tmp[i]+2.5)
            elif tmp[i]<=2:
                c.append(tf.constant([1.0]))
            else:
                c.append((tmp[i]-2)**2+2.5)
        #c=tf.convert_to_tensor(c)
        '''
        c=tf.where(tf.logical_and(tmp<=0.5,tmp >0),b*2,c)
        #c=tf.where(tmp>0.5 and tmp<2,c,c*2)
        c=tf.where(tf.logical_and(tmp>2,tmp<=3),b*4,c)
        c=tf.where(tf.logical_and(tmp>3,tmp<=4),b*8,c)
        c=tf.where(tf.logical_and(tmp>4,tmp<=5),b*16,c)
        c=tf.where(tmp>5,b*32,c)

        #l2 = K.mean(K.square(tf.multiply(c,y_pred) - tf.multiply(c,y_true)), axis=-1)
        l2 = tf.reduce_mean(tf.multiply(c, K.square(y_pred-y_true)))
        return l2

def nloss(y_true,y_pred):
    '''
    c = we(y_true)
    tmp=y_true[:,-1,0,0]
    c=tf.ones_like(y_true[:,-1,0,0])
    c=tf.Variable(c)
    for i in range(len(tmp)):
        if tmp[i]<=0.5:
            c[i].assign(-tmp[i]+2.5)
        elif tmp[i]<=2:
            c[i].assign(1)
        else:
            c[i].assign((tmp[i]-2)**2+2)
    c=[]
    for i in range(len(tmp)):
        if tmp[i]<=0.5:
            c.append(-tmp[i]+2.5)
        elif tmp[i]<=2:
            c.append(tf.constant([1.0]))
        else:
            c.append((tmp[i]-2)**2+2.5)
    c=tf.convert_to_tensor(c)
    '''
    tmp=y_true
    c=tf.ones_like(y_true)
    c=tf.where(tmp<0.5 and tmp >0,c,c*2)
    c=tf.where(tmp>0.5 and tmp<2,c,c*2)
    c=tf.where(tmp>2 and tmp<3,c,c*4)
    c=tf.where(tmp>3 and tmp<4,c,c*8)
    c=tf.where(tmp>4 and tmp<5,c,c*16)
    l2 = K.mean(K.square(c*(y_pred - y_true)), axis=-1)
    l1 = K.mean(K.abs(c*(y_pred - y_true)), axis=-1)
    a = tf.ones_like(y_true)
    b = tf.zeros_like(y_true)

    mask1 = tf.where(y_true>0.1,a,b )
    mask2 = K.abs(1-mask1)
    # std restraint
    s1 = K.std((y_true-y_pred)*mask1)+K.std((y_true-y_pred)*mask2)    
    # lw = K.mean(K.abs(y_true-y_pred)*mask1,axis=-1)+K.mean(K.abs(y_true-y_pred)*mask2,axis=-1)
    return l1+l2+s1

# In[]
def plot_loss(history_callback,save_path=None,model='model'):
    font2 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }

    history_dict=history_callback.history

    loss_value=history_dict['loss']
    val_loss_value=history_dict['val_loss']
    
    acc_value=history_dict['accuracy']
    val_acc_value=history_dict['val_accuracy']      
    


    epochs=range(1,len(val_loss_value)+1)
    if not save_path is None:
        np.savez(save_path+'acc_loss_%s'%model,
                 loss=loss_value,val_loss=val_loss_value,
                 acc_po=acc_value,val_acc_po=val_acc_value)

    
    # acc polarity 
    figure, ax = plt.subplots(figsize=(8,6))
    plt.plot(epochs,acc_value,'b--',label='Training acc of polarity')
    plt.plot(epochs,val_acc_value,'r--',label='Validation acc of polarity')    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Epochs',font2)
    plt.ylabel('Accuracy',font2)
    plt.legend(prop=font2,loc='lower right')
    if not save_path is None:
        plt.savefig(save_path+'ACC_%s.png'%model,dpi=600)
    plt.show()    
    
    # loss 
    figure, ax = plt.subplots(figsize=(8,6))
    plt.plot(epochs,loss_value,'b',label='Training loss')
    plt.plot(epochs,val_loss_value,'r',label='Validation loss')
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Epochs',font2)
    plt.ylabel('Loss',font2)
    plt.legend(prop=font2)
    if not save_path is None:
        plt.savefig(save_path+'Loss_%s.png'%model,dpi=600)    
    plt.show()    
    
# In[]
def plot_loss_pm(history_callback,save_path=None,model='model'):
    font2 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }
    
    history_dict=history_callback.history
    
    loss_value=history_dict['loss']
    val_loss_value=history_dict['val_loss']
    
    loss_pk=history_dict['pick_loss']
    val_loss_pk=history_dict['val_pick_loss']    
    
    loss_po=history_dict['mag_loss']
    val_loss_po=history_dict['val_mag_loss']      
    
    try:
        acc_pk=history_dict['pick_accuracy']
        val_acc_pk=history_dict['val_pick_accuracy']
        acc_po=history_dict['mag_accuracy']
        val_acc_po=history_dict['val_mag_accuracy']        
    
    except:
        acc_value=history_dict['accuracy']
        val_acc_value=history_dict['val_accuracy']  
    
    epochs=range(1,len(acc_pk)+1)
    if not save_path is None:
        np.savez(save_path+'acc_loss_%s'%model,
                 loss=loss_value,val_loss=val_loss_value,
                 loss_pk=loss_pk,val_loss_pk=val_loss_pk,
                 loss_po=loss_po,val_loss_po=val_loss_po,
                 acc_pk=acc_pk,val_acc_pk=val_acc_pk,
                 acc_po=acc_po,val_acc_po=val_acc_po)
    
    # acc picking
    figure, ax = plt.subplots(figsize=(8,6))
    plt.plot(epochs,acc_pk,'b',label='Training acc of picking')
    plt.plot(epochs,val_acc_pk,'r',label='Validation acc of picking')  
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Epochs',font2)
    plt.ylabel('Accuracy',font2)
    plt.legend(prop=font2,loc='lower right')
    if not save_path is None:
        plt.savefig(save_path+'ACC_PK_%s.png'%model,dpi=600)
    plt.show()
    
    # acc polarity 
    figure, ax = plt.subplots(figsize=(8,6))
    plt.plot(epochs,acc_po,'b--',label='Training acc of polarity')
    plt.plot(epochs,val_acc_po,'r--',label='Validation acc of polarity')    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Epochs',font2)
    plt.ylabel('Accuracy',font2)
    plt.legend(prop=font2,loc='lower right')
    if not save_path is None:
        plt.savefig(save_path+'ACC_MG_%s.png'%model,dpi=600)
    plt.show()    
    
    # loss 
    figure, ax = plt.subplots(figsize=(8,6))
    plt.plot(epochs,loss_value,'b',label='Training loss')
    plt.plot(epochs,val_loss_value,'r',label='Validation loss')
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Epochs',font2)
    plt.ylabel('Loss',font2)
    plt.legend(prop=font2)
    if not save_path is None:
        plt.savefig(save_path+'Loss_%s.png'%model,dpi=600)    
    plt.show()    
    
    # loss Picking
    figure, ax = plt.subplots(figsize=(8,6))
    plt.plot(epochs,loss_pk,'b--',label='Training loss of picking')
    plt.plot(epochs,val_loss_pk,'r--',label='Validation loss of picking')    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Epochs',font2)
    plt.ylabel('Loss',font2)
    plt.legend(prop=font2)
    if not save_path is None:
        plt.savefig(save_path+'Loss_PK_%s.png'%model,dpi=600)    
    plt.show()    
    
    # loss Polarity
    figure, ax = plt.subplots(figsize=(8,6))
    plt.plot(epochs,loss_po,'b-.',label='Training loss of polarity')
    plt.plot(epochs,val_loss_po,'r-.',label='Validation loss of polarity')    
    
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('Epochs',font2)
    plt.ylabel('Loss',font2)
    plt.legend(prop=font2)
    if not save_path is None:
        plt.savefig(save_path+'Loss_MG_%s.png'%model,dpi=600)    
    plt.show()    
    
            
# In[]
try:
    from keras.utils import Sequence
except:
    from keras.utils.all_utils import Sequence
class DataGenerator_mag2(Sequence):

    def __init__(self, dtfl1,df,ev_list,batch_size=128,shuffle=True,dpss=[300],fl=1,cus_no=0,tpr=0,
                 ratio=0,noi_win=300,lm=0):
        """
        # Arguments
        ---
            files: filename.
            batch_size: . """

        self.batch_size = batch_size
        self.ev_list = ev_list
        self.dtfl1 = dtfl1
        self.df =df
        self.noi_win=noi_win
        self.dpss=dpss
        self.shuffle=shuffle
        self.ratio=ratio
        self.lm=lm
        self.fl=fl
        self.cus_no=cus_no
        self.tpr=tpr
        
    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.ev_list)// self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """
        dtfl1=h5py.File(self.dtfl1,'r')
        batch_size  =self.batch_size   
        ev_list_temp=np.array(self.ev_list)[index*batch_size:(index+1)*batch_size].tolist()
                                
        # read batch data
        X, Y= self._read_data(dtfl1,ev_list_temp,self.dpss[0])
        return ({'input': X}, {'mag':Y }) 
        dtfl1.close()
        #batch_size  =self.batch_size   
        #ev_list_temp=np.array(self.ev_list)[index*batch_size:(index+1)*batch_size].tolist()

        # read batch data
        #X, Y= self._read_data(ev_list_temp,self.dpss[0])
        #return ({'input': X}, {'mag':Y }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.ev_list)
            
    def _add_noise(self,sig,db,k):
        n=6000
        np.random.seed(k)
        noise=np.random.normal(size=(3,n))
        s2=np.sum(sig**2)/len(sig)
        n2=np.sum(noise[2,:]**2)/len(noise[2,:])
        a=(s2/n2/(10**(db/10)))**(0.5)
        noise=noise*a
        return noise
    
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            x_max=np.max(abs(data1))
            if x_max!=0.0:
                data2[i,:,:]=data1/x_max 
        return data2
        
        
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1      
    
    def _read_data(self,dtfl1, batch_files,lo):

        batch_size=self.batch_size
        noi_win=self.noi_win

        #------------------------#    
        train_x=np.zeros((batch_size,lo+noi_win,3))
        pt=np.zeros((batch_size,))  
        # st=np.zeros((batch_size,))
        mg=np.ones((batch_size,lo+noi_win,1))*self.lm
        # evt_nm=[]
        #------------------------#    
        for c, evi in enumerate(batch_files):
            nodata=0
            try:
                # evt_nm.append(evi)
                dataset = np.array(dtfl1.get('data/'+str(evi)))
                if np.size(dataset,1)<np.size(dataset,0):
                    dataset=dataset.transpose(1,0) 
            except:
                nodata=1
            if nodata==1:
                continue
                
                # print(evi)
            for ii in range(3):
                try:
                    dataset[ii,:]=self._taper(dataset[ii,:],1,100) #100
                except:
                    nodata=1
                    continue
                    # print(evi)
            if nodata==1:
                continue                    
                    
            try:    
                dataset =self._bp_filter(dataset,2,1,20,0.01)
            except:
                0
                # print(np.shape(dataset))
            
            df1=self.df.loc[self.df.trace_name==evi]
            try:
                pt[c] = df1['trace_P_arrival_sample'].tolist()[0]
            except:
                pt[c] = df1['p_arrival_sample'].tolist()[0]
                
            
            # try:
            #     st[c] = df1['trace_S_arrival_sample'].tolist()[0]
            # except:
            #     st[c] = -2
            
            mag= df1['source_magnitude'].tolist()[0]+1.0
            if self.fl==1:
                fl=np.random.randint(0,2)
            else:
                fl=1
            
            if fl==0:
                stm=int(pt[c]-lo-noi_win)
                if stm<0:
                    stm=0
                    try:
                        train_x[c,:,:3] = dataset[:,stm:stm+(lo+noi_win)].transpose((1,0)) 
                        mg[c,int(pt[c]):,0] = mag
                    except:
                        0
                        # print(evi)
                else:
                    stm=stm-noi_win
                    if stm<0:stm=0
                    try:
                        train_x[c,:,:3] = dataset[:,stm:stm+(lo+noi_win)].transpose((1,0))  
                    except:
                        0
                        # print(evi)
            else:
                if self.ratio:
                    if self.cus_no:
                        ran_win=int(self.cus_no)
                    else:
                        ran_win=int(self.ratio%10)*100                    
                    
                    # ran_win=int(self.ratio%10)*100
                else:
                    # l1=int((lo+noi_win)/100)
                    # l2=1
                    # ran_win=np.random.randint(l2,l1)*100
                    l1=int((lo+noi_win)/100-1)*100
                    l2=100
                    ran_win=np.random.randint(l2,l1)*1 
                    
                stm=int(pt[c]-ran_win)
                
                if stm<0:
                    stm=0
                    try:
                        if self.tpr:
                            trdata=np.array([self._taper(dataset[ii,stm:stm+(lo+noi_win)],1,50) for ii in range(3)])
                            train_x[c,:,:3] =trdata.transpose((1,0))
                        else:
                            train_x[c,:,:3] = dataset[:,stm:stm+(lo+noi_win)].transpose((1,0)) 
                        mg[c,int(pt[c]):,0] = mag
                    except:
                        0
                        # print(evi)
                else:
                    try:
                        if self.tpr:
                            trdata=np.array([self._taper(dataset[ii,stm:stm+(lo+noi_win)],1,50) for ii in range(3)])
                            train_x[c,:,:3] =trdata.transpose((1,0))
                        else:
                            train_x[c,:,:3] = np.array(dataset)[:,stm:stm+(lo+noi_win)].transpose((1,0)) 
                        mg[c,ran_win:,0] = mag 
                    except:
                        0
                        # print(evi)
          
        return np.expand_dims(np.nan_to_num(train_x),axis=2),np.expand_dims(np.nan_to_num(mg),axis=2) # ,evt_nm      
# In[]
try:
    from keras.utils import Sequence
except:
    from keras.utils.all_utils import Sequence
class DataGenerator_mag3(Sequence):

    def __init__(self, dtfl1,df,ev_list,batch_size=128,shuffle=True,dpss=[300],fl=1,cus_no=0,tpr=0,
                 ratio=0,noi_win=300,lm=0):
        """
        # Arguments
        ---
            files: filename.
            batch_size: . """

        self.batch_size = batch_size
        self.ev_list = ev_list
        self.dtfl1 = dtfl1
        self.df =df
        self.noi_win=noi_win
        self.dpss=dpss
        self.shuffle=shuffle
        self.ratio=ratio
        self.lm=lm
        self.fl=fl
        self.cus_no=cus_no
        self.tpr=tpr
        
    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.ev_list)// self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """
        dtfl=h5py.File(self.dtfl1,'r')
        batch_size  =self.batch_size   
        ev_list_temp=np.array(self.ev_list)[index*batch_size:(index+1)*batch_size].tolist()
                                
        # read batch data
        X, Y= self._read_data(dtfl,ev_list_temp,self.dpss[0])
        return ({'input': X}, {'mag':Y }) 
        dtfl.close()
        #batch_size  =self.batch_size   
        #ev_list_temp=np.array(self.ev_list)[index*batch_size:(index+1)*batch_size].tolist()

        # read batch data
        #X, Y= self._read_data(ev_list_temp,self.dpss[0])
        #return ({'input': X}, {'mag':Y }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.ev_list)
            
    def _add_noise(self,sig,db,k):
        n=6000
        np.random.seed(k)
        noise=np.random.normal(size=(3,n))
        s2=np.sum(sig**2)/len(sig)
        n2=np.sum(noise[2,:]**2)/len(noise[2,:])
        a=(s2/n2/(10**(db/10)))**(0.5)
        noise=noise*a
        return noise
    
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            x_max=np.max(abs(data1))
            if x_max!=0.0:
                data2[i,:,:]=data1/x_max 
        return data2
        
        
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1      
    
    def _read_data(self,dtfl, batch_files,lo):

        batch_size=self.batch_size
        noi_win=self.noi_win

        #------------------------#    
        train_x=np.zeros((batch_size,lo+noi_win,3))
        pt=np.zeros((batch_size,))  
        # st=np.zeros((batch_size,))
        mg=np.ones((batch_size,lo+noi_win,1))*self.lm
        # evt_nm=[]
        #------------------------#    
        for c, evi in enumerate(batch_files):
            # evt_nm.append(evi)
            dataset = np.array(dtfl.get('data/'+str(evi)))
            try:
                if np.size(dataset,1)<np.size(dataset,0):
                    dataset=dataset.transpose(1,0) 
            except:
                # time.sleep(0.01*np.random.randint(3))
                
                # dataset = np.array(dtfl.get('data/'+str(evi)))
                # try:
                #     if np.size(dataset,1)<np.size(dataset,0):
                #         dataset=dataset.transpose(1,0)                 
                # except:
                print(evi)
                continue
                
            for ii in range(3):
                dataset[ii,:]=self._taper(dataset[ii,:],1,100) #100

  
            dataset =self._bp_filter(dataset,2,1,20,0.01)

            df1=self.df.loc[self.df.trace_name==evi]
            try:
                pt[c] = df1['trace_P_arrival_sample'].tolist()[0]
            except:
                pt[c] = df1['p_arrival_sample'].tolist()[0]

            mag= df1['source_magnitude'].tolist()[0]+1.0
            if self.fl==1:
                fl=np.random.randint(0,2)
            else:
                fl=1
            
            if fl==0:
                stm=int(pt[c]-lo-noi_win)
                if stm>0:
                    stm=stm-noi_win
                    if stm<0:
                        stm=0
                        train_x[c,:,:3] = dataset[:,stm:stm+(lo+noi_win)].transpose((1,0))  
            else:
                if self.ratio:
                    if self.cus_no:
                        ran_win=int(self.cus_no)
                    else:
                        ran_win=int(self.ratio%10)*100                    

                else:

                    l1=int((lo+noi_win)/100-1)*100
                    l2=100
                    ran_win=np.random.randint(l2,l1)*1 
                    
                stm=int(pt[c]-ran_win)
                if stm<0:
                    stm=0
                    if self.tpr:
                        trdata=np.array([self._taper(dataset[ii,stm:stm+(lo+noi_win)],1,50) for ii in range(3)])
                        train_x[c,:,:3] =trdata.transpose((1,0))
                    else:
                        train_x[c,:,:3] = dataset[:,stm:stm+(lo+noi_win)].transpose((1,0)) 
                    mg[c,int(pt[c]):,0] = mag

                else:
                    if self.tpr:
                        trdata=np.array([self._taper(dataset[ii,stm:stm+(lo+noi_win)],1,50) for ii in range(3)])
                        train_x[c,:,:3] =trdata.transpose((1,0))
                    else:
                        train_x[c,:,:3] = np.array(dataset)[:,stm:stm+(lo+noi_win)].transpose((1,0)) 
                    mg[c,ran_win:,0] = mag 
          
        return np.expand_dims(train_x,axis=2),np.expand_dims(mg,axis=2) #,evt_nm 
# In[] 
try:
    from keras.utils import Sequence
except:
    from keras.utils.all_utils import Sequence
class DataGenerator_mag4(Sequence):

    def __init__(self, dtfl1,df,ev_list,batch_size=128,shuffle=True,dpss=[300],fl=1,cus_no=0,tpr=0,
                 ratio=0,noi_win=300,lm=0):
        """
        # Arguments
        ---
            files: filename.
            batch_size: . """

        self.batch_size = batch_size
        self.ev_list = ev_list
        self.dtfl1 = dtfl1
        self.df =df
        self.noi_win=noi_win
        self.dpss=dpss
        self.shuffle=shuffle
        self.ratio=ratio
        self.lm=lm
        self.fl=fl
        self.cus_no=cus_no
        self.tpr=tpr
        
    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.ev_list)// self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """
        # dtfl=h5py.File(self.dtfl1,'r')
        dtfl=[]
        batch_size = self.batch_size   
        ev_list_temp=np.array(self.ev_list)[index*batch_size:(index+1)*batch_size].tolist()
                                
        # read batch data
        X, Y= self._read_data(dtfl,ev_list_temp,self.dpss[0])
        return ({'input': X}, {'mag':Y }) 
        # dtfl.close()
        #batch_size  =self.batch_size   
        #ev_list_temp=np.array(self.ev_list)[index*batch_size:(index+1)*batch_size].tolist()

        # read batch data
        #X, Y= self._read_data(ev_list_temp,self.dpss[0])
        #return ({'input': X}, {'mag':Y }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.ev_list)
            
    def _add_noise(self,sig,db,k):
        n=6000
        np.random.seed(k)
        noise=np.random.normal(size=(3,n))
        s2=np.sum(sig**2)/len(sig)
        n2=np.sum(noise[2,:]**2)/len(noise[2,:])
        a=(s2/n2/(10**(db/10)))**(0.5)
        noise=noise*a
        return noise
    
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            x_max=np.max(abs(data1))
            if x_max!=0.0:
                data2[i,:,:]=data1/x_max 
        return data2
        
        
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1      
    
    def _read_data(self,dtfl, batch_files,lo):

        batch_size=self.batch_size
        noi_win=self.noi_win

        #------------------------#    
        train_x=np.zeros((batch_size,lo+noi_win,3))
        pt=np.zeros((batch_size,))  
        # st=np.zeros((batch_size,))
        mg=np.ones((batch_size,lo+noi_win,1))*self.lm
        # evt_nm=[]
        #------------------------#    
        for c, evi in enumerate(batch_files):
            # evt_nm.append(evi)
            
            dtfl=h5py.File(self.dtfl1,'r')
            dataset = np.array(dtfl.get('data/'+str(evi)))
            dtfl.close()

            try:
                if np.size(dataset,1)<np.size(dataset,0):
                    dataset=dataset.transpose(1,0) 
            except:
                # time.sleep(0.01*np.random.randint(3))
                
                # dataset = np.array(dtfl.get('data/'+str(evi)))
                # try:
                #     if np.size(dataset,1)<np.size(dataset,0):
                #         dataset=dataset.transpose(1,0)                 
                # except:
                print(evi)
                continue
                
            for ii in range(3):
                dataset[ii,:]=self._taper(dataset[ii,:],1,100) #100

  
            dataset =self._bp_filter(dataset,2,1,20,0.01)

            df1=self.df.loc[self.df.trace_name==evi]
            try:
                pt[c] = df1['trace_P_arrival_sample'].tolist()[0]
            except:
                pt[c] = df1['p_arrival_sample'].tolist()[0]

            mag= df1['source_magnitude'].tolist()[0]+1.0
            if self.fl==1:
                fl=np.random.randint(0,2)
            else:
                fl=1
            
            if fl==0:
                stm=int(pt[c]-lo-noi_win)
                if stm>0:
                    stm=stm-noi_win
                    if stm<0:
                        stm=0
                        train_x[c,:,:3] = dataset[:,stm:stm+(lo+noi_win)].transpose((1,0))  
            else:
                if self.ratio:
                    if self.cus_no:
                        ran_win=int(self.cus_no)
                    else:
                        ran_win=int(self.ratio%10)*100                    

                else:

                    l1=int((lo+noi_win)/100-1)*100
                    l2=100
                    ran_win=np.random.randint(l2,l1)*1 
                    
                stm=int(pt[c]-ran_win)
                if stm<0:
                    stm=0
                    if self.tpr:
                        trdata=np.array([self._taper(dataset[ii,stm:stm+(lo+noi_win)],1,50) for ii in range(3)])
                        train_x[c,:,:3] =trdata.transpose((1,0))
                    else:
                        train_x[c,:,:3] = dataset[:,stm:stm+(lo+noi_win)].transpose((1,0)) 
                    mg[c,int(pt[c]):,0] = mag

                else:
                    if self.tpr:
                        trdata=np.array([self._taper(dataset[ii,stm:stm+(lo+noi_win)],1,50) for ii in range(3)])
                        train_x[c,:,:3] =trdata.transpose((1,0))
                    else:
                        train_x[c,:,:3] = np.array(dataset)[:,stm:stm+(lo+noi_win)].transpose((1,0)) 
                    mg[c,ran_win:,0] = mag 
          
        return np.expand_dims(train_x,axis=2),np.expand_dims(mg,axis=2) #,evt_nm 
        
# In[] 
try:
    from keras.utils import Sequence
except:
    from keras.utils.all_utils import Sequence
class DataGenerator_mag(Sequence):

    def __init__(self, dtfl1,df,ev_list,batch_size=128,shuffle=True,fl=1,cus_no=0,tpr=0,
                 dpss=[300],ratio=0,noi_win=300,lm=-0):
        """
        # Arguments
        ---
            files: filename.
            batch_size: . """

        self.batch_size = batch_size
        self.ev_list = ev_list
        self.dtfl1 = dtfl1
        self.df =df
        self.noi_win=noi_win
        self.dpss=dpss
        self.shuffle=shuffle
        self.ratio=ratio
        self.lm=lm
        self.fl=fl
        self.cus_no=cus_no
        self.tpr=tpr
        
    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.ev_list)// self.batch_size

    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """
        dtfl1=h5py.File(self.dtfl1,'r')
        batch_size  =self.batch_size   
        ev_list_temp=np.array(self.ev_list)[index*batch_size:(index+1)*batch_size].tolist()
                                
        # read batch data
        X, Y= self._read_data(dtfl1,ev_list_temp,self.dpss[0])
        return ({'input': X}, {'mag':Y }) 
        dtfl1.close()
        #batch_size  =self.batch_size   
        #ev_list_temp=np.array(self.ev_list)[index*batch_size:(index+1)*batch_size].tolist()

        # read batch data
        #X, Y= self._read_data(ev_list_temp,self.dpss[0])
        #return ({'input': X}, {'mag':Y }) 

    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.ev_list)
            
    def _add_noise(self,sig,db,k):
        n=6000
        np.random.seed(k)
        noise=np.random.normal(size=(3,n))
        s2=np.sum(sig**2)/len(sig)
        n2=np.sum(noise[2,:]**2)/len(noise[2,:])
        a=(s2/n2/(10**(db/10)))**(0.5)
        noise=noise*a
        return noise
    
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            x_max=np.max(abs(data1))
            if x_max!=0.0:
                data2[i,:,:]=data1/x_max 
        return data2
        
        
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1      
    
    def _read_data(self,dtfl1, batch_files,lo):

        batch_size=self.batch_size
        noi_win=self.noi_win

        #------------------------#    
        train_x=np.zeros((batch_size,lo+noi_win,3))
        pt=np.zeros((batch_size,))  
        # st=np.zeros((batch_size,))
        mg=np.ones((batch_size,lo+noi_win,1))*self.lm
        # evt_nm=[]
        #------------------------#    
        for c, evi in enumerate(batch_files):
            nodata=0
            try:
                dataset = np.array(dtfl1.get('data/'+str(evi)))
                # evt_nm.append(evi)
                if np.size(dataset,1)<np.size(dataset,0):
                    dataset=dataset.transpose(1,0) 
            except:
                nodata=1
            if nodata==1:
                continue
            
            for ii in range(3):
                try:
                    dataset[ii,:]=self._taper(dataset[ii,:],1,100) 
                except:
                    nodata=1
                    continue
                    # print(evi)
            if nodata==1:
                continue                    
                    
            try:    
                dataset =self._bp_filter(dataset,2,1,20,0.01)
            except:
                print(np.shape(dataset))
            
            df1=self.df.loc[self.df.trace_name==evi]
            try:
                pt[c] = df1['trace_P_arrival_sample'].tolist()[0]
            except:
                pt[c] = df1['p_arrival_sample'].tolist()[0]    
            
            # try:
            #     st[c] = df1['trace_S_arrival_sample'].tolist()[0]
            # except:
            #     st[c] = -2
            
            mag= df1['source_magnitude'].tolist()[0]
            if self.fl==1:
                fl=np.random.randint(0,2)
            else:
                fl=1
            if fl==0:
                stm=int(pt[c]-lo-noi_win)
                if stm<0:
                    stm=0
                    try:
                        train_x[c,:,:3] = dataset[:,stm:stm+(lo+noi_win)].transpose((1,0)) 
                        mg[c,int(pt[c]):,0] = mag
                    except:
                        print(evi)
                    
                else:
                    stm=stm-noi_win
                    if stm<0:stm=0
                    try:
                        train_x[c,:,:3] = dataset[:,stm:stm+(lo+noi_win)].transpose((1,0))  
                    except:
                        print(evi)
            else:
                if self.ratio:
                    # ran_win=int(self.ratio%10)*100
                    if self.cus_no:
                        ran_win=int(self.cus_no)
                    else:
                        ran_win=int(self.ratio%10)*100  
                else:
                    # l1=int((lo+noi_win)/100)
                    # l2=1
                    # ran_win=np.random.randint(l2,l1)*100
                    l1=int((lo+noi_win)/100-1)*100
                    l2=100
                    ran_win=np.random.randint(l2,l1)*1                 
                stm=int(pt[c]-ran_win)
                if stm<0:
                    stm=0
                    try:
                        if self.tpr:
                            trdata=np.array([self._taper(dataset[ii,stm:stm+(lo+noi_win)],1,50) for ii in range(3)])
                            train_x[c,:,:3] =trdata.transpose((1,0))   
                        else:
                            train_x[c,:,:3] = dataset[:,stm:stm+(lo+noi_win)].transpose((1,0)) 
                        mg[c,int(pt[c]):,0] = mag
                    except:
                        print(evi)
                    
                else:
                    try:
                        if self.tpr:
                            trdata=np.array([self._taper(dataset[ii,stm:stm+(lo+noi_win)],1,50) for ii in range(3)])
                            train_x[c,:,:3] =trdata.transpose((1,0)) 
                        else:
                            train_x[c,:,:3] = np.array(dataset)[:,stm:stm+(lo+noi_win)].transpose((1,0)) 
                        mg[c,ran_win:,0] = mag  
                    except:
                        print(evi)

        return np.expand_dims(np.nan_to_num(train_x),axis=2),np.expand_dims(np.nan_to_num(mg),axis=2)
# In[]
def gaussian(x, sigma,  u):
    y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
    return y/np.max(abs(y))
gaus=gaussian(np.linspace(-5, 5, 100),0.5,0)
# plt.plot(gaus)
# gaus=gaus[25-10:25+10]

try:
    from keras.utils import Sequence
except:
    from keras.utils.all_utils import Sequence
    
class DataGenerator_PM(Sequence):
    
    def __init__(self, dtfl1,df,ev_list,batch_size=128,shuffle=True,fl=1,cus_no=0,tpr=0,
                 dpss=[300],ratio=0,noi_win=300,lm=0):
        """
        # Arguments
        ---
            files: filename.
            batch_size: . """
        
        self.batch_size = batch_size
        self.ev_list = ev_list
        self.dtfl1 = dtfl1
        self.df =df
        self.noi_win=noi_win
        self.dpss=dpss
        self.shuffle=shuffle
        self.ratio=ratio
        self.lm=lm
        self.fl=fl
        self.cus_no=cus_no
        self.tpr=tpr
    
    def __len__(self):
        """return: steps num of one epoch. """
        
        return len(self.ev_list)// self.batch_size
    
    def __getitem__(self, index):
        """Gets the `index-th` batch.
        ---
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch data. """
        dtfl1=h5py.File(self.dtfl1,'r')
        batch_size  =self.batch_size   
        ev_list_temp=np.array(self.ev_list)[index*batch_size:(index+1)*batch_size].tolist()
        
        # read batch data
        X, Y, Z= self._read_data(dtfl1,ev_list_temp,self.dpss[0])
        return ({'input': X}, {'mag':Y,'pick':Z }) 
        dtfl1.close()
        #batch_size  =self.batch_size   
        #ev_list_temp=np.array(self.ev_list)[index*batch_size:(index+1)*batch_size].tolist()
        
        # read batch data
        #X, Y= self._read_data(ev_list_temp,self.dpss[0])
        #return ({'input': X}, {'mag':Y }) 
    
    def on_epoch_end(self):
        """shuffle data after one epoch. """
        if self.shuffle == True:
            np.random.shuffle(self.ev_list)
    
    def _add_noise(self,sig,db,k):
        n=6000
        np.random.seed(k)
        noise=np.random.normal(size=(3,n))
        s2=np.sum(sig**2)/len(sig)
        n2=np.sum(noise[2,:]**2)/len(noise[2,:])
        a=(s2/n2/(10**(db/10)))**(0.5)
        noise=noise*a
        return noise
    
    def _bp_filter(self,data,n,n1,n2,dt):
        wn1=n1*2*dt
        wn2=n2*2*dt
        b, a = signal.butter(n, [wn1,wn2], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData
    
    def _normal3(self,data):  
        data2=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            data1=data[i,:,:]
            x_max=np.max(abs(data1))
            if x_max!=0.0:
                data2[i,:,:]=data1/x_max 
        return data2
    
    
    def _taper(self,data,n,N):
        nn=len(data)
        if n==1:
            w=math.pi/N
            F0=0.5
            F1=0.5
        elif n==2:
            w=math.pi/N
            F0=0.54
            F1=0.46
        else:
            w=math.pi/N/2
            F0=1
            F1=1
        win=np.ones((nn,1))
        for i in range(N):
            win[i]=(F0-F1*math.cos(w*(i-1)))
        win1=np.flipud(win)
        
        data1=data*win.reshape(win.shape[0],)
        data1=data1*win1.reshape(win1.shape[0],)
        return data1      
    
    def _read_data(self,dtfl1, batch_files,lo):
        
        gh=50
        batch_size=self.batch_size
        noi_win=self.noi_win
        
        #------------------------#    
        train_x=np.zeros((batch_size,lo+noi_win,3))
        pt=np.zeros((batch_size,))  
        # st=np.zeros((batch_size,))
        mg=np.ones((batch_size,lo+noi_win,1))*self.lm
        pk=np.zeros((batch_size,lo+noi_win,1))
        # evt_nm=[]
        #------------------------#    
        for c, evi in enumerate(batch_files):
            nodata=0
            try:
                dataset = np.array(dtfl1.get('data/'+str(evi)))
                # evt_nm.append(evi)
                if np.size(dataset,1)<np.size(dataset,0):
                    dataset=dataset.transpose(1,0) 
            except:
                nodata=1
            if nodata==1:
                continue
            
            for ii in range(3):
                try:
                    dataset[ii,:]=self._taper(dataset[ii,:],1,20) #100
                except:
                    nodata=1
                    continue
                    # print(evi)
            if nodata==1:
                continue                    
            
            try:    
                dataset =self._bp_filter(dataset,2,1,20,0.01)
            except:
                print(np.shape(dataset))
            
            df1=self.df.loc[self.df.trace_name==evi]
            try:
                pt[c] = df1['trace_P_arrival_sample'].tolist()[0]
            except:
                pt[c] = df1['p_arrival_sample'].tolist()[0]    
            
            # try:
            #     st[c] = df1['trace_S_arrival_sample'].tolist()[0]
            # except:
            #     st[c] = -2
            
            mag= df1['source_magnitude'].tolist()[0]+1.0
            if self.fl==1:
                fl=np.random.randint(0,2)
            else:
                fl=1
            if fl==0 and pt[c]>800:
                stm=int(pt[c]-lo-noi_win)
                if stm<100:
                    stm=100
                    try:
                        pt[c]=pt[c]-stm
                        train_x[c,:,:3] = dataset[:,stm:stm+(lo+noi_win)].transpose((1,0)) 
                        mg[c,int(pt[c]):,0] = mag
                        tt1=int(pt[c])-gh
                        tt2=tt1+gh*2
                        if tt2>lo+noi_win:
                            tt2=lo+noi_win 
                        pk[c,tt1:tt2,0] = gaus[:tt2-tt1]
                    except:
                        0
                        # print(0,evi)
                
                else:
                    stm=stm-noi_win
                    if stm<100:stm=100
                    try:
                        train_x[c,:,:3] = dataset[:,stm:stm+(lo+noi_win)].transpose((1,0))  
                    except:
                        print(1,evi)
            else:
                if self.ratio:
                    # ran_win=int(self.ratio%10)*100
                    if self.cus_no:
                        ran_win=int(self.cus_no)
                    else:
                        ran_win=int(self.ratio%10)*100  
                else:
                    # l1=int((lo+noi_win)/100)
                    # l2=1
                    # ran_win=np.random.randint(l2,l1)*100
                    l1=int((lo+noi_win)/100-1)*100
                    l2=100
                    ran_win=np.random.randint(l2,l1)*1                 
                stm=int(pt[c]-ran_win)
                if stm<100:
                    stm=100
                    try:
                        pt[c]=pt[c]-stm
                        if self.tpr:
                            trdata=np.array([self._taper(dataset[ii,stm:stm+(lo+noi_win)],1,50) for ii in range(3)])
                            train_x[c,:,:3] =trdata.transpose((1,0))   
                        else:
                            train_x[c,:,:3] = dataset[:,stm:stm+(lo+noi_win)].transpose((1,0)) 
                        mg[c,int(pt[c]):,0] = mag
                        
                        tt1=int(pt[c])-gh
                        tt2=tt1+2*gh
                        if tt2>lo+noi_win:
                            tt2=lo+noi_win 
                            pk[c,tt1:tt2,0] = gaus[:tt2-tt1] 
                        elif tt1<0:
                            tt1=0
                            pk[c,tt1:tt2,0] = gaus[-tt2:] 
                        else:
                            pk[c,tt1:tt2,0] = gaus
                        
                    except:
                        print(2,evi)
                
                else:
                    try:
                        if self.tpr:
                            trdata=np.array([self._taper(dataset[ii,stm:stm+(lo+noi_win)],1,50) for ii in range(3)])
                            train_x[c,:,:3] =trdata.transpose((1,0)) 
                        else:
                            train_x[c,:,:3] = np.array(dataset)[:,stm:stm+(lo+noi_win)].transpose((1,0)) 
                        mg[c,ran_win:,0] = mag 
                        
                        tt1=ran_win-gh
                        tt2=tt1+2*gh
                        if tt2>lo+noi_win:
                            tt2=lo+noi_win 
                            pk[c,tt1:tt2,0] = gaus[:tt2-tt1] 
                        elif tt1<0:
                            tt1=0
                            pk[c,tt1:tt2,0] = gaus[-tt2:] 
                        else:
                            pk[c,tt1:tt2,0] = gaus                        
                    except:
                        print(3,evi)
        
        return np.expand_dims(np.nan_to_num(train_x),axis=2),np.expand_dims(np.nan_to_num(mg),axis=2),np.expand_dims(np.nan_to_num(pk),axis=2)