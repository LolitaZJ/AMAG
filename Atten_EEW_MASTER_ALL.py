#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 20:29:20 2024

@author: zhangj2

# fast run!!!
nohup python Atten_EEW_MASTER_ALL.py --batch_size=1024 --loss=mse --save_name=AMAG_d4k5 --depth=4 --kernel_size=5 --ratio=0 --mode=train --GPU=3 > AMAG_d4k5.txt 2>&1 

# all STEAD data
nohup python Atten_EEW_MASTER_ALL.py --fast False --batch_size=1024 --loss=mse --save_name=AMAG_d4k5 --depth=4 --kernel_size=5 --ratio=0 --mode=train --GPU=3 > AMAG_d4k5.txt 2>&1 

# INSTANCE data
change data_path csv_path; set --stead False


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
from scipy import signal
from scipy import interpolate
import random
import math
import pandas as pd 
import h5py

import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')


from Atten_EEW_Model2 import mag_model
from Atten_EEW_Utils import DataGenerator_mag2, plot_loss,mse_mae,wmae,wmse,wmse_wmae,myErr
from Atten_EEW_Utils import mse_std,mae_std,msemae_std,loss_std

# In[]
# Set GPU
#==========================================# 
def start_gpu(args):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print('Physical GPU：', len(gpus))
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print('Logical GPU：', len(logical_gpus))

#==========================================#
# Set Configures
#==========================================# 
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU",
                        default="3",
                        help="set gpu ids") 
    
    parser.add_argument("--input_size",
                        default=(600,1,3),
                        help="input size (x,y,z)/(x,y)")    
    
    parser.add_argument("--model_name",
                        default="mag_model",
                        help="mag_model")
                        
    parser.add_argument("--mode",
                        default="train",
                        help="train/test")
                        
    parser.add_argument("--save_name",
                        default="AMAG_V01",
                        help="save model name")
    
    parser.add_argument("--save_path",
                        default="./fig/",
                        help="save path for loss/acc/others")  
    
    parser.add_argument("--save_model",
                        default="./model/",
                        help="save path for model")     
    
    
    parser.add_argument("--data_path",
                        default="/data2/share/STEAD/merged.hdf5",
                        help="data path")   

    parser.add_argument("--csv_path",
                        default="/data2/share/STEAD/merged.csv",
                        help="csv path") 
    
    parser.add_argument("--kernel_size",
                        default=5,
                        type=int,
                        help="kernel size (7,1) (5,1) (3,1)") 

    parser.add_argument("--depth",
                        default=4,
                        type=int,
                        help="depth 5,4,3")

    parser.add_argument("--epochs",
                        default=100,
                        type=int,
                        help="number of epochs (default: 100)")
    
    parser.add_argument("--batch_size",
                        default=1024,
                        type=int,
                        help="batch size")
    
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="learning rate")
    
    parser.add_argument("--patience",
                        default=10,
                        type=int,
                        help="early stopping")
    
    parser.add_argument("--monitor",
                        default="val_loss",
                        help="monitor the val_loss/loss/acc/val_acc")  
    
    parser.add_argument("--monitor_mode",
                        default="min",
                        help="min/max/auto") 
    
    parser.add_argument("--loss",
                        default='mse',
                        help="loss fucntion (mse/mae/nloss1/nloss2)")  

    parser.add_argument("--ratio",
                        default=0,
                        type=int,
                        help="len_noise,len_signal")
    
    parser.add_argument("--dpss",
                        default=300,
                        type=int,
                        help="len_signal") 
    
    parser.add_argument("--noi_win",
                        default=300,
                        type=int,
                        help="len_noise")    
        
    parser.add_argument("--test",
                        default=True,
                        help="test data")         

    parser.add_argument("--stead",
                        default=True,
                        help="stead data") 
    parser.add_argument("--fast",
                        default=True,
                        help="fast traing")     
    args = parser.parse_args()
    return args

# In[] main
if __name__ == '__main__':
    # In[]
    
    args = read_args()
    start_gpu(args)
    # args.ratio=33
    print('=================================')
    print(args.depth,args.kernel_size,args.loss,args.ratio)
    print('=================================')
    if args.ratio!=0:
        args.save_name='%s_Ratio_%d_EEW'%(args.save_name,args.ratio)
        print(args.save_name)
        args.dpss=int(args.ratio/10)*100
        args.noi_win=int(args.ratio%10)*100
    else:
        print(args.save_name)
        
    model=mag_model(time_input=(args.noi_win+args.dpss,1,3),kernel_size=(args.kernel_size,1),depths=args.depth)
    # model.summary() 
    #---------------------------------#
    np.random.seed(7)
    
    if args.stead:
        dtfl1=args.data_path
        df=pd.read_csv(args.csv_path)        
        df['snr_db']=df['snr_db'].str.split('[\[\]\s\n]+').str[-2].astype('float')
        ev_list = df.loc[(df.source_magnitude_type=='ml') & (  df['source_magnitude']>0 ) & (  df['snr_db']>10 ) ]['trace_name'].tolist()
            
    else:
        dtfl1=args.data_path
        df=pd.read_csv(args.csv_path)
        ev_list = df.loc[(df.source_magnitude_type=='ML') & (  df['source_magnitude']>0 ) & (  df['trace_Z_snr_db']>10 )]['trace_name'].tolist()
    
    np.random.shuffle(ev_list)
    
    inx=int(len(ev_list)*0.8)
    inx1=int(len(ev_list)*0.9)
    inx2=int(len(ev_list))

    print('========')
    print('Dataset samples %d'%len(ev_list))
    print('Training samples %d'%inx)
    print('Validation samples %d'%(inx1-inx))
    print('Testing samples %d'%(inx2-inx1))
    print('========')
    train_list=ev_list[:inx]
    valid_list=ev_list[inx:inx1]
    test_list=ev_list[inx1:inx2]
    
    # for fast training
    # if args.fast:
    #     train_list=train_list[:10240]
    #     valid_list=valid_list[:1024]
    #     test_list=test_list[:1024]
    
    gen_train = DataGenerator_mag2(dtfl1,df,train_list,batch_size=args.batch_size,dpss=[args.dpss],ratio=args.ratio,noi_win=args.noi_win)
    gen_valid = DataGenerator_mag2(dtfl1,df,valid_list,batch_size=args.batch_size,dpss=[args.dpss],ratio=args.ratio,noi_win=args.noi_win)
    
    
    if args.mode=="train":
        model.compile(loss=args.loss,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
        if args.loss=='wmse':
            model.compile(loss=wmse,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
        if args.loss=='wmae':
            model.compile(loss=wmae,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
        if args.loss=='wmse_wmae':
            model.compile(loss=wmse_wmae,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
        if args.loss=='mse_mae':
            model.compile(loss=mse_mae,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
            
        if args.loss=='mae_std':
            model.compile(loss=mae_std,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
        if args.loss=='mse_std':
            model.compile(loss=mse_std,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
        if args.loss=='msemae_std':
            model.compile(loss=msemae_std,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])            
        if args.loss=='loss_std':
            model.compile(loss=loss_std,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])    
            
        if not os.path.exists(args.save_model):
            os.mkdir(args.save_model)              
    
        saveBestModel= ModelCheckpoint(args.save_model+'%s.h5'%args.save_name, monitor=args.monitor, verbose=1, save_best_only=True,mode=args.monitor_mode)
        estop = EarlyStopping(monitor=args.monitor, patience=args.patience, verbose=0, mode=args.monitor_mode)
        callbacks_list = [saveBestModel,estop]
        
        # fit
        begin = datetime.datetime.now()

        batch_size=args.batch_size
        steps_per_epoch=int(len(train_list)/batch_size)
        validation_steps=int(len(valid_list)/batch_size)
        history_callback=model.fit_generator(
                                            generator=gen_train,  
                                            steps_per_epoch=steps_per_epoch,
                                            epochs=args.epochs, 
                                              verbose=1,
                                               workers=4,
                                               use_multiprocessing=True, 
                                              callbacks=callbacks_list,
                                              validation_data=gen_valid, 
                                              validation_steps=validation_steps)

        end = datetime.datetime.now()
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)        
                
        plot_loss(history_callback,save_path=args.save_path,model=args.save_name)
        
        print('Training time:',end-begin)   
    #=========================================================#  
    if args.test:
        # read data
        test_size=1024
        gen_test = DataGenerator_mag2(dtfl1,df,test_list,batch_size=test_size,dpss=[args.dpss],ratio=args.ratio,noi_win=args.noi_win)

        x_test=[]
        y_test=[]
        tmp1=iter(gen_test)
        for _ in range( len(gen_test)):
            tmp=next(tmp1)
            x_test.extend(tmp[0]['input'])
            y_test.extend(tmp[1]['mag'])
        
        x_test=np.array(x_test)
        y_test=np.array(y_test)
        
        # load model
        #model=load_model('./model/%s.h5'%args.save_name ,custom_objects=SeqSelfAttention.get_custom_objects())
        model=load_model(args.save_model+'%s.h5'%args.save_name ,
                         custom_objects={'SeqSelfAttention':SeqSelfAttention,
                                         'mse_mae':mse_mae,
                                         'wmse':wmse,
                                         'wmae':wmae,
                                         'wmse_wmae':wmse_wmae,
                                         'mse_std':mse_std,
                                         'mae_std':mae_std, 
                                         'msemae_std':msemae_std,
                                         'loss_std':loss_std
                                         })
        p_test=model.predict(gen_test)
        
        # plot results
        ll=args.noi_win+args.dpss
        tt=np.arange(ll)*0.01
        for i in range(0,16,4):
            print(i)
            fig, ax = plt.subplots(3,1, figsize=(8, 6))  
            ax[0].plot(tt,x_test[i,:,0,0]/np.max(abs(x_test[i,:,0,0])),c='k',label='Original data')
            ax[0].set_ylim(-1.1, 1.1  )
            ax[0].set_xlim(0, ll//100)
            ax[1].plot(tt,p_test[i,:,0,0],c='k',label='Predicted') 
            ax[2].plot(tt,y_test[i,:,0,0],c='k',label='Ground True')
            
            for ax1 in ax[1:]:
                ax1.legend()
                ax1.axvline(0, c='k')
                ax1.set_xlim(0, ll//100)
                ax1.set_ylim( np.min([np.min(y_test[i,:,0,0]),np.min(p_test[i,:,0,0])])-0.1, 0.1+np.max( [np.max(y_test[i,:,0,0]),np.max(p_test[i,:,0,0])] ))
            ax1.set_xlabel('Time (s)')
            if not args.save_path is None:
                plt.savefig(args.save_path+'res_%s_%d.png'%(args.save_name,i),dpi=600)
                     
            
        p_value=[ np.max(p_test[i,:,0,0]) for i in range(len(p_test))  ]
        
        x_value=[ np.max(y_test[i,:,0,0]) for i in range(len(y_test))  ]
        
        #================================#       
        fig, ax = plt.subplots(1,1, figsize=(6, 6))     
        plt.plot(x_value,p_value,'ko')
        plt.plot([-1,6],[-1,6],'r-.')
        plt.xlim([-1,6])
        plt.ylim([-1,6])
        plt.xlabel('Ground True')
        plt.ylabel('Prediction')
        plt.savefig(args.save_path+'diagonal_%s.png'%(args.save_name),dpi=600)
        plt.show()    
        
        #================================#
        inx=[ i for i in range(len(x_value)) if x_value[i]>0.5 ]
        
        err_mag=np.array(p_value)[inx]-np.array(x_value)[inx]
        
        fig, ax = plt.subplots(1,1, figsize=(8, 6))     
        plt.hist(err_mag)
        ll=int(np.max(abs(err_mag))+0.5)
        # plt.plot([-5,5],[-5,5],'r-.')
        plt.xlim([-ll,ll])
        # plt.ylim([-4.5,5])
        plt.xlabel('Error of magnitude estimation')
        plt.ylabel('Frequency')
        plt.savefig(args.save_path+'err_%s.png'%(args.save_name),dpi=600)
        plt.show()  
        
        print('=============%s================'%args.save_name)
        print('MAE: %f'%np.mean(abs(err_mag)))   
        print('===============================')             
# In[]
        
        
        
    
