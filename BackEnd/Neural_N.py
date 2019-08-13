# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:33:32 2019

@author: Dodzilla
"""

import os
from scipy.io import wavfile
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from keras.callbacks import ModelCheckpoint

class Informatii_Filtre:
    def __init__(self, nfilt=26, nfeat=13, nfft=512, rate=16000):
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join('..','Models','conv.model')
        self.p_path = os.path.join('..','Pickles','conv.p')
        
def Verificare_Date(config):
    if os.path.isfile(config.p_path):
        with open(config.p_path,'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None
    
def Generare_Model(forme):
    model = Sequential()
    model.add(Conv2D(16, (3,3),activation='relu', strides=(1,1),padding='same', input_shape=forme))
    model.add(Conv2D(32,(3,3),activation='relu',strides=(1, 1),padding='same'))
    model.add(Conv2D(64,(3,3),activation='relu',strides=(1, 1),padding='same'))
    model.add(Conv2D(128,(3,3),activation='relu',strides=(1, 1),padding='same'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    return model

def Construire_Modele(lista,df,inf,classes):
    x = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for file in tqdm(lista):
        rate, wav = wavfile.read('Clean/'+ str(file)+'.wav')
        Class = df.at[file, 'Class']
        chunks = [wav[x:x+inf.step] for x in range(0,len(wav),inf.step)]
        if len(chunks) == 1 and len(chunks[0]<inf.step) :
            new_chunks = list(chunks[0])
            while(len(new_chunks)<inf.step):
                for i in range(0,abs(len(new_chunks)-inf.step)):
                    new_chunks.append(new_chunks[i])
            chunks =np.array([new_chunks])
        if len(chunks) > 1 and len(chunks[-1])<inf.step :
            new_chunks = list(chunks[-1])
            while(len(new_chunks)<inf.step):
                for i in range(0,abs(len(new_chunks)-inf.step)):
                    new_chunks.append(new_chunks[i])
            chunks[-1] =np.array(new_chunks)
        for i in range(0,len(chunks)):
            x_sample = mfcc(chunks[i],  rate, numcep=inf.nfeat, nfilt=inf.nfilt,
                        nfft=inf.nfft)
            x.append(x_sample)
            y.append(classes.index(Class))
            _min = min(np.amin(x_sample), _min)
            _max = max(np.amax(x_sample), _max)
    inf.min = _min
    inf.max = _max
    x, y = np.array(x), np.array(y)
    x = (x- _min) / (_max - _min)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    y = to_categorical(y, num_classes = 10)
    with open(inf.p_path, 'wb') as handle:
        pickle.dump(inf, handle, protocol = 2)
    return x, y

def Salvare_Modele(lista,df,inf,classes):
    x, y = Construire_Modele(lista,df,inf,classes)
    y_flat = np.argmax(y, axis=1)
    input_shape = (x.shape[1], x.shape[2], 1)
    model = Generare_Model(input_shape)
    class_w = compute_class_weight('balanced',np.unique(y_flat),y_flat)
    ch_point = ModelCheckpoint(inf.model_path, monitor='val_acc', 
                                   verbose=1, mode='max', 
                             save_best_only=True,save_weights_only=False,period=1)
    model.fit(x, y,epochs=20, batch_size=32,shuffle=True,
          class_weight=class_w)
    model.save(inf.model_path)
    
     
    