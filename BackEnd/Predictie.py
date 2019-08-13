# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 18:59:32 2019

@author: Dodzilla
"""


from scipy.io import wavfile
import numpy as np
from python_speech_features import mfcc
import pickle

def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num 
def build_prediction(audio_file,handle,classes,model):
    
   y_true = []
   y_pred = []
   rate, wav = wavfile.read(audio_file)
   c = classes
   inf = pickle.load(handle)
   y_prob = []
   x=[]
   for i in range(0, wav.shape[0]-inf.step, inf.step):
            sample = wav[i:i+inf.step]
            x = mfcc(sample, rate, numcep=inf.nfeat, nfilt=inf.nfilt, nfft=inf.nfft)
            x = (x - inf.min) / (inf.max - inf.min)
            x = x.reshape(1, x.shape[0], x.shape[1], 1)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
   y_true = most_frequent(y_pred)
   return  c[y_true], y_pred

def raport_pred(y_pred,classes):
    fn = dict()
    num_fn =dict()
    final= dict()
    suma = 0
    for i in range(0,len(classes)):
        fn[i] = 0
        num_fn[i] = classes[i]
    for i in y_pred:
        fn[i] += 1
        suma +=1
    for i in range(0,len(classes)):
        final[num_fn[i]] = round(fn[i]/suma,3)
    return final