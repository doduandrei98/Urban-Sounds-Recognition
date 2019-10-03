# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:24:21 2019

@author: Dodzilla
"""

import soundfile
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa

def desen_diagrama(df,class_dist):
    fig , ax = plt.subplots()
    ax.set_title('Distribuirea Claselor',y=1.08)
    ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',shadow=False,startangle=90)
    ax.axis('equal')
    # plt.show()
    plt.savefig(os.path.join('../Images/', 'Diagram.png'), bbox_inches='tight')
    plt.close(fig)
    
def desen_semnale(semnale,classes):
    figura, axe = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
    figura.suptitle('Serii Timp', size=16)
    j = 0
    for x in range(2):
        for y in range(0,(classes+1)//2):
            axe[x,y].set_title(list(semnale.keys())[j])
            axe[x,y].plot(list(semnale.values())[j])
            axe[x,y].get_xaxis().set_visible(False)
            axe[x,y].get_yaxis().set_visible(False)
            j += 1
    # plt.show()
    plt.savefig(os.path.join('../Images/', 'Semnale.png'), bbox_inches='tight')
    plt.close(figura)

def desen_fft(fft,classes):
    figura, axe = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
    figura.suptitle('Transformare Fourier', size=16)
    j = 0
    for x in range(2):
        for y in range(0,(classes+1)//2):
            data = list(fft.values())[j]
            Y, freq = data[0], data[1]
            axe[x,y].set_title(list(fft.keys())[j])
            axe[x,y].plot(freq, Y)
            axe[x,y].get_xaxis().set_visible(False)
            axe[x,y].get_yaxis().set_visible(False)
            j += 1
    # plt.show()
    plt.savefig(os.path.join('../Images/', 'FFT.png'), bbox_inches='tight')
    plt.close(figura)

def desen_fbank(fbank,classes):
    figura, axe = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
    figura.suptitle('Coeficient Bank Filtru ', size=16)
    j = 0
    for x in range(2):
        for y in range(0,(classes+1)//2):
            axe[x,y].set_title(list(fbank.keys())[j])
            axe[x,y].imshow(list(fbank.values())[j],cmap='inferno', interpolation='nearest')
            axe[x,y].get_xaxis().set_visible(False)
            axe[x,y].get_yaxis().set_visible(False)
            j += 1
    # plt.show()
    plt.savefig(os.path.join('../Images/', 'Fbank.png'), bbox_inches='tight')
    plt.close(figura)

def desen_mfccs(mfccs,classes):
    figura, axe = plt.subplots(nrows=2, ncols=5, sharex=False,sharey=True, figsize=(20,5))
    figura.suptitle('Frecventa Mel', size=16)
    j = 0
    for x in range(2):
        for y in range(0,(classes+1)//2):
            axe[x,y].set_title(list(mfccs.keys())[j])
            axe[x,y].imshow(list(mfccs.values())[j],cmap='inferno', interpolation='nearest')
            axe[x,y].get_xaxis().set_visible(False)
            axe[x,y].get_yaxis().set_visible(False)
            j += 1
    # plt.show()
    plt.savefig(os.path.join('../Images/', 'MFC.png'), bbox_inches='tight')
    plt.close(figura)

def envelope(signal , rate, threshold):
    mask = []
    signal = pd.Series(signal).apply(np.abs)
    signal_mean = signal.rolling(window= int(rate/10),min_periods=1,center=True).mean()
    for mean in signal_mean:
        if mean > threshold :
            mask.append(True)
        else:
            mask.append(False)
    return mask

def calc_fft(signal , rate):
    n = len(signal)
    freq = np.fft.rfftfreq(n , 1/rate)
    magnitude = abs(np.fft.rfft(signal)/n)
    return (magnitude,freq)

'''urban-sound-classification/Train/'''

def convertire(locatie, lista,destinatie):
    for f in lista:
        signal, rate = soundfile.read(locatie+'/'+str(f)+'.wav')
        soundfile.write(destinatie+'/'+str(f)+'.wav', signal, rate, subtype='PCM_16')
            
def curatare(sursa,lista,destinatie):
   if len(os.listdir(destinatie))==0:
       for f in tqdm(lista):
           signal ,rate = librosa.load(sursa+'/'+str(f)+'.wav',sr=16000)
           mask = envelope(signal , rate , 0.0005)
           wavfile.write(filename=destinatie+'/'+str(f)+'.wav',
                         rate=rate, data=signal[mask])


def drow_all(classes,df,class_dist):
    
    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}

    for c in classes:
        wav_file=df[df.Class == c].iloc[0,0]
        signal,rate=librosa.load('../Train/'+str(wav_file)+'.wav',sr=44100)
        mask = envelope(signal , rate , 0.005)
        signal = signal[mask]
        signals[c] = signal
        fft[c] = calc_fft(signal , rate)
        bank=logfbank(signal[:rate], rate , nfilt=26, nfft=1103)
        fbank[c] = bank
        mel = mfcc(signal[:rate] , rate , numcep=13, nfilt=26, nfft=1103).T
        mfccs[c] = mel
    desen_diagrama(df,class_dist)
    desen_semnale(signals,len(classes))
    desen_fft(fft,len(classes))
    desen_fbank(fbank,len(classes))
    desen_mfccs(mfccs,len(classes))
    
def drow_histograma(final):
    fig = plt.figure()
    plt.bar(list(final.keys()),list(final.values()))
    plt.xlabel("Sunete")
    plt.ylabel("Probabilitate")
    plt.title('Histograma')
    plt.ylim(0,1)
    plt.grid(axis = 'y')
    plt.xticks(rotation = 90)
    plt.tight_layout()
    x= plt.xticks()[0]
    for ind, val in enumerate(list(final.values())):
        plt.text(x[ind] - 0.25, val + 0.01, str(val))
    #plt.show()
    plt.savefig(os.path.join('../Images/', 'Histogram.png'), bbox_inches='tight')
    plt.close(fig)
    
