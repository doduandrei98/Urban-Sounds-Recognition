# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:26:31 2019

@author: Dodzilla
"""

import pandas as pd
import os
import sys
sys.path.append("..")
import numpy as np
from scipy.io import wavfile
import Edit_Module as em
import Neural_N as NN
from keras.models import load_model
import Predictie as pr
import matplotlib.pyplot as plt
#import plotly.express as px
from importlib import reload



df = pd.read_excel('../train.xlsx')
df.set_index('ID',inplace=True)
classes = list(np.unique(df.Class))
for f in df.index:
    rate, signal = wavfile.read('../Clean/'+str(f)+'.wav')
    df.at[f, 'length'] = signal.shape[0]/rate
inf = NN.Informatii_Filtre()
class_dist = df.groupby(['Class'])['length'].mean()
prob_dist = class_dist / class_dist.sum()

lista = df.index
p_path = os.path.join('..','pickles','conv.p')
'''NN.Salvare_Modele(lista, df, inf, classes) '''    
'''em.drow_all(classes, df,class_dist)'''

handle = open(p_path,'rb')  
model = load_model(inf.model_path)

y_true ,y_pred = pr.build_prediction('../Ctest/5.wav', handle, classes, model)

final = pr.raport_pred(y_pred, classes)
em.drow_histograma(final)




