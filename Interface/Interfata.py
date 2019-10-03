# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:47:20 2019

@author: Dodzilla
"""

import pandas as pd
import os
import sys
sys.path.append("../BackEnd")
import wx
import wx.lib.scrolledpanel
import os
import numpy as np
from scipy.io import wavfile
import Edit_Module as em
import Neural_N as NN
from keras.models import load_model
import Predictie as pr
import matplotlib.pyplot as plt
#import plotly.express as px
from importlib import reload
import librosa

df = pd.read_excel('../train.xlsx')
df.set_index('ID',inplace=True)
classes = list(np.unique(df.Class))
for f in df.index:
    rate, signal = wavfile.read('../Clean/'+str(f)+'.wav')
    df.at[f, 'length'] = signal.shape[0]/rate
inf = NN.Informatii_Filtre()
class_dist = df.groupby(['Class'])['length'].mean()
df.reset_index(inplace=True)
prob_dist = class_dist / class_dist.sum()

p_path = os.path.join('..','pickles','conv.p')

model = load_model(inf.model_path)
signals = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes:
    wav_file=df[df.Class == c].iloc[0,0]
    signal,rate=librosa.load('../Train/'+str(wav_file)+'.wav',sr=44100)
    mask = em.envelope(signal , rate , 0.005)
    signal = signal[mask]
    signals[c] = signal
    fft[c] = em.calc_fft(signal , rate)
    bank=em.logfbank(signal[:rate], rate , nfilt=26, nfft=1103)
    fbank[c] = bank
    mel = em.mfcc(signal[:rate] , rate , numcep=13, nfilt=26, nfft=1103).T
    mfccs[c] = mel

class Interfata(wx.Frame):

    def __init__(self, parent, title):
        super(Interfata, self).__init__(parent, title=title,
            size=(wx.DisplaySize()[0]//1.1, wx.DisplaySize()[1]//1.1))
        self.Centre()
        self.Init()

    def Init(self):
        nb = wx.Notebook(self)
        nb.AddPage(MyPanel1(nb), "Histograma")
        #nb.AddPage(MyPanel2(nb), "Analiza")
        nb.AddPage(MyPanel3(nb), "Vizualiare")
        #self.Centre()
        self.Show(True)

class MyPanel1(wx.Panel):
    def __init__(self, parent):
        super(MyPanel1, self).__init__(parent)
        self.Template()
        self.Centre()

    def Template(self):
        panel = self
        panel.SetBackgroundColour('#4f5049')
        panel.currentDirectory = os.getcwd()


        panel.vbox = wx.BoxSizer(wx.VERTICAL)
        panel.tbox = wx.BoxSizer(wx.HORIZONTAL)
        panel.bbox = wx.BoxSizer(wx.VERTICAL)
        panel.browser = wx.BoxSizer(wx.HORIZONTAL)
        panel.buttons = wx.BoxSizer(wx.HORIZONTAL)
        panel.midP = wx.Panel(panel)
        panel.midP.SetBackgroundColour('#ededed')
        panel.midP.box = wx.BoxSizer(wx.HORIZONTAL)
        panel.midP.IMG = 0

        panel.vbox.Add(panel.tbox, wx.ID_ANY, wx.EXPAND | wx.ALL, 20)
        panel.tbox.Add(panel.midP, wx.ID_ANY, wx.EXPAND | wx.ALL)
        panel.vbox.Add(panel.bbox, flag=wx.ALIGN_CENTER | wx.EXPAND | wx.BOTTOM, border=20)
        panel.bbox.Add(panel.browser, flag=wx.EXPAND | wx.RIGHT | wx.LEFT, border=200)
        panel.bbox.Add(panel.buttons, flag=wx.ALIGN_CENTER | wx.TOP, border=20)
        panel.tc = wx.TextCtrl(panel)
        panel.browser.Add(panel.tc, proportion=1, flag=wx.RIGHT | wx.ALIGN_CENTER, border=8)
        panel.bt1 = wx.Button(panel, label='Browser', size=(70, 25))
        panel.browser.Add(panel.bt1)
        panel.bt2 = wx.Button(panel, label='OK', size=(70, 25))
        panel.bt3 = wx.Button(panel, label='Close', size=(70, 25))
        panel.buttons.Add(panel.bt2,flag=wx.RIGHT, border=20)
        panel.buttons.Add(panel.bt3)
        panel.bt1.Bind(wx.EVT_BUTTON, self.onSaveFile)
        panel.bt2.Bind(wx.EVT_BUTTON, self.Rulare)
        panel.bt3.Bind(wx.EVT_BUTTON, self.CloseApp)

        panel.SetSizer(panel.vbox)
        panel.midP.SetSizer(panel.midP.box)

    def CloseApp(self,event):
        f = self.GrandParent
        f.Destroy()

    def Rulare(self,event):
        if 'png' in vars(self.midP):
            self.midP.png.Destroy()
            vars(self.midP).pop('png')
        handle = open(p_path, 'rb')
        y_true, y_pred = pr.build_prediction(self.tc.GetValue(), handle, classes, model)
        handle.close()
        final = pr.raport_pred(y_pred, classes)
        em.drow_histograma(final)
        self.midP.png = wx.StaticBitmap(self.midP, -1, wx.Bitmap('../Images/Histogram.png', wx.BITMAP_TYPE_ANY))
        self.midP.box.Add(self.midP.png, wx.ID_ANY, wx.EXPAND | wx.ALL, 20)
        self.midP.Refresh()
        self.Layout()
        self.d = list(self.midP.png.GetBestSize())



    def onSaveFile(self,event):
        dlg = wx.FileDialog(
            self, message="Save file as ...",
            defaultDir=self.currentDirectory,
            defaultFile="", style=wx.FD_SAVE
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.tc.SetValue(path)

        dlg.Destroy()

class MyPanel3(wx.Panel):
    def __init__(self, parent):
        super(MyPanel3, self).__init__(parent)
        self.Template()
        self.Centre()

    def Template(self):
        panel = self
        panel.SetBackgroundColour('#4f5049')
        panel.currentDirectory = os.getcwd()

        panel.vbox = wx.BoxSizer(wx.VERTICAL)
        panel.tbox = wx.BoxSizer(wx.HORIZONTAL)
        panel.buttons = wx.BoxSizer(wx.HORIZONTAL)

        panel.midP = wx.Panel(panel)
        panel.midP.SetBackgroundColour('#ededed')
        panel.midP.box = wx.BoxSizer(wx.HORIZONTAL)

        panel.vbox.Add(panel.tbox, wx.ID_ANY, wx.EXPAND | wx.ALL, 20)
        panel.tbox.Add(panel.midP, wx.ID_ANY, wx.EXPAND | wx.ALL)
        panel.vbox.Add(panel.buttons, flag=wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, border=20)

        panel.bt1 = wx.Button(panel, label='Diagrama', size=(70, 25))
        panel.bt2 = wx.Button(panel, label='Semnale', size=(70, 25))
        panel.bt3 = wx.Button(panel, label='FFT', size=(70, 25))
        panel.bt4 = wx.Button(panel, label='Fbank', size=(70, 25))
        panel.bt5 = wx.Button(panel, label='Mfcc', size=(70, 25))
        panel.bt6 = wx.Button(panel, label='Close', size=(70, 25))

        panel.buttons.Add(panel.bt1, flag=wx.RIGHT, border=20)
        panel.buttons.Add(panel.bt2, flag=wx.RIGHT, border=20)
        panel.buttons.Add(panel.bt3, flag=wx.RIGHT, border=20)
        panel.buttons.Add(panel.bt4, flag=wx.RIGHT, border=20)
        panel.buttons.Add(panel.bt5)
        panel.buttons.Add(panel.bt6, flag=wx.LEFT, border=50)

        panel.bt1.Bind(wx.EVT_BUTTON, self.Diagrama)
        panel.bt2.Bind(wx.EVT_BUTTON, self.Semnale)
        panel.bt3.Bind(wx.EVT_BUTTON, self.FFT)
        panel.bt4.Bind(wx.EVT_BUTTON, self.Fbank)
        panel.bt5.Bind(wx.EVT_BUTTON, self.Mfccs)
        panel.bt6.Bind(wx.EVT_BUTTON, self.CloseApp)

        panel.SetSizer(panel.vbox)
        panel.midP.SetSizer(panel.midP.box)

    def Diagrama(self, event):
        if 'png' in vars(self.midP):
            self.midP.png.Destroy()
            vars(self.midP).pop('png')
        em.desen_diagrama(df, class_dist)
        self.midP.png = wx.StaticBitmap(self.midP, -1, wx.Bitmap('../Images/Diagram.png', wx.BITMAP_TYPE_ANY))
        self.midP.box.Add(self.midP.png, wx.ID_ANY, wx.EXPAND | wx.ALL, 20)
        self.midP.Refresh()
        self.Layout()
        self.d = list(self.midP.png.GetBestSize())

    def Semnale(self, event):
        if 'png' in vars(self.midP):
            self.midP.png.Destroy()
            vars(self.midP).pop('png')
        em.desen_semnale(signals,len(classes))
        self.midP.png = wx.Image('../Images/Semnale.png',wx.BITMAP_TYPE_ANY)
        self.d = self.midP.png.GetSize()
        self.p = self.midP.GetSize()
        if self.d[0] > self.p[0]:
            self.midP.png = self.midP.png.Scale(self.p[0] - 10*self.p[0]//100,self.d[1],wx.IMAGE_QUALITY_HIGH)
            self.d = self.midP.png.GetSize()
        if self.d[1] > self.p[1]:
            self.midP.png = self.midP.png.Scale(self.d[0],self.p[1],wx.IMAGE_QUALITY_HIGH)
        self.d = self.midP.png.GetSize()

        self.midP.png = wx.StaticBitmap(self.midP, -1, wx.BitmapFromImage(self.midP.png))
        self.midP.box.Add(self.midP.png, wx.ID_ANY, wx.EXPAND | wx.ALL, 20)
        self.midP.Refresh()
        self.Layout()



    def FFT(self, event):
        if 'png' in vars(self.midP):
            self.midP.png.Destroy()
            vars(self.midP).pop('png')
        em.desen_fft(fft,len(classes))
        self.midP.png = wx.Image('../Images/FFT.png', wx.BITMAP_TYPE_ANY)
        self.d = self.midP.png.GetSize()
        self.p = self.midP.GetSize()
        if self.d[0] > self.p[0]:
            self.midP.png = self.midP.png.Scale(self.p[0] - 10 * self.p[0] // 100, self.d[1], wx.IMAGE_QUALITY_HIGH)
            self.d = self.midP.png.GetSize()
        if self.d[1] > self.p[1]:
            self.midP.png = self.midP.png.Scale(self.d[0], self.p[1], wx.IMAGE_QUALITY_HIGH)
        self.d = self.midP.png.GetSize()

        self.midP.png = wx.StaticBitmap(self.midP, -1, wx.BitmapFromImage(self.midP.png))
        self.midP.box.Add(self.midP.png, wx.ID_ANY, wx.EXPAND | wx.ALL, 20)
        self.midP.Refresh()
        self.Layout()

    def Fbank(self, event):
        if 'png' in vars(self.midP):
            self.midP.png.Destroy()
            vars(self.midP).pop('png')
        em.desen_fbank(fbank,len(classes))
        self.midP.png = wx.Image('../Images/Fbank.png', wx.BITMAP_TYPE_ANY)
        self.d = self.midP.png.GetSize()
        self.p = self.midP.GetSize()
        if self.d[0] > self.p[0]:
            self.midP.png = self.midP.png.Scale(self.p[0] - 10 * self.p[0] // 100, self.d[1], wx.IMAGE_QUALITY_HIGH)
            self.d = self.midP.png.GetSize()
        if self.d[1] > self.p[1]:
            self.midP.png = self.midP.png.Scale(self.d[0], self.p[1], wx.IMAGE_QUALITY_HIGH)
        self.d = self.midP.png.GetSize()

        self.midP.png = wx.StaticBitmap(self.midP, -1, wx.BitmapFromImage(self.midP.png))
        self.midP.box.Add(self.midP.png, wx.ID_ANY, wx.EXPAND | wx.ALL, 20)
        self.midP.Refresh()
        self.Layout()

    def Mfccs(self, event):
        if 'png' in vars(self.midP):
            self.midP.png.Destroy()
            vars(self.midP).pop('png')
        em.desen_mfccs(mfccs,len(classes))
        self.midP.png = wx.Image('../Images/MFC.png', wx.BITMAP_TYPE_ANY)
        self.d = self.midP.png.GetSize()
        self.p = self.midP.GetSize()
        if self.d[0] > self.p[0]:
            self.midP.png = self.midP.png.Scale(self.p[0] - 10 * self.p[0] // 100, self.d[1], wx.IMAGE_QUALITY_HIGH)
            self.d = self.midP.png.GetSize()
        if self.d[1] > self.p[1]:
            self.midP.png = self.midP.png.Scale(self.d[0], self.p[1], wx.IMAGE_QUALITY_HIGH)
        self.d = self.midP.png.GetSize()

        self.midP.png = wx.StaticBitmap(self.midP, -1, wx.BitmapFromImage(self.midP.png))
        self.midP.box.Add(self.midP.png, wx.ID_ANY, wx.EXPAND | wx.ALL, 20)
        self.midP.Refresh()
        self.Layout()

    def CloseApp(self, event):
        f = self.GrandParent
        f.Destroy()








def main():

    app = wx.App()
    ex = Interfata(None, title='Urban Sounds Recognision')
    ex.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()