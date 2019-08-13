# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:47:20 2019

@author: Dodzilla
"""
import wx
import wx.lib.scrolledpanel


class Interfata(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, title = 'Sound Recognision', size = (wx.DisplaySize()[0],wx.DisplaySize()[1]))
        self.SetBackgroundColour(wx.BLACK)
        nb = wx.Notebook(self)
        nb.AddPage(Histograma(nb), "Histograma")
        nb.AddPage(Antrenare(nb), "Antrenare")
        nb.AddPage(Detalii_Date(nb), "Detalii_Date")

class Histograma(wx.Panel):
   def __init__(self, parent):
      super(Histograma, self).__init__(parent)
      self.SetBackgroundColour('#FDDF99')


class Antrenare(wx.Panel):
   def __init__(self, parent):
      super(Antrenare, self).__init__(parent)
      self.SetBackgroundColour('#3b5998')

class Detalii_Date(wx.Panel):
   def __init__(self, parent):
      super(Detalii_Date, self).__init__(parent)
      self.SetBackgroundColour('#00daff')


app = wx.App()
Int = Interfata(parent = None, id = 1)
Int.Show()
app.MainLoop()