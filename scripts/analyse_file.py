#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:48:52 2021

@author: km357
"""
import os, sys
# from datetime import datetime, timedelta
import numpy as np
import pandas as pd
# import xarray as xr
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# from xhistogram.xarray import histogram as xhist
# import timeit
from scipy.ndimage import uniform_filter, binary_opening, binary_dilation
home = os.path.expanduser("~")
sys.path.append(
    os.path.join(home, 'Documents','Python3', 'gpym'))
import gpym

import warnings
warnings.filterwarnings("ignore")


dpr_path = os.path.join( home, 'Data', 'GPM', 
                  'V06', '%d', '%02d', '%02d', 'radar') 

date_str, gran_str  = '151031', '009505'
date = pd.to_datetime(date_str, format = '%y%m%d')
   
tmp_dir = dpr_path % (date.year, date.month, date.day)
files_dpr = [f for f in os.listdir(tmp_dir) if
             (gran_str in f ) and (f.startswith('2A.GPM.DPR.V'))]
dpr_file_n = os.path.join(tmp_dir,files_dpr[0])

swaths = ['NS','MS','HS',]
dsetDPR = {sw: gpym.io.open_dataset(
    dpr_file_n, subgroup = sw, read_attrs = True) 
    for sw in swaths}


plt.figure()
for sw in swaths:
    marker = 'o'
    if sw == 'HS': marker = 'x'
    plt.scatter(dsetDPR[sw].Longitude[:2,:].values.ravel(), 
             dsetDPR[sw].Latitude[:2,:].values.ravel(),
             marker = marker, label = sw)

dsetDPR['NS'] = dsetDPR['NS'].isel(nray = np.arange(12,37))
dsetDPR['NS'] = dsetDPR['NS'].rename({'nray':'nrayMS'})
sw = 'NS'
plt.scatter(dsetDPR[sw].Longitude[:2,:].values.ravel(), 
         dsetDPR[sw].Latitude[:2,:].values.ravel(),
         marker = '.', label = sw)
plt.legend()
plt.grid()

tmp = (dsetDPR['NS'].zFactorMeasured>10).astype(int)
# tmp = tmp.where(
#     dsetDPR['NS'].nbin<dsetDPR['NS'].binClutterFreeBottom, 0)
b_o_size = 5
struc = np.zeros((1,3,3))
struc[0,1,:] =1
struc[0,:,1] =1
flag_op = (binary_opening(tmp, structure = np.ones((1,1,b_o_size)))  |
           binary_opening(tmp, structure = struc) )

tmp = (dsetDPR['NS'].zFactorMeasured>15).astype(int)
flag_dl = binary_dilation(tmp, 
        structure = np.ones((1,b_o_size,b_o_size*2+1)))

flag_op = (flag_op & flag_dl)

plt.figure()
dsetDPR['NS'].zFactorMeasured.isel(nrayMS = 12).where(
    dsetDPR['NS'].zFactorMeasured.isel(nrayMS = 12)>10).plot(
    x = 'nscan', y = 'Altitude', vmin =10, vmax = 55, cmap = 'jet')

plt.figure()
dsetDPR['NS'].zFactorMeasured.isel(nrayMS = 12).where(
    flag_op[:,12,:]).plot(
    x = 'nscan', y = 'Altitude', vmin =10, vmax = 55, cmap = 'jet')




