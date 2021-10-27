#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 13:51:37 2021

@author: km357
"""
import os, sys
from datetime import datetime, timedelta
# import re
# import shutil
import xarray as xr
home = os.path.expanduser("~")
sys.path.append(os.path.join(home, 'Documents','Python3','my_modules','gpym'))
from gpym import io
from xhistogram.xarray import histogram
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

date = datetime(2015,1,1)

sys.stdout.write('%s\r' % date)
sys.stdout.flush()

date_str = '%04d-%02d-%02d' % (date.year, date.month, date.day)
dpr_dir = '/data/doppler/GPM/V06/%04d/%02d/%02d/radar' % (
            date.year, date.month, date.day)
files = [f for f in os.listdir(dpr_dir) 
         if f.startswith('2A.GPM.DPR.V') and f.endswith('.HDF5') ]

dsets = []
for file in files:
    fn = os.path.join(dpr_dir, file)
    dset_tmp = io.open_dataset(fn, subgroup = 'NS', read_attrs = True)
    dset_nadir = dset_tmp.isel(nray=24)
    
    tmp_hist = histogram(dset_nadir.sigmaZeroMeasured.where(dset_nadir.flagPrecip==0),
                         bins = [np.arange(-10,30,0.25),])
    
    # plt.figure()
    # tmp_hist.plot()
    dsets.append(tmp_hist)
   
    
    
dset = dsets[0]*1
for ii in range(1, len(dsets)):
    dset += dsets[ii] 
    
plt.figure()
dset.plot()
plt.savefig(os.path.join(home, 'sigma_0_Ku_GPM_no_rain.png'))

    