#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  15 17:32:16 2021

@author: km357
"""
import os, sys
from datetime import datetime, timedelta
# import re
# import shutil
import numpy as np
import xarray as xr
from xhistogram.xarray import histogram as xhist
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
import timeit
# from scipy.signal import savgol_filter

home = os.path.expanduser("~")
sys.path.append(os.path.join(home, 'Documents','Python3', 'gpym'))
import gpym

import warnings
warnings.filterwarnings("ignore")

out_dir = os.path.join(home, 'Documents','Python3','OUTPUT','gpm', 'DWR_prof')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


bins_dfr = np.arange(-10,35,0.25)
bins_t = np.arange(220,300,1)

bins_t_ice = np.arange(-50,0,1.)
bins_z_rain = np.arange(15,55,1.)
bins_dfr_rain = np.arange(-5,25,1.)
bins_sfc_type = np.array([0,100,200])

year = 2014
year = int(sys.argv[1])

date = datetime(year,1,1)
time0 = datetime(1970,1,1)

count = 0


while date<datetime(year+1,1,1):# (datetime.today()-timedelta(days = 1)):
  
    
    date_str = '%04d-%02d-%02d' % (date.year, date.month, date.day)
    dpr_dir = '/data/doppler/GPM/V06/%04d/%02d/%02d/radar' % (
        date.year, date.month, date.day)
    if not os.path.exists(dpr_dir):
        date += timedelta(days=1)
        continue
    files = [f for f in os.listdir(dpr_dir)
                if f.startswith('2A.GPM.DPR.V') and f.endswith('.HDF5') ]
    files.sort()
    
    for file in files:
        
        fn = os.path.join(dpr_dir, file)
        starttime = timeit.default_timer()        
        try:  
                        
            dsetKu = gpym.io.open_dataset(fn, subgroup = 'NS', read_attrs = True)
            dsetKu = dsetKu.isel(nray = np.arange(12,37))
            dsetKu = dsetKu.rename({'nray': 'nrayMS'}) 
            
            dsetKa = gpym.io.open_dataset(fn, subgroup = 'MS', read_attrs = True)
        except Exception as ex:
            print(ex)
            continue
            
        
        ds_flag = ((dsetKu.flagPrecip == 11) & 
                   (dsetKu.typePrecip//10000000 ==1) & 
                   (dsetKu.qualityTypePrecip == 1) & 
                   (dsetKu.binBBBottom<dsetKu.binClutterFreeBottom-8) & 
                   (dsetKu.binBBBottom>120) & 
                   (dsetKu.binStormTop<dsetKu.binBBTop-8))         
        sel_ind = np.where(ds_flag)
        if sel_ind[0].size<1:
            continue
        
           
        dsetKu_sel = dsetKu.isel(nscan = xr.DataArray(sel_ind[0], dims = 'prof'),
                             nrayMS = xr.DataArray(sel_ind[1], dims = 'prof'))
        dsetKa_sel = dsetKa.isel(nscan = xr.DataArray(sel_ind[0], dims = 'prof'),
                             nrayMS = xr.DataArray(sel_ind[1], dims = 'prof'))
                 
        DWRm_sel = dsetKu_sel.zFactorMeasured - dsetKa_sel.zFactorMeasured  
        DWRm_sel.name = 'DWR'
        
        clutter_free_flag = (dsetKu_sel.nbin<dsetKu_sel.binClutterFreeBottom-1)  
        good_sens_flag = ((dsetKu_sel.zFactorMeasured>15) & 
                          (dsetKa_sel.zFactorMeasured>21) )
        DWRm_sel = DWRm_sel.where(clutter_free_flag & good_sens_flag)
        DWRm_sel = DWRm_sel.load()
        
        phase_sel = dsetKu_sel.phase*1.
        phase_sel = phase_sel.where((phase_sel>50) & (phase_sel<100))-100
        phase_sel.name = 'iceTemperature'
       
        da_ones = DWRm_sel.fillna(0.)*0.+1.
        da_ones.name = 'ones'
        
        
        flag_below_BB = ((dsetKu_sel.nbin>dsetKu_sel.binBBBottom) & 
                         (dsetKu_sel.nbin<=dsetKu_sel.binBBBottom+8))

        Z_prof = dsetKu_sel.zFactorMeasured.where(flag_below_BB)
        Z_sel = Z_prof.mean(dim='nbin', skipna = True)*da_ones
        Z_sel.name = 'Z_rain'
        
        
        
        DWR_prof = DWRm_sel.copy()
        DWR_prof = DWR_prof.where(flag_below_BB)
        DWR_full = DWR_prof.mean(dim='nbin', skipna = True)*da_ones
        DWR_full.name = 'DWR_rain'
        
        sfc_full = dsetKu_sel.landSurfaceType*da_ones
        sfc_full.name = 'SFC_type'
        
        tmp_hist_dm = xhist(DWRm_sel,phase_sel,  DWR_full, sfc_full,
                bins=[bins_dfr,bins_t_ice, bins_dfr_rain, bins_sfc_type], )
        
        tmp_hist_rr = xhist(DWRm_sel,phase_sel,Z_sel, sfc_full,
                bins=[bins_dfr,bins_t_ice, bins_z_rain, bins_sfc_type], )

        if count==0:
            hist_total_dm = tmp_hist_dm*1.
            hist_total_rr = tmp_hist_rr*1.
        else:
            hist_total_dm += tmp_hist_dm
            hist_total_rr += tmp_hist_rr
               
        
        print("%s; %.1f s to process" % (file,
            (timeit.default_timer() - starttime)))
        count +=1
    date += timedelta(days=1)
      
            
dset_hist = xr.Dataset(data_vars = {arr.name: arr 
                    for arr in [hist_total_dm, hist_total_rr]})
fn_out = os.path.join(out_dir, 'histograms_DWR_T_rain_Z_DWR_year_%d.nc' % year)
dset_hist.to_netcdf(fn_out)

