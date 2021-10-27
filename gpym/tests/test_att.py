#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 17:20:59 2021

@author: km357
"""
import os, sys
from datetime import datetime
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

freqs = {'Ku': 13.6, 'Ka': 35.5, }
bands = list(freqs.keys())
home = os.path.expanduser("~")
sys.path.append(os.path.join(home, 'Documents','Python3', 'gpym'))

import gpym

# from gpym import server
import warnings
warnings.filterwarnings("ignore")

# make datasets display nicely
xr.set_options(display_style="html")

date = datetime(2017,12,22)
orbit = 21690


output_dir = os.path.join(home, 'Documents','Python3', 'OUTPUT','gpym',
                          '%06d' % orbit)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



gpm_dir = os.path.join(home, 'Data', 'GPM')
dpr_dir = os.path.join(gpm_dir, 'V06', '%04d', '%02d', '%02d', 'radar') % (
    date.year, date.month, date.day)

pps_client = gpym.server.PPS(usr = 'km357@leicester.ac.uk', 
                             save_dir = gpm_dir)
  

date_str = '%04d-%02d-%02d' % (date.year, date.month, date.day)


pps_client.download_files(date_str, prod = '2A.GPM.DPR.V', 
                    gran = orbit, verbose = True) 

pps_client.download_files(date_str, prod = '2A-ENV.GPM.DPR.V', 
                    gran = orbit, verbose = True) 

    
files = []
if os.path.exists(dpr_dir):
    files = [f for f in os.listdir(dpr_dir) 
        if (f.startswith('2A.GPM.DPR.V') and 
            f.endswith('.HDF5') and '{:06d}'.format(orbit) in f)]
    

file_dpr = files[0]
fn_dpr = os.path.join(dpr_dir, file_dpr)
fn_env = fn_dpr.replace('2A.GPM.DPR.V','2A-ENV.GPM.DPR.V')
           
ind_MS = np.arange(12,37)
ds_dpr = gpym.io.open_dataset(fn_dpr, subgroup = 'NS')
ds_dpr = ds_dpr.isel(nray = ind_MS)
ds_dpr = ds_dpr.rename({'nray':'nrayMS'})

ds_env = gpym.io.open_dataset(fn_env, subgroup = 'NS')
ds_env = ds_env.isel(nray = ind_MS)
ds_env = ds_env.rename({'nray':'nrayMS'})

ds_ka = gpym.io.open_dataset(fn_dpr, subgroup='MS')

ind_sc = np.where((ds_dpr.Longitude[:,12]>-65) & 
                  (ds_dpr.Longitude[:,12]<-61))[0]
ds_dpr = ds_dpr.isel(nscan = ind_sc)
ds_env = ds_env.isel(nscan = ind_sc)
ds_ka = ds_ka.isel(nscan = ind_sc)

fig, ax  = plt.subplots()
ds_dpr.precipRateNearSurface.plot(cmap = 'jet', ax = ax, vmax = 10, 
                                  x= 'Longitude', y = 'Latitude')

for ii in [1,3]:
    for ds, band in zip([ds_dpr, ds_ka], ['Ku', 'Ka']):
        fig, ax  = plt.subplots()
        ds.piaNP.isel(nNP=ii).plot(cmap = 'jet', ax = ax,  
                                          x= 'Longitude', y = 'Latitude')
        ax.set_title('{}, nNP = {}'.format(band,ii))
        plt.tight_layout()
        fig_name = os.path.join(output_dir, 
            'piaNP_{:d}_{:s}_orbit_{:d}.png'.format(ii, band, orbit))
        fig.savefig(fig_name)


    
specific_att_gases = xr.Dataset({band: (ds_env.airPressure.dims,
                                  np.zeros(ds_env.airPressure.shape),
                                  {'unit':'dB/km'}) for band in bands},
                          coords = ds_env.airPressure.coords)

specific_att_cloud = xr.Dataset({band: (ds_env.airPressure.dims,
                                  np.zeros(ds_env.airPressure.shape),
                                  {'unit':'dB/km'}) for band in bands},
                          coords = ds_env.airPressure.coords)

   

for band in bands:
    print('specific attenuation computations at the {}-band...'.format(band))
    
    specific_att_gases[band][:] = gpym.tools.specific_attenuation_gases(
                                ds_env, band)
    
    specific_att_cloud[band][:] = gpym.tools.specific_attenuation_cloud(
                                ds_env, band)
    
    
print('done')

delta_h = -ds_env.Altitude.diff(dim = 'nbin').values[0]
for band in bands:
    
    pia_gases = specific_att_gases[band].sum(dim = 'nbin')*delta_h*2
    fig, ax  = plt.subplots()
    pia_gases.plot(cmap = 'jet', ax = ax,  
                x= 'Longitude', y = 'Latitude')
    ax.set_title('2-way PIA gases')
    plt.tight_layout()
    fig_name = os.path.join(output_dir, 
        'piaNP_gases_calc_ENV_file_{:s}_orbit_{:d}.png'.format( band, orbit))
    fig.savefig(fig_name)
    
    pia_cloud = specific_att_cloud[band].sum(dim = 'nbin')*delta_h*2
    fig, ax  = plt.subplots()
    pia_cloud.plot(cmap = 'jet', ax = ax,  
                x= 'Longitude', y = 'Latitude')
    ax.set_title('2-way PIA cloud')
    plt.tight_layout()
    fig_name = os.path.join(output_dir, 
        'piaNP_liquid_cloud_calc_ENV_file_{:s}_orbit_{:d}.png'.format( band, orbit))
    fig.savefig(fig_name)


   

ii = 6
for ds, band in zip([ds_dpr, ds_ka], ['Ku', 'Ka']):
    fig, ax  = plt.subplots()
    ds.zFactorMeasured.isel(nrayMS=ii).plot(
        cmap = 'nipy_spectral', ax = ax,  vmin = 12, vmax = 50, 
        x= 'Longitude', y = 'Altitude',)
    ax.set_xlim(-65, -62)
    ax.set_ylim(0, 16.5)
    ax.set_title('{}, nrayMS = {}'.format(band,ii))
    plt.tight_layout()
    fig_name = os.path.join(output_dir, 
        'zFactorMeasured_{:s}_nrayMS_{:d}_orbit_{:d}.png'.format( band, ii, orbit))
    fig.savefig(fig_name)
    
    fig, ax  = plt.subplots()
    specific_att_gases[band].isel(nrayMS=ii).plot(
        cmap = 'nipy_spectral', ax = ax,  vmin = 0,
        x= 'Longitude', y = 'Altitude',)
    ax.set_xlim(-65, -62)
    ax.set_ylim(0, 16.5)
    ax.set_title('specific attenuation gases, nrayMS = {}'.format(ii))
    plt.tight_layout()
    fig_name = os.path.join(output_dir, 
        'specificAttenuationGases_{:s}_nrayMS_{:d}_orbit_{:d}.png'.format( band, ii, orbit))
    fig.savefig(fig_name)
    
    fig, ax  = plt.subplots()
    specific_att_cloud[band].isel(nrayMS=ii).plot(
        cmap = 'nipy_spectral', ax = ax,  vmin = 0,
        x= 'Longitude', y = 'Altitude',)
    ax.set_xlim(-65, -62)
    ax.set_ylim(0, 16.5)
    ax.set_title('specific attenuation cloud, nrayMS = {}'.format(ii))
    plt.tight_layout()
    fig_name = os.path.join(output_dir, 
        'specificAttenuationLiquidCloud_{:s}_nrayMS_{:d}_orbit_{:d}.png'.format( band, ii, orbit))
    fig.savefig(fig_name)
  
