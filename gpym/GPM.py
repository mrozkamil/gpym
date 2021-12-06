#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:25:49 2021

@author: km357
"""
import os

from . import io
import numpy as np
import xarray as xr

class DPR():
    def __init__(self, filename):
        
        module_path = os.path.dirname(__file__)
       
        self.NS = io.open_dataset(filename, subgroup = 'NS', read_attrs = True)
        self.MS = io.open_dataset(filename, subgroup = 'MS', read_attrs = True)
        self.HS = io.open_dataset(filename, subgroup = 'HS', read_attrs = True)
        
        LUT_fn = os.path.join(module_path, 'scattering', 'rain', 
                              'PSD_integrated', 'gamma', 'mimics_insitu',
                              'NASA_DSD_dataset_2DVD_5min_stats_Gamma.nc')
        self.LUT_rain = xr.load_dataset(LUT_fn)
        
        LUT_fn = os.path.join(module_path, 'scattering', 'ice', 
                              'PSD_integrated', 'gamma', 'mimics_insitu',
                              'insitu_PSD_dataset_Dle_stats_Gamma.nc')
        self.LUT_ice = xr.load_dataset(LUT_fn)
        
    def _restrict_NS_to_MS(self,):
        if 'nray' in self.NS.dims:
            self.NS = self.NS.isel(nray = np.arange(12,37))
            self.NS  = self.NS .rename({'nray': 'nrayMS'}) 
    def _dB(self,x):
        return 10*np.log10(x)
    def _inv_dB(self,x):
        return 10**(x/10)
    def _simulate_Ze_k(self, Dm_dB, PR_dB, 
                       flag_rain, flag_ice, k_ML = None,
                       alpha = 0.015, band = 'Ku',):

        if Dm_dB.shape == PR_dB.shape:
            dims = ['dim%d' % dd for dd in range(len(Dm_dB.shape))] 
            
        if type(Dm_dB)==np.ndarray:
            Dm_dB_da = xr.DataArray(Dm_dB, dims = dims)
        elif type(Dm_dB) == xr.core.dataarray.DataArray:
            Dm_dB_da = Dm_dB.copy()
        
        if type(PR_dB)==np.ndarray:
            PR_dB_da = xr.DataArray(PR_dB, dims = dims)
        elif type(PR_dB) == xr.core.dataarray.DataArray:
            PR_dB_da = PR_dB.copy()
            
        z_name = 'mean_Z_%s' % band
        k_name = 'mean_k_%s' % band
        

        Ze_ice = self.LUT_ice[z_name].interp(Alpha = alpha,
            Dm_dB_bin = Dm_dB_da, PR_dB_bin = PR_dB_da).where(flag_ice,-99.)
        k_ice = self.LUT_ice[k_name].interp(Alpha = alpha,
            Dm_dB_bin = Dm_dB_da, PR_dB_bin = PR_dB_da).where(flag_ice, 0.)
        
        
        Ze_rain = self.LUT_rain[z_name].interp(
            Dm_dB_bin = Dm_dB_da, PR_dB_bin = PR_dB_da).where(flag_rain, -99.)
        k_rain = self.LUT_rain[k_name].interp(
            Dm_dB_bin = Dm_dB_da, PR_dB_bin = PR_dB_da).where(flag_rain,0.)
        
        Ze = self._dB( self._inv_dB(Ze_ice) + self._inv_dB(Ze_rain))
        spec_att = k_ice + k_rain
        if k_ML is not None:
            spec_att += k_ML
        return Ze, spec_att
    
    def _simulate_Zm(self, Dm_dB, PR_dB, 
                     flag_rain, flag_ice, k_ML = None,
                       alpha = 0.015, band = 'Ku',
                       range_spacing = 0.25, range_dim = 'dim0',):
        
        Ze, spec_att = self._simulate_Ze_k(                            
             Dm_dB = Dm_dB, PR_dB = PR_dB, 
             flag_rain = flag_rain, flag_ice = flag_ice, 
              k_ML = k_ML, alpha = alpha, band = band,)
        
        one_way_att = spec_att.cumsum(dim = range_dim)*range_spacing
        
        Zm = Ze - 2*one_way_att
        return Zm
        
        
        