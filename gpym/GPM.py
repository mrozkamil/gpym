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
    def _simulate_refl(self, flag_rain, flag_ice, 
                       Dm_dB, PR_dB, alpha = 0.015, 
                       range_axis = -1, dh = 0.25, band = 'Ka'):
        
        # Z_e = np.full(Dm_dB.size, -99.)
        # one_way_att = np.full(Dm_dB.size, 0.)
        if type(Dm_dB)==np.ndarray:
            Dm_dB_da = xr.DataArray(Dm_dB, dims = 'Z')
        elif type(Dm_dB) == xr.core.dataarray.DataArray:
            Dm_dB_da = Dm_dB.copy()
        
        if type(PR_dB)==np.ndarray:
            PR_dB_da = xr.DataArray(PR_dB, dims = 'Z')
        elif type(PR_dB) == xr.core.dataarray.DataArray:
            PR_dB_da = PR_dB.copy()
            
        z_name = 'mean_Z_%s' % band
        k_name = 'mean_k_%s' % band
        
    
        
        Z_e_ice = self.LUT_ice[z_name].interp(Alpha = alpha,
            Dm_dB_bin = Dm_dB_da, PR_dB_bin = PR_dB_da).where(flag_ice,-99.)
        one_way_att_ice = self.LUT_ice[k_name].interp(Alpha = alpha,
            Dm_dB_bin = Dm_dB_da, PR_dB_bin = PR_dB_da).where(flag_ice, 0.)
        
        
        Z_e_rain = self.LUT_rain[z_name].interp(
            Dm_dB_bin = Dm_dB_da, PR_dB_bin = PR_dB_da).where(flag_rain, -99.)
        one_way_att_rain = self.LUT_rain[k_name].interp(
            Dm_dB_bin = Dm_dB_da, PR_dB_bin = PR_dB_da).where(flag_rain,0.)
        
        Z_e = self._dB( self._inv_dB(Z_e_ice) + self._inv_dB(Z_e_rain))
        one_way_att = one_way_att_ice+one_way_att_rain
        
        

        return Z_e, one_way_att