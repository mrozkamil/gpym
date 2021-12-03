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
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.NS = io.open_dataset(filename, subgroup = 'NS', read_attrs = True)
        self.MS = io.open_dataset(filename, subgroup = 'MS', read_attrs = True)
        self.HS = io.open_dataset(filename, subgroup = 'HS', read_attrs = True)
        
        LUT_fn = os.path.join(current_dir, 'scattering', 'rain', 
                              'PSD_integrated', 'gamma', 'data',
                              'NASA_DSD_dataset_2DVD_5min_stats_Gamma.nc')
        self.LUT_rain = xr.load_dataset(LUT_fn)
        
        
        LUT_fn = os.path.join(current_dir, 'scattering', 'ice', 
                              'PSD_integrated', 'gamma', 'data',
                              'NASA_DSD_dataset_2DVD_5min_stats_Gamma.nc')
        self.LUT_rain = xr.load_dataset(LUT_fn)
        
    def _restrict_NS_to_MS(self,):
        if 'nray' in self.NS.dims:
            self.NS = self.NS.isel(nray = np.arange(12,37))
            self.NS  = self.NS .rename({'nray': 'nrayMS'}) 
    # def _simulate_refl(self, )
        
