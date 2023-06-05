#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:25:49 2021

@author: km357
"""

from . import io
import numpy as np
import h5py
import xarray as xr

class GPM():
    def __getitem__(self, key):
         return getattr(self,key)
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __init__(self, filename , subgroups = None, consistency_test = False, concat_dim = 'nscan'): 
        self.swaths = []
        self.open_dataset(filename, subgroups = subgroups, 
                consistency_test = consistency_test, concat_dim = concat_dim)
              

    def add_subgroup(self, filename, subgroup, concat_dim = 'nscan',
        consistency_test = False):
        ### loading radar data     
        if isinstance(filename, str):            
            tmp_ds = io.open_dataset(
                filename, subgroup = subgroup, read_attrs = True, 
                consistency_test = consistency_test)           
        if isinstance(filename, list):
            tmp_ds = io.open_mfdataset(
                filename, subgroup = subgroup, read_attrs = True,
                consistency_test = consistency_test) 
        if not subgroup in dir(self):
            self[subgroup] = tmp_ds
            self.swaths.append(subgroup)
        else:
            self[subgroup] = xr.concat([self[subgroup], tmp_ds],
                dim = concat_dim, combine_attrs = 'drop_conflicts')
        tmp_cords = list(self[subgroup].coords)
        if not ('_nscan' in tmp_cords) and ('nscan' in tmp_cords):
            tmp_cords.append('_nscan')
            self[subgroup]['_nscan'] = ('nscan',
                    np.arange(self[subgroup]['nscan'].size))
            self[subgroup] =  self[subgroup].set_coords(tmp_cords)
            
    def open_dataset(self, filename, subgroups=None, concat_dim = 'nscan',
            consistency_test = False):
        if isinstance(filename, str):   
            fn = filename
        if isinstance(filename, list):           
            fn = filename[0]            
        if subgroups is None:
            f = h5py.File(fn, 'r')  
            subgroups = []
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    subgroups.append(key)
            f.close()              
        for subgroup in subgroups:
            self.add_subgroup(filename, subgroup = subgroup, 
                              concat_dim = concat_dim,
                              consistency_test = consistency_test) 
            
    def add_dataset(self, filename, subgroups=None, concat_dim = 'nscan',
            consistency_test = False):
        self.open_dataset(filename, subgroups = subgroups, 
                          concat_dim = concat_dim,
                          consistency_test = consistency_test)
    
       
        
class DPR(GPM):    
    def __init__(self, filename, subgroups = None):
        super().__init__(filename, subgroups = subgroups)
        
        keys = ['MS', 'HS', 'NS', 'FS', ]
        methods = [m_name for m_name in dir(self) 
                   if not m_name.startswith('__')]
        for key in keys:
            if key not in methods:
                continue           
            if 'nbin' in self[key].dims:
                var_n = 'nbin'
                dh = 0.125
            elif 'nbinHS' in self[key].dims:
                var_n = 'nbinHS'
                dh = 0.25
            else:
                var_n = None
            if var_n:
                height = (self[key][var_n].size - self[key][var_n]-1)*dh
                # print(var_n)
                # print(key)
                # print(height)
                # print(self[key]['Altitude'] )
                self[key]['Altitude'] =  height
                self[key]['Altitude'].attrs = {
                    'DimensionNames': var_n,
                     'Units': 'km',
                     'units': 'km',
                     'CodeMissingValue': '-9999.9'}
                coords = list(self[key].coords)                
                coords.append('Altitude')  
                self[key] =  self[key].set_coords(coords)
                self[key].attrs['range_spacing'] = dh
            
                
        ### restricting the NS data to the MS swath
        if 'NS' in methods: 
            self.MNS = self.NS.isel(nray = np.arange(12,37))
            self.MNS  = self.MNS .rename({'nray': 'nrayMS'})             
        self.swath_name = {'Ka': 'MS', 'Ku': 'MNS'}        

   
       
      
        
        