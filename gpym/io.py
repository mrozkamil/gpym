#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:17:24 2021

@author: kamil.mroz@le.ac.uk
"""
# import os, re
# from pandas import to_datetime 
import netCDF4
import xarray as xr
# import numpy as np
xr.set_options(display_style="html")
                 
def _rename_dims(dset, consistency_test = False):
    vars_to_drop = []
    for var_n in dset.variables:
        if 'DimensionNames' in dset[var_n].attrs.keys():
            dim_names_new = dset[var_n].attrs['DimensionNames'].replace(
                ' ','').split(',')
            dim_names_old = list(dset[var_n].dims) 
            # check if a list of dims has no repeated entry
            dim_names_old_set = set(dim_names_old)
            if len(dim_names_old_set)==len(dim_names_old):
                swap_dict = {dim_n_old : dim_n_new 
                    for dim_n_old, dim_n_new in zip(dim_names_old, dim_names_new)
                    if dim_n_old != dim_n_new}
                dset[var_n] = dset[var_n].rename(swap_dict)
            else:
                if consistency_test:
                    # print('%s has repeated dim, loaded to memory...' % var_n)
                    tmp_data = dset[var_n].data
                    dset[var_n] = (dim_names_new, tmp_data)
                else:
                    vars_to_drop.append(var_n)
    dset = dset.drop(vars_to_drop)
    return dset

def open_dataset(filename, subgroup = 'NS', read_attrs = True, consistency_test = False):          
   
    # height = 0.125*(175 - np.arange(176))  
              
    ncf = netCDF4.Dataset(filename, diskless=True, persist=False, mode='r')   # type: ignore 
    attrs = {}   
    if read_attrs:         
        attr_list = ['FileHeader', 'FileInfo', 'InputRecord', 'JAXAInfo',
                     'NavigationRecord']
        all_keys = list(dir(ncf))
        for key in attr_list:
            if key in all_keys:                
                tmp_attr = ncf.__dict__[key][:]
                tmp_list = tmp_attr.split('\n')
                for item in tmp_list:
                    if '=' in item:
                        k,v = item.split('=')
                        v = v.replace(';','')
                        vprim = v.replace('.','',1).replace('-','',1)
                        if vprim.isnumeric():
                            v = float(v)
                        attrs[k] = v           
    
   
    if subgroup in list(ncf.groups.keys()):
        nch = ncf.groups.get(subgroup)
        dset0 = xr.open_dataset(xr.backends.NetCDF4DataStore(nch))    # type: ignore        
        dset0 = _rename_dims(dset0,  consistency_test = consistency_test)
                          
        for subsubgroup in nch.groups.keys():  
            # print(subsubgroup)
            ncg = nch.groups.get(subsubgroup)
            dset1 = xr.open_dataset(xr.backends.NetCDF4DataStore(ncg), 
                        decode_timedelta = False,)  # type: ignore
           
            dset1 = _rename_dims(dset1,  consistency_test = consistency_test)
            if subsubgroup == 'ScanTime':
                
                time_vars = ('Year', 'Month', 'DayOfMonth',
                     'Hour',  'Minute', 'Second', 'MilliSecond')
                dtype_dict = {'Year': '<M8[Y]', 
                              'Month': '<m8[M]',
                              'DayOfMonth': '<m8[D]', 
                              'Hour': '<m8[h]',
                              'Minute': '<m8[m]', 
                              'Second': '<m8[s]', 
                              'MilliSecond': '<m8[ms]', }
                
                time_dict = {tkey: dset1[tkey].values.astype('int16')
                                 for tkey in time_vars}
                
                time_dict['Year'] += -1970
                time_dict['Month'] += -1
                time_dict['DayOfMonth'] += -1          
                dset0['time'] = sum(time_dict[key].astype(dtype_dict[key]) 
                                    for key in time_vars)
            else:
                var_list_0 = list(dset0.variables)
                var_list_1 = list(dset1.variables)
                for var_n in var_list_1:
                    if var_n in var_list_0:
                        var_n_mod = '{}_{}'.format(var_n, subsubgroup)                        
                        dset1 = dset1.rename({var_n: var_n_mod})                        
                dset0 = xr.merge([dset0, dset1])
                
        coords = ['Longitude', 'Latitude', 'time']
        
         
        dset0 = dset0.set_coords(coords)   
        dset0.attrs = attrs    
        return dset0
    else:
        print('subgroup "{}" not present in the file {}'.format(
            subgroup, filename.split('/')[-1]))
        print('available groups are:')
        print(list(ncf.groups.keys()))

def open_mfdataset(file_list, subgroup = 'NS', read_attrs = True,
                   dim = 'nscan'):
    if len(file_list)>0:       
        dsets = []
        for filename in file_list:
            dsets.append(open_dataset(filename, subgroup = subgroup, 
                             read_attrs = read_attrs))
        dset = xr.concat(dsets, dim = dim, 
                         combine_attrs = 'drop_conflicts')
        return dset

