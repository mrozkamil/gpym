#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:23:10 2022

@author: km357
"""

import os
import xarray as xr

def save_to_netcdf(dset_hist, fn):    
    variables = [key for key in dset_hist.variables 
             if key not in dset_hist.dims]
    encoding = {key: {'dtype': 'int32','compression': 'gzip', } 
                for key in variables}
    dset_hist.to_netcdf(fn, engine = 'h5netcdf', encoding= encoding)
    
home = os.path.expanduser("~")

main_dir = os.path.join(home, 'Documents','Python3','OUTPUT','gpm', 
                        'Z_diff_stats', 'one_day_pdf')
fns_out = [f for f in os.listdir(main_dir) if (f.endswith('.nc') and
           f.startswith('histograms_ZKa'))]
fns_out.sort()

subgroup = 'MS'

for iii,fn in enumerate(fns_out):
    fname = os.path.join(main_dir, fn)
    dset_hist = xr.open_dataset(fname)
    fn_new = fname.replace('_ZKa_', '_ZKa_%s_' % subgroup)
    
    variables = [key for key in dset_hist.variables 
             if key not in dset_hist.dims and '_%s_' % subgroup in key]
    replacement = {var_n: var_n.replace('_%s' % subgroup, '') 
                   for var_n in variables}
    
    dset_hist_new = dset_hist[replacement.keys()].rename(replacement)
    save_to_netcdf(dset_hist_new, fn_new)
   
    
