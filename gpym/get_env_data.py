#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:32:16 2021

@author: km357
"""
import os, sys
from datetime import datetime, timedelta
# import re
# import shutil
home = os.path.expanduser("~")
sys.path.append(os.path.join(home, 'Documents','Python3','my_modules','gpym'))
from gpym import io


import warnings
warnings.filterwarnings("ignore")

date = datetime(2018,12,14)
time0 = datetime(1970,1,1)

compress = False


while date<datetime(2018,12,15):# (datetime.today()-timedelta(days = 1)):
    if date.day == 1:
        sys.stdout.write('%s\r' % date)
        sys.stdout.flush()
    
    print()
    date_str = '%04d-%02d-%02d' % (date.year, date.month, date.day)
    cmd = ('python /data/doppler/GPM/ScriptShare/get_GPM_files.py %s radar ' + 
        '2A-ENV.GPM.DPR.V 06') % date_str
    os.system(cmd)
    
    
    if compress:
        dpr_dir = '/data/doppler/GPM/V06/%04d/%02d/%02d/radar' % (
            date.year, date.month, date.day)
        files = [f for f in os.listdir(dpr_dir) 
                 if f.startswith('2A-ENV.GPM.DPR.V') and f.endswith('.HDF5') ]
        
        for file in files:
            print(file)
            fn = os.path.join(dpr_dir, file)
            fn_out = fn.replace('.HDF5','.nc')
            dset = io.open_dataset(fn, subgroup = 'NS', read_attrs = True)
            
            encoding = {key: {"zlib": True, "complevel": 6, 
                              'least_significant_digit': 3,
                              }
                            for key in dset.variables}  
            encoding = {key: {"compression": "lzf", "compression_opts": 9}
                            for key in dset.variables}  
            
            
            for key in ['airTemperature', 'airPressure', 'waterVapor',
                        'cloudLiquidWater']:
                encoding[key]['chunksizes'] = (1,1,176)
            # encoding = {key: {'dtype': 'int16',                              
            #              '_FillValue': -9999,}
            #             for key in dset.variables}  
            nscan_s = dset.nscan.size
            
            # encoding['Latitude']['scale_factor'] = 0.003
            # encoding['Longitude']['scale_factor'] = 0.006
        
            # encoding['airPressure']['scale_factor'] = 0.05
            
            
            # for key in dset.variables:                
            #     if 'Temperature' in key:
            #         encoding[key]['scale_factor'] = 0.002
            
            
            dset['surfaceWind'] = dset.surfaceWind.rename({'phony_dim_4': 'nwind'})
            dset = dset.isel({'phony_dim_4':  0})
            
            dset['cloudLiquidWater'] = dset['cloudLiquidWater']*1e3
            dset['cloudLiquidWater'].attrs['units'] = 'g/m^3'
            dset['cloudLiquidWater'].attrs['Units'] = 'g/m^3'
            # encoding['cloudLiquidWater']['scale_factor'] = 0.00001
            dset['waterVapor'] = dset['waterVapor']*1e3
            dset['waterVapor'].attrs['units'] = 'g/m^3'
            dset['waterVapor'].attrs['Units'] = 'g/m^3'  
            # encoding['waterVapor']['scale_factor'] = 0.00001
            dset.to_netcdf(fn_out, mode='w', encoding = encoding)
            dset.to_netcdf(fn_out, mode='w', encoding = encoding,
                           engine = 'h5netcdf')
            # engine = 'h5netcdf'
            # encoding = {key: {'scale_factor': 1e-2, 'dtype': 'int16',}                    
            #                 for key in dset.variables}
            
            # encoding['Latitude']['scale_factor'] = 1e-3
            # encoding['Longitude']['scale_factor'] = 1e-3
            
            # for key in dset.variables:
            #     if (('Pressure' in key) or ('Temperature' in key) or 
            #        ('water' in key ) or ('Water' in key )):
            #         encoding[key]['dtype'] = 'uint16'      
            # dset.to_netcdf(fn_out, mode='w', encoding = encoding)
                  
            
            os.chmod(fn_out, 0o744)        
            os.remove(fn) 
            aaa
    date += timedelta(days = 1)
    
    
