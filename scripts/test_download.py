#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:38:01 2021

@author: km357
"""
import os, sys

print(os.environ['PYTHONPATH'].split(os.pathsep))
home = os.path.expanduser("~")
os.environ["PYTHONPATH"] = os.path.join(home,'conda3','envs', 'gpym')
print(os.environ['PYTHONPATH'].split(os.pathsep))
if '/cm/local/apps/cuda/libs/current/pynvml' in sys.path:
    sys.path.remove('/cm/local/apps/cuda/libs/current/pynvml')

sys.path.append(os.path.join(home, 'Documents','Python3', 'gpym'))
print(sys.path)

from datetime import datetime, timedelta
import xarray as xr
import gpym
import warnings
warnings.filterwarnings("ignore")

# make datasets display nicely
xr.set_options(display_style="html")

    
gpm_dir = os.path.join(home, 'Data', 'GPM')

log_in =  'dcw17@leicester.ac.uk'
log_in = 'km357@leicester.ac.uk'
pps_client = gpym.server.PPS(usr = log_in, 
                             save_dir = gpm_dir)
date = datetime(2017,12,31)

while date<datetime(2022,1,1):
    
    date += timedelta(days=1)
    
    date_str = '%04d-%02d-%02d' % (date.year, date.month, date.day)
    try:
        # pps_client.download_files(date_str, prod = '2A.GPM.DPR.V',
        #                       verbose = True) 
        pps_client.download_files(date_str, prod = '1C.GPM.GMI.XCAL',
                              verbose = True, ver = 5, directory = '1C') 
        
    except Exception as ex:
        print('ERROR:')
        print(ex)
        print()
        
    
    
    
    # pps_client.download_files(date_str, prod = '2A-ENV.GPM.DPR.V', 
    #                     gran = orbit, verbose = True) 
