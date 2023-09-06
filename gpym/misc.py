#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 13:35:29 2021

@author: km357
"""

from scipy.interpolate import interp1d
import numpy as np
from scipy.special import expit, logit

band_range_dict = {
    'HF': (3e-3, 30e-3),
    'VHF': (30e-3, 300e-3),
    'UHF': (300e-3, 3.),
    'L': (1., 2.),
    'S': (2., 4.),
    'C': (4.,8.),
    'X': (8.,12.),
    'Ku': (12., 18.),
    'K': (18., 27.),
    'Ka': (27., 40.),
    'V': (40.,75.),
    'W': (75., 110.),
    'G': (110., 300.)}

band_usual_freq = {'S' : 2.8, 'C': 5.6,  'X': 9.6, 'Ku': 13.6,  'K': 24.,
      'Ka': 35.55, 'W': 94.0,  'G': 200}
for band in band_range_dict.keys():
    if band not in band_usual_freq.keys():
        band_usual_freq[band] = 1/(.5/band_range_dict[band][0]+
                             .5/band_range_dict[band][1])

def band_range(band):
    return band_range_dict[band]

def band_name(freq_GHz):
    if ((freq_GHz<band_range_dict['HF'][0]) or
        (freq_GHz>band_range_dict['G'][1])):
        print ('Provided frequency is outside of the considered range (3 MHz to 300 GHz)')
        return
    y_arr = np.arange(len(band_range_dict))
    x_arr = np.array([fr[0] for fr in band_range_dict.values()])
    bands = list(band_range_dict.keys())
    tmp_interp = interp1d(x_arr, y_arr, kind = 'previous',
                          fill_value="extrapolate")
    index = tmp_interp(freq_GHz)

    return bands[int(index)]

def dB(x):
    return 10*np.log10(x)

def inv_dB(x):
    return 10**(x/10)

def map_real_2_interval( x, yl = 0., yh = 1.):
    return expit(x)*(yh-yl)+yl

def map_interval_2_real( y, yl = 0., yh = 1.):
    y = np.minimum(np.maximum(y,yl),yh)
    return logit((y-yl)/(yh-yl))


