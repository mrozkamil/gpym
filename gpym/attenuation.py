#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:27:50 2021

@author: km357
"""

import itur.models.itu676 as itu676
import itur.models.itu840 as itu840

from . import misc
freqs = misc.band_usual_freq


def gases(ds_env, band, ):
    """    

    Parameters
    ----------
    ds_env : xarray dataset genereted from the 2A-ENV.GPM.DPR file
        Use gpym.io.open_dataset to create this object.
        Please use ds_env.sel() or ds_env.isel() for subsets of the file.
    band : str
        either 'Ku or 'Ka'

    Returns
    -------
    numpy array
        Specific attenuation estimate due to atmospheric gases

    """
    P = ds_env.airPressure.values
    T = ds_env.airTemperature.values
    rho_air = ds_env.waterVapor.isel(nwater=0).values*1e3
    
    fs = freqs[band]
    return itu676.gamma_exact(fs, P=P, rho=rho_air, T=T)
   
   
def liquid_cloud(ds_env, band):
    """    

    Parameters
    ----------
    ds_env : xarray dataset genereted from the 2A-ENV.GPM.DPR file
        Use gpym.io.open_dataset to create this object.
        Please use ds_env.sel() or ds_env.isel() for subsets of the file.
    band : str
        either 'Ku or 'Ka'

    Returns
    -------
    numpy array
        Specific attenuation estimate due to liquid clouds 

    """       
    
    T = ds_env.airTemperature.values
    rho_cl = ds_env.cloudLiquidWater.isel(nwater=0).values*1e3    
    fs = freqs[band]
    sp_att_coef_cl = itu840.specific_attenuation_coefficients(f = fs, 
                                                         T = T-273.14)    
    return sp_att_coef_cl*rho_cl
    