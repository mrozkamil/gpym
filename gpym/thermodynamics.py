#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:12:17 2020

@author: km357
"""
from astropy import units 

dry_air_const = 287.058 *units.Joule/units.kg/units.Kelvin
water_vapour_const = 461.5 *units.Joule/units.kg/units.Kelvin

def specific_humidity_to_vapour_pressure(P,q, approx = False):
    
    """Parameters:	
    P (number or Quantity) – Total atmospheric pressure (hPa)
    q (number or Quantity) – Specific humidity (kg/kg)    
    Returns:	
    e – Water vapour pressure (hPa)
    """
    Rd_Rv_ratio = dry_air_const/water_vapour_const
    if approx:
        denom = Rd_Rv_ratio
    else:
        denom = q*(1-Rd_Rv_ratio) + Rd_Rv_ratio
    return (q*P/denom)
def vapour_pressure_to_vapour_density(T,e):
    """Parameters:	
    T (number or Quantity) – Absolute temperature (K)
    e – Water vapour pressure (hPa)    
    Returns:	
    rho_v – Water vapour density (kg/m^3)
    """
    return (e/T/water_vapour_const).decompose()

def vapour_density(T,P,q, approx = False):
    """Parameters:	
    T (number or Quantity) – Absolute temperature (K)
    P (number or Quantity) – Total atmospheric pressure (hPa)
    q (number or Quantity) – Specific humidity (kg/kg)    
    Returns:	
    rho_v – Water vapour density (kg/m^3)
    """
    e = specific_humidity_to_vapour_pressure(P,q, approx = approx)
    return vapour_pressure_to_vapour_density(T,e)