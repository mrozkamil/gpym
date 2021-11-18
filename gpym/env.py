#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 07:18:09 2019

@author: km357
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
def standard_atmos_press(altitude,T_b = 288.15, P_b = 101325, 
                          lap_rate = -0.0065, M = 0.0289644,
                          R_s = 8.3144598, g_0 = 9.80665):
    """
    input: altitude in [m]
    T_b Sea level standard temperature [k] 
    P_b Sea level standard atmospheric pressure [Pa]
    lap_rate = -0.0065 [K/m]
    M Molar mass of dry air [kg/mol]
    g_0 Earth gravitational acceleration [m/s^2]
    R_s Universal gas constant [J/mol/K]
    returns 
    pressure in [Pa]
    """
    expon = g_0*M/R_s/lap_rate    
    return P_b*((1- altitude*lap_rate/T_b)**expon)

def water_vapour_saturation_pressure(T):
    """
    input temperature in K
    output pressure in Pa
    """
    t = T-273.15
    return 610.78*np.power(10, 7.5*t/(t+237.3))

def air_density(T = 283.15, RH = 0.5, pressure = 101300):
    """
    input: pressure in Pa, T in K, RH in fractions ie 0<=RH<=1
    returns: air density in kg/m^3
    """
    p_sat = water_vapour_saturation_pressure(T)
    vapour_press = p_sat*RH
    M_d = 0.028964 # Molar mass of dry air, kg/mol
    M_v = 0.018016 # Molar mass of water vapor,  kg/mol
    p_d =  pressure - vapour_press # Partial pressure of dry air (Pa)
    R = 8.314 #Universal gas constant,  J/(K·mol)
    return (p_d*M_d + vapour_press*M_v)/(R*T)

def _fit_funct(D,a,b,c):
    return a-b*np.exp(-D*c)

def _fit_funct2(D,a,b,c):
    return a*(1-np.exp(-(D/b)**c))
def _blend_function(D):
    return (np.tanh(D)+1)*0.5
    
class rain_terminal_velocity():
    def __init__(self):
        D_smpl = np.array([-0.009126,-0.007826,0,0.007826,0.009126,
                            0.02, 0.03, 0.04, 0.05,
                           0.06, 0.07, 0.08, 0.09, 0.1,
                           0.12, 0.14, 0.16, 0.18, 0.2, 
                           0.22, 0.24, 0.26, 0.28, 0.3,
                           0.32, 0.34, 0.36, 0.38, 0.4,
                           0.42, 0.44, 0.46, 0.48, 0.5,
                           0.52, 0.54, 0.56, 0.58])*1e1
        V_smpl = np.array([25, 18, 0, 18, 25,
                            72, 117, 162, 206,
                           247, 287, 327, 367, 403,
                           464, 517, 565, 609, 649, 
                           690, 727, 757, 782, 806,
                           826, 844, 860, 872, 883,
                           892, 898, 903, 907, 909,
                           912, 914, 916, 917])*1e-2
        
        params = curve_fit(_fit_funct, D_smpl[-15:], V_smpl[-15:], 
                           bounds=([9.,9.,0.], [12., 35., 3.]))[0]
        params2 = curve_fit(_fit_funct2, D_smpl[-15:], V_smpl[-15:], 
                           bounds=([9.,0.,0.], [12., 2., 2.]))[0]
        D_large = np.arange(6,10,0.2)
        V_large = _fit_funct2(D_large,params2[0],params2[1],params2[2])
        g_k_fun = UnivariateSpline(np.concatenate((D_smpl,D_large,)),
                                   np.concatenate((V_smpl,V_large,)),
                                   s = 5e-3)     

        self.gunn_kinzer_table = (D_smpl, V_smpl)
        self.models = {'Kessler69': (lambda D: 13*np.sqrt(0.1*D)),
                  'Altas&Ulbrich77': (lambda D: 17.67*np.power(0.1*D, 0.67)),
                  'Willis84': (lambda D: 4.854*D*np.exp(-0.195*D)),
                  'Best50': (lambda D: 9.58*(1-np.exp(-(D/1.71)**1.147))),
                  'Atlas73': (lambda D: 9.65-10.3*np.exp(-0.6*D)),
                  'Brandes2002': (lambda D: 1e-2*(-10.21 + 493.2 * D 
                                    - 95.51 * D**2 + 7.934*D**3-0.2362*D**4)),
                  'Gossard': (lambda D: 9.65*(1-np.exp(-0.53 * D))),
                  'Stokes': (lambda D: 30*D**2),
                  'Gunn&Kinzer': (lambda D: g_k_fun(D)),
                  'large_fit': (lambda D: _fit_funct(D,params[0],params[1],params[2])),
                  'large_fit2': (lambda D: _fit_funct2(D,params2[0],params2[1],params2[2])),
                  }
        
    def __call__(self, D, method = 'Gunn&Kinzer'):
        vel = self.models[method](D)        
        flag = (D<0.1)
        if np.any (flag):
            blend = _blend_function(1e2*(D-0.1))
            vel_blend = self.models['Stokes'](D)
            vel = vel*blend + vel_blend*(1-blend)
        if method == 'Gunn&Kinzer':
            flag = (D>self.gunn_kinzer_table[0][-1])
            if np.any (flag):
                blend = _blend_function(1e1*(D-self.gunn_kinzer_table[0][-3]))
                vel_blend = self.models['large_fit2'](D)
                vel = vel*(1-blend) + vel_blend*(blend)
        return vel
            