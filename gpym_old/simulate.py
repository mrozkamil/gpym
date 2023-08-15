#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:51:49 2022

@author: km357
"""
import os
import xarray as xr
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter
import numpy as np
from . import misc

module_path = '/Users/km357/Documents/Python3/gpym/gpym'
# module_path = os.path.dirname(__file__)

# class radar_simulator():
    
#     def __init__(self, filename = None):
        
#         ### loading scattering LUT
#         module_path = os.path.dirname(__file__)
#         hyd_type = ['ice', 'rain']
#         fn = {'ice': 'insitu_PSD_dataset_Dle_stats_Gamma.nc',
#                   'rain': 'NASA_DSD_dataset_2DVD_5min_stats_Gamma.nc'}
#         self.scat_LUT = {}
#         for hyd in hyd_type:
#             LUT_fn = os.path.join(module_path, 'scattering_tables', hyd, 
#                               'PSD_Integrated', 'gamma', 'mimics_insitu',
#                               fn[hyd])
#             self.scat_LUT[hyd] = xr.load_dataset(LUT_fn) # ice
#             self.scat_LUT[hyd] = self.scat_LUT[hyd].rename({'Dm_dB_bin': 'Dm_dB',
#                                              'PR_dB_bin': 'PR_dB'})
            
#         ### generating a new coordinate
#         self.scat_LUT['ice']['Alpha_dB'] = misc.dB(self.scat_LUT['ice']['Alpha'])
#         tmp_cords = list(self.scat_LUT['ice'].coords)
#         tmp_cords.append('Alpha_dB')
#         self.scat_LUT['ice'] = self.scat_LUT['ice'].set_coords(tmp_cords)
#         self.scat_LUT['ice'] = self.scat_LUT['ice'].swap_dims({"Alpha": "Alpha_dB"})

#         ### define the scattering LUT interpolation limits 
#         self.scat_LUT_lims = {'ice':{},'rain':{}}
#         for hyd in hyd_type:
#             for var_n in ['Dm_dB', 'PR_dB','Alpha_dB']:
#                 if var_n in self.scat_LUT[hyd].coords:
#                     self.scat_LUT_lims[hyd][var_n] = [
#                         self.scat_LUT[hyd][var_n].values.min(),
#                         self.scat_LUT[hyd][var_n].values.max()]
            
#         tm_da = self.scat_LUT['ice'].histogram_Dm_dB_PR_dB.sum(dim = 'PR_dB')
#         lims_da = [tm_da.Dm_dB.where(tm_da>0).min(dim = 'Dm_dB'),
#             tm_da.Dm_dB.where(tm_da>0).max(dim = 'Dm_dB')]
#         for ii, lim_da in enumerate(lims_da):
#             self.scat_LUT_lims['ice']['Dm_dB'][ii] = RGI(
#                 [lim_da[c].data for c in lim_da.dims], lim_da.data, 
#                 method = 'linear', bounds_error=False, fill_value=-99.)
            
        
#         # self.scat_LUT['ice'] = self.scat_LUT['ice']['Z'].fillna(-99.)
        
        
#         self.scat_RGI = {'ice':{}, 'rain':{}}
#         sim_vars = ['Z', 'k']
#         fill_val = {'Z': -999., 'k': 0.}
#         bands = ['Ku', 'Ka']
#         for hyd in hyd_type:
#             for var_n in sim_vars:
#                 self.scat_RGI[hyd][var_n] = {}
#                 self.scat_RGI[hyd][var_n+'_valid'] = {}
#                 for band in bands:
#                     v_name = 'mean_%s_%s' % (var_n,band)
#                     arr = self.scat_LUT[hyd][v_name].copy()
#                     arr_valid = np.isfinite(arr.values).astype('float')
#                     arr = arr.fillna(fill_val[var_n])

#                     self.scat_RGI[hyd][var_n][band] = RGI(
#                         [arr[c].data for c in arr.dims],arr.data, 
#                         method = 'linear',  bounds_error=False, 
#                         fill_value=fill_val[var_n])
#                     self.scat_RGI[hyd][var_n+'_valid'][band] = RGI(
#                         [arr[c].data for c in arr.dims],arr_valid, 
#                         method = 'linear',  bounds_error=False, 
#                         fill_value=fill_val[var_n])

#     def BB_ext(self, PR, band = 'Ku'):
#         if band == 'Ku':
#             total_ext = 0.048 * PR**1.05
#         elif band == 'Ka':
#             total_ext = 0.66 * PR**1.1
#         return total_ext
    
    
#     def _simulate(self,Dm_dB, PR_dB, Alpha_dB = -18.,
#             hydro = 'rain', var = 'Z', band = 'Ku'):
#         # fill_val = {'Z': -99., 'k': 0.}
#         if hydro == 'ice':
#             if isinstance(Alpha_dB, (int, float, complex)):
#                 Alpha_dB_ice = Alpha_dB*np.ones(Dm_dB.shape)
#             elif type(Alpha_dB)==np.ndarray:
#                 Alpha_dB_ice = np.copy(Alpha_dB)
#             ret_val = self.scat_RGI[hydro][var][band]((Dm_dB, PR_dB, Alpha_dB_ice))
             
#         elif hydro == 'rain':
#             ret_val = self.scat_RGI[hydro][var][band]((Dm_dB, PR_dB,))
#         # ret_val[~np.isfinite(ret_val)] = fill_val[var]
#         return ret_val
        
#     def simulate_Ze_k(self, 
#             Dm_dB_rain, PR_dB_rain,  flag_rain, 
#             Dm_dB_ice,  PR_dB_ice, Alpha_dB, flag_ice, 
#             k_ML = None,  band = 'Ku',):

#         Ze = np.full(flag_rain.shape, 1e-99)
#         spec_att = np.zeros(flag_rain.shape)
            
#         Ze_ice = self._simulate(Dm_dB = Dm_dB_ice, 
#             PR_dB = PR_dB_ice, Alpha_dB = Alpha_dB,
#             hydro = 'ice', var = 'Z', band = band)
        
#         k_ice = self._simulate(Dm_dB = Dm_dB_ice, 
#             PR_dB = PR_dB_ice, Alpha_dB = Alpha_dB,
#             hydro = 'ice', var = 'k', band = band)
        
#         Ze[flag_ice] += misc.inv_dB(Ze_ice)
#         spec_att[flag_ice] +=k_ice
        
#         Ze_rain = self._simulate(
#             Dm_dB = Dm_dB_rain,  PR_dB = PR_dB_rain, 
#             hydro = 'rain', var = 'Z', band = band)
#         k_rain = self._simulate(
#             Dm_dB = Dm_dB_rain,  PR_dB = PR_dB_rain, 
#             hydro = 'rain', var = 'k', band = band)
        
#         Ze[flag_rain] += misc.inv_dB(Ze_rain)
#         spec_att[flag_rain] +=k_rain
        
#         if k_ML is not None:
#             spec_att += k_ML
#         return misc.dB(Ze), spec_att
    
#     def simulate_Zm(self, 
#             Dm_dB_rain, PR_dB_rain,  flag_rain, 
#             Dm_dB_ice,  PR_dB_ice, Alpha_dB, flag_ice, 
#             k_ML = None,  band = 'Ku',
#             range_spacing = 0.125, range_axis = 0,
#             return_effective = False):
        
#         Ze, spec_att = self.simulate_Ze_k(  
#             Dm_dB_rain = Dm_dB_rain, PR_dB_rain = PR_dB_rain,
#             flag_rain = flag_rain, 
#             Dm_dB_ice = Dm_dB_ice,  PR_dB_ice = PR_dB_ice, 
#             Alpha_dB = Alpha_dB, flag_ice = flag_ice, 
#             k_ML = k_ML,  band = band,)
        
#         one_way_att = np.cumsum(spec_att, axis = range_axis)*range_spacing
        
#         Zm = np.copy(Ze)
#         Zm[1:] += -2.*one_way_att[:-1]
#         if return_effective:
#             return Zm, 2*one_way_att[-1], Ze, spec_att
#         else:
#             return Zm, 2*one_way_att[-1]
        
        
class radar_simulator_logdm_logsm():
    
    def _prepocess_rain_lut(self, ds):
        ds_new = ds.assign_coords(T = ds.attrs['temp_K'])
        ds_new['ref_in_real'] = ds.attrs['ref_in_real']
        ds_new['ref_in_imag'] = ds.attrs['ref_in_imag']
        ds_new = ds_new.expand_dims('T')
        ds_new['T'].attrs['units'] = 'K'
        ds_new = ds_new.transpose('Dm_dB', 'Sm_p_dB', 'T', 
                                  missing_dims='ignore')
        return ds_new
    
    def _prepocess_ice_lut(self, ds):
        ds_new = ds.assign_coords(Alpha_dB = 10*ds.attrs['log10_alpha'])
        ds_new = ds_new.rename({'Dm': 'Dm_ice', 'Dm_rain': 'Dm'})
        ds_new['ref_in_real'] = ds.attrs['ref_in_real']
        ds_new['ref_in_imag'] = ds.attrs['ref_in_imag']
        ds_new = ds_new.expand_dims('Alpha_dB')
        ds_new = ds_new.transpose('Dm_dB', 'Sm_p_dB', 'Alpha_dB',
                                  missing_dims='ignore')
        return ds_new
    
    
    
    def __init__(self, get_derivatives = False):
        
        ### loading scattering LUT
       
        hyd_type = ['ice', 'rain']
        bands = ['Ku', 'Ka']
        
        lut_dir = os.path.join(module_path, 'scattering_tables', '%s', '%s')
        lut_file_names = {hyd: {band: [] for band in bands} 
                          for hyd in hyd_type}
        self.scat_LUT = {hyd: {band: dict() for band in bands} 
                         for hyd in hyd_type}
        self.scat_LUT_lims = {hyd: {band: dict() for band in bands} 
                              for hyd in hyd_type}
        self.scat_LUT_size = {hyd: {band: dict() for band in bands} 
                              for hyd in hyd_type}
        self.scat_LUT_regular_grid = {hyd: {band: dict() for band in bands} 
                              for hyd in hyd_type}
        self.scat_RGI = {hyd: {band: dict() for band in bands} 
                              for hyd in hyd_type}
        for band in bands:
            for hyd in hyd_type:
                
                #LUT file list
                LUT_dir = lut_dir % (hyd,band)
                tmp_list = [os.path.join(LUT_dir,fn) 
                            for fn in os.listdir(LUT_dir)
                            if 'Williams' in fn and fn.endswith('.nc')]
                tmp_list.sort()
                lut_file_names[hyd][band] = tmp_list
                #open LUT files for ice and rain
                if hyd == 'rain':
                    self.scat_LUT[hyd][band] = xr.open_mfdataset(
                        lut_file_names[hyd][band],
                        preprocess = self._prepocess_rain_lut, 
                        concat_dim='T',  combine='nested') # rain
                    self.scat_LUT[hyd][band] = self.scat_LUT[hyd][band].sortby('T')
                if hyd == 'ice':
                    self.scat_LUT[hyd][band] = xr.open_mfdataset(
                        lut_file_names[hyd][band],
                        preprocess = self._prepocess_ice_lut, 
                        concat_dim='Alpha_dB',  combine='nested') # ice
                    self.scat_LUT[hyd][band] = self.scat_LUT[hyd][band].sortby('Alpha_dB')
                    self.scat_LUT[hyd][band]['IWC_dB'] = 10*np.log10(
                        self.scat_LUT[hyd][band]['IWC'])
                
                
                    
                self.scat_LUT[hyd][band] = self.scat_LUT[hyd][band].load()
                self.scat_LUT[hyd][band]['k_dB'] = 10*np.log10(
                    self.scat_LUT[hyd][band]['k'])
                self.scat_LUT[hyd][band]['PR_dB'] = 10*np.log10(
                    self.scat_LUT[hyd][band]['PR'])
                # get LUT variables limits
                for var_n in ['Dm_dB', 'Sm_p_dB','Alpha_dB', 'T']:
                    if var_n in self.scat_LUT[hyd][band].coords:
                        self.scat_LUT_lims[hyd][var_n] = np.array([
                            self.scat_LUT[hyd][band][var_n].values.min(),
                            self.scat_LUT[hyd][band][var_n].values.max()])
                        self.scat_LUT_size[hyd][band][var_n] = (
                            self.scat_LUT[hyd][band][var_n].size)
                        self.scat_LUT_regular_grid[hyd][band][var_n] = np.all(
                            np.diff(self.scat_LUT[hyd][band][var_n],n=2)<1e6)
                        
 
                        
                # generate regular grid interpolants 
                # They are faster than xarray interpolation 
                # but slower than map_coordinates based on spline approximation
                for var_n in self.scat_LUT[hyd][band].variables:
                    arr = self.scat_LUT[hyd][band][var_n].copy()
                    dims = arr.dims
                    if 'Dm_dB' in dims and 'Sm_p_dB' in dims and (
                            'T' in  dims or 'Alpha_dB' in dims):
                        # print(var_n)
                        self.scat_RGI[hyd][band][var_n] = RGI(
                            [arr[c].data for c in arr.dims],arr.data, 
                            method = 'linear',  bounds_error=False, 
                            fill_value=arr.data.min())
                        if get_derivatives:
                            for ind_dim, dim in enumerate(dims):
                                D_arr = savgol_filter(arr.data, 
                                    window_length = 5, polyorder = 2, deriv = 1,
                                    axis = ind_dim)
                                D_x = savgol_filter(arr[dim].data, 
                                    window_length = 5, polyorder = 2, deriv = 1,)
                                
                                expand_dims = [0,1,2]
                                expand_dims.remove(ind_dim)
                                
                                
                                Darr_Dx = D_arr/np.expand_dims(D_x, axis=expand_dims)
                                
                                var_n_deriv = 'd_%s_d_%s' % (var_n, dim)
                                self.scat_LUT[hyd][band][var_n_deriv] = (dims, Darr_Dx)
                                
                                self.scat_RGI[hyd][band][var_n_deriv] = RGI(
                                    [arr[c].data for c in arr.dims],Darr_Dx, 
                                    method = 'linear',  bounds_error=False, 
                                    fill_value=0.)
                                
    
    def __call__(self, hydro = 'rain', var = 'Z', band = 'Ku', 
                 interp_method = 'SPL',order = 1 , 
                 **kwargs, ):
        allowed_kwargs = ['Dm_dB' , 'Sm_p_dB', 'T', 'Alpha_dB','WC_dB']
        kwargs_list = list(kwargs.keys())
        for key in kwargs_list:
            if key not in allowed_kwargs:
                print('%s is not in the list of allowed variables:')
                print(allowed_kwargs)
                return
        if 'Alpha_dB' in kwargs_list:
            hydro = 'ice'
        
        if hydro =='rain':
            var_list = ['Dm_dB' , 'Sm_p_dB', 'T', ]
            if 'T' not in kwargs_list:
                kwargs['T'] = 273.15*np.ones(kwargs['Dm_dB'].shape)
            
        if hydro =='ice':
            var_list =  ['Dm_dB' , 'Sm_p_dB', 'Alpha_dB',]
            if 'Alpha_dB' not in kwargs_list:
                kwargs['Alpha_dB'] = -16.*np.ones(kwargs['Dm_dB'].shape)
        if interp_method == 'RGI':
            X = tuple([kwargs[vn] for vn in var_list])
            method={0: 'nearest', 1:'linear', 3:'cubic', 5: 'quintic'}
            interp_val = self.scat_RGI[hydro][band][var](X, method = method[order])
            
        if interp_method == 'SPL':
            X = np.array([(kwargs[vn] - self.scat_LUT_lims[hydro][vn][0])/
                 np.diff(self.scat_LUT_lims[hydro][vn])*
                 (self.scat_LUT_size[hydro][band][vn]-1) for vn in var_list])
            
            interp_val = map_coordinates(self.scat_LUT[hydro][band][var].data, 
                                         coordinates = X, mode = 'nearest',
                                         order = order)
        
        if var in ['Z', 'k_dB', 'PR_dB', 'IWC_dB']:
            if 'WC_dB' in kwargs_list:
               interp_val += kwargs['WC_dB']
        if var in ['k', 'PR', 'IWC']:
            if 'WC_dB' in kwargs_list:
                WC = 10**(0.1*kwargs['WC_dB'])
                interp_val *= WC
        return interp_val
        

    def BB_ext(self, PR, band = 'Ku'):
        """
        Matrosov (2008) "Assessment of Radar Signal Attenuation Caused by 
        the Melting Hydrometeor Layer" 
        """
        if band == 'Ku':
            total_ext = 0.048 * PR**1.05
        elif band == 'Ka':
            total_ext = 0.66 * PR**1.1
        return total_ext
    
    
        
    def simulate_Ze_k(self, 
            Dm_dB_rain, WC_dB_rain,  flag_rain, 
            Dm_dB_ice,  WC_dB_ice, Alpha_dB, flag_ice, 
            k_ML = None,  band = 'Ku',):

        Ze = np.full(flag_rain.shape, 1e-99)
        spec_att = np.zeros(flag_rain.shape)
            
        Ze_ice = self(Dm_dB = Dm_dB_ice, 
            WC_dB = WC_dB_ice, Alpha_dB = Alpha_dB,
            hydro = 'ice', var = 'Z', band = band)
        
        k_ice = self(Dm_dB = Dm_dB_ice, 
            WC_dB = WC_dB_ice, Alpha_dB = Alpha_dB,
            hydro = 'ice', var = 'k', band = band)
        
        Ze[flag_ice] += misc.inv_dB(Ze_ice)
        spec_att[flag_ice] +=k_ice
        
        Ze_rain = self(
            Dm_dB = Dm_dB_rain,  WC_dB = WC_dB_rain, 
            hydro = 'rain', var = 'Z', band = band)
        k_rain = self._simulate(
            Dm_dB = Dm_dB_rain,  WC_dB = WC_dB_rain, 
            hydro = 'rain', var = 'k', band = band)
        
        Ze[flag_rain] += misc.inv_dB(Ze_rain)
        spec_att[flag_rain] +=k_rain
        
        if k_ML is not None:
            spec_att += k_ML
        return misc.dB(Ze), spec_att
    
    def simulate_Zm(self, 
            Dm_dB_rain, WC_dB_rain,  flag_rain, 
            Dm_dB_ice,  WC_dB_ice, Alpha_dB, flag_ice, 
            k_ML = None,  band = 'Ku',
            range_spacing = 0.125, range_axis = 0,
            return_effective = False):
        
        Ze, spec_att = self.simulate_Ze_k(  
            Dm_dB_rain = Dm_dB_rain, WC_dB_rain = WC_dB_rain,
            flag_rain = flag_rain, 
            Dm_dB_ice = Dm_dB_ice,  WC_dB_ice = WC_dB_ice, 
            Alpha_dB = Alpha_dB, flag_ice = flag_ice, 
            k_ML = k_ML,  band = band,)
        
        one_way_att = np.cumsum(spec_att, axis = range_axis)*range_spacing
        
        Zm = np.copy(Ze)
        Zm[1:] += -2.*one_way_att[:-1]
        if return_effective:
            return Zm, 2*one_way_att[-1], Ze, spec_att
        else:
            return Zm, 2*one_way_att[-1]
        
import time
samples = 100
Dm_dB = np.linspace(-3,0)
Sm_p_dB = np.zeros_like(Dm_dB)
T = np.ones_like(Dm_dB)*283.15
order = 1;

hydro = 'rain'
band = 'Ku'
var = 'Z'

radar = radar_simulator_logdm_logsm()

start = time.time()
for n in range(samples):
    Dm_dB +=1/samples
    out = radar(Dm_dB = Dm_dB, Sm_p_dB = Sm_p_dB, T = T, hydro = hydro,
                var = var, band = band, interp_method = 'RGI', order = order)
end = time.time()
print ('RegularGridInterpolator: %0.1f [us]' % (((end-start)/samples)*1e6))

start = time.time()
for n in range(samples):
    Dm_dB +=1/samples
    out = radar(Dm_dB = Dm_dB, Sm_p_dB = Sm_p_dB, T = T, hydro = hydro,
                var = var, band = band, interp_method = 'SPL', order = order)
end = time.time()
print ('map_coordinates: %0.1f [us]' % (((end-start)/samples)*1e6))
    