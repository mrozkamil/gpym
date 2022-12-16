#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:34:33 2022

@author: km357
"""
import os
import numpy as np
import xarray as xr
from scipy.optimize import minimize

from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from scipy.interpolate import RegularGridInterpolator as RGI
import matplotlib.pyplot as plt
import dill
from . import misc
from . import simulate
import timeit

module_path = '/Users/km357/Scripts/Python3/gpym/gpym'
# module_path = os.path.dirname(__file__)

radar_sim = simulate.radar_simulator()

class OE():
    def __init__(self, filename = None):
        
        ### loading scattering LUT
        
        hyd_type = ['ice', 'rain']
        fn = {'ice': 'insitu_PSD_dataset_Dle_stats_Gamma.nc',
                  'rain': 'NASA_DSD_dataset_2DVD_5min_stats_Gamma.nc'}
        self.scat_LUT = {}
        for hyd in hyd_type:
            LUT_fn = os.path.join(module_path, 'scattering_tables', hyd, 
                              'PSD_Integrated', 'gamma', 'mimics_insitu',
                              fn[hyd])
            self.scat_LUT[hyd] = xr.load_dataset(LUT_fn) # ice
            self.scat_LUT[hyd] = self.scat_LUT[hyd].rename({'Dm_dB_bin': 'Dm_dB',
                                             'PR_dB_bin': 'PR_dB'})
            
        ### generating a new coordinate
        self.scat_LUT['ice']['Alpha_dB'] = misc.dB(self.scat_LUT['ice']['Alpha'])
        tmp_cords = list(self.scat_LUT['ice'].coords)
        tmp_cords.append('Alpha_dB')
        self.scat_LUT['ice'] = self.scat_LUT['ice'].set_coords(tmp_cords)
        self.scat_LUT['ice'] = self.scat_LUT['ice'].swap_dims({"Alpha": "Alpha_dB"})
        self.scat_LUT['ice']['Alpha_dB'] = np.round(
            self.scat_LUT['ice']['Alpha_dB'].values, decimals =1)

        ZKu = self.scat_LUT['rain']['mean_Z_Ku'].values
        ZKa = self.scat_LUT['rain']['mean_Z_Ka'].values
        dDWR = savgol_filter(ZKu-ZKa, window_length=5, 
                             polyorder = 1, deriv=1, axis = 0)
        ind_good = np.where((ZKu>-99.) & np.isfinite(ZKu) &
                            (ZKa>-99.) & np.isfinite(ZKa) &
                            (dDWR>0))
        
        coords = np.stack([ZKu[ind_good], ZKa[ind_good]]).T
        self.rainTree = cKDTree(coords)
        
        self.Dm_dB_rain_tree =  self.scat_LUT['rain']['mean_Z_Ku'
                                ].Dm_dB.values[ind_good[0]]
        self.PR_dB_rain_tree =  self.scat_LUT['rain']['mean_Z_Ku'
                                ].PR_dB.values[ind_good[1]]
        
        
        self.Dm_dB_ice_tree, self.PR_dB_ice_tree, self.iceTree = {}, {}, {}
        for ind_a, aa in enumerate(self.scat_LUT['ice']['Alpha_dB'].values):
            
            # print(aa)
            ZKu = self.scat_LUT['ice']['mean_Z_Ku'].values[:,:,ind_a]
            ZKa = self.scat_LUT['ice']['mean_Z_Ka'].values[:,:,ind_a]
            dDWR = savgol_filter(ZKu-ZKa, window_length=5, 
                                 polyorder = 1, deriv=1, axis = 0)
            
            ind_good = np.where((ZKu>-99.) & np.isfinite(ZKu) &
                                (ZKa>-99.) & np.isfinite(ZKa) &
                                (dDWR>0))
            
            coords = np.stack([ZKu[ind_good], ZKa[ind_good]]).T
            
            self.iceTree[aa] = cKDTree(coords)
            
            self.Dm_dB_ice_tree[aa] =  self.scat_LUT['ice']['mean_Z_Ku'
                                    ].Dm_dB.values[ind_good[0]]
            self.PR_dB_ice_tree[aa] =  self.scat_LUT['ice']['mean_Z_Ku'
                                    ].PR_dB.values[ind_good[1]]
            
            # plt.figure()
            # plt.scatter(self.PR_dB_ice_tree[aa], self.Dm_dB_ice_tree[aa], s =1,
            #             c = ZKu[ind_good]-ZKa[ind_good])
            # plt.colorbar()
            
            # plt.figure()
            # plt.scatter(ZKu[ind_good],ZKu[ind_good] - ZKa[ind_good], 
            #     c = self.scat_LUT['rain']['mean_Z_Ku'].Dm_dB.values[ind_good[0]], 
            #     s = 1, vmin = -3, vmax = 4)
            # plt.colorbar()
            
            # aaaa
            

        ### a-priori information and a first guess LUT
        fn_ap = {'ice': 'insitu_PSD_dataset_stats_PCA_Dle_Dm_PR_Alpha.pkl',
                  'rain': 'NASA_DSD_dataset_2DVD_5min_PCA_Dm_PR.pkl'}
        fn_fg = {'ice':  'insitu_PSD_dataset_Dle_expected_unknowns.nc',
                  'rain': 'NASA_DSD_dataset_2DVD_5min_expected_unknowns.nc'}
        self.PCA, self.fg_LUT = {}, {}
        
        for hyd in hyd_type:
            stats_dir = os.path.join(module_path,'scattering_tables', 
                        hyd, 'PSD_Integrated', 'insitu', 'stats')
            
            ap_name = os.path.join(stats_dir, fn_ap[hyd])
            with open(ap_name, 'rb') as file_obj:
                self.PCA[hyd] = dill.load(file_obj)
                
            fg_name = os.path.join(stats_dir,fn_fg[hyd])
            self.fg_LUT[hyd] = xr.load_dataset(fg_name)
            
        self.fg_RGI, self.fg_std_RGI = {}, {}
        for hyd in hyd_type:
            self.fg_RGI[hyd], self.fg_std_RGI[hyd] = {}, {}
            for x_name, new_name in zip(['Dm_dB', 'PR_dB', 'logAlphaMG'],
                     ['Dm_dB', 'PR_dB', 'Alpha_dB']):
                self.fg_RGI[hyd][new_name], self.fg_std_RGI[hyd][new_name] = {}, {}
                for v_name in ['Z_Ku','Z_Ka','Z_Ku_Z_Ka']:
                    mean_name = 'mean_%s_for_%s' % (x_name, v_name)
                    std_name = 'std_%s_for_%s' % (x_name, v_name)
                    if mean_name in self.fg_LUT[hyd].variables:
                        band = v_name.replace('Z_','')
                        arr = self.fg_LUT[hyd][mean_name]
                        if x_name == 'logAlphaMG': arr = 10*arr
                        self.fg_RGI[hyd][new_name][band] = RGI(
                            [arr[c].data for c in arr.dims],arr.data, 
                            method = 'linear',  bounds_error=False, 
                            fill_value=-99.)
                        arr = self.fg_LUT[hyd][std_name]
                        if x_name == 'logAlphaMG': arr = 10*arr
                        self.fg_std_RGI[hyd][new_name][band] = RGI(
                            [arr[c].data for c in arr.dims],arr.data, 
                            method = 'linear',  bounds_error=False, 
                            fill_value=-99.)
                        
        alphas = radar_sim.scat_LUT['ice']['Alpha_dB'].values
        self.fg_exp_v_sims = {}
        for hyd in hyd_type:
            self.fg_exp_v_sims[hyd] = {}
            Dm_dB_tmp = self.fg_LUT[hyd]['mean_Dm_dB_for_Z_Ku']
            PR_dB_tmp = self.fg_LUT[hyd]['mean_PR_dB_for_Z_Ku']
            if hyd == 'ice':
                Dm_dB_tmp, ALPHAS = np.meshgrid(Dm_dB_tmp, alphas)
                PR_dB_tmp, ALPHAS = np.meshgrid(PR_dB_tmp, alphas)
                
            for v_name in ['Z_Ku','Z_Ka']:
                band = v_name.replace('Z_','')                
                self.fg_exp_v_sims[hyd][v_name] = radar_sim._simulate(
                    Dm_dB_tmp, PR_dB_tmp, Alpha_dB = ALPHAS, 
                    hydro = hyd, var = 'Z', band = band)                 
                


    # def _smooth_Z(self,Z,Z_thr = 10):
    #     Z_new = np.maximum(Z,Z_thr)
    #     return savgol_filter(Z_new, window_length= 5, polyorder=1, )
    
    
    def _Tikhonov_matrix(self, n, diff = 1):
        A = np.eye(n)- np.eye(n,k=1)
        A[-1,:] = 0
        Ap = np.linalg.matrix_power(A,diff)
        return np.matmul(Ap.T, Ap)
    
    
    
    def _fg_stats(self, hydro = 'rain', var_name = 'Dm_dB',  
                     Zku=None, Zka=None,):
        estimate, estimate_std = [],[]
        
        if Zku is not None:
            estimate.append(self.fg_RGI[var_name]['Ku']((Zku,)))
            estimate_std.append(self.fg_std_RGI[var_name]['Ku']((Zku,)))
        if Zka is not None:
            estimate.append(self.fg_RGI[var_name]['Ka']((Zka,)))
            estimate_std.append(self.fg_std_RGI[var_name]['Ka']((Zka,)))
        if Zka is not None and Zku is not None:
            estimate.append(self.fg_RGI[var_name]['Ku_Ka']((Zku,Zka)))
            estimate_std.append(self.fg_std_RGI[var_name]['Ku_Ka']((Zku,Zka)))
        
        estimate_arr = np.array(estimate)
        est_weight_arr = 1/np.array(estimate_std)**2
        flag_bad = ((est_weight_arr<1e-2) | ~np.isfinite(est_weight_arr) |
                     ~np.isfinite(estimate_arr))
        est_weight_arr[flag_bad] = 0
        estimate_arr[flag_bad] = 0
        final_est = np.sum(estimate_arr*est_weight_arr, axis = 0)/np.sum(
            est_weight_arr, axis = 0)
        return final_est
    
    def _fg_stats_simul(self, hydro = 'rain', var_name = 'Dm_dB',  
                     Zku=None, Zka=None, Alpha_dB = -15.):
        
        
        x = self.fg_LUT[hydro]['mean_%s_for_Z_Ku' % var_name].values
        if hydro=='ice':
            ind_alpha = np.argmin(( Alpha_dB-
                radar_sim.scat_LUT['ice']['Alpha_dB'].values)**2)
        
        
        if Zku is not None and Zka is None:            
            if hydro=='ice':                
                y = self.fg_exp_v_sims[hydro]['Z_Ku'][ind_alpha]
            else:
                y = self.fg_exp_v_sims[hydro]['Z_Ku']
            x_close = np.array([x[np.nanargmin((yy-y)**2)] for yy in Zku])            
            
        if Zka is not None and Zku is None:            
            if hydro=='ice':                
                y = self.fg_exp_v_sims[hydro]['Z_Ka'][ind_alpha]
            else:
                y = self.fg_exp_v_sims[hydro]['Z_Ka']
            x_close = np.array([x[np.nanargmin((yy-y)**2)] for yy in Zka])            
           
        if Zka is not None and Zku is not None:   
            if hydro=='ice':                
                y0 = self.fg_exp_v_sims[hydro]['Z_Ku'][ind_alpha]
                y1 = self.fg_exp_v_sims[hydro]['Z_Ka'][ind_alpha]
            else:
                y0 = self.fg_exp_v_sims[hydro]['Z_Ku']
                y1 = self.fg_exp_v_sims[hydro]['Z_Ka']   
            
            x_close = np.array([x[np.nanargmin((yy0-y0)**2 + (yy1-y1)**2)] 
                                for (yy0, yy1) in zip(Zku, Zka)])           
       
        return x_close
    
    
    def _attenuation_correction(self,Zm_v, ind_rain, ind_ice, ind_ml, 
                                delta_PIA, range_spacing = 0.125):
        
        poly_rain_k_db = {'Ka': np.array([-19.854, 5.914, -0.423, 0.138])[::-1],
                          'Ku': np.array([-37.723, 11.539, -1.162, 0.085])[::-1]}
        
        Zm_tmp = {} 
        for band in ['Ku', 'Ka']:
            tmp_Z = np.copy(Zm_v[band])
            if band == 'Ku':
                thr = 9
                
            elif band == 'Ka':
                thr = 15
            
            flag_snr_low = (tmp_Z < thr)
            tmp_Z[flag_snr_low] = thr
            flag_bad = np.full(tmp_Z.size, True)
            Zm_tmp[band] = np.copy(tmp_Z)
            for poly_o in [4,3,2,1]:
                Zm_sm = savgol_filter(x = tmp_Z, 
                    window_length=poly_o*2+1, polyorder=poly_o)
                Zm_tmp[band][flag_bad] = Zm_sm[flag_bad]
                flag_bad = (savgol_filter(x = (flag_snr_low).astype(float), 
                                      window_length=poly_o*2+1, polyorder=0)>0.)
        
            Zm_tmp[band][flag_bad] = Zm_v[band][flag_bad]

            
        Ze_v = {band: Zm_tmp[band]*1. for band in ['Ku','Ka']}
        
        
        
        
        for ii in range(3):
            spec_att = {band: np.zeros(Zm_v[band].shape) for band in ['Ku','Ka']}
            att_corr = {band: np.zeros(Zm_v[band].shape) for band in ['Ku','Ka']}
            
            for band in ['Ka', 'Ku']:
                k_rain_dB = np.polyval(poly_rain_k_db[band], Ze_v[band][ind_rain]/10)
                spec_att[band][ind_rain] = misc.inv_dB(k_rain_dB)
                if spec_att[band][ind_rain].size>0:
                    spec_att[band][ind_ml] = spec_att[band][ind_rain][0]*2
                
                att_corr[band] = np.cumsum(spec_att[band])*2*range_spacing
            
            diff_pia_sim = att_corr['Ka'][-1] - att_corr['Ku'][-1]
            
            scaling = delta_PIA/diff_pia_sim
            for band in ['Ka', 'Ku']:
                att_corr[band] *= scaling
                Ze_v[band] = Zm_tmp[band]+att_corr[band]
            
            
            
           
            # plt.plot(Ze_v['Ku'], '-', label = ii, color = 'C%d' % ii)
            # plt.plot(Ze_v['Ka'], '--', color = 'C%d' % ii)
            # plt.ylim(10,50)
            # plt.grid()
            # plt.legend()
        
        # plt.figure()
        # plt.plot(Zm_v['Ku'], '-k')
        # plt.plot(Zm_v['Ka'], '--k')
        # plt.plot(Ze_v['Ku'], '-b')
        # plt.plot(Ze_v['Ka'], '--b')
        # plt.ylim(0,50)

        return Ze_v
    
    def _fg_alpha(self, Dm_dB_ice_bot, PR_dB_ice_bot,
                  zku_ice_m, zka_ice_m, ):
       
        alphas = radar_sim.scat_LUT['ice']['Alpha_dB'].values
        
        zku_alpha = radar_sim._simulate(Dm_dB_ice_bot, PR_dB_ice_bot, 
                  Alpha_dB = alphas, hydro = 'ice', var = 'Z', band = 'Ku')
        
        zka_alpha = radar_sim._simulate(Dm_dB_ice_bot, PR_dB_ice_bot, 
                  Alpha_dB = alphas, hydro = 'ice', var = 'Z', band = 'Ka')
        
        diff_val = ((zku_alpha-zku_ice_m)**2 + (zka_alpha-zka_ice_m)**2)
        ind_min = np.argmin(diff_val)
        min_diff = diff_val[ind_min]
        iter_n = 0
        # rain_PC0 = self.PCA['rain'].components_[0]
        rain_PC0 = np.ones(2)
        min_diff_next=0
        while min_diff>2. and iter_n<10 and (min_diff_next<min_diff):
            
            step_size = np.log10(min_diff*2+1)/2
            direct = np.sign(zku_ice_m-zku_alpha[ind_min])
            
            PR_dB_ice_bot += direct*step_size*rain_PC0[0]
            Dm_dB_ice_bot += direct*step_size*rain_PC0[1]
            
            
            zku_alpha = radar_sim._simulate(Dm_dB_ice_bot, PR_dB_ice_bot, 
                      Alpha_dB = alphas, hydro = 'ice', var = 'Z', band = 'Ku')
            
            zka_alpha = radar_sim._simulate(Dm_dB_ice_bot, PR_dB_ice_bot, 
                      Alpha_dB = alphas, hydro = 'ice', var = 'Z', band = 'Ka')
            
            diff_val = ((zku_alpha-zku_ice_m)**2 + (zka_alpha-zka_ice_m)**2)
            ind_min = np.argmin(diff_val)
            min_diff_next = diff_val[ind_min]
            
            if min_diff_next<min_diff:
                min_diff = min_diff_next+1e-9
           
            # print("alpha = %.1f: %.2f" % (alphas[ind_min], min_diff))
            iter_n+=1
        
        Alpha_dB_ice_fg = alphas[ind_min]
        return Alpha_dB_ice_fg
        
    
    def _fg(self, Ze_v, ind_rain, ind_ice, ind_ml, ):
        
        coord_q = np.stack([Ze_v['Ku'][ind_rain],
                            Ze_v['Ka'][ind_rain]]).T
        
        query_ind = self.rainTree.query(coord_q)[1]
        Dm_dB_rain_fg = self.Dm_dB_rain_tree[query_ind]
        PR_dB_rain_fg = self.PR_dB_rain_tree[query_ind]
        
        
        # ZKu = self.scat_LUT['rain']['mean_Z_Ku'].values
        # ZKa = self.scat_LUT['rain']['mean_Z_Ka'].values
        
        # dDWR = savgol_filter(ZKu-ZKa, window_length=5, 
        #                      polyorder = 1, deriv=1, axis = 0)
        # ind_good = np.where((ZKu>-99.) & np.isfinite(ZKu) &
        #                     (ZKa>-99.) & np.isfinite(ZKa) &
        #                     (dDWR>0))
       
        # self.Dm_dB_rain_tree =  self.scat_LUT['rain']['mean_Z_Ku'
        #                         ].Dm_dB.values[ind_good[0]]
        
        # plt.figure()
        # plt.scatter(ZKu[ind_good],ZKa[ind_good], 
        #     c = self.scat_LUT['rain']['mean_Z_Ku'].Dm_dB.values[ind_good[0]], 
        #     s = 1, vmin = -3, vmax = 4)
        # plt.scatter(Ze_v['Ku'][ind_rain], Ze_v['Ka'][ind_rain], 
        #              c = Dm_dB_rain_fg, s = 16 , vmin = -3, vmax = 4,
        #              marker = 'x')
        
        # plt.colorbar()
        
        
        Dm_dB_ice_bot = np.median(Dm_dB_rain_fg[:])
        PR_dB_ice_bot = np.median(PR_dB_rain_fg[:])
        
        zka_ice_m = np.mean(Ze_v['Ka'][ind_ice][-3:])
        zku_ice_m = np.mean(Ze_v['Ku'][ind_ice][-3:])
        
        alphas = radar_sim.scat_LUT['ice']['Alpha_dB'].values
        
        zku_alpha = radar_sim._simulate(Dm_dB_ice_bot, PR_dB_ice_bot, 
                  Alpha_dB = alphas, hydro = 'ice', var = 'Z', band = 'Ku')
        
        zka_alpha = radar_sim._simulate(Dm_dB_ice_bot, PR_dB_ice_bot, 
                  Alpha_dB = alphas, hydro = 'ice', var = 'Z', band = 'Ka')
        
        diff_val = ((zku_alpha-zku_ice_m)**2 + (zka_alpha-zka_ice_m)**2)
        ind_min = np.argmin(diff_val)
        min_diff = diff_val[ind_min]
        iter_n = 0
        rain_PC0 = self.PCA['rain'].components_[0]
        min_diff_next=0
        while min_diff>2. and iter_n<30 and (min_diff_next<min_diff):
            
            step_size = np.log10(min_diff*2+1)/2
            direct = np.sign(zku_ice_m-zku_alpha[ind_min])
            
            PR_dB_ice_bot += direct*step_size*rain_PC0[0]
            Dm_dB_ice_bot += direct*step_size*rain_PC0[1]
            
            
            zku_alpha = radar_sim._simulate(Dm_dB_ice_bot, PR_dB_ice_bot, 
                      Alpha_dB = alphas, hydro = 'ice', var = 'Z', band = 'Ku')
            
            zka_alpha = radar_sim._simulate(Dm_dB_ice_bot, PR_dB_ice_bot, 
                      Alpha_dB = alphas, hydro = 'ice', var = 'Z', band = 'Ka')
            
            diff_val = ((zku_alpha-zku_ice_m)**2 + (zka_alpha-zka_ice_m)**2)
            ind_min = np.argmin(diff_val)
            min_diff_next = diff_val[ind_min]
            
            if min_diff_next<min_diff:
                min_diff = min_diff_next+1e-9
           
            print("alpha = %.1f: %.2f" % (alphas[ind_min], min_diff))
            iter_n+=1
        
        Alpha_dB_ice_fg = alphas[ind_min]
        
       
        
        Ze_ka = np.copy(Ze_v['Ka'][ind_ice])
        Ze_ku = np.copy(Ze_v['Ku'][ind_ice])
        tmp_DWR = Ze_ku-Ze_ka
        
        ind_snr_l = Ze_ka<15
        ind_snr_h = Ze_ka>18
        ind_snr_hl = ind_snr_h & (Ze_ku<24)
        
        tmp_poly = np.polyfit(Ze_ku[ind_snr_h],
                              tmp_DWR[ind_snr_h], deg = 2)
        # tmp_poly_p = np.polyder(tmp_poly)
        # Zku_crit= -tmp_poly_p[1]/tmp_poly_p[0]
        
        tmp_poly1 = np.polyfit(Ze_ku[ind_snr_hl],
                              tmp_DWR[ind_snr_hl], deg = 1)
        
        
        # plt.figure()
        # plt.plot(Ze_ku[ind_snr_h],tmp_DWR[ind_snr_h], 'x')
        # plt.plot(Ze_ku,tmp_DWR, 'o')
        # plt.plot(Ze_ku, np.polyval(tmp_poly1, Ze_ku), '.')
        # plt.plot(Ze_ku, np.polyval(tmp_poly, Ze_ku), '.')
        # plt.ylim(-2,15)
        
        
        tmp_dwr = np.maximum(0,np.minimum(
            np.polyval(tmp_poly1, Ze_v['Ku'][ind_ice][ind_snr_l]), 
            np.polyval(tmp_poly, Ze_v['Ku'][ind_ice][ind_snr_l])))
        
        Ze_ka[ind_snr_l] =  Ze_v['Ku'][ind_ice][ind_snr_l]-tmp_dwr
        
        
        coord_q = np.stack([Ze_ku,
                            Ze_ka]).T
        
        
        query_ind = self.iceTree[Alpha_dB_ice_fg].query(coord_q)[1]
        
        Dm_dB_ice_fg = self.Dm_dB_ice_tree[Alpha_dB_ice_fg][query_ind]
        PR_dB_ice_fg = self.PR_dB_ice_tree[Alpha_dB_ice_fg][query_ind]
        
       
        
        return (Dm_dB_rain_fg, PR_dB_rain_fg, Dm_dB_ice_fg, PR_dB_ice_fg, 
                Alpha_dB_ice_fg,)
    
    def _form_x(self, Dm_dB, PR_dB, Alpha_dB, BB_ext_dB):
        """
        first Dm vector then PR vect then Alpha ice
        
        """
        x = np.r_[Dm_dB, PR_dB,Alpha_dB, BB_ext_dB]
        return x
        

    def _split_x(self, x,):
        """
        first Dm vector then PR vect then Alpha ice
        
        """
        # Alpha_dB_ice = x[-1]
        Dm_dB, PR_dB = np.split(x[:-3], 2)
        Alpha_dB = x[-3]
        BB_ext_dB = x[-2:]
        
        return Dm_dB, PR_dB, Alpha_dB, BB_ext_dB

    def _std_Z_DPR(self,Z, band = 'Ku', M = 110):
        if band == 'Ku':
            SNR = Z-15
        elif band == 'Ka':
            SNR == Z-21
        #Hogan et al. (2005)
        std_Z=4.343/np.sqrt(M)*(1+misc.inv_dB(-SNR))
        return np.sqrt(std_Z**2 + 0.7**2)
    def _transform_rain_x(self, Dm_dB_rain, PR_dB_rain):
        
        
        X_tr = np.stack([Dm_dB_rain, PR_dB_rain, ]).T
        poly_dm_pr = np.array([ 3.77025051, -0.36799125])
        
        # d_line = np.abs(poly_dm_pr[0]*Dm_dB_rain - PR_dB_rain + poly_dm_pr[1])/(
        #     np.sqrt(poly_dm_pr[0]**2 +1))
        
        theta = np.arctan(poly_dm_pr[0])        
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        X_tr[:,1] += -poly_dm_pr[1]                
        X_tr = np.matmul( X_tr, R)
        # plt.figure()
        # plt.plot(X_tr[:,0], X_tr[:,1], '.')
        # plt.plot(X_tr[:,0], d_line, 'x')
        mean_tg_ang = 0.6887236887123311
        X_tr[:,0] += -X_tr[:,1]/mean_tg_ang 
        # plt.figure()
        # plt.scatter(Dm_dB_rain, PR_dB_rain)
        # plt.plot(Dm_dB_rain, np.polyval(poly_dm_pr, Dm_dB_rain))
        # plt.figure()
        # plt.plot(X_tr[:,0], X_tr[:,1], '.')
        
        return X_tr
    def _dist_from_expected_relation(self, Dm_dB_rain, PR_dB_rain):

        poly_dm_pr = np.array([ 3.77025051, -0.36799125])
        
        d_line = np.abs(poly_dm_pr[0]*Dm_dB_rain - PR_dB_rain + poly_dm_pr[1])/(
            np.sqrt(poly_dm_pr[0]**2 +1))
        return d_line
        
        
    def retrieve_Dm_PR_1D(self, dpr_obj, nscan, nrayMS, 
                          vert_resolution_km = 0.5,
                          make_plot = False, fig_dir = None,
                          method = 'Powell'):
        
        
        if 'Dm_dB' not in list(dpr_obj.MS.variables):
            dpr_obj.MS['Dm_dB'] =(dpr_obj.MS.zFactorMeasured.dims,
                                np.full(dpr_obj.MS.zFactorMeasured.shape, -99.))
        if 'PR_dB' not in list(dpr_obj.MS.variables):
              dpr_obj.MS['PR_dB'] =(dpr_obj.MS.zFactorMeasured.dims,
                                np.full(dpr_obj.MS.zFactorMeasured.shape, -99.))  
        if 'Alpha_dB' not in list(dpr_obj.MS.variables):
              dpr_obj.MS['Alpha_dB'] =(dpr_obj.MS.Longitude.dims,
                                  np.full(dpr_obj.MS.Longitude.shape, -99.))
        if 'zKuSim' not in list(dpr_obj.MS.variables):
                dpr_obj.MS['zKuSim'] =(dpr_obj.MS.zFactorMeasured.dims,
                                  np.full(dpr_obj.MS.zFactorMeasured.shape, -99.))  
        if 'zKaSim' not in list(dpr_obj.MS.variables):
                dpr_obj.MS['zKaSim'] =(dpr_obj.MS.zFactorMeasured.dims,
                                  np.full(dpr_obj.MS.zFactorMeasured.shape, -99.))  
        if 'zKuEffSim' not in list(dpr_obj.MS.variables):
                dpr_obj.MS['zKuEffSim'] =(dpr_obj.MS.zFactorMeasured.dims,
                                  np.full(dpr_obj.MS.zFactorMeasured.shape, -99.))  
        if 'zKaEffSim' not in list(dpr_obj.MS.variables):
                dpr_obj.MS['zKaEffSim'] =(dpr_obj.MS.zFactorMeasured.dims,
                                  np.full(dpr_obj.MS.zFactorMeasured.shape, -99.))  
          
        if 'piaKuSim' not in list(dpr_obj.MS.variables):
              dpr_obj.MS['piaKuSim'] =(dpr_obj.MS.Longitude.dims,
                                  np.full(dpr_obj.MS.Longitude.shape, -99.))
        if 'piaKaSim' not in list(dpr_obj.MS.variables):
              dpr_obj.MS['piaKaSim'] =(dpr_obj.MS.Longitude.dims,
                                  np.full(dpr_obj.MS.Longitude.shape, -99.))
        if 'piaKuBB' not in list(dpr_obj.MS.variables):
              dpr_obj.MS['piaKuBB'] =(dpr_obj.MS.Longitude.dims,
                                  np.full(dpr_obj.MS.Longitude.shape, -99.))
        if 'piaKaBB' not in list(dpr_obj.MS.variables):
              dpr_obj.MS['piaKaBB'] =(dpr_obj.MS.Longitude.dims,
                                  np.full(dpr_obj.MS.Longitude.shape, -99.))
        
        if 'CF_n' not in list(dpr_obj.MS.variables):
              dpr_obj.MS['CF_n'] =(dpr_obj.MS.Longitude.dims,
                                  np.full(dpr_obj.MS.Longitude.shape, -99.))
        
        
        retr_grid_km = np.arange(0,20,vert_resolution_km)
        
        
        sn = dpr_obj.swath_name
        
        bands = ['Ku', 'Ka']
        range_spacing = dpr_obj['MS'].attrs['range_spacing']
        
        col_ds = {band: dpr_obj[sn[band]].isel(nscan = nscan, nrayMS = nrayMS)
                    for band in bands}
        
        
        # Zm_v = {band: self._smooth_Z(
        #     Z = col_ds[band].zFactorMeasured.values,Z_thr = Z_thr)
        #     for (band, Z_thr) in zip(bands,[9,15])}
        
        
        #### read measured reflectivities
        Zm_v = {band: col_ds[band].zFactorMeasured.values*1. for band in bands}
        
        std_Z = {band: self._std_Z_DPR(Zm_v[band]) for band in bands} 
        weight_Z = {band: 1/std_Z[band]**2 for band in bands} 
        
        
        
        #### gasses attenuation correction
        for band in bands:
            gass_att_corr = np.nancumsum(
                col_ds[band].attenuationNP.values)*range_spacing
            Zm_v[band] += 2*gass_att_corr
        
        
        PIAm = {band: col_ds[band].PIAhybrid.values.item()  for band in bands} 
        # std_PIA = {band: col_ds[band].stddevHY.values.item()  for band in bands} 
        # weight_PIA = {band: 1/std_PIA[band]**2 for band in bands}
  
        #### read differential pia estimate
        delta_PIA = 5*col_ds['Ku'].PIAhybrid.values.item()
        std_delta_PIA = 5*col_ds['Ku'].stddevHY.values.item()
        weight_delta_PIA = 1./std_delta_PIA**2.
        if not np.isfinite(delta_PIA+weight_delta_PIA):
            delta_PIA = 0.
            weight_delta_PIA = 0.
        
        
       
        #### partitioning data to rain, melting and ice
        flag_ml = ((col_ds['Ku'].phase>=100) & 
                    (col_ds['Ku'].phase<200)).values 
        flag_ice = ((col_ds['Ku'].phase<100) & (Zm_v['Ku']>10) & (
            col_ds['Ku'].nbin >= col_ds['Ku'].binStormTop-8)).values
        flag_rain = (col_ds['Ku'].phase.values>=200) & (Zm_v['Ku']>10) & (
           col_ds[band].nbin <= col_ds['Ku'].binRealSurface.values )
        
        ind_ml = np.where(flag_ml)[0]
        ind_rain = np.where(flag_rain)[0]
        ind_ice = np.where(flag_ice)[0]
        
        flag_hyd = (flag_rain | flag_ice | flag_ml)
        
        if ind_ml.size==0:            
            ind_ml = np.array([np.maximum(0,np.where(flag_rain)[0][0]-1),])
            flag_ml[ind_ml] = True
        if ind_ice.size==0:            
            ind_ice = np.array([np.maximum(0,np.where(flag_ml)[0][0]-1),])
            flag_ice[ind_ice] = True       
        if ind_rain.size==0:
            ind_rain = np.array([np.minimum(175,np.where(flag_hyd)[0][-1]+1),])
        
                
        flag_hyd = (flag_rain | flag_ice | flag_ml)
        
        

        for band in bands:
            flag_above = (col_ds[band].nbin < 
                          col_ds[band].binStormTop-4).values
            weight_Z[band][flag_ml | flag_above] = 0.
            flag_clutter = (col_ds[band].nbin > 
                    col_ds[band].binClutterFreeBottom).values
            
            ind_extrap_input = np.where(flag_rain &
                                        ~flag_clutter)[0]
            ind_extrap = np.where(flag_rain & flag_clutter)[0]
            
            if ind_extrap_input.size>16:
                ind_extrap_input = ind_extrap_input[-16:]
                
            if ind_extrap_input.size>7:
                tmp_poly = np.polyfit(ind_extrap_input, 
                            Zm_v[band][ind_extrap_input] , deg =1)
                
                Zm_v[band][ind_extrap] = np.polyval(
                    tmp_poly, ind_extrap)
                
                weight_Z[band][ind_extrap] = 0.5
            else:
                weight_Z[band][ind_extrap] = 0.
            # weight_PIA[band] *= 1+np.sum(weight_Z[band]>3)
        weight_Z['Ku'][Zm_v['Ku']<12] = 0.
        weight_Z['Ka'][Zm_v['Ka']<18] = 0.
        
        len_Y = (weight_Z['Ku']>0.).sum() + (weight_Z['Ka']>0.).sum() +1
        
    
       
        delta_PIA_tmp = delta_PIA*1.
        for ii in range(10):
            expected_pr_dB = (misc.dB(delta_PIA_tmp/
                (2*ind_rain.size*range_spacing))*0.935+ 6.546)
            BB_ext = np.array([radar_sim.BB_ext( 
                misc.inv_dB(expected_pr_dB), band= band) for band in bands])
            delta_PIA_tmp = np.maximum(0.1, delta_PIA - (BB_ext[1]-BB_ext[0]))
            # print(expected_pr_dB)
        
        
        # weight_delta_PIA *= (weight_Z['Ka'][ind_rain].sum())*.25
       
        # rain_h_km = dpr_obj['MS'].Altitude[flag_rain].values
        # ice_h_km = dpr_obj['MS'].Altitude[flag_ice].values
        
        Ze_v = self._attenuation_correction(Zm_v, ind_rain, ind_ice, ind_ml, 
                                            delta_PIA)
        # (Dm_dB_rain_fg, PR_dB_rain_fg, Dm_dB_ice_fg, PR_dB_ice_fg, 
        #         Alpha_dB_ice_fg,) = self._fg(Ze_v, ind_rain, ind_ice, ind_ml,)
        
        
        Dm_dB_rain_fg = self._fg_stats_simul( 
            hydro = 'rain', var_name = 'Dm_dB',  
            Zku=Ze_v['Ku'][ind_rain], Zka=Ze_v['Ka'][ind_rain],)    
        
        PR_dB_rain_fg = self._fg_stats_simul( 
            hydro = 'rain', var_name = 'PR_dB',  
            Zku=Ze_v['Ku'][ind_rain], Zka=Ze_v['Ka'][ind_rain],)
        ind_low_snr = (Ze_v['Ka'][ind_rain]<18)
        Dm_dB_rain_fg[ind_low_snr] = self._fg_stats_simul( 
            hydro = 'rain', var_name = 'Dm_dB',  
            Zku=Ze_v['Ku'][ind_rain], Zka=None,)[ind_low_snr]
        PR_dB_rain_fg[ind_low_snr] = self._fg_stats_simul( 
            hydro = 'rain', var_name = 'PR_dB',  
            Zku=Ze_v['Ku'][ind_rain], Zka=None,)[ind_low_snr]
        
        
        Dm_dB_ice_bot = np.polyval(
            np.polyfit(ind_rain, Dm_dB_rain_fg, deg =1), ind_ice[-1])
        PR_dB_ice_bot = np.polyval(
            np.polyfit(ind_rain, PR_dB_rain_fg, deg =1), ind_ice[-1])
        
        zka_ice_m = np.mean(Ze_v['Ka'][ind_ice][-2:])
        zku_ice_m = np.mean(Ze_v['Ku'][ind_ice][-2:])
        
        
        Alpha_dB_ice_fg = self._fg_alpha( Dm_dB_ice_bot, PR_dB_ice_bot,
                      zku_ice_m, zka_ice_m, )
        
        # Alpha_dB_ice_fg = -18.24
        # Alpha_dB_ice_fg = -12.63
        # Alpha_dB_ice_fg = -2.4
        
        Dm_dB_ice_fg = self._fg_stats_simul( 
            hydro = 'ice', var_name = 'Dm_dB',  Alpha_dB = Alpha_dB_ice_fg,
            Zku=Ze_v['Ku'][ind_ice], Zka=Ze_v['Ka'][ind_ice],)        
        PR_dB_ice_fg = self._fg_stats_simul( 
            hydro = 'ice', var_name = 'PR_dB',  Alpha_dB = Alpha_dB_ice_fg,
            Zku=Ze_v['Ku'][ind_ice], Zka=Ze_v['Ka'][ind_ice],)
        ind_low_snr = (Ze_v['Ka'][ind_ice]<18)
        Dm_dB_ice_fg[ind_low_snr] = self._fg_stats_simul( 
            hydro = 'ice', var_name = 'Dm_dB',  Alpha_dB = Alpha_dB_ice_fg,
            Zku=Ze_v['Ku'][ind_ice], Zka=None,)[ind_low_snr]
        PR_dB_ice_fg[ind_low_snr] = self._fg_stats_simul( 
            hydro = 'ice', var_name = 'PR_dB',  Alpha_dB = Alpha_dB_ice_fg,
            Zku=Ze_v['Ku'][ind_ice], Zka=None,)[ind_low_snr]
        
        
        # dpr_obj.MS['CF_n'][nscan, nrayMS,]
        
        
        
        storm_bot= dpr_obj['MS'].Altitude[flag_rain | flag_ice].min(
            ).values.item()
        storm_top = dpr_obj['MS'].Altitude[flag_rain | flag_ice].max(
            ).values.item()
        storm_ml_h = dpr_obj['MS'].Altitude[ind_ml].max().values.item()
        
        retr_top = retr_grid_km[retr_grid_km>=storm_top].min()
        retr_bot = retr_grid_km[retr_grid_km<=storm_bot].max()
        
        retr_nodes_km = retr_grid_km[(retr_grid_km>=retr_bot) & 
                                     (retr_grid_km<=retr_top)]
        retr_nodes_solid = (retr_nodes_km>storm_ml_h)
        
        Dm_dB_all_fg = np.zeros(retr_nodes_km.shape)
        PR_dB_all_fg = np.zeros(retr_nodes_km.shape)
       

        Dm_dB_all_fg[~retr_nodes_solid] = np.interp(
            retr_nodes_km[~retr_nodes_solid], 
            dpr_obj['MS'].Altitude[flag_rain].values[::-1], Dm_dB_rain_fg[::-1])
        PR_dB_all_fg[~retr_nodes_solid] = np.interp(
            retr_nodes_km[~retr_nodes_solid], 
            dpr_obj['MS'].Altitude[flag_rain].values[::-1], PR_dB_rain_fg[::-1])
        
        Dm_dB_all_fg[retr_nodes_solid] = np.interp(
            retr_nodes_km[retr_nodes_solid], 
            dpr_obj['MS'].Altitude[flag_ice].values[::-1], Dm_dB_ice_fg[::-1])
        PR_dB_all_fg[retr_nodes_solid] = np.interp(
            retr_nodes_km[retr_nodes_solid], 
            dpr_obj['MS'].Altitude[flag_ice].values[::-1], PR_dB_ice_fg[::-1])
        
        
        mean_rr = np.mean(misc.inv_dB(PR_dB_rain_fg))
        BB_ext_dB_fg = misc.dB(np.array([radar_sim.BB_ext( mean_rr, band= band) 
                        for band in bands]))
        
        x0 = self._form_x(Dm_dB = Dm_dB_all_fg, 
                          PR_dB = PR_dB_all_fg, 
                          Alpha_dB = Alpha_dB_ice_fg,
                          BB_ext_dB = BB_ext_dB_fg)
        
        # Tikh = self._Tikhonov_matrix(n = retr_nodes_km.size, diff = 1)
        # Tikh2 = self._Tikhonov_matrix(n = retr_nodes_km[::2].size, diff = 1)
        # Tikh3 = self._Tikhonov_matrix(n = retr_nodes_km[::3].size, diff = 1)
        d_kernels =[np.array([-1,1]), 
                    np.array([-1,0,1]), 
                    np.array([-1,0,0,1])] 
        
        
        def CF_retr(x, return_x = False, return_y = False, 
                    Alpha_dB = Alpha_dB_ice_fg): 
            
            CF = 0.
            Dm_dB, PR_dB, Alpha_dB, BB_ext_dB, = self._split_x(x)
            
            Dm_dB_HR = np.interp(dpr_obj['MS'].Altitude.values,
                                   retr_nodes_km, Dm_dB)
            PR_dB_HR = np.interp(dpr_obj['MS'].Altitude.values,
                                   retr_nodes_km, PR_dB)
            BB_ext = misc.inv_dB(BB_ext_dB)
          
            if return_x:
                return Dm_dB_HR, PR_dB_HR, Alpha_dB, BB_ext
            
            Dm_dB_rain = Dm_dB_HR[flag_rain]
            PR_dB_rain = PR_dB_HR[flag_rain]
            
            Dm_dB_ice = Dm_dB_HR[flag_ice]
            PR_dB_ice = PR_dB_HR[flag_ice]
    
            mean_pr = misc.inv_dB( np.mean(PR_dB_rain))
            
            BB_ext_dB_exp = misc.dB(np.array(
                [radar_sim.BB_ext( mean_pr, band= band) 
                            for band in bands]))
            Zs_v, PIAs = {}, {}
            if return_y:
                Ze_s, spec_att = {}, {}
            CF_o = 0.
            # plt.figure()
            for ib, band in enumerate(bands):
                # k_ML = (ml_fract*self._BB_ext( mean_pr, band= band)*
                #         BB_ext_sc[band])
                k_ML = np.zeros(Zm_v[band].size)
                k_ML[ind_ml] = BB_ext[ib]/ind_ml.size/range_spacing*.5
              
                tmp_out = radar_sim.simulate_Zm( 
                    Dm_dB_rain = Dm_dB_rain, PR_dB_rain = PR_dB_rain,
                    flag_rain = flag_rain, 
                    Dm_dB_ice = Dm_dB_ice,  PR_dB_ice = PR_dB_ice, 
                    Alpha_dB = Alpha_dB, flag_ice = flag_ice,
                    k_ML = k_ML, band = band,
                    range_spacing = range_spacing, 
                    range_axis = 0, return_effective=return_y)
                Zs_v[band], PIAs[band] = tmp_out[:2]
                if return_y:
                    Ze_s[band], spec_att[band] = tmp_out[2:]
                
                CF_o += np.sum(weight_Z[band]*((Zm_v[band]- Zs_v[band])**2))
                # print(CF_o)
            if return_y:
                return Zs_v, PIAs, Ze_s, spec_att
            #     plt.plot(weight_Z[band]*((Zm_v[band]- Zs_v[band])**2),
            #              dpr_obj['MS'].Altitude[:].values, label = band, )
            # plt.legend()
            delta_pia_sim = PIAs['Ka']-PIAs['Ku']
            CF_o += weight_delta_PIA*(delta_PIA-delta_pia_sim)**2
            # print(CF_o)
            CF += CF_o
            
            
            
            # X_tr = self._transform_rain_x(Dm_dB_rain, PR_dB_rain)  
            dist_from_exp_rel = self._dist_from_expected_relation(
                Dm_dB_rain, PR_dB_rain) 
            
            CF_b = 0.
            CF_b = 1e2*5.*np.sum(np.diff(dist_from_exp_rel)**2)/0.99**2
            CF_b += np.sum((BB_ext_dB_exp-BB_ext_dB)**2)/(0.5**2)
            CF_b += 1e1*(np.diff(BB_ext_dB)[0]-11.3)**2
            # CF_b += np.sum((expected_pr_dB-PR_dB_rain)**2)
            
            tmp_vect = np.stack([PR_dB_ice, Dm_dB_ice,
                                  np.ones(flag_ice.sum())*Alpha_dB/10,]).T
            PCA_val = self.PCA['ice'].transform(tmp_vect)
            PCA_dev = PCA_val**2/self.PCA['ice'].explained_variance_
            CF_b += PCA_dev.sum()
            
            # tmp_vect = np.stack([PR_dB_rain, Dm_dB_rain,]).T
            # PCA_val = self.PCA['rain'].transform(tmp_vect)
            # PCA_dev = PCA_val**2/self.PCA['rain'].explained_variance_
            # CF_b += PCA_dev.sum()
            CF += CF_b
            
            CF_c = 0
            for tmp_vect in [PR_dB[retr_nodes_solid], PR_dB[~retr_nodes_solid],
                    Dm_dB[retr_nodes_solid], Dm_dB[~retr_nodes_solid]]:
                for ii in range(3):                    
                    d_vect = np.convolve(tmp_vect, d_kernels[ii], mode = 'valid')
                    CF_c += np.sum(d_vect**2)
                
                    # CF_c += np.inner(np.matmul( Tikh,tmp_vect,),tmp_vect)
                    # CF_c += np.inner(np.matmul( Tikh2,tmp_vect[::2],),
                    #                   tmp_vect[::2])
                    # CF_c += np.inner(np.matmul( Tikh3,tmp_vect[::3],),
                    #                   tmp_vect[::3])
            CF += CF_c*.01
            
            return CF
        
        
        CF_x0 = CF_retr(x0)
        ret_shape = dpr_obj.MS.Longitude.shape
        starttime = timeit.default_timer()
        for ii in range(-1,2):
            nr =  nrayMS+ii
           
            for jj in range(-3,4):
                ns = nscan+jj
                
                if nr>0 and nr<ret_shape[1] and ns>0 and ns<ret_shape[0]:
                    if dpr_obj.MS['CF_n'][ns,nr]>0:
                        # print(dpr_obj.MS['CF_n'][ns,nr])
                        
                        Dm_dB_fg_neigh= np.interp(
                            retr_nodes_km, 
                            dpr_obj['MS'].Altitude.values[::-1], 
                            dpr_obj.MS['Dm_dB'][ns, nr,].values[::-1])
                        
                        PR_dB_fg_neigh= np.interp(
                            retr_nodes_km, 
                            dpr_obj['MS'].Altitude.values[::-1], 
                            dpr_obj.MS['PR_dB'][ns, nr,].values[::-1])
                        
                        ind_tmp = (Dm_dB_fg_neigh<-10)
                        Dm_dB_fg_neigh[ind_tmp] = Dm_dB_all_fg[ind_tmp]
                        PR_dB_fg_neigh[ind_tmp] = PR_dB_all_fg[ind_tmp]
                        
                        xi = self._form_x(
                            Dm_dB = Dm_dB_fg_neigh,  
                            PR_dB = PR_dB_fg_neigh, 
                            Alpha_dB = dpr_obj.MS['Alpha_dB'][ns, nr,],
                            BB_ext_dB = misc.dB(np.array(
                                [dpr_obj.MS['piaKuBB'][ns, nr,],
                                 dpr_obj.MS['piaKaBB'][ns, nr,]])))
                        
                        tmp_a = dpr_obj.MS['Alpha_dB'][ns, nr,].values.item()
                        CF_xi = CF_retr(xi, Alpha_dB = tmp_a)
                        
                        # print('%.1f - %.1f (%d,%d)' % (CF_x0, CF_xi,ns, nr))
                        if CF_xi<CF_x0:
                            print('%.1f -> %.1f (%d,%d)' % (CF_x0, CF_xi, ns, nr))
                            CF_x0 = CF_xi*1.
                            x0 = xi*1.
                            Alpha_dB_ice_fg = tmp_a
                    
        time_meth = timeit.default_timer() - starttime
        print("testing neighbours: %1.3fs" % ( time_meth,))
        
       
        # x0 = self._form_x(Dm_dB = Dm_dB_all_fg, 
        #                   PR_dB = PR_dB_all_fg, 
        #                   BB_ext_dB = BB_ext_dB_fg)
        
        iter_val, iter_time = [], []
        
        max_iter = 100.
        def callbackF(x):
            iter_val.append(CF_retr(x))
            iter_time.append(timeit.default_timer() - starttime)
            
        # fig, ax = plt.subplots()
        # ax.grid()
       
        if method in ['BFGS', 'CG']:
            options={'maxiter': max_iter, 'disp': False, 
                      'gtol': 0.01,  'norm': np.inf}
        elif method in ['Powell']:
            options={'maxiter': max_iter, 'disp': False, 
                     'ftol': 0.001, }

        iter_val, iter_time = [CF_x0,], [0.]
        starttime = timeit.default_timer()
        res = minimize(CF_retr, x0 = x0, method = method, 
            options=options, callback = callbackF,
            args = (False, False, Alpha_dB_ice_fg),)
        time_meth = timeit.default_timer() - starttime
        print("%s method: %1.1fs, CF: %.1f, %d iter." % (
            method, time_meth, iter_val[-1], len(iter_val)))
        # x0 = res.x
        
        # ax.plot(iter_time, iter_val, 'x-',
        #          label = "%s: %1.1fs; CF: %.1f, %d iter" % (
        #     method, time_meth,iter_val[-1], len(iter_val)))
        
     
        
        Dm_dB_HR, PR_dB_HR, Alpha_dB, BB_ext= CF_retr(res.x,  return_x = True)
        Dm_dB_HR[~flag_hyd] = -99.
        PR_dB_HR[~flag_hyd] = -99.
        
        Zs_v, PIAs, Ze_s, spec_att_s = CF_retr(res.x, return_y = True)
        
        dpr_obj.MS['CF_n'][nscan, nrayMS,] = CF_retr(res.x)/len_Y
        
        dpr_obj.MS['Dm_dB'][nscan, nrayMS,] = Dm_dB_HR
        dpr_obj.MS['PR_dB'][nscan, nrayMS,]= PR_dB_HR
        dpr_obj.MS['Alpha_dB'][nscan, nrayMS,] = Alpha_dB
        
        dpr_obj.MS['piaKuSim'][nscan, nrayMS,] = PIAs['Ku']
        dpr_obj.MS['piaKaSim'][nscan, nrayMS,] = PIAs['Ka']
        
        dpr_obj.MS['piaKuBB'][nscan, nrayMS,] = BB_ext[0]
        dpr_obj.MS['piaKaBB'][nscan, nrayMS,] = BB_ext[1]
        
        dpr_obj.MS['zKuSim'][nscan, nrayMS,] = Zs_v['Ku']
        dpr_obj.MS['zKaSim'][nscan, nrayMS,] = Zs_v['Ka']
        
        dpr_obj.MS['zKuEffSim'][nscan, nrayMS,] = Ze_s['Ku']
        dpr_obj.MS['zKaEffSim'][nscan, nrayMS,] = Ze_s['Ka']
        
        
        if make_plot:
            
            if fig_dir is None:
                fig_dir = os.getcwd()
         
            tmp_txt = '$\delta PIA_m = %.1f$; $\delta PIA_s = %.1f$;\n' % (
                PIAm['Ka']-PIAm['Ku'], PIAs['Ka']-PIAs['Ku'])
            
            tmp_txt += '$BB_{ext}^{Ku} = %.1f$; $BB_{ext}^{Ka} = %.1f$;' %  (
                *BB_ext,)
            
            DWRm = Zm_v['Ku']-Zm_v['Ka']
            DWRm[Zm_v['Ku']<9] = np.nan
            DWRm[Zm_v['Ka']<15] = np.nan
            Zs_v['Ku'][Zs_v['Ku']<-30]=np.nan
            Zs_v['Ka'][Zs_v['Ka']<-30]=np.nan
            
            # Zc_v = {band: col_ds[band].zFactorCorrected.values for band in bands}
            
            fn_tmp = dpr_obj.MS.attrs['FileName'].replace('.HDF5', 
                      '.nrMS%d.ns%d' % (nrayMS, nscan))
               
            fn_out = os.path.join(fig_dir, fn_tmp+'.Zsim.png')
            fig, ax = plt.subplots()
            ax.plot(Zm_v['Ku'], col_ds['Ka'].Altitude, '-b', label = '$Z_m^{Ku}$')
            ax.plot(Zm_v['Ka'], col_ds['Ka'].Altitude, '-g', label = '$Z_m^{Ka}$')
            ax.plot(Ze_s['Ku'], col_ds['Ka'].Altitude, ':b', label = '$Z_e^{Ku}$')
            ax.plot(Ze_s['Ka'], col_ds['Ka'].Altitude, ':g', label = '$Z_e^{Ka}$')
            # ax.plot(Zc_v['Ku'], col_ds['Ka'].Altitude, '-.b', label = '$Z_c^{Ku}$')
            # ax.plot(Zc_v['Ka'], col_ds['Ka'].Altitude, '-.g', label = '$Z_c^{Ka}$')
            ax.plot(Zs_v['Ku'], col_ds['Ka'].Altitude, '--b', label = '$Z_s^{Ku}$')
            ax.plot(Zs_v['Ka'], col_ds['Ka'].Altitude, '--g', label = '$Z_s^{Ka}$')
            
            
            ax.set_xlim(0,55)
            ax.set_ylim(0,retr_nodes_km[-1])
            ax.grid()
            ax.text(25,retr_nodes_km[-1]*0.8, tmp_txt, fontsize =14,
                    fontweight = 'bold')
            ax.set_ylabel('Altitude [km]')
            ax.legend(loc = 'lower right')
            ax.set_xlabel('$Z$ [dBZ]')  
            
            color = 'k'
            ax2 = ax.twiny() # instantiate a second axes that shares the same y-axis
            ax2.set_xlabel('$DWR$ [dB]', color=color)  # we already handled the x-label with ax1
            ax2.tick_params(axis='x', labelcolor=color)
            ax2.plot(DWRm, col_ds['Ka'].Altitude, '-k', 
                    label = '$DWR_m$')
            ax2.plot(Zs_v['Ku']-Zs_v['Ka'], col_ds['Ka'].Altitude, '--k', 
                    label = '$DWR_s$')  
            
            ax2.set_xlim(-5,22.5)
            ax2.set_xticks([0,5,10,15,20])
            plt.tight_layout()
            
            plt.savefig(fn_out, dpi =200) 
            
            
            fn_out = os.path.join(fig_dir, fn_tmp+'.Xvect.png')
            fig, ax = plt.subplots()
            
            ax.plot(misc.inv_dB(PR_dB_HR), col_ds['Ka'].Altitude[:], '-g',
                    label = '$PR$ OE')
            
            ax.plot(col_ds['Ku'].precipRate, col_ds['Ku'].Altitude, '--g',
                    label = '$PR$ DPR')
            
          
            ax.plot(misc.inv_dB(PR_dB_rain_fg), 
                    col_ds['Ka'].Altitude[ind_rain], ':g',
                    label = '$PR$ FG')
            
            ax.plot(misc.inv_dB(PR_dB_ice_fg), 
                    col_ds['Ka'].Altitude[ind_ice], ':g',)
            
            ax.set_xlabel('$PR$ [mm/h]', color='g')
            ax.tick_params(axis='x', labelcolor='g')
                 # ax.legend()
            ax.set_xlim(0,15)
            ax.set_xticks(np.arange(0,15, 2.5))
            ax.set_ylim(0,retr_nodes_km[-1])
            ax.set_ylabel('Altitude [km]')
            ax.grid()      
            
            ax2 = ax.twiny()  # instantiate a second axes that shares the same y-axis

            color = 'b'
            ax2.set_xlabel('$D_m$', color=color)  # we already handled the x-label with ax1
            ax2.tick_params(axis='x', labelcolor=color)
            # ax2.legend(loc = 'right')
            
            ax2.set_xlim(0,3)
            ax2.set_xticks(np.arange(0,3, 0.5))
            ax2.plot(misc.inv_dB(Dm_dB_HR), col_ds['Ka'].Altitude[:], '-b',
                    label = '$D_m$ OE')
            ax2.plot(col_ds['Ku'].paramDSD.isel( nDSD = 1), 
                    col_ds['Ku'].Altitude, '--b', label = '$D_m$ DPR')
            ax2.plot(misc.inv_dB(Dm_dB_rain_fg), 
                    col_ds['Ka'].Altitude[ind_rain], ':b',
                    label = '$D_m$ FG')
            ax2.plot(misc.inv_dB(Dm_dB_ice_fg), 
                    col_ds['Ka'].Altitude[ind_ice], ':b',)
            ax2.set_title(r'$\alpha = $%.2f; OE "-"; DPR "--"; FG ":"' % 
                          Alpha_dB_ice_fg)
            
            plt.tight_layout()
            plt.savefig(fn_out, dpi =200) 
        
        
            
            
    #         mean_dm_dB =  Dm_dB_rain.mean()
    #         mean_pr_dB =  PR_dB_rain.mean()
            
    #         tmp_zku = self._simulate(mean_dm_dB, mean_pr_dB, 
    #                  Alpha_dB = self.scat_LUT['ice']['Alpha_dB'].values,
    #                 hydro = 'ice', var = 'Z', band = 'Ku')
    #         tmp_zka = self._simulate(mean_dm_dB, mean_pr_dB, 
    #                  Alpha_dB = self.scat_LUT['ice']['Alpha_dB'].values,
    #                 hydro = 'ice', var = 'Z', band = 'Ka')
            
    #         tmp_zka_ice_m = np.mean(Zm_v['Ka'][flag_ice][-3:])
    #         tmp_zku_ice_m = np.mean(Zm_v['Ku'][flag_ice][-3:])
            
            
    #         alpha_ind = np.argmin((tmp_zku-tmp_zku_ice_m)**2 +
    #                               (tmp_zka-tmp_zka_ice_m)**2)
            
    #         CF_alpha = 1e2*(Alpha_dB[-1]-
    #                 self.scat_LUT['ice']['Alpha_dB'].values[alpha_ind])**2
            
    #         CF += CF_alpha
            
    #         CF_c = 0
    #         for tmp_vect in [PR_dB_ice, Dm_dB_ice]:
    #             CF_c += np.inner(np.matmul( Tikh_i,tmp_vect,),tmp_vect)
    #         for tmp_vect in [PR_dB_rain, Dm_dB_rain]:
    #             CF_c += np.inner(np.matmul( Tikh_r,tmp_vect,),tmp_vect)
    #         # CF_c += 1e1*(PR_dB_ice[-1]-mean_pr_dB)**2
    #         # CF_c += 1e1*(Dm_dB_ice[-1]-mean_dm_dB)**2
            
    #         # tmp_vect = np.concatenate([PR_dB_ice, PR_dB_rain])
    #         # CF_c += np.inner(np.matmul( Tikh,tmp_vect,),tmp_vect)
    #         # tmp_vect = np.concatenate([Dm_dB_ice, Dm_dB_rain])
    #         # CF_c += np.inner(np.matmul( Tikh,tmp_vect,),tmp_vect)
            
    #         CF += CF_c*10.


    #         CF_expect = (np.sum((PR_dB_rain- mean_pr_dB)**2)+
    #                 np.sum((Dm_dB_rain - mean_dm_dB)**2))*5.
            
    #         exp_dm_dB_ice = np.interp(np.arange(ilen), [0,ilen-1],
    #                                   [-1., mean_dm_dB])
    #         exp_pr_dB_ice = np.interp(np.arange(ilen), [0,ilen-1],
    #                                   [-1., mean_pr_dB])
            
    #         CF_expect += (np.sum((PR_dB_ice - exp_pr_dB_ice)**2)+
    #                 np.sum((Dm_dB_ice - exp_dm_dB_ice)**2))*0.5*10.
            
    #         CF += CF_expect
            
    #         return CF
 
        
    #     Alpha_dB_0 = -18.
    #     Dm_ice_lims = [
    #         self.scat_LUT_lims['ice']['Dm_dB'][ii](Alpha_dB_0*np.ones(ilen))
    #                    for ii in range(2)]
       
    #     Dm_dB_ice_fg = self._first_guess(hydro = 'ice', var_name = 'Dm_dB',  
    #                  Zku=Zm_v['Ku'][flag_ice], Zka=Zm_v['Ka'][flag_ice])
        
    #     tmp_x_dm_ice = self._map_int_2_R(Dm_dB_ice_fg, *Dm_ice_lims)
        
    #     PR_dB_ice_fg = self._first_guess(hydro = 'ice', var_name = 'PR_dB',  
    #                  Zku=Zm_v['Ku'][flag_ice], Zka=Zm_v['Ka'][flag_ice])
        
        
    #     Z_tmp_ice = self._simulate(Dm_dB = Dm_dB_ice_fg, 
    #                                PR_dB = PR_dB_ice_fg, 
    #                                Alpha_dB = Alpha_dB_0,
    #                                hydro = 'ice', var = 'Z', band = 'Ku')
    #     PR_corr = Zm_v['Ku'][flag_ice] - Z_tmp_ice
        
    #     PR_dB_ice_fg += PR_corr
        
    #     tmp_x_pr_ice = self._map_int_2_R(PR_dB_ice_fg, 
    #                         *self.scat_LUT_lims['ice']['PR_dB'])
        
    #     bb_frac = flag_ml.sum()*2/(flag_ml.sum()*2+
    #                                             flag_rain.sum())
    #     rain_frac = flag_rain.sum()/(flag_ml.sum()*2+
    #                                             flag_rain.sum())
        
    #     Ze_v = {}
    #     for band in ['Ka', 'Ku']:
           
    #         Ze_rain = Zm_v[band][flag_rain]
    #         plt.figure()
    #         plt.plot(Ze_rain)
    #         two_way_ext = PIAm[band]*rain_frac/flag_rain.sum()*np.ones(flag_rain.sum())
    #         pia_corr = np.cumsum(two_way_ext)+ bb_frac*PIAm[band]
    #         Ze_rain += pia_corr
    #         plt.plot(Ze_rain)
    #         for aa in range(3):
    #             if band == 'Ka':
    #                 ext_tmp_dB = -23.46 +0.71 *Ze_rain
    #             elif band=='Ku':
    #                 ext_tmp_dB = -32.29 +0.67 *Ze_rain
                    
    #             two_way_ext = 10**(ext_tmp_dB/10)*2*range_spacing
    #             two_way_ext *= PIAm[band]*rain_frac/two_way_ext.sum()
    #             pia_corr = np.cumsum(two_way_ext)+bb_frac*PIAm[band]
    #             Ze_rain = Zm_v[band][flag_rain]+pia_corr
    #             plt.plot(Ze_rain, label = aa)
                
    #         Ze_v[band] = Ze_rain
    #         plt.legend()
    #         plt.grid()
                
                
    #     coord_q = np.stack([Ze_v['Ku'], Ze_v['Ka']]).T
    #     query_ind = self.rainTree.query(coord_q)[1]
                 
    #     Dm_dB_rain_fg = self.Dm_dB_tree[query_ind]

    #     # Dm_dB_rain_fg = self._first_guess(hydro = 'rain', var_name = 'Dm_dB',  
    #     #              Zku=Zm_v['Ku'][flag_rain], Zka=Zm_v['Ka'][flag_rain])
        
    #     tmp_x_dm_rain = self._map_int_2_R(Dm_dB_rain_fg, 
    #                     *self.scat_LUT_lims['rain']['Dm_dB'])
        
        
    #     # PR_dB_rain_fg = self._first_guess(hydro = 'rain', var_name = 'PR_dB',  
    #     #              Zku=Zm_v['Ku'][flag_rain], Zka=Zm_v['Ka'][flag_rain])
        
    #     PR_dB_rain_fg = self.PR_dB_tree[query_ind]
        
    #     tmp_x_pr_rain = self._map_int_2_R(PR_dB_rain_fg, 
    #                     *self.scat_LUT_lims['rain']['PR_dB'])
        
    #     tmp_x_alpha = self._map_int_2_R(
    #         Alpha_dB_0, *self.scat_LUT_lims['ice']['Alpha_dB'])
   
    #     x0 = np.concatenate([tmp_x_dm_rain, tmp_x_pr_rain,
    #                           tmp_x_dm_ice, tmp_x_pr_ice, 
    #                           np.array([tmp_x_alpha, 0,0])])
    #     x0[~np.isfinite(x0)] = 0
   
        
   
    #     fig, ax_cb = plt.subplots()
    #     col_ds['Ku'].zFactorMeasured.plot(ax = ax_cb, y = 'Altitude')
    #     col_ds['Ka'].zFactorMeasured.plot(ax = ax_cb, y= 'Altitude')
    #     ax_cb.set_xlim(10,55)
    #     ax_cb.grid()
        
    #     def callbackF(xk):
    #         CF = 0.
    #         BB_ext_sc = {}
    #         (Dm_dB_rain, PR_dB_rain, 
    #          Dm_dB_ice, PR_dB_ice, Alpha_dB,
    #          BB_ext_sc['Ku'], BB_ext_sc['Ka']) = split_x(xk)
    #         mean_pr = misc.inv_dB( PR_dB_rain.mean())
           
    #         Zs_v, PIAs = {}, {}
    #         for band in bands:
    #             # k_ML = (ml_fract*self._BB_ext( mean_pr, band= band)*
    #             #         BB_ext_sc[band])
    #             k_ML = np.zeros(Zm_v[band].size)
    #             k_ML[ind_ml[0]] = (self._BB_ext( mean_pr, band= band)*
    #                     BB_ext_sc[band])
    #             Zs_v[band], PIAs[band] = self._simulate_Zm( 
    #                 Dm_dB_rain = Dm_dB_rain, PR_dB_rain = PR_dB_rain,
    #                 flag_rain = flag_rain, 
    #                 Dm_dB_ice = Dm_dB_ice,  PR_dB_ice = PR_dB_ice, 
    #                 Alpha_dB = Alpha_dB, flag_ice = flag_ice,
    #                 k_ML = k_ML, band = band,
    #                 range_spacing = range_spacing, range_axis = 0,)
    #         ax_cb.plot(Zs_v['Ku'],  col_ds['Ku'].Altitude, '-',
    #                        label = self._Nfeval, lw = 1, )
    #         ax_cb.plot(Zs_v['Ka'],  col_ds['Ka'].Altitude, '--',
    #                        label = self._Nfeval, lw = 1, )
    #         for band in bands:               
    #             CF += np.sum(weight_Z[band]*((Zs_v[band] - Zm_v[band])**2))
                              
    #         print ('{0:3d}: \t {1:3.2f}'.format(self._Nfeval, CF))
    #         self._Nfeval += 1
            
       
            
    #     callbackF(x0)
    #     # res = minimize(CF_retr, x0 = x0, method = 'CG', 
    #     #     options={'maxiter': 10, 'disp': False,}, 
    #     #     callback= callbackF)
    #     res = minimize(CF_retr, x0 = x0, method = 'Powell', 
    #         options={'maxiter': 15, 'disp': False,'ftol': 0.001,}, 
    #         callback= callbackF)
    #     ax_cb.legend()
    #     # res = minimize(CF_retr, x0 = res.x, method = 'Nelder-Mead', 
    #     #     options={'maxiter': 1000, 'disp': False,}, )
        
       
        
    #     # res = minimize(CF_retr, x0 = x0, method = 'BFGS', 
    #     #     options={'maxiter': 100, 'disp': False, }, 
    #     #     callback= callbackF)
        
   
        
        
    #     BB_ext_sc = {}
    #     (Dm_dB_rain_res, PR_dB_rain_res, Dm_dB_ice_res, 
    #          PR_dB_ice_res, Alpha_dB_res, 
    #          BB_ext_sc['Ku'], BB_ext_sc['Ka']) = split_x(res.x)
        
        
    #     self.MS['Dm_dB'][nscan, nrayMS,][flag_ice] = Dm_dB_ice_res
    #     self.MS['Dm_dB'][nscan, nrayMS,][flag_rain] = Dm_dB_rain_res
    #     self.MS['PR_dB'][nscan, nrayMS,][flag_ice] = PR_dB_ice_res
    #     self.MS['PR_dB'][nscan, nrayMS,][flag_rain] = PR_dB_rain_res
    #     self.MS['Alpha_dB'][nscan, nrayMS,] = Alpha_dB_res[-1]
        
    #     mean_pr = misc.inv_dB( PR_dB_rain_res.mean())
    #     Zs_v, PIAs = {}, {}
    #     for band in bands:
    #         # k_ML = (ml_fract*self._BB_ext( mean_pr, band= band)*
    #         #         BB_ext_sc[band])
    #         k_ML = np.zeros(Zm_v[band].size)
    #         k_ML[ind_ml[0]] = (self._BB_ext( mean_pr, band= band)*
    #                 BB_ext_sc[band])
            
    #         Zs_v[band], PIAs[band] = self._simulate_Zm( 
    #             Dm_dB_rain = Dm_dB_rain_res, 
    #             PR_dB_rain = PR_dB_rain_res,
    #             flag_rain = flag_rain, 
    #             Dm_dB_ice = Dm_dB_ice_res, 
    #             PR_dB_ice = PR_dB_ice_res, 
    #             Alpha_dB = Alpha_dB_res, flag_ice = flag_ice,
    #             k_ML = k_ML, band = band,
    #             range_spacing = range_spacing, range_axis = 0,)
        
    #     self.MS['piaKuSim'][nscan, nrayMS,] = PIAs['Ku']
    #     self.MS['piaKaSim'][nscan, nrayMS,] = PIAs['Ka']
        
    #     self.MS['zKuSim'][nscan, nrayMS,] = Zs_v['Ku']
    #     self.MS['zKaSim'][nscan, nrayMS,] = Zs_v['Ka']
        
        
        
    #     if make_plot:
           
    #         # fig, ax = plt.subplots()
    #         # for band in bands:
    #         #     ax.plot(np.sqrt(weight_Z[band]*
    #         #                     ((Zs_v[band] - Zm_v[band])**2)),
    #         #             col_ds['Ka'].Altitude)
            
    #         tmp_txt = ''
    #         for band in bands:
    #             tmp_txt += '$PIA_{%s}^m$: %.1f, $PIA_{%s}^s$: %.1f\n' % (
    #                 band, PIAm[band], band, PIAs[band])
                
                
    #         fig, ax = plt.subplots()
    #         col_ds['Ku'].zFactorMeasured.plot(ax = ax, y = 'Altitude')
    #         col_ds['Ka'].zFactorMeasured.plot(ax = ax, y= 'Altitude')
            
    #         col_ds['Ka'].zKuSim.plot(ax = ax, y= 'Altitude')
    #         col_ds['Ka'].zKaSim.plot(ax = ax, y= 'Altitude')
    #         # ax.plot(Zm_v['Ku'], col_ds['Ka'].Altitude, '--')
    #         # ax.plot(Zm_v['Ka'], col_ds['Ka'].Altitude, '--')
    #         ax.set_xlim(10,55)
    #         ax.grid()
    #         ax.text(30,10, tmp_txt)
            
    #         fig, ax = plt.subplots()
    #         (10**(self['MS']['Dm_dB'][nscan, nrayMS,]/10)).plot(
    #             ax = ax, y= 'Altitude',label = 'Dm')
    #         (10**(self['MS']['PR_dB'][nscan, nrayMS,]/10)).plot(
    #             ax = ax, y= 'Altitude',label = 'PR')
    #         ax.plot((Alpha_dB_res+18)/(5+18), 
    #                 col_ds['Ka'].Altitude[flag_ice], 
    #                 label = r'$\alpha$')
    #         ax.legend()
    #         ax.set_xlim(-0.5,8)
    #         ax.grid()
    
class OE_pca():
    def __init__(self, filename = None):
        
        ### loading scattering LUT
        
        hyd_type = ['ice', 'rain']
        bands = ['Ku', 'Ka']
        
        lut_dir = os.path.join(module_path, 'scattering_tables', '%s', '%s')
        
        fn = {hyd: {band: lut_dir % ()  for band in bands} for hyd in hyd_type}
        fn = {'ice': 'insitu_PSD_dataset_Dle_stats_Gamma.nc',
              'rain': 'NASA_DSD_dataset_2DVD_5min_stats_Gamma.nc'}
        
        hyd_type = list(fn.keys())
        
        self.scat_LUT = {}
        for hyd in hyd_type:
            LUT_fn = os.path.join(module_path, 'scattering_tables', hyd, 
                              'PSD_Integrated', 'gamma', 'mimics_insitu',
                              fn[hyd])
            self.scat_LUT[hyd] = xr.load_dataset(LUT_fn) # ice
            self.scat_LUT[hyd] = self.scat_LUT[hyd].rename({'Dm_dB_bin': 'Dm_dB',
                                             'PR_dB_bin': 'PR_dB'})
            
        ### generating a new coordinate
        self.scat_LUT['ice']['Alpha_dB'] = misc.dB(self.scat_LUT['ice']['Alpha'])
        tmp_cords = list(self.scat_LUT['ice'].coords)
        tmp_cords.append('Alpha_dB')
        self.scat_LUT['ice'] = self.scat_LUT['ice'].set_coords(tmp_cords)
        self.scat_LUT['ice'] = self.scat_LUT['ice'].swap_dims({"Alpha": "Alpha_dB"})
        self.scat_LUT['ice']['Alpha_dB'] = np.round(
            self.scat_LUT['ice']['Alpha_dB'].values, decimals =1)

        ZKu = self.scat_LUT['rain']['mean_Z_Ku'].values
        ZKa = self.scat_LUT['rain']['mean_Z_Ka'].values
        dDWR = savgol_filter(ZKu-ZKa, window_length=5, 
                             polyorder = 1, deriv=1, axis = 0)
        ind_good = np.where((ZKu>-99.) & np.isfinite(ZKu) &
                            (ZKa>-99.) & np.isfinite(ZKa) &
                            (dDWR>0))
        
        coords = np.stack([ZKu[ind_good], ZKa[ind_good]]).T
        self.rainTree = cKDTree(coords)
        
        self.Dm_dB_rain_tree =  self.scat_LUT['rain']['mean_Z_Ku'
                                ].Dm_dB.values[ind_good[0]]
        self.PR_dB_rain_tree =  self.scat_LUT['rain']['mean_Z_Ku'
                                ].PR_dB.values[ind_good[1]]
        
        
        self.Dm_dB_ice_tree, self.PR_dB_ice_tree, self.iceTree = {}, {}, {}
        for ind_a, aa in enumerate(self.scat_LUT['ice']['Alpha_dB'].values):
            
            # print(aa)
            ZKu = self.scat_LUT['ice']['mean_Z_Ku'].values[:,:,ind_a]
            ZKa = self.scat_LUT['ice']['mean_Z_Ka'].values[:,:,ind_a]
            dDWR = savgol_filter(ZKu-ZKa, window_length=5, 
                                 polyorder = 1, deriv=1, axis = 0)
            
            ind_good = np.where((ZKu>-99.) & np.isfinite(ZKu) &
                                (ZKa>-99.) & np.isfinite(ZKa) &
                                (dDWR>0))
            
            coords = np.stack([ZKu[ind_good], ZKa[ind_good]]).T
            
            self.iceTree[aa] = cKDTree(coords)
            
            self.Dm_dB_ice_tree[aa] =  self.scat_LUT['ice']['mean_Z_Ku'
                                    ].Dm_dB.values[ind_good[0]]
            self.PR_dB_ice_tree[aa] =  self.scat_LUT['ice']['mean_Z_Ku'
                                    ].PR_dB.values[ind_good[1]]
            
            # plt.figure()
            # plt.scatter(self.PR_dB_ice_tree[aa], self.Dm_dB_ice_tree[aa], s =1,
            #             c = ZKu[ind_good]-ZKa[ind_good])
            # plt.colorbar()
            
            # plt.figure()
            # plt.scatter(ZKu[ind_good],ZKu[ind_good] - ZKa[ind_good], 
            #     c = self.scat_LUT['rain']['mean_Z_Ku'].Dm_dB.values[ind_good[0]], 
            #     s = 1, vmin = -3, vmax = 4)
            # plt.colorbar()
            
            # aaaa
            

        ### a-priori information and a first guess LUT
        fn_ap = {'ice': 'insitu_PSD_dataset_stats_PCA_Dle_Dm_PR_Alpha.pkl',
                  'rain': 'NASA_DSD_dataset_2DVD_5min_PCA_Dm_PR.pkl'}
        fn_fg = {'ice':  'insitu_PSD_dataset_Dle_expected_unknowns.nc',
                  'rain': 'NASA_DSD_dataset_2DVD_5min_expected_unknowns.nc'}
        self.PCA, self.fg_LUT = {}, {}
        
        for hyd in hyd_type:
            stats_dir = os.path.join(module_path,'scattering_tables', 
                        hyd, 'PSD_Integrated', 'insitu', 'stats')
            
            ap_name = os.path.join(stats_dir, fn_ap[hyd])
            with open(ap_name, 'rb') as file_obj:
                self.PCA[hyd] = dill.load(file_obj)
                
            fg_name = os.path.join(stats_dir,fn_fg[hyd])
            self.fg_LUT[hyd] = xr.load_dataset(fg_name)
            
        self.fg_RGI, self.fg_std_RGI = {}, {}
        for hyd in hyd_type:
            self.fg_RGI[hyd], self.fg_std_RGI[hyd] = {}, {}
            for x_name, new_name in zip(['Dm_dB', 'PR_dB', 'logAlphaMG'],
                     ['Dm_dB', 'PR_dB', 'Alpha_dB']):
                self.fg_RGI[hyd][new_name], self.fg_std_RGI[hyd][new_name] = {}, {}
                for v_name in ['Z_Ku','Z_Ka','Z_Ku_Z_Ka']:
                    mean_name = 'mean_%s_for_%s' % (x_name, v_name)
                    std_name = 'std_%s_for_%s' % (x_name, v_name)
                    if mean_name in self.fg_LUT[hyd].variables:
                        band = v_name.replace('Z_','')
                        arr = self.fg_LUT[hyd][mean_name]
                        if x_name == 'logAlphaMG': arr = 10*arr
                        self.fg_RGI[hyd][new_name][band] = RGI(
                            [arr[c].data for c in arr.dims],arr.data, 
                            method = 'linear',  bounds_error=False, 
                            fill_value=-99.)
                        arr = self.fg_LUT[hyd][std_name]
                        if x_name == 'logAlphaMG': arr = 10*arr
                        self.fg_std_RGI[hyd][new_name][band] = RGI(
                            [arr[c].data for c in arr.dims],arr.data, 
                            method = 'linear',  bounds_error=False, 
                            fill_value=-99.)
                        
        alphas = radar_sim.scat_LUT['ice']['Alpha_dB'].values
        self.fg_exp_v_sims = {}
        for hyd in hyd_type:
            self.fg_exp_v_sims[hyd] = {}
            Dm_dB_tmp = self.fg_LUT[hyd]['mean_Dm_dB_for_Z_Ku']
            PR_dB_tmp = self.fg_LUT[hyd]['mean_PR_dB_for_Z_Ku']
            if hyd == 'ice':
                Dm_dB_tmp, ALPHAS = np.meshgrid(Dm_dB_tmp, alphas)
                PR_dB_tmp, ALPHAS = np.meshgrid(PR_dB_tmp, alphas)
                
            for v_name in ['Z_Ku','Z_Ka']:
                band = v_name.replace('Z_','')                
                self.fg_exp_v_sims[hyd][v_name] = radar_sim._simulate(
                    Dm_dB_tmp, PR_dB_tmp, Alpha_dB = ALPHAS, 
                    hydro = hyd, var = 'Z', band = band)                 
                


    # def _smooth_Z(self,Z,Z_thr = 10):
    #     Z_new = np.maximum(Z,Z_thr)
    #     return savgol_filter(Z_new, window_length= 5, polyorder=1, )
    
    
    def _Tikhonov_matrix(self, n, diff = 1):
        A = np.eye(n)- np.eye(n,k=1)
        A[-1,:] = 0
        Ap = np.linalg.matrix_power(A,diff)
        return np.matmul(Ap.T, Ap)
        