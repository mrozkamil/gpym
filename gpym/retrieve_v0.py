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
from scipy.integrate import cumtrapz
from scipy.linalg import sqrtm
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from scipy.interpolate import RegularGridInterpolator as RGI
import matplotlib.pyplot as plt
# import dill
from . import misc
from . import simulate
from . import representation 
import timeit

module_path = '/Users/km357/Documents/Python3/gpym/gpym'
# module_path = os.path.dirname(__file__)

class OE():
    def __init__(self, filename = None):
        
        # each radar bin is represented by 3 unknowns in a form of a column
        
        self.radar_sim = simulate.radar_simulator()
        
        # physical variables are 
        # 10log10 WC [g/m3]
        # 10log10 sigma_m (PSD width) [mm]
        # 10log10 D_m [mm]
        self.phys_vars = ['WC_dB', 'Sm_dB', 'Dm_dB']
        
        # Williams variables are 
        # 10log10 WC [g/m3]
        # 10log10 sigma_m prime (PSD width) [mm] (it removes a correlation between Sigma_m and D_m)
        # 10log10 D_m [mm]
        self.Williams_vars = ['WC_dB', 'Sm_p_dB', 'Dm_dB']
        
        
        self.base_change_mat = {b1:{} for b1 in ['PCA', 'phys', 'Will']}
        self.shift_vect = {b1:{} for b1 in ['PCA', 'phys', 'Will']}
        
        # columns of the array are the principal components
        self.base_change_mat['PCA']['phys'] = np.array(
            [[ 0.73479976,  0.55868832,  0.38462536],
             [-0.67753545,  0.63120603,  0.37752439],
            [ 0.03185939,  0.53800215, -0.84234118]]).T
        
        # mean values of the physical parameters 
        # 'WC_dB', 'Sm_dB', 'Dm_dB'
        self.shift_vect['PCA']['phys'] = np.array(
            [[-9.35146661, -3.79017189,  0.90294262],]).T
            
        # new variable sigma_m prime defined as
        # Sm_p_dB = Sm_dB - (1.4525519586 * Dm_dB - 5.102)
        self.base_change_mat['phys']['Will'] = np.array(
            [[1, 0, 0], 
             [0,1,-1.4525519586],
             [0,0,1]])
        self.shift_vect['phys']['Will'] = np.array([[0, 5.102, 0],]).T
        
        self.base_change_mat['PCA']['Will'] = np.matmul(
            self.base_change_mat['phys']['Will'],
            self.base_change_mat['PCA']['phys'],)
        
        self.shift_vect['PCA']['Will'] = np.matmul(
            self.base_change_mat['phys']['Will'],
            self.shift_vect['PCA']['phys']
            ) + self.shift_vect['phys']['Will'] 
       
        
            
        # PC variance (the first element has an increased variance 
        # to account for deficienties in the in-situ measurements)
        self._pca_variance = np.array(
            [12.2340293442*100, 5.14467090, 0.175293022])
        
   
        pc0_table = ((np.linspace(-7, 7 , 101) - 
                      self.shift_vect['PCA']['phys'][2])/
                    self.base_change_mat['PCA']['phys'][2,0])
        
        pc_arr = np.stack([pc0_table, 
                np.zeros_like(pc0_table), 
                np.zeros_like(pc0_table)], axis = 0)

        phys_arr = self._transform_bases(pc_arr, 
                            base1 = 'PCA', base2 = 'phys')
        Will_arr = self._transform_bases(phys_arr, 
                            base1 = 'phys', base2 = 'Will')
        Z_table = {}
        for hydro in ['rain', 'ice']:
            Z_table[hydro] = {}
            for band in ['Ku','Ka']:
                Z_table[hydro][band] = self.radar_sim(
                    hydro = hydro, var = 'Z', band = band, 
                    WC_dB = Will_arr[0,:], Sm_p_dB = Will_arr[1,:], 
                    Dm_dB = Will_arr[2,:])
        
        self.KDTree = {}
        for hydro in ['rain', 'ice']:
            self.KDTree[hydro] = {}
            for band in ['Ku','Ka']:
                self.KDTree[hydro][band] = cKDTree(
                    Z_table[hydro][band][:,np.newaxis])

            coords = np.stack([Z_table[hydro]['Ku'], 
                                Z_table[hydro]['Ka']]).T
            self.KDTree[hydro]['DPR'] = cKDTree(coords)
        self.KDTree_pc0 = pc0_table
        
        fact = np.log(10)/10
        # self.Ze_sim_unc = {'Ka': 0.5, 'Ku': 1.2}
        self.Ze_std = {'rain': {'Ka': .32, 'Ku': .48}} #calculated with NASA dataset
        self.Ze_std['ice'] = {band: np.sqrt(val*2 + 1.0**2)
                        for band, val in self.Ze_std['rain'].items()}
        self.Ze_std['melt'] = {'Ka': np.inf, 'Ku': np.inf}
        # self.k_sim_unc = {'Ka': 0.14, 'Ku': 0.21}
        self.k_dB_std = {'rain': {'Ka': 0.1, 'Ku': 0.15},
                         'melt': {'Ka': 0.97, 'Ku': 0.97}} #calculated with NASA dataset
        self.k_dB_std['ice'] = {band: np.sqrt(val*2 + 1.0**2)
                        for band, val in self.k_dB_std['rain'].items()}
        
        self.k_fract_unc = {hydro: {band: fact* val
                        for band, val in self.k_dB_std[hydro].items()}
                                for hydro in self.k_dB_std.keys()}
        
        
        self.k_dB_Z_rel_std = {'Ka': 1.9, 'Ku': 1.5}
        self.k_dB_Z_rel_frac_unc = {band: fact*val
                        for band, val in self.k_dB_Z_rel_std.items()}
    
    def _Tikhonov_matrix(self, n, diff = 2, w = None):
        A = np.eye(n)- np.eye(n,k=1)
        A[-1,:] = 0
        Ap = np.linalg.matrix_power(A,diff)
        if w is None:
            return np.matmul(Ap.T, Ap)
        else:
            W = np.diag(w)
            return np.matmul(Ap.T,np.matmul(W, Ap))
    
    def _transform_Sm_dB_Dm_dB_2_Sm_p_dB(self, Sm_dB, Dm_dB):
        Sm_p_dB = Sm_dB - (1.4525519586 * Dm_dB - 5.102)
        return Sm_p_dB
    def _transform_Sm_p_dB_Dm_dB_2_Sm_dB(self, Sm_p_dB, Dm_dB):
        Sm_dB = Sm_p_dB + (1.4525519586 * Dm_dB - 5.102)  
        return Sm_dB
    
    def _transform_bases(self, data, base1='PCA', base2='phys'):
        new_data = np.matmul(self.base_change_mat[base1][base2], 
                data) + self.shift_vect[base1][base2]
        return new_data
    
    
    def HB_correction(self, ZdB_m,  flag_hydro, PIA_dB, band = 'Ku', 
                      dh_km = 0.125):
        """
        HB attenuation correction routine
        we assume that k = alpha * Z_lin^beta 
        ( <==> 10 log10(k) = 10 log10(alpha) + beta*Z_dB)
        beta is constant in the column
        """
        log_alpha = {'Ku':  {'rain': -32.20,
                              'melt': 10*np.log10(1.5) - 32.20,
                              'ice': -41.09,},
                    'Ka':  {'rain': -28.89,
                            'melt': 10*np.log10(2.5) - 28.89,
                            'ice': -30.33}}
                              
        beta = {'Ku': 0.6971, 'Ka': 0.8819}           
        
        k_exp_dB = np.ones_like(ZdB_m)*-999.
        
        for hydro in flag_hydro.keys():
            if flag_hydro[hydro].sum()>0:
                k_exp_dB[flag_hydro[hydro]] = (
                    beta[band]*ZdB_m[flag_hydro[hydro]] + 
                    log_alpha[band][hydro])
        k_exp = 10**(0.1*k_exp_dB)
        zeta = 0.2*np.log(10)*beta[band]*np.cumsum(k_exp)*dh_km 
        zeta_surf = zeta[-1]
        
        eps_surf = (1-10**(-0.1*PIA_dB*beta[band]))/zeta_surf
        # zeta_surf *= eps_surf
        correction = -10./beta[band]*np.log10(1-eps_surf*zeta)
        ZdB_corr = ZdB_m + correction
        return ZdB_corr
                          
    
    def _split_x(self, x, lx, nx = 3):
        """
        pc0, pc1, pc2, Alpha ice, BB_ext
        
        """
        split_ind = (np.arange(nx,dtype = 'int')+1)*lx
        return np.split(x, split_ind)

    def _std_Z_DPR(self,Z, band = 'Ku', M = 110):
        if band == 'Ku':
            SNR = Z-15.46
        elif band == 'Ka':
            SNR = Z-19.18
        #Hogan et al. (2005)
        std_Z = 4.343/np.sqrt(M)*(1+misc.inv_dB(-SNR))
        
        return std_Z 
    
    def _fg_pc0(self, Ze_v, flag_hydro):
        
        pc_val = np.ones_like(Ze_v['Ku'])*-99.9
        flag_zka_low = (Ze_v['Ka']<18.)
        
        for hydro in flag_hydro.keys():
            if flag_hydro[hydro].sum()>0:
                coord_q = np.stack([Ze_v['Ku'][flag_hydro[hydro]], 
                                    Ze_v['Ka'][flag_hydro[hydro]]]).T
                if hydro == 'melt':
                    hydro_q = 'rain'
                else:
                    hydro_q = hydro
                query_ind = self.KDTree[hydro_q]['DPR'].query(coord_q)[1]
                pc_val[flag_hydro[hydro]] = self.KDTree_pc0[query_ind]
                
                query_ind = self.KDTree[hydro_q]['Ku'].query(
                    np.expand_dims(coord_q[:,0],1))[1]
                pc_val[flag_hydro[hydro] & flag_zka_low] = self.KDTree_pc0[
                    query_ind[flag_zka_low[flag_hydro[hydro]]]]

        return pc_val
        
    def _Forw_model(self, WC_dB, Sm_p_dB, Dm_dB, T_K, Alpha_dB, BB_att_1_way, 
                    flag_hydro, band = 'Ku', range_spacing = 0.125):
        
        size_z = flag_hydro[list(flag_hydro.keys())[0]].shape
        Ze, k = -99.9*np.ones(size_z), np.zeros(size_z)
        
        for hydro in flag_hydro.keys():
            if hydro =='melt': 
                
                k [flag_hydro[hydro]] = BB_att_1_way/range_spacing/flag_hydro[hydro].sum()
                
            elif hydro =='rain' :
            
                Z_tmp = self.radar_sim(
                    hydro = hydro, var = 'Z', band = band, 
                    WC_dB = WC_dB[flag_hydro[hydro]], 
                    Sm_p_dB = Sm_p_dB[flag_hydro[hydro]], 
                    Dm_dB = Dm_dB[flag_hydro[hydro]],
                    T = T_K[flag_hydro[hydro]],)
            
                k_tmp = self.radar_sim(
                    hydro = hydro, var = 'k', band = band, 
                    WC_dB = WC_dB[flag_hydro[hydro]], 
                    Sm_p_dB = Sm_p_dB[flag_hydro[hydro]], 
                    Dm_dB = Dm_dB[flag_hydro[hydro]],
                    T = T_K[flag_hydro[hydro]],)
                Ze[flag_hydro[hydro]] = Z_tmp
                k [flag_hydro[hydro]] = k_tmp
            elif hydro =='ice' :
                
                Z_tmp = self.radar_sim(
                    hydro = hydro, var = 'Z', band = band, 
                    WC_dB = WC_dB[flag_hydro[hydro]], 
                    Sm_p_dB = Sm_p_dB[flag_hydro[hydro]], 
                    Dm_dB = Dm_dB[flag_hydro[hydro]],
                    Alpha_dB = Alpha_dB)
            
                k_tmp = self.radar_sim(
                    hydro = hydro, var = 'k', band = band, 
                    WC_dB = WC_dB[flag_hydro[hydro]], 
                    Sm_p_dB = Sm_p_dB[flag_hydro[hydro]], 
                    Dm_dB = Dm_dB[flag_hydro[hydro]],
                    Alpha_dB = Alpha_dB)
                
                Ze[flag_hydro[hydro]] = Z_tmp
                k [flag_hydro[hydro]] = k_tmp
            
        one_way_att = np.cumsum(k)*range_spacing
         
        Zm = np.copy(Ze)
        Zm[1:] += -2.*one_way_att[:-1]
        return Zm, one_way_att[-1]*2, Ze , k
            
        
            

        
    def retrieve_PC_1D(self, dpr_obj, nscan, nrayMS, 
                          vert_resolution_km = 0.5,
                          make_plot = False, fig_dir = None,                          
                          method = 'SLSQP', deg = 1, nx = 3,
                         plot_diag = False):
        
        
        backward_corr = True

        var_3D_list = ['WC_dB', 'Sm_dB', 'Dm_dB', 'PR_dB',
                       'zKuSim', 'zKaSim', 'zKuEffSim', 'zKaEffSim'] 
        
        for var in var_3D_list:
            if var not in list(dpr_obj.MS.variables):
                dpr_obj.MS[var] =(dpr_obj.MS.zFactorMeasured.dims,
                        np.full(dpr_obj.MS.zFactorMeasured.shape, -999.9))
                
        var_2D_list = ['Alpha_dB', 'piaKuSim',  'piaKaSim', 
                       'piaKaBB' , 'piaKuBB', 'CF_n'] 

        for var in var_2D_list:
            if var not in list(dpr_obj.MS.variables):
                  dpr_obj.MS[var] =(dpr_obj.MS.Longitude.dims,
                    np.full(dpr_obj.MS.Longitude.shape, -999.9))
        
        
        # retr_grid_km = np.arange(0,20,vert_resolution_km)

        sn = dpr_obj.swath_name
        
        bands = ['Ku', 'Ka']
        range_spacing = dpr_obj['MS'].attrs['range_spacing']
        
        col_ds = {band: dpr_obj[sn[band]].isel(
            nscan = nscan, nrayMS = nrayMS).load() for band in bands}
        
        
        # Zm_v = {band: self._smooth_Z(
        #     Z = col_ds[band].zFactorMeasured.values,Z_thr = Z_thr)
        #     for (band, Z_thr) in zip(bands,[9,15])}
        
        
        #### read measured reflectivities
        Zm_v = {band: col_ds[band].zFactorMeasured.data*1. for band in bands}
        phase = col_ds['Ku'].phase.data*1.
        T_K = np.zeros_like(phase)
        T_K[phase<=100] = phase[phase<=100]-100+273.15
        T_K[phase>=200] = phase[phase>=200]-200+273.15
        T_K[(phase<200) & (phase>100)] = 273.15
        
        std_Z = {band: self._std_Z_DPR(Zm_v[band]) for band in bands} 
        weight_Z = {band: 1/std_Z[band]**2 for band in bands} 
        

        #### gasses attenuation correction
        for band in bands:
            gass_att_corr = np.nancumsum(
                col_ds[band].attenuationNP.values)*range_spacing
            Zm_v[band] += 2*gass_att_corr
        
        
        PIAm = {band: col_ds[band].PIAhybrid.values.item()  for band in bands} 
        std_PIA = {band: col_ds[band].stddevHY.values.item()  for band in bands} 
        # weight_PIA = {band: 1/std_PIA[band]**2 for band in bands}
  
        #### read differential pia estimate
        delta_PIA = 5*col_ds['Ku'].PIAhybrid.values.item()
        std_delta_PIA = 5*col_ds['Ku'].stddevHY.values.item()
        weight_delta_PIA = 1./std_delta_PIA**2.
        if not np.isfinite(delta_PIA+weight_delta_PIA):
            delta_PIA = 0.
            weight_delta_PIA = 0.
        
       
        #### partitioning data to rain, melting and ice
        flag_hydro = {
            'melt': ((col_ds['Ku'].phase>=100) & 
                    (col_ds['Ku'].phase<200)).data,
            'ice': ((col_ds['Ku'].phase<100) & (Zm_v['Ku']>10) & (
                    col_ds['Ku'].nbin >= col_ds['Ku'].binStormTop-8)).data,
            'rain': (col_ds['Ku'].phase.values>=200) & (Zm_v['Ku']>10) & (
                col_ds[band].nbin <= col_ds['Ku'].binRealSurface.values )}
        
        # ind_hydro = {key: np.where(val) for key, val in flag_hydro.items()}
        tmp_ind =  np.where(flag_hydro['rain'] | 
                            flag_hydro['melt'] | flag_hydro['ice'])[0]
        flag_any_hydro = np.full(Zm_v['Ku'].size, False)
        flag_any_hydro[tmp_ind[0]:tmp_ind[-1]] = True 
        ind_any_hydro = np.where(flag_any_hydro)[0]
        
        flag_bad_Ka = ((Zm_v['Ka']>Zm_v['Ku']) & flag_hydro['ice'])
        Zm_v['Ka'][flag_bad_Ka] = Zm_v['Ku'][flag_bad_Ka]
        
        fl_hydro = {hyd: fl[flag_any_hydro] for hyd, fl in flag_hydro.items()}
        
        # cloud_top_h = col_ds['Ku'].Altitude[flag_any_hydro].data.max()
        alt_v = col_ds['Ku'].Altitude[flag_any_hydro].data
        fine_nodes = alt_v[::-1]
        coarse_nodes = np.arange(vert_resolution_km,fine_nodes[-1], 
                                 vert_resolution_km)
        if coarse_nodes[-1] == fine_nodes[-1]:
            coarse_nodes = coarse_nodes[:-1]
        
        spl_repr = representation.spline(fine_nodes = fine_nodes,
                    coarse_nodes = coarse_nodes,
                    deg = deg, make_plot=False)
        
        lx = spl_repr.ind_node_signif.size # number of the retrieval levels
      
        for band in bands:
            flag_above = (col_ds[band].nbin < 
                          col_ds[band].binStormTop-4).values
            weight_Z[band][flag_hydro['melt'] | flag_above] = 0.
            flag_clutter = (col_ds[band].nbin > 
                    col_ds[band].binClutterFreeBottom).values
            
            ind_extrap_input = np.where(flag_hydro['rain'] &  ~flag_clutter)[0]
            ind_extrap = np.where(flag_hydro['rain'] & flag_clutter)[0]
            
            if ind_extrap_input.size>16:
                ind_extrap_input = ind_extrap_input[-16:]
                
            if ind_extrap_input.size>7:
                tmp_poly = np.polyfit(ind_extrap_input, 
                            Zm_v[band][ind_extrap_input] , deg =1)
                
                Zm_v[band][ind_extrap] = np.polyval(
                    tmp_poly, ind_extrap)
                
                weight_Z[band][ind_extrap] = 3
            else:
                weight_Z[band][ind_extrap] = 0.
            # weight_PIA[band] *= 1+np.sum(weight_Z[band]>3)
        weight_Z['Ku'][Zm_v['Ku']<12] = 0.
        weight_Z['Ka'][Zm_v['Ka']<18] = 0.
        
        std_Zm = {band : 1/np.sqrt(weight_Z[band]) for band in bands} 
        w_BB = (1/np.array([1.,1.,])**2)
        
        len_Y = (weight_Z['Ku']>0.).sum() + (weight_Z['Ka']>0.).sum() +1
        
        
        Ze_v ={band: self.HB_correction(ZdB_m = Zm_v[band], 
            flag_hydro = flag_hydro, PIA_dB = PIAm[band], band = band, 
                          dh_km = range_spacing) for band in bands}
        att_corr = {band: Ze_v[band] - Zm_v[band] for band in bands}
        att_corr_bottom_up = {band: att_corr[band][-1]- att_corr[band]
                              for band in bands}
        
        var_Ze = {band: std_Zm[band]**2 for band in bands}
        for band in bands:
            for hydro in flag_hydro.keys():
                var_Ze[band][flag_hydro[hydro]] += self.Ze_std[hydro][band]
   
        var_Ze_model = {band: var_Ze[band][flag_any_hydro]  for band in bands}
        
        for band in bands:
            var_Ze[band] += (std_PIA[band]**2+
                  (att_corr_bottom_up[band]*self.k_dB_Z_rel_frac_unc[band])**2 )
            
            
        weight_Ze = {band: np.maximum(0,1./var_Ze[band][flag_any_hydro]) for band in bands}
                 
        # k_per_bin = {band: np.diff(att_corr[band], prepend = 0.) for band in bands}
        
        pc0_fg = self._fg_pc0( Ze_v = Ze_v, flag_hydro = flag_hydro)
        pc0_fg_rep = spl_repr.get_rep(fine_nodes, 
                            pc0_fg[flag_any_hydro][::-1])
        
               
        pc0 = spl_repr(pc0_fg_rep)[::-1]           
        
        PC_arr_fg = np.stack([pc0,pc0*0,pc0*0], axis = 0)
        Will_arr_fg = self._transform_bases(PC_arr_fg, 
                            base1 = 'PCA', base2 = 'Will')
        
        # PR_fg = self.radar_sim(
        #     hydro = 'rain', var = 'PR', band = band, 
        #     WC_dB = Will_arr_fg[0,:], Sm_p_dB = Will_arr_fg[1,:], 
        #     Dm_dB = Will_arr_fg[2,:])
        #PR can be computed and BB extinction
        # mean_rr = np.mean(PR_fg[fl_hydro['rain']])
        # BB_ext_dB_ap = misc.dB(np.array(
        #     [self.radar_sim.BB_ext( mean_rr, band= band) 
        #                 for band in bands]))
        BB_ext_dB_ap = misc.dB(np.array(
            [att_corr[band][flag_hydro['melt']][-1] - 
             att_corr[band][flag_hydro['melt']][0]
                        for band in bands]))
        
        att_corr['Ka'][flag_hydro['melt']][[0,-1]]
        
        
        ind_below_bb = np.where(fl_hydro['rain'])[0][slice(7)]
        ind_above_bb = np.where(fl_hydro['ice'])[0][::-1][slice(7)][::-1]
        
        Zm_ice = {}
        for ib, band in enumerate(bands):
            Zm_ice[band] = np.polyval(
                np.polyfit(ind_above_bb, 
                    Ze_v[band][flag_any_hydro][ind_above_bb], 1), 
                ind_above_bb[-1])
        
        
        mean_wc_db =  np.mean(Will_arr_fg[0,ind_below_bb])
        mean_sm_db =  np.mean(Will_arr_fg[1,ind_below_bb])
        mean_dm_db =  np.mean(Will_arr_fg[2,ind_below_bb])
        
        ones_arr = np.ones(self.radar_sim.scat_LUT['ice']['Ku'].Alpha_dB.shape)
        Zs_ice = {}
        alpha_test = self.radar_sim.scat_LUT['ice']['Ku'].Alpha_dB.data
        for ib, band in enumerate(bands):
            Zs_ice[band] = self.radar_sim(
                hydro = 'ice', var = 'Z', band = band, 
                WC_dB = mean_wc_db*ones_arr, 
                Sm_p_dB = mean_sm_db*ones_arr, 
                Dm_dB = mean_dm_db*ones_arr,
                Alpha_dB = alpha_test)
           
        CF_alpha_tmp = (Zs_ice['Ku']-Zm_ice['Ku'])**2 + (Zs_ice['Ka']-Zm_ice['Ka'])
        ind_min_alpha = np.argmin(CF_alpha_tmp)
        Alpha_dB_ap = alpha_test[ind_min_alpha]
        
        
        pc_fg_list = [pc0_fg_rep,]
        for ii in range(1,nx):
            pc_fg_list.append(np.zeros_like(pc0_fg_rep))
        x_fg = np.concatenate(pc_fg_list + 
                            [np.array([Alpha_dB_ap,]), BB_ext_dB_ap])
        
        
        ######finish from here
        # x = x_fg*1
        
        w = np.ones(fine_nodes.size)
        w[fl_hydro['rain']] = 4.**2.
        l_melt = fl_hydro['melt'].sum()
        w[fl_hydro['melt']] = np.interp(
            np.arange(l_melt), [0,l_melt], [1., 6.] )
        
        w_pc = np.maximum(0.1, np.minimum(1.,
                    1 - (alt_v - alt_v[fl_hydro['ice']][-1])*0.2))**2
        
        
        w_pc[Zm_v['Ku'][flag_any_hydro]<12] = 0.
        w_pc[Zm_v['Ka'][flag_any_hydro]<18] = 0.
        
        T1_mat = self._Tikhonov_matrix(n = fine_nodes.size, 
                                       diff = 1, w = w)
        T2_mat = self._Tikhonov_matrix(n = fine_nodes.size, 
                                       diff = 2)
        
        T_mat = (T1_mat + T2_mat)*9.
        
       
      
        
        def CF_retr(x,flag_hydro = flag_hydro, 
                    return_x = False, return_y = False,
                    plot_CF = False): 
            
           
            CF = 0.
            split_list = self._split_x(x,lx = lx, nx = nx)
            
            pc_rep_list = split_list[:nx]
            if nx<3:
                while len(pc_rep_list)<3:
                    pc_rep_list.append(np.zeros_like(pc_rep_list[0]))
            
            parms = split_list[nx]
            
            Alpha_dB_bb_top, *BB_ext_dB = parms
    
            pc_list = [spl_repr(pc_rep)[::-1] for pc_rep in pc_rep_list]

            PC_arr = np.stack(pc_list, axis = 0)
            Will_arr = self._transform_bases(PC_arr, 
                                base1 = 'PCA', base2 = 'Will')
            
            
            if return_x:
                out_x = dict(WC_dB = Will_arr[0,:], Sm_p_dB = Will_arr[1,:], 
                             Dm_dB = Will_arr[2,:], Alpha_dB = Alpha_dB_bb_top,
                             BB_ext = {'Ku': misc.inv_dB(BB_ext_dB[0]),
                                       'Ka': misc.inv_dB(BB_ext_dB[1]),},
                             PCs  = PC_arr )
                
                if not return_y:
                    return out_x
            
            Alpha_dB = np.interp(T_K[flag_hydro['ice']], 
                [0,273.15-30, 273.15],
                [-18., -18, Alpha_dB_bb_top])
            
            Zatt_s, pia_s, Ze_s, k_s, var_Zm  = {}, {}, {}, {}, {}
            for ib, band in enumerate(bands):
                Zatt_s[band], pia_s[band], Ze_s[band], k_s[band] = self._Forw_model(
                    WC_dB = Will_arr[0,:], Sm_p_dB = Will_arr[1,:], 
                    Dm_dB= Will_arr[2,:],
                    T_K=T_K[flag_any_hydro], Alpha_dB=Alpha_dB, 
                    BB_att_1_way = 10**(0.1*BB_ext_dB[ib])*0.5, 
                    flag_hydro=fl_hydro, band = band, range_spacing = 0.125)
                
                att_std = k_s[band]*range_spacing
                for hydro in fl_hydro.keys():
                    att_std[fl_hydro[hydro]] *= self.k_fract_unc[hydro][band]
                int_att_var = np.cumsum(att_std**2)
                
                var_Zm[band] = (var_Ze_model[band] + int_att_var)
                
                
            if return_y:
                out_y = dict(Zm = Zatt_s, Ze = Ze_s,
                             pia = pia_s, k = k_s, 
                             var_Zm = var_Zm, )
                if not return_x:
                    return out_y
                else:
                    return out_x, out_y
            
            
            
            CF_b = (w_pc[np.newaxis,:]*PC_arr**2/self._pca_variance[:,np.newaxis]).sum()
            CF += CF_b
            
            CF_T = sum([np.dot(np.matmul(T_mat,pc),pc) for pc in pc_list])

            CF += CF_T
            
           
            # PR_fg_fine = self.radar_sim(
            #     hydro = 'rain', var = 'PR', band = 'Ku', 
            #     WC_dB = Will_arr[0,:], Sm_p_dB = Will_arr[1,:], 
            #     Dm_dB = Will_arr[2,:])
            
            # mean_rr = np.mean(PR_fg_fine[fl_hydro['rain']])
            # BB_ext_dB_ap = misc.dB(np.array(
            #     [self.radar_sim.BB_ext( mean_rr, band= band) 
            #                 for band in bands]))
            
            CF_bb = (w_BB*(BB_ext_dB - BB_ext_dB_ap)**2).sum()
            CF += CF_bb + 3**2*(BB_ext_dB[1] - BB_ext_dB[0] - 10.8)**2
            
            # CF_MOSS = 5**2*np.sum((PC_arr[:,ind_below_bb].mean(axis = 1)- 
            #            PC_arr[:,ind_above_bb.mean(axis = 1))**2)
            CF_MOSS = 5**2*np.sum((PC_arr[:,ind_below_bb].mean(axis = 1)- 
                       PC_arr[:,ind_above_bb[-1]])**2)
            
            CF += CF_MOSS
            
            CF_o = 0.
            for ib, band in enumerate(bands):
     
                weight_Zs = np.maximum(0,1./var_Zm[band])
                
                CF_o += np.sum(weight_Zs*(
                    (Zm_v[band][flag_any_hydro]- Zatt_s[band])**2))
                
                if backward_corr:
                    CF_o += np.sum(weight_Ze[band]*(
                        (Ze_v[band][flag_any_hydro]- Ze_s[band])**2))
            
            CF_o += weight_delta_PIA*(
                (pia_s['Ka'] -pia_s['Ku']) - delta_PIA)**2

            CF += CF_o
            
            
            CF_Alpha = 0.5**2*(Alpha_dB_bb_top-Alpha_dB_ap)**2
            CF += CF_Alpha
            
            if plot_CF:
                pca_w = (w_pc[np.newaxis,:]*PC_arr**2/self._pca_variance[:,np.newaxis])
                CF_Tick = [] 
                plt.figure()
                for ii in range(3):
                    
                    plt.plot(pca_w[ii,:], '--', label = 'PC%d' % ii)
                    T_mat_sqrt = sqrtm(T_mat)
                    vect = np.matmul(T_mat_sqrt,pc_list[ii])**2
                    plt.plot(vect,'-.', label = 'PC%d Tikh' % ii)
                    CF_Tick.append(sum(vect))
                    
                
                for band in bands:
                    plt.plot(1./var_Zm[band]*(
                        (Zm_v[band][flag_any_hydro]- Zatt_s[band])**2), 
                        label = 'Z%s' % band)
                    plt.plot(weight_Ze[band]*(
                        (Ze_v[band][flag_any_hydro]- Ze_s[band])**2), ':',
                        label = 'Ze%s' % band)
                    plt.plot(1./var_Zm[band], '.', label = 'W %s' % band)
                    
                
                plt.grid()
                plt.legend()
                # print(sum(CF_Tick))
                # print(CF_T)
            return CF 
        
        
            
        # options={'maxiter': 1, 'disp': True,  'ftol': 1e-01, }
        # starttime = timeit.default_timer()
        # res = minimize(CF_retr, x0 = x_fg, method = 'Powell', 
        #     options=options,  args = (flag_hydro,),)
        # time_meth = timeit.default_timer() - starttime
        # print("%1.1fs" % ( time_meth, ))
        # options={'maxiter': 50, 'disp': True,  'ftol': 1e-01, }
        # # starttime = timeit.default_timer()
        # res = minimize(CF_retr, x0 = res.x, method = 'SLSQP', 
        #     options=options,  args = (flag_hydro,),)
        # time_meth = timeit.default_timer() - starttime
        # print("%1.1fs" % ( time_meth, ))
        
       
        
       
        CF_x0 = CF_retr(x_fg)
        ret_shape = dpr_obj.MS.Longitude.shape
        for ii in range(-1,2):
            nr =  nrayMS+ii           
            for jj in range(-1,2):
                ns = nscan+jj                
                if nr>0 and nr<ret_shape[1] and ns>0 and ns<ret_shape[0]:
                    if dpr_obj.MS['CF_n'][ns,nr]>0:
                        # print(dpr_obj.MS['CF_n'][ns,nr])
                        Phys_arr_t =  np.stack(
                            [dpr_obj.MS['WC_dB'][ns,nr].data,
                             dpr_obj.MS['Sm_dB'][ns,nr].data,
                             dpr_obj.MS['Dm_dB'][ns,nr].data], axis = 0)
                        
                        PC_arr_t =  np.matmul(self.base_change_mat['PCA']['phys'].T,
                                    Phys_arr_t - self.shift_vect['PCA']['phys'])
                        pc_fg_rep = []
                        for pp in range(PC_arr_t.shape[0]):                            
                            pc_fg_rep.append( spl_repr.get_rep(fine_nodes, 
                                            PC_arr_t[pp,flag_any_hydro][::-1]))
                            
                        x_prev = np.concatenate(pc_fg_rep[:nx] +
                            [np.array([dpr_obj.MS['Alpha_dB'].data[ns,nr],]),                             
                            BB_ext_dB_ap])
                            
                        CF_xp = CF_retr(x_prev)
                        
                        if CF_xp<CF_x0:   
                            x_fg = 1.*x_prev
                            CF_x0 = 1.*CF_xp
                            print('x0 updated' )
                        
        
        # method = 'Powell'
        # method = 'SLSQP'
        starttime = timeit.default_timer()       
        options={'maxiter': 100, 'disp': True,  'ftol': 5e-02, }       
        res = minimize(CF_retr, x0 = x_fg, method = method, 
            options=options,  args = (flag_hydro,),)
        time_meth = timeit.default_timer() - starttime
        print("%1.1fs" % ( time_meth, ))
        
        
        
        out_x, out_y = CF_retr(x = res.x,flag_hydro = flag_hydro, 
                    return_x = True, return_y = True, plot_CF = plot_diag)
        if plot_diag:
            CF_retr(x = res.x,flag_hydro = flag_hydro, 
                    plot_CF = plot_diag)
        
        out_x['Sm_dB'] = self._transform_Sm_p_dB_Dm_dB_2_Sm_dB(
            Sm_p_dB = out_x['Sm_p_dB'], Dm_dB = out_x['Dm_dB'] )
        
        out_x['PR_dB'] = self.radar_sim(
            hydro = 'rain', var = 'PR_dB', band = 'Ku', 
            WC_dB = out_x['WC_dB'], Sm_p_dB = out_x['Sm_p_dB'], 
            Dm_dB = out_x['Dm_dB'])
        
        if make_plot:
            if backward_corr:
                save_str = 'SRT_backward'
            else:
                save_str = 'no_SRT_backward'
            
          
            
            plt.figure()
            for ib, band in enumerate(bands):
               
                 
                 zm_plot = Zm_v[band][flag_any_hydro]*1.            
                 plt.plot(zm_plot, alt_v, '-', color = 'C%d' % ib)
                 plt.fill_betweenx(y = alt_v,
                                   x1 = zm_plot - np.sqrt(out_y['var_Zm'][band] ),
                                   x2 = zm_plot + np.sqrt(out_y['var_Zm'][band] ),
                                   color = 'C%d' % ib, alpha = 0.3 )
                 
                 
                 zm_plot = out_y['Zm'][band]*1.
                 zm_plot[fl_hydro['melt']] = np.nan
                 plt.plot(zm_plot,alt_v, '--', color = 'C%d' % ib)
                 
                
            
            plt.grid()
            plt.ylim(0, 10,)
            plt.xlim(14,46)
            plt.title(' alph.= %.1f; $PIA_s$: %.1f; %.1f; $PIA_m$: %.1f; %.1f; \n att BB sim : %.1f, %.1f; ' % (
                out_x['Alpha_dB'],
                out_y['pia']['Ku'], out_y['pia']['Ka'], PIAm['Ku'], PIAm['Ka'],
                out_x['BB_ext']['Ku'],  out_x['BB_ext']['Ka'],
                ))
            plt.tight_layout(pad = 0.1)
            plt.savefig(os.path.join(fig_dir,'Z_meas_Z_sim_nrMS_%d_ns_%d_%s.png' % (
                nrayMS, nscan, save_str)),)
            
            plt.figure()
            for ib, band in enumerate(bands):
               
                 
                 zm_plot = Ze_v[band][flag_any_hydro]*1.          
                 plt.plot(zm_plot, alt_v, '-', color = 'C%d' % ib)
                 plt.fill_betweenx(y = alt_v,
                                   x1 = zm_plot - np.sqrt(var_Ze[band][flag_any_hydro] ),
                                   x2 = zm_plot + np.sqrt(var_Ze[band][flag_any_hydro]),
                                   color = 'C%d' % ib, alpha = 0.3 )
                 
                 
                 zm_plot = out_y['Ze'][band]*1.
                 zm_plot[fl_hydro['melt']] = np.nan
                 plt.plot(zm_plot,alt_v, '--', color = 'C%d' % ib)
                 
                
            
            plt.grid()
            plt.ylim(0, 10,)
            plt.xlim(14,46)
            plt.title((' alph.= %.1f; $PIA_s$: %.1f; %.1f; $PIA_m$: %.1f; %.1f;' + 
                      '\n att BB sim : %.1f, %.1f; delta PIA meas.and sim.: %.1f, %.1f' )% (
                out_x['Alpha_dB'],
                out_y['pia']['Ku'], out_y['pia']['Ka'], PIAm['Ku'], PIAm['Ka'],
                out_x['BB_ext']['Ku'],  out_x['BB_ext']['Ka'], 
                delta_PIA, out_y['pia']['Ka'] - out_y['pia']['Ku']
                ))
            plt.tight_layout(pad = 0.1)
            plt.savefig(os.path.join(fig_dir,'Z_e_Z_sim_nrMS_%d_ns_%d_%s.png' % (
                nrayMS, nscan, save_str)),)
            
            
            plt.figure()
            for ib in range(3):      
                 
                 PC = out_x['PCs'][ib,:]/np.sqrt(self._pca_variance[ib] )
                 plt.plot(PC,alt_v, '-', color = 'C%d' % ib,
                          label = 'PC%d' % ib)                                 
            plt.legend()
            plt.grid()
            plt.ylim(0, 10,)
            plt.xlim(-3,3)   
            plt.xlabel('PC/std(PC)')
            plt.tight_layout(pad = 0.1)
            plt.savefig(os.path.join(fig_dir,'PCs_retr_ap_nrMS_%d_ns_%d_%s.png' % (
                nrayMS, nscan, save_str)),)
            
            
                
            
           
            plt.figure()
            plt.grid()
            plt.plot(out_x['WC_dB']+10,alt_v, label = 'WC_dB')
            plt.plot(out_x['Sm_p_dB'],alt_v, label = 'Sm_p_dB')
            plt.plot(out_x['Dm_dB'],alt_v, label = 'Dm_dB')        
            
            plt.plot(Will_arr_fg[0]+10,alt_v, '--',  color = 'C0')
            plt.plot(Will_arr_fg[1],alt_v, '--', color = 'C1')
            plt.plot(Will_arr_fg[2],alt_v, '--', color = 'C2')     
            
            plt.plot(out_x['WC_dB'][fl_hydro['melt']]+10,alt_v[fl_hydro['melt']], 'x',  color = 'C0')
            plt.plot(out_x['Sm_p_dB'][fl_hydro['melt']],alt_v[fl_hydro['melt']], 'x', color = 'C1')
            plt.plot(out_x['Dm_dB'][fl_hydro['melt']],alt_v[fl_hydro['melt']], 'x', color = 'C2')  
           
           
            plt.title('retrieval: "-", first guess: "--", melting: "x"')
            plt.legend()
            plt.xlim(-5, 10)
            plt.ylim(0, 10)
            plt.tight_layout(pad = 0.1)
            plt.savefig(os.path.join(fig_dir,'retr_nrMS_%d_ns_%d_%s.png' % (
                nrayMS, nscan, save_str)),)
            
            
            
            
            plt.figure()
            plt.grid()      
            plt.plot(misc.inv_dB(out_x['Dm_dB'])*10, alt_v, label = '10*Dm')
            plt.plot(col_ds['Ku'].paramDSD.isel(nDSD=1)[flag_any_hydro]*10, alt_v,'--',  color = 'C0', )
            plt.plot(misc.inv_dB(out_x['PR_dB']),alt_v, label = 'PR')      
            plt.plot(col_ds['Ku'].precipRate[flag_any_hydro],alt_v, '--',  color = 'C1', )
            plt.title('retrieval: "-", DPR: "--"')
            plt.legend()
            plt.ylim(0, 10)
            plt.xlim(0, 25)
            plt.tight_layout(pad = 0.1)
            plt.savefig(os.path.join(fig_dir,'retr_vs_DPR_nrMS_%d_ns_%d_%s.png' % (
                nrayMS, nscan, save_str)),)
        

        
        
              
                      
        var_3D_list_x = ['WC_dB', 'Sm_dB', 'Dm_dB', 'PR_dB']
        for var in var_3D_list_x:
            dpr_obj.MS[var][nscan,nrayMS,ind_any_hydro] = out_x[var]
            
        for band in bands:
            var_out = 'z%sSim' % band
            dpr_obj.MS[var_out][nscan,nrayMS,ind_any_hydro] = out_y['Zm'][band]
            var_out = 'z%sEffSim' % band
            dpr_obj.MS[var_out][nscan,nrayMS,ind_any_hydro] = out_y['Ze'][band]
            
            var_out = 'pia%sSim' % band
            dpr_obj.MS[var_out][nscan,nrayMS] = out_y['pia'][band]
            var_out = 'pia%sBB' % band
            dpr_obj.MS[var_out][nscan,nrayMS] = out_x['BB_ext'][band]
            
            
        dpr_obj.MS['Alpha_dB'][nscan,nrayMS] = out_x['Alpha_dB']
        dpr_obj.MS['CF_n'][nscan,nrayMS] = res.fun/len_Y       
                
                          
         
            