#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:34:33 2022

@author: km357
"""
import os
import numpy as np
import numpy.typing as npt
# import xarray as xr
from scipy.optimize import minimize
# from scipy.integrate import cumtrapz
# from scipy.linalg import sqrtm
# from scipy.signal import savgol_filter
from scipy.interpolate import CubicHermiteSpline as CHS
from scipy.spatial import cKDTree
from scipy.special import expit

# from scipy.interpolate import RegularGridInterpolator as RGI
import matplotlib.pyplot as plt
import dill
import timeit
from sklearn.covariance import MinCovDet


from . import misc
from . import simulate
from . import representation 


home = os.path.expanduser("~")
module_path = os.path.dirname(__file__)


class ScaledPCA:
    def __init__(self, n_components = None, dsd_stat_fn = None):
        self.n_components = n_components
        if dsd_stat_fn:
            with open(dsd_stat_fn , 'rb') as f:                
                self.robust_cov=dill.load(f)
            self._calc_params()
        else:
            self.robust_cov = None    
            # self.components_ = None
            # self.mean_ = None
            # self.std_ = None            
            # self.explained_variance_ = None
        
    def fit(self, X):
        # fit a MCD robust estimator to data
        self.robust_cov = MinCovDet().fit(X)        
        self._calc_params()
        
    def _calc_params(self,):
        # number of PCA components
        if self.n_components is None:
            self.n_components = self.robust_cov.covariance_.shape[0] # type: ignore
        # calculate the mean and covariance matrix
        self.mean_ = self.robust_cov.location_ # type: ignore

        # data normalization
        self.std_ = np.sqrt(np.diag(self.robust_cov.covariance_)) # type: ignore

        # covariance matrix of scaled data
        cov_matrix = self.robust_cov.covariance_/(self.std_*self.std_[:,np.newaxis]) # type: ignore
        # calculate the eigenvectors and eigenvalues of the covariance matrix        
        eig_vals, eig_vect = np.linalg.eig(cov_matrix) 
         # sort the eigenvectors in descending order of eigenvalues
        idx = np.argsort(eig_vals)[::-1][:self.n_components] # type: ignore
        self.components_ = eig_vect[:, idx] # type: ignore
        self.explained_variance_ = eig_vals[idx] 
        
    def transform(self, X):
        # center the data
        X_centered = X - self.mean_
        # normalize input
        X_centered *= 1./self.std_ # type: ignore        
        # project onto the principal components
        X_transformed = np.matmul(X_centered, self.components_) # type: ignore
        
        return X_transformed
    
    def inverse_transform(self, X_transformed):
        # project back to the original space
        X_original = np.matmul(X_transformed, self.components_.T)
        # account for normalization
        X_original *= self.std_        
        # add back the mean of the training data
        X_original = X_original + self.mean_        
        return X_original
        
    
class basis_repr():
    """
    basis_elements is a matrix whose columns are vector basis elements, they dont need to be orthogonal 
    """
    def __init__(self, basis_elements,):

        self.basis = basis_elements
        XTX = np.matmul(basis_elements.T, basis_elements)
        self.proj_mat = np.linalg.solve(XTX,basis_elements.T)
        self.proj_mat_T = self.proj_mat.T

    def transform(self,X):
        X_transformed = np.matmul(X, self.proj_mat_T) # type: ignore        
        return X_transformed
    def inverse_transform(self,X_transformed):
        # project back to the original space
        X_original = np.matmul(X_transformed, self.basis.T)
        return X_original

class OE():
    def __init__(self, filename = None, apriori = 'GV_radars'):
        
        
        self.DPR_res = 0.125 #km
        self.radar_sim = simulate.radar_simulator(get_derivatives = False)
        self.bands = ['Ku', 'Ka',]
        
        # each radar bin is represented by 3 unknowns in a form of a column vector
        # physical variables are 
        # 10log10 PR [g/m3]
        # 10log10 sigma_m (PSD width) [mm]
        # 10log10 D_m [mm]
        self.phys_vars = ['PR_dB', 'Sm_dB', 'Dm_dB']
        
        # Williams variables are 
        # 10log10 PR [mm/h]
        # 10log10 sigma_m prime (PSD width) [mm] (it removes a correlation between Sigma_m and D_m)
        # 10log10 D_m [mm]
        self.Williams_vars = ['PR_dB', 'Sm_p_dB', 'Dm_dB']
        stat_fn = os.path.join(module_path, 'dsd_stats', 
                    'NASA_DSD_dataset_all_5min_MinCovDet_analysis_var_scaled.pkl')
        if apriori == 'GV_disdrometers':
            stat_fn = os.path.join(module_path, 'dsd_stats', 
                    'NASA_DSD_dataset_all_5min_MinCovDet_analysis_var_scaled.pkl')
        elif apriori == 'GV_radars':
            stat_fn = os.path.join(module_path, 'dsd_stats',
                'GPM_GV_dataset_MinCovDet_analysis.pkl')
        elif apriori == 'GV_radars_old':
            stat_fn = os.path.join(module_path, 'dsd_stats',
                'GPM_GV_dataset_MinCovDet_analysis_test.pkl')
                
            
        self.ScaledPCA = ScaledPCA(dsd_stat_fn = stat_fn)
        
        """
        # our scattering tables are generated assuming a constant mass flux, 
        # and that the velocity of particles was calculated at 1013hPa. 
        # this implies that the forward model need a pressure/velocity correction for 
        # the a-priori to be equal to a zero vector 
        P0 = 101300      #reference pressure [Pa]
        T0 = 273.15+20   #reference temperaute [K]
        L0 = -6.5*1e-3    #lapse rate [K/m]
        h0 = 0           #reference height [m]
        Rs = 8.3144598   #universal gas constant  [J/(mol·K)]
        g0 = 9.80665     #gravitational acceleration [m/s2]
        M  = 0.0289644   #molar mass of Earth's air  [kg/mol]

        self.DPR_res = 0.125 #km
        self.DPR_alt = np.arange(175,-1,-1)*self.DPR_res
        sim_press_low = P0*(((T0+ (self.DPR_alt*1e3 - h0)*L0)/T0)**(-g0*M/Rs/L0))

        P1 = 22632.10
        T1 = 216.65
        h1 = 11e3

        sim_press_high = P1*np.exp(-g0*M*(self.DPR_alt*1e3 - h1)/Rs/T1)

        sim_press = sim_press_low*1.
        tmp_flag = (sim_press_high>sim_press_low) & (self.DPR_alt*1e3>h1)
        sim_press[tmp_flag] = sim_press_high[tmp_flag]

        # plt.figure()
        # plt.plot(self.DPR_alt, sim_press_low)
        # plt.plot(self.DPR_alt, sim_press_high)
        # plt.plot(self.DPR_alt, sim_press, '--k')
        # plt.grid()
        self.WC_db_ap_press_corr = misc.dB((sim_press/P0)**0.4)

        """
      
        n_points = 201
        pc0_table = np.linspace(-10, 10 , n_points)
        pc_arr = np.stack([pc0_table, 
                np.zeros(n_points), 
                np.zeros(n_points)], axis = 1)

        phys_arr = self.ScaledPCA.inverse_transform(pc_arr)
        sm_p_tmp = self._transform_Sm_dB_Dm_dB_2_Sm_p_dB(Sm_dB = phys_arr[:,1], Dm_dB = phys_arr[:,2])
        Will_arr = phys_arr.copy()
        Will_arr[:,1] = sm_p_tmp 

        # plt.figure()
        # plt.plot(phys_arr[:,0], phys_arr[:,2], label = 'Dm_dB')
        # plt.plot(phys_arr[:,0], phys_arr[:,1], label = 'Sm_dB')
        # plt.legend()
        # plt.grid()
        # plt.xlabel('PR_dB')
                           
        Z_table = {}
        for hydro in ['rain', 'ice']:
            Z_table[hydro] = {}
            for band in ['Ku','Ka']:
                Z_table[hydro][band] = self.radar_sim(
                    hydro = hydro, var = 'Z', band = band, 
                    PR_dB = Will_arr[:,0], Sm_p_dB = Will_arr[:,1], 
                    Dm_dB = Will_arr[:,2])
        
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
        self.KDTree_PR_dB = phys_arr[:,0]
        self.KDTree_Sm_dB = phys_arr[:,1]
        self.KDTree_Dm_dB = phys_arr[:,2]
        
        
        fact = np.log(10)/10
        # self.Ze_sim_unc = {'Ka': 0.5, 'Ku': 1.2}
        self.Ze_std = {'rain': {'Ka': .32, 'Ku': .48}} #calculated with NASA dataset
        self.Ze_std['ice'] = {band: np.sqrt(val*2 + 1.0**2)
                        for band, val in self.Ze_std['rain'].items()}
        self.Ze_std['melt'] = {'Ka': np.inf, 'Ku': np.inf}
        # self.k_sim_unc = {'Ka': 0.14, 'Ku': 0.21}
        self.k_dB_std = {'rain': {'Ka': 0.1, 'Ku': 0.15},
                         'melt': {'Ka': 0., 'Ku': 0.}} #calculated with NASA dataset
        self.k_dB_std['ice'] = {band: np.sqrt(val*2 + 1.0**2)
                        for band, val in self.k_dB_std['rain'].items()}

        self.k_fract_unc = {hydro: {band: fact* val
                        for band, val in self.k_dB_std[hydro].items()}
                                for hydro in self.k_dB_std.keys()}
        
        self.k_dB_Z_rel_std = {'Ka': 1.9, 'Ku': 1.5}
        self.k_dB_Z_rel_frac_unc = {band: fact*val
                        for band, val in self.k_dB_Z_rel_std.items()}
        
        
        
        # self.retr_reds = retr_resol_km
        # fine_nodes = np.arange(0., 22., self.DPR_res)
        # self.spl_func_width = (deg+1)*retr_resol_km
        # coarse_nodes = np.arange(retr_resol_km,fine_nodes[-1], retr_resol_km)

        # self.spl_repr = representation.spline(fine_nodes = fine_nodes,
        #             coarse_nodes = coarse_nodes,
        #             deg = deg, make_plot=True)

    
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
        Sm_p_dB = Sm_dB - (1.6 * Dm_dB - 5.)
        return Sm_p_dB
    def _transform_Sm_p_dB_Dm_dB_2_Sm_dB(self, Sm_p_dB, Dm_dB):
        Sm_dB = Sm_p_dB + (1.6 * Dm_dB - 5.)  
        return Sm_dB    
   
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
                          
    """
    def _split_x(self, x, lx, nx_full = 3, nx_single = 0) -> list:
        
        pc0, pc1, pc2, Alpha ice, BB_ext
        
        if nx_full+nx_single>3:
            print('number of unknowns exceeds 3')
            return []
        split_list = [lx*(ii+1) for ii in range(nx_full)]
        for ii in range(nx_single):
            split_list.append(ii+1+nx_full*lx)
        return np.split(x, split_list)

    def _form_x_vect(self, x_repr, spline_repr, lx, nx_full = 3, nx_single = 0,) -> np.ndarray:

        #we split x_repr vector into PC representation and other parameters
        split_list = self._split_x(x_repr, lx = lx, 
                        nx_full = nx_full, nx_single = nx_single)
        # spline representation of principal components in the column
        pc_rep_list = [np.zeros(spline_repr.ind_node_signif.size) for ii in range(3)]
        # full vectors
        for ii in range(nx_full):
            pc_rep_list[ii][:lx] = split_list[ii]
        # single values for a whole column (if any)
        for ii in range(nx_single):
            pc_rep_list[ii+nx_full][:lx] = split_list[ii+nx_full]

        # spline representation of vector x is transformed to a list of PCs at the native resolution
        pc_list = [spline_repr(pc_rep)  for pc_rep in pc_rep_list]
        pc_list.append(split_list[-1])
        return np.concatenate(pc_list)
    """

    def _form_x(self, x_repr, spl_repr, ) -> tuple[dict, npt.NDArray[np.float_]]:
        # state vector
        ind_x = 0
        nx =  spl_repr['PC0'].basis.shape[0]
        PC_arr = np.zeros((nx, 3) )
        for ii, pc_name in enumerate(['PC0', 'PC1', 'PC2']):
            ix = spl_repr[pc_name].basis.shape[1]      
            tmp_x = x_repr[ind_x:ind_x+ix]
            tmp_pc =  spl_repr[pc_name].inverse_transform(tmp_x) 
            PC_arr[:,ii] = tmp_pc
            ind_x += ix

        x = np.concatenate(
            [PC_arr[:,0], PC_arr[:,1], PC_arr[:,2], x_repr[ind_x:]])
               
        Phys_arr = self.ScaledPCA.inverse_transform(PC_arr)
        Will_arr = Phys_arr.copy()        
        Will_arr[:,1] = self._transform_Sm_dB_Dm_dB_2_Sm_p_dB(
            Sm_dB = Phys_arr[:,1], Dm_dB = Phys_arr[:,2])
        
        Alpha_dB_bb_top, *BB_ext = x_repr[ind_x:]      
        BB_ext_dB = np.array([BB_ext[0], BB_ext[0]+ BB_ext[1]])
        Alpha_dB = spl_repr['Alpha_dB'].inverse_transform(np.array([Alpha_dB_bb_top,]))


        x_dict = dict(PR_dB = Will_arr[:,0], Sm_p_dB = Will_arr[:,1], 
                    Dm_dB = Will_arr[:,2], Alpha_dB = Alpha_dB,
                    BB_ext  =  {'Ku': misc.inv_dB(BB_ext_dB[0]),
                                'Ka': misc.inv_dB(BB_ext_dB[1]),},
                    PCs  = PC_arr )
        return x_dict, x
    

    def _form_y_vect(self, Zm, Ze, dPIA, ) -> np.ndarray:
        Zm_v = np.concatenate([Zm[band] for band in self.bands])
        Ze_v = np.concatenate([Ze[band] for band in self.bands])
        return np.r_[Zm_v, Ze_v, dPIA]
    
    def _form_y_dict(self, x_dict, flag_any_hydro, T_K, fl_hydro) -> dict:
        
        Zatt_s, pia_s, Ze_s, k_s, int_att_var  = {}, {}, {}, {}, {}
        for ib, band in enumerate(self.bands):
            Zatt_s[band], pia_s[band], Ze_s[band], k_s[band] = self._Forw_model(
                PR_dB = x_dict['PR_dB'], Sm_p_dB = x_dict['Sm_p_dB'], 
                Dm_dB= x_dict['Dm_dB'],
                T_K=T_K[flag_any_hydro], Alpha_dB=x_dict['Alpha_dB'], 
                BB_att_1_way = x_dict['BB_ext'][band]*0.5, 
                flag_hydro=fl_hydro, band = band, range_spacing = self.DPR_res)
        y_dict = dict(Zm = Zatt_s, Ze = Ze_s,
                        pia = pia_s, k = k_s, )
        return y_dict


    def _std_Z_DPR(self,Z, band = 'Ku', M = 110):
        if band == 'Ku':
            SNR = Z-15.46
        else: #band == 'Ka'
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
        
    def _Forw_model(self, PR_dB, Sm_p_dB, Dm_dB, T_K, Alpha_dB, BB_att_1_way, 
                    flag_hydro, band = 'Ku', range_spacing = 0.125):
        
        size_z = flag_hydro[list(flag_hydro.keys())[0]].shape
        Ze, k = -99.9*np.ones(size_z), np.zeros(size_z)
        simulated_hydros = [hydro for hydro in flag_hydro.keys() 
            if hydro in ['rain', 'ice']]
            
        for hydro in simulated_hydros:
                Z_tmp = self.radar_sim(
                    hydro = hydro, var = 'Z', band = band, 
                    PR_dB = PR_dB[flag_hydro[hydro]], 
                    Sm_p_dB = Sm_p_dB[flag_hydro[hydro]], 
                    Dm_dB = Dm_dB[flag_hydro[hydro]],
                    T = T_K[flag_hydro[hydro]],
                    Alpha_dB = Alpha_dB[flag_hydro[hydro]])
            
                k_tmp = self.radar_sim(
                    hydro = hydro, var = 'k', band = band, 
                    PR_dB = PR_dB[flag_hydro[hydro]], 
                    Sm_p_dB = Sm_p_dB[flag_hydro[hydro]], 
                    Dm_dB = Dm_dB[flag_hydro[hydro]],
                    T = T_K[flag_hydro[hydro]],
                    Alpha_dB = Alpha_dB[flag_hydro[hydro]])
                Ze[flag_hydro[hydro]] = Z_tmp
                k [flag_hydro[hydro]] = k_tmp
            
        if 'melt' in flag_hydro.keys(): 
            att_per_bin = BB_att_1_way/flag_hydro['melt'].sum()
            k[flag_hydro['melt']] = att_per_bin/range_spacing
               
            
        one_way_att = np.cumsum(k)*range_spacing
         
        Zm = np.copy(Ze)
        Zm[1:] += -2.*one_way_att[:-1]
        return Zm, one_way_att[-1]*2, Ze , k
    """
    def _dForw_model(self, PR_dB, Sm_p_dB, Dm_dB, T_K, Alpha_dB, 
                    flag_hydro, x = 'Dm_dB', band = 'Ku', 
                    range_spacing = 0.125):
        # x in ['PR_dB', 'Dm_dB', 'Sm_p_dB', 'Alpha_dB']
        size_z = flag_hydro[list(flag_hydro.keys())[0]].size
        Jac_Ze = np.zeros(size_z, )
        Jac_k = np.zeros(size_z, )
        der_dB_const = 0.1*np.log(10)
        
        simulated_hydros = [hydro for hydro in flag_hydro.keys() 
            if hydro in ['rain', 'ice']]
        
        if x in ['Dm_dB', 'Sm_p_dB', 'Sm_dB', 'Alpha_dB']:
            size_x = size_z*1
            for hydro in simulated_hydros:
                
                if x == 'Alpha_dB' and hydro == 'rain': continue
                der_name = 'd_Z_d_%s' % x
                dZ_tmp = self.radar_sim(
                    hydro = hydro, var = der_name, band = band, 
                    PR_dB = PR_dB[flag_hydro[hydro]], 
                    Sm_p_dB = Sm_p_dB[flag_hydro[hydro]], 
                    Dm_dB = Dm_dB[flag_hydro[hydro]],
                    T = T_K[flag_hydro[hydro]],
                    Alpha_dB = Alpha_dB[flag_hydro[hydro]],)
                der_name = 'd_k_d_%s' % x
                dk_tmp = self.radar_sim(
                    hydro = hydro, var = der_name, band = band, 
                    PR_dB = PR_dB[flag_hydro[hydro]], 
                    Sm_p_dB = Sm_p_dB[flag_hydro[hydro]], 
                    Dm_dB = Dm_dB[flag_hydro[hydro]],
                    T = T_K[flag_hydro[hydro]],
                    Alpha_dB = Alpha_dB[flag_hydro[hydro]],)
                Jac_Ze[flag_hydro[hydro]] = dZ_tmp
                Jac_k[flag_hydro[hydro]] = dk_tmp

        elif x=='PR_dB':
            size_x = size_z*1

            for hydro in simulated_hydros:
                dZ_tmp = np.ones((flag_hydro[hydro].sum()))
                PR = 10**(0.1*PR_dB[flag_hydro[hydro]])
                
                k_tmp = self.radar_sim(
                    hydro = hydro, var = 'k', band = band, 
                    PR_dB = PR_dB[flag_hydro[hydro]], 
                    Sm_p_dB = Sm_p_dB[flag_hydro[hydro]], 
                    Dm_dB = Dm_dB[flag_hydro[hydro]],
                    T = T_K[flag_hydro[hydro]],
                    Alpha_dB = Alpha_dB[flag_hydro[hydro]],)
                    
                Jac_Ze[flag_hydro[hydro]] = dZ_tmp
                Jac_k[flag_hydro[hydro]] = der_dB_const * WC* k_tmp

        Jac_Zm = np.diag(Jac_Ze)
        Jac_PIA = 2*Jac_k[np.newaxis,:]*range_spacing
        Jac_Zm += np.tril (np.ones_like(Jac_Zm)*Jac_PIA)
        return Jac_Zm, Jac_PIA, Jac_Ze
        """
    def _CF_1D(self,
                x_repr,  x_ap, R_ap_inv,
                y_m, R_m_inv, T_K,
                weight_Tikhonov, 
                fl_hydro, flag_any_hydro, 
                spl_repr,  ) -> float:   

        """
        x_repr is spline reprezentation of x; plus additional parameters
        y_m is the measurement vector: Z measured (@ Ku and Ka), Z corrected for attenuation, differential PIA
        R_m_inv is the measurement error covariance matrix (it's diagonal so we provide it in a form of a vector);
            it does not include errors due to simulated attenuation 
        """
                    
        # an initial value of the cost function
        CF = 0.
        nx = spl_repr['PC0'].basis.shape[0]
        # state vector        
        x_dict, x = self._form_x(x_repr, spl_repr, )
        # measurement vector
        y_dict = self._form_y_dict(x_dict, flag_any_hydro, T_K, fl_hydro)
    
        int_att_var, var_Zm  = {}, {}
        for ib, band in enumerate(self.bands):
            att_std = y_dict['k'][band]*1.
            for hydro in fl_hydro.keys():
                att_std[fl_hydro[hydro]] *= self.k_fract_unc[hydro][band]
            int_att_var[band] = np.cumsum(att_std**2)*self.DPR_res*0.
            var_Zm[band] = 1/R_m_inv[ib*nx:(ib+1)*nx] + int_att_var[band]

        # cost function of the measurements
        y = self._form_y_vect(Zm = y_dict['Zm'], Ze = y_dict['Ze'], 
                dPIA = y_dict['pia']['Ka'] - y_dict['pia']['Ku'])

        # the attenuation uncertainty must be added to the error covariance matrix
        R_att = np.concatenate([int_att_var[band] for band in self.bands])
        tmp_R = 1/R_m_inv[:2*nx] + R_att
        R_m_inv[:2*nx] = 1/tmp_R
        CF_o = np.dot((y_m - y)**2,R_m_inv)
        CF += CF_o

        # cost function corresponding to a-priori information; weight_PC is a weight in the column,
        # the higher from the ML the lower the weight
        CF_b = np.dot((x-x_ap)**2,R_ap_inv)
        CF += CF_b
        
        # cost function corresponding to the smoothing condition (Tikhonov matrix)
        Tikhonov_components = np.array([np.dot(np.matmul(weight_Tikhonov,x_dict['PCs'][:,ii]),
                x_dict['PCs'][:,ii]) for ii in range(3)])
        CF_T = np.sum(Tikhonov_components)
        CF += CF_T

        # # cost function of the melting only assumption
        # CF_MOSS = 5**2*np.sum((PC_arr[:,fl_hydro['rain']].mean(axis = 1)- 
        #            PC_arr[:,fl_hydro['ice']][:,-1])**2)
        # CF += CF_MOSS
        return CF 


        
    def retrieve_PC_1D(self, dpr_obj, nscan, nrayMS, 
                        make_plot = False, fig_dir = home,                          
                        method = 'SLSQP', retr_resol_ice = 0.4, retr_resol_rain = 1.6, 
                        retr_resol_scale_x1 = 3., maxiter = 30, ):
        
        var_3D_list = [ 'Sm_dB', 'Dm_dB', 'PR_dB', 'WC_dB', 
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
     
        sn = dpr_obj.swath_name
        
        bands = self.bands
        range_spacing = dpr_obj['MS'].attrs['range_spacing']
        
        col_ds = {band: dpr_obj[sn[band]].isel(
            nscan = nscan, nrayMS = nrayMS).load() for band in bands}
        
              
        
        #### read measured reflectivities
        Zm_v = {band: col_ds[band].zFactorMeasured.data*1. for band in bands}
        phase = col_ds['Ku'].phase.data*1.
        T_K = np.zeros_like(phase)
        T_K[phase<=100] = phase[phase<=100]-100+273.15
        T_K[phase>=200] = phase[phase>=200]-200+273.15
        T_K[(phase<200) & (phase>100)] = 273.15

        #Freezing level height
        if np.any(phase>=200) & (col_ds['Ku'].binBBTop.data>0):
            FL_height = (176-col_ds['Ku'].binBBTop.data)*0.125 
        else:
            FL_height = (176-col_ds['Ku'].binZeroDeg.data)*0.125  
        
        
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
        weight_delta_PIA = 1./std_delta_PIA**2
        if not np.isfinite(delta_PIA+weight_delta_PIA):
            delta_PIA = 0.
            print('delta PIA unreliable')
            weight_delta_PIA = 0.
            
        #### partitioning data to rain, melting and ice
        flag_hydro = {
            'melt': ((col_ds['Ku'].phase>=100) & 
                    (col_ds['Ku'].phase<200)).data,
            'ice': ((col_ds['Ku'].phase<100) & (Zm_v['Ku']>10) & (
                    col_ds['Ku'].nbin >= col_ds['Ku'].binStormTop)).data,
            'rain': (col_ds['Ku'].phase.values>=200) & (Zm_v['Ku']>10) & (
                col_ds['Ku'].nbin <= col_ds['Ku'].binRealSurface.values )}

        ind_ice = np.where(flag_hydro['ice'])[0]
        if ind_ice.size>0:
            ii = ind_ice[0]-1
            while (Zm_v['Ku'][ii]>9.5) and (ii>=0):
                # print(ii)
                flag_hydro['ice'][ii] = True
                ii += -1
                
        ind_ice = np.where(flag_hydro['ice'])[0]
                
        
        flag_hydro['ice'][ind_ice[-1]] = False
        flag_hydro['melt'][ind_ice[-1]] = True
        
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
        nx = alt_v.size # number of the retrieval levels 

        nodes_pc0 = [alt_v[-1],]
        nodes_pc1 = [alt_v[-1],]
        
        for lev_height, vert_res in zip([FL_height, alt_v[0]], 
                [retr_resol_rain, retr_resol_ice]):
        
            while nodes_pc0[-1]<lev_height:
                nodes_pc0.append(nodes_pc0[-1]+vert_res)
            while nodes_pc1[-1]<lev_height:
                nodes_pc1.append(nodes_pc1[-1]+vert_res*retr_resol_scale_x1)
        
        if nodes_pc0[-1]>=alt_v[0]:
            nodes_pc0[-1]=alt_v[0]
        else:
            nodes_pc0.append(alt_v[0])
            
        if nodes_pc1[-1]>=alt_v[0]:
            nodes_pc1[-1]=alt_v[0]
        else:
            nodes_pc1.append(alt_v[0])
        
        
              
        spl_nodes = {'PC0': np.array(nodes_pc0),
                     'PC1': np.array(nodes_pc1),
                     'PC2': np.array([alt_v[-1], alt_v[0]])}
        
        spl_basis = {}
        spl_repr = {}
        for x_name, tmp_x in spl_nodes.items():           
            spl_basis[x_name] = np.empty((nx, tmp_x.size, )) # columns are the basis vectors
            if tmp_x.size>1:
                for ii in range(tmp_x.size):
                    tmp_y = np.zeros_like(tmp_x)
                    tmp_y[ii] = 1.
                    dydx = np.gradient(tmp_y,tmp_x)
                    # print(dydx)
                    # print(tmp_x)
                    # print(tmp_y)
                    # dydx = np.gradient(tmp_y,)
                    dydx[0] = 0.
                    # dydx[-1] = 0.
                    spl = CHS(x = tmp_x, y = tmp_y, dydx = dydx)
                    spl_basis[x_name][:, ii] = spl(alt_v)
            elif tmp_x.size==1:
                spl_basis[x_name][:, 0] = np.ones(alt_v.size)
            spl_repr[x_name] = basis_repr(spl_basis[x_name])
        
        spl_basis['Alpha_dB'] = np.ones((nx,1))
        spl_repr['Alpha_dB'] = basis_repr(spl_basis['Alpha_dB'])
        # Alpha_dB_base = np.interp(T_K[flag_any_hydro], 
        #         [0,273.15-30, 273.15, 400.],
        #         [0., 0., 1., 1.])

       
        
            
      
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
                weight_Z[band][ind_extrap] = 1./(.4**2) #1./(.75**2)
            else:
                weight_Z[band][ind_extrap] = 1./(.4**2) #1./(1.5**2)
                Zm_v[band][ind_extrap] = np.mean(Zm_v[band][ind_extrap_input])
            # weight_PIA[band] *= 1+np.sum(weight_Z[band]>3)
        weight_Z['Ku'][Zm_v['Ku']<10] = 0.
        weight_Z['Ka'][Zm_v['Ka']<18] = 0.        
        std_Zm = {band : 1/np.sqrt(weight_Z[band]) for band in bands}                 
        len_Y = (weight_Z['Ku']>0.).sum() + (weight_Z['Ka']>0.).sum() +1

        Ze_v ={band: self.HB_correction(ZdB_m = Zm_v[band], 
            flag_hydro = flag_hydro, PIA_dB = PIAm[band], band = band, 
            dh_km = range_spacing) for band in bands}
        att_corr = {band: Ze_v[band] - Zm_v[band] for band in bands}
        tmp_k = {band: np.diff(att_corr[band], prepend = 0)/self.DPR_res 
                for band in bands}
        var_bottom_up = {band: 
            np.cumsum((tmp_k[band][::-1]*self.k_dB_Z_rel_frac_unc[band])**2)[::-1]*
                self.DPR_res for band in bands}
        
        var_Ze = {band: std_Zm[band]**2 for band in bands}
        for band in bands:
            for hydro in flag_hydro.keys():
                var_Ze[band][flag_hydro[hydro]] += self.Ze_std[hydro][band]**2
   
        weight_Ze_model = {band: np.maximum(0, 1/var_Ze[band][flag_any_hydro]) 
            for band in bands}

        
        for band in bands:
            var_Ze[band] += (std_PIA[band]**2 + var_bottom_up[band])

        weight_Ze = {band: np.maximum(0,1./var_Ze[band][flag_any_hydro]) for band in bands}

                 
        # k_per_bin = {band: np.diff(att_corr[band], prepend = 0.) for band in bands}
        
        pc0_fg = self._fg_pc0( Ze_v = {band: Ze_v[band]
                for band in bands}, flag_hydro = flag_hydro)
        pc0_fg = np.maximum(-5., pc0_fg)
        pc0_fg_rep = spl_repr['PC0'].transform(pc0_fg[flag_any_hydro])  
        pc0 = spl_repr['PC0'].inverse_transform(pc0_fg_rep)        

        # plt.figure()
        # plt.plot(pc0_fg[flag_any_hydro], alt_v)
        # plt.plot(pc0, alt_v)
        # plt.grid()
        
        
        
        PC_arr_fg = np.stack([pc0,pc0*0,pc0*0], axis = 1)
        Phys_arr_fg = self.ScaledPCA.inverse_transform(PC_arr_fg)
        Will_arr_fg = Phys_arr_fg.copy()        
        Will_arr_fg[:,1] = self._transform_Sm_dB_Dm_dB_2_Sm_p_dB(
            Sm_dB = Phys_arr_fg[:,1], Dm_dB = Phys_arr_fg[:,2])


        BB_ext_dB_HB = misc.dB(np.array(
            [att_corr[band][flag_hydro['rain']][0] - 
            att_corr[band][flag_hydro['ice']][-1] for band in bands]))
        BB_ext_dB_ap = np.array([BB_ext_dB_HB[1]-6. , 6.,])        
                
        BB_ext_dB_var = np.array([2.**2, 3**2]) #fractional uncertainty
        
        # att_corr['Ka'][flag_hydro['melt']][[0,-1]]
        
        ind_below_bb = np.where(fl_hydro['rain'])[0][slice(7)]
        ind_above_bb = np.where(fl_hydro['ice'])[0][slice(-7,None)]       
        
        Zm_ice = {}
        for ib, band in enumerate(bands):
            Zm_ice[band] = np.polyval(
                np.polyfit(ind_above_bb, 
                    Ze_v[band][flag_any_hydro][ind_above_bb], 1), 
                ind_above_bb[-1])
        
        
        mean_pr_db =  np.mean(Will_arr_fg[ind_below_bb,0])
        mean_sm_db =  np.mean(Will_arr_fg[ind_below_bb,1])
        mean_dm_db =  np.mean(Will_arr_fg[ind_below_bb,2])
        
        ones_arr = np.ones(self.radar_sim.scat_LUT['ice']['Ku']['Alpha_dB'].shape)
        Zs_ice = {}
        alpha_test = self.radar_sim.scat_LUT['ice']['Ku']['Alpha_dB'].data
        for ib, band in enumerate(bands):
            Zs_ice[band] = self.radar_sim(
                hydro = 'ice', var = 'Z', band = band, 
                PR_dB = mean_pr_db*ones_arr, 
                Sm_p_dB = mean_sm_db*ones_arr, 
                Dm_dB = mean_dm_db*ones_arr,
                Alpha_dB = alpha_test)
           
        # CF_alpha_tmp = ((Zs_ice['Ku'] - Zm_ice['Ku'])**2 + 
        #                 (Zs_ice['Ka'] - Zm_ice['Ka'])**2)
        CF_alpha_tmp = ((Zs_ice['Ku'] - Zm_ice['Ku'])**2)
        ind_min_alpha = np.argmin(CF_alpha_tmp)
        Alpha_dB_ap = alpha_test[ind_min_alpha]

        # plt.figure()
        # plt.plot(alpha_test,CF_alpha_tmp)
        # plt.plot(alpha_test,(Zs_ice['Ka'] - Zm_ice['Ka'])**2)
        # plt.plot(alpha_test,(Zs_ice['Ku'] - Zm_ice['Ku'])**2)
        # plt.grid()

        Alpha_dB_var = 3.**2 #just a random estimate, factor of 2 uncertainty
        
        pc_fg_list = [pc0_fg_rep, 
                    np.zeros(spl_repr['PC1'].basis.shape[1],),
                    np.zeros(spl_repr['PC2'].basis.shape[1],)]        

        x_rep_fg = np.concatenate(pc_fg_list + 
                            [np.array([Alpha_dB_ap,]), BB_ext_dB_ap])

        x_ap = np.concatenate([np.zeros(nx), np.zeros(nx), np.zeros(nx)] + 
                            [np.array([Alpha_dB_ap,]), BB_ext_dB_ap])

        # min_w = 0.25
        # weight_PC = min_w+ (1.**2-min_w)*expit( 
        #     (ind_any_hydro - ind_any_hydro[fl_hydro['ice']][-1])/8*6+6)
            
        
        weight_PC = np.ones(nx)
        # weight_PC[Zm_v['Ka'][flag_any_hydro]<18] = 0.
        weight_PC[Zm_v['Ku'][flag_any_hydro]<10] = 0.
        R_ap_inv = np.concatenate([weight_PC/(tmp_var*sc_fact) 
                            for (tmp_var, sc_fact) in zip(self.ScaledPCA.explained_variance_, [4.,1.,1.])] + 
                            [np.array([1/Alpha_dB_var,]), 1/BB_ext_dB_var])
        
        # weight of a-priori PCs, more weight in rain 
        w_max = 4.**2.
        w_min = 1.**2.


        w = np.ones(nx)
        w[fl_hydro['ice']] = 0.
        l_melt = fl_hydro['melt'].sum()
        w[fl_hydro['melt']] = np.interp(
            np.arange(l_melt), [0,l_melt], [0., 1.] )

        w1 = w*(w_max-w_min)*4.+w_min
        w2 = (1.- w)*(w_max-w_min)+w_min

        # w = np.ones(nx)*w_max
        # w[fl_hydro['ice']] = 1.
        # l_melt = fl_hydro['melt'].sum()
        # w[fl_hydro['melt']] = np.interp(
        #     np.arange(l_melt), [0,l_melt], [1., w_max] )

        T1_mat = self._Tikhonov_matrix(n = nx, diff = 1, w = w1) # measures deviation from a constant value 
        T2_mat = self._Tikhonov_matrix(n = nx, diff = 2, w = w2)  # measures deviation from a linear change 
        weight_Tikhonov = T2_mat+T1_mat

        # Tikhonov_matrix = np.zeros((3*nx + 3, 3*nx + 3))
        

        # weight_PC = np.maximum(0.1, np.minimum(1.,
        #             1 - (alt_v - alt_v[fl_hydro['ice']][-1])*0.2))**2
        # weight_PC[Zm_v['Ku'][flag_any_hydro]<12] = 0.
        # weight_PC[Zm_v['Ka'][flag_any_hydro]<18] = 0.
        
        
            
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
        
        Zm = {band: Zm_v[band][flag_any_hydro] for band in self.bands}
        Ze = {band: Ze_v[band][flag_any_hydro] for band in self.bands}
        y_m = self._form_y_vect(Zm = Zm, Ze = Ze, dPIA = delta_PIA)

        R_m_inv = self._form_y_vect(Zm = weight_Ze_model, Ze = weight_Ze, 
                    dPIA = weight_delta_PIA*np.sum(weight_Ze_model['Ku']/4))

        args = (x_ap, R_ap_inv,
                y_m, R_m_inv, T_K,
                weight_Tikhonov, 
                fl_hydro, flag_any_hydro, 
                spl_repr, )


        # CF_x0 = self._CF_1D(x_rep_fg,  *args) 
        # print(x_ap.shape)

        # ret_shape = dpr_obj.MS.Longitude.shape
        # for ii in range(-1,2):
        #     nr =  nrayMS+ii           
        #     for jj in range(-1,2):
        #         ns = nscan+jj                
        #         if nr>0 and nr<ret_shape[1] and ns>0 and ns<ret_shape[0]:
        #             if dpr_obj.MS['CF_n'][ns,nr]>0:
        #                 # print(dpr_obj.MS['CF_n'][ns,nr])
        #                 Phys_arr_t =  np.stack(
        #                     [dpr_obj.MS['PR_dB'][ns,nr].data,
        #                      dpr_obj.MS['Sm_dB'][ns,nr].data,
        #                      dpr_obj.MS['Dm_dB'][ns,nr].data], axis = 0)
                        
        #                 PC_arr_t =  np.matmul(self.base_change_mat['PCA']['phys'].T,
        #                             Phys_arr_t - self.shift_vect['PCA']['phys'])
        #                 pc_fg_rep = []
        #                 for pp in range(nx_full):                            
        #                     pc_fg_rep.append( spline_repr.get_rep(fine_nodes, 
        #                                     PC_arr_t[pp,flag_any_hydro]))
        #                 for pp in range(nx_single):
        #                     pc_fg_rep.append(np.array([np.mean(spline_repr.get_rep(fine_nodes, 
        #                                     PC_arr_t[pp+nx_full,flag_any_hydro])),]))
                                        
        #                 x_prev = np.concatenate(pc_fg_rep +
        #                     [np.array([dpr_obj.MS['Alpha_dB'].data[ns,nr],]),                             
        #                     BB_ext_dB_ap])
                            
        #                 CF_xp = self._CF_1D(x_prev,  *args)                        

        #                 if CF_xp<CF_x0:   
        #                     print('CF: %.1f -> %.1f x0 updated' % (CF_x0, CF_xp) )
        #                     x_rep_fg = 1.*x_prev
        #                     CF_x0 = 1.*CF_xp
        #                     # print(x_rep_fg.shape)

        # method = 'Powell'
        # method = 'SLSQP'


        bounds = [(-8., 8.) for ii in range(x_rep_fg.size-3)]
        bounds += [(-18.,-3.), (-20.,10.), (3., 12.)]

        for jj in range(x_rep_fg.size):
            if x_rep_fg[jj]< bounds[jj][0]:
                x_rep_fg[jj]= bounds[jj][0]+0.1
            elif x_rep_fg[jj]> bounds[jj][1]:
                x_rep_fg[jj]= bounds[jj][1]-0.1
        
        starttime = timeit.default_timer()  

        options={'maxiter': maxiter, 'disp': True,  'ftol': 5e-03, } 
        # iter = []
        # def callback(x_repr):
        #     x = self._form_x_vect(x_repr, 
        #         spline_repr, lx = lx, nx_full = nx_full, nx_single = nx_single)
        #     nx = spline_repr.fine_nodes.size

        #     x_dict = self._form_x_dict(x, nx, flag_any_hydro, )
        #     y_dict = self._form_y_dict(x_dict, Alpha_dB_base, flag_any_hydro, T_K, fl_hydro)
        #     iter.append(y_dict['pia']['Ka'] - y_dict['pia']['Ku'])
        #     plt.figure(99)
        #     plt.plot(len(iter), iter[-1])
        


        res = minimize(self._CF_1D, x0 = x_rep_fg, method = method, 
            options=options,  args = args,  bounds = bounds, )
        time_meth = timeit.default_timer() - starttime
        print("%1.1fs" % ( time_meth, ))

        """
        res = minimize(self._CF_1D, x0 = x_rep_fg, method = 'Powell', 
            options = {'disp': True,  'ftol': 5e-03, },  args = args,  bounds = bounds, )

        res = minimize(self._CF_1D, x0 = res.x, method = 'TNC', 
            options = {'disp': True,  'ftol': 5e-03, },  args = args,  bounds = bounds, )
        
        """
                        
        out_x, x = self._form_x(res.x, spl_repr, )
        # measurement vector
        out_y = self._form_y_dict(out_x, flag_any_hydro, T_K, fl_hydro)
                
        out_x['Sm_dB'] = self._transform_Sm_p_dB_Dm_dB_2_Sm_dB(
            Sm_p_dB = out_x['Sm_p_dB'], Dm_dB = out_x['Dm_dB'] )
        
        out_x['WC_dB'] = self.radar_sim(
            hydro = 'rain', var = 'RWC_dB', band = 'Ku', 
            PR_dB = out_x['PR_dB'], Sm_p_dB = out_x['Sm_p_dB'], 
            Dm_dB = out_x['Dm_dB'])
        
        IWC_dB = self.radar_sim(
            hydro = 'ice', var = 'IWC_dB', band = 'Ku', 
            PR_dB = out_x['PR_dB'], Sm_p_dB = out_x['Sm_p_dB'], 
            Dm_dB = out_x['Dm_dB'])
        out_x['WC_dB'][fl_hydro['ice']] = IWC_dB[fl_hydro['ice']]
        out_x['WC_dB'][fl_hydro['melt']] = np.nan # transistion to be defined
        
        
        if make_plot:            
            save_str = 'SRT_backward'
            title_str = r'$\alpha$= %.1f; $PIA_s$: %.1f; %.1f; $PIA_m$: %.1f; %.1f; $BB_{ext}$ : %.1f, %.1f; ' % (
                out_x['Alpha_dB'][-1],
                out_y['pia']['Ku'], out_y['pia']['Ka'], PIAm['Ku'], PIAm['Ka'],
                out_x['BB_ext']['Ku'],  out_x['BB_ext']['Ka'], )
            plt.figure()
            for ib, band in enumerate(bands):     
                zm_plot = Zm_v[band][flag_any_hydro]*1.      
                zm_std = std_Zm[band][flag_any_hydro] 
                plt.plot(zm_plot, alt_v, '-', color = 'C%d' % ib)
                plt.fill_betweenx(y = alt_v,
                        x1 = zm_plot - zm_std, x2 = zm_plot + zm_std,
                        color = 'C%d' % ib, alpha = 0.3 )
                zm_plot = out_y['Zm'][band]*1.
                zm_plot[fl_hydro['melt']] = np.nan                
                plt.plot(zm_plot,alt_v, '--', color = 'C%d' % ib) 
                zm_plot = out_y['Ze'][band]*1.
                zm_plot[fl_hydro['melt']] = np.nan
                plt.plot(zm_plot,alt_v, ':', color = 'C%d' % ib)
            plt.grid()
            plt.ylim(0, alt_v[0]+0.25,)
            plt.xlim(12,50)
            plt.title(title_str)
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
            plt.ylim(0, alt_v[0]+0.25,)
            plt.xlim(12,50)
            plt.title(title_str)
            plt.tight_layout(pad = 0.1)
            plt.savefig(os.path.join(fig_dir,'Z_e_Z_sim_nrMS_%d_ns_%d_%s.png' % (
                nrayMS, nscan, save_str)),)
            
            
            plt.figure()
            for ib in range(3):  
                PC = out_x['PCs'][:,ib]
                plt.plot(PC,alt_v, '-', color = 'C%d' % ib,
                        label = 'PC%d' % ib)                                 
            plt.legend()
            plt.grid()
            plt.ylim(0, alt_v[0]+0.25,)
            plt.xlim(-5,5)   
            plt.xlabel('PC')
            plt.tight_layout(pad = 0.1)
            plt.savefig(os.path.join(fig_dir,'PCs_retr_ap_nrMS_%d_ns_%d_%s.png' % (
                nrayMS, nscan, save_str)),)
            
            
           
            plt.figure()
            plt.grid()
            plt.plot(out_x['PR_dB'],alt_v, label = 'PR_dB')
            plt.plot(out_x['Sm_p_dB'],alt_v, label = 'Sm_p_dB')
            plt.plot(out_x['Dm_dB'],alt_v, label = 'Dm_dB')        
            
            plt.plot(Will_arr_fg[:,0],alt_v, '--',  color = 'C0')
            plt.plot(Will_arr_fg[:,1],alt_v, '--', color = 'C1')
            plt.plot(Will_arr_fg[:,2],alt_v, '--', color = 'C2')     
            
            plt.plot(out_x['PR_dB'][fl_hydro['melt']],alt_v[fl_hydro['melt']], 'x',  color = 'C0')
            plt.plot(out_x['Sm_p_dB'][fl_hydro['melt']],alt_v[fl_hydro['melt']], 'x', color = 'C1')
            plt.plot(out_x['Dm_dB'][fl_hydro['melt']],alt_v[fl_hydro['melt']], 'x', color = 'C2')  
           
           
            plt.title('retrieval: "-", first guess: "--", melting: "x"')
            plt.legend()
            plt.xlim(-10, 15)
            plt.ylim(0, alt_v[0]+0.25)
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
            plt.ylim(0, alt_v[0]+0.25,)
            plt.xlim(0, 30)
            plt.tight_layout(pad = 0.1)
            plt.savefig(os.path.join(fig_dir,'retr_vs_DPR_nrMS_%d_ns_%d_%s.png' % (
                nrayMS, nscan, save_str)),)
                    
        var_3D_list_x = ['Sm_dB', 'Dm_dB', 'PR_dB', 'WC_dB']
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
            
            
        dpr_obj.MS['Alpha_dB'][nscan,nrayMS] = out_x['Alpha_dB'][-1]
        dpr_obj.MS['CF_n'][nscan,nrayMS] = res.fun/len_Y       
                
                          
         
            