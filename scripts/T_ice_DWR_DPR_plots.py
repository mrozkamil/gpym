#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  15 17:32:16 2021

@author: km357
"""
import os
import numpy as np
import xarray as xr
# from xhistogram.xarray import histogram as xhist
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

# import timeit
from scipy.signal import savgol_filter

home = os.path.expanduser("~")
# sys.path.append(os.path.join(home, 'Documents','Python3', 'gpym'))
# import gpym

import warnings
warnings.filterwarnings("ignore")


def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]
        
def plot_hist(hist, return_mean = False, plot_hist = True):
    
    x_name = 'DWR_bin'
    y_name = 'iceTemperature_bin'
    coords_loops = [f for f in hist.coords if ((f not in coords_plots) 
                                               and (hist[f].size ==1))]
    fn_str = ''
    title_str = ''
    for coord_n in coords_loops:
        fn_str += coord_n.replace('bin', '{:.2f}_'.format(hist[coord_n].values))
        title_str += coord_n.replace('_bin', '={}; '.format(hist[coord_n].values))
    
    
    
    fn_out = os.path.join(out_dir, 'DWR_T_%s.png' % (fn_str[:-1]))
    
            
    CDF = hist.cumsum(dim = x_name
                            )/hist.sum(dim = x_name)
    MEAN = (hist*hist.DWR_bin).sum(dim = x_name
                            )/hist.sum(dim = x_name)
    
    temp_c = hist[y_name].values
    delta_T = np.diff(temp_c)[0]
    
    dMEAN_dT = savgol_filter(MEAN.values, window_length=7,
                              polyorder = 2, deriv =1, delta=delta_T,
                              mode = 'interp')
    
    MEAN = MEAN.where(hist.sum(dim = x_name)>50.)
    dMEAN_dT[np.where(hist.sum(dim = x_name)<=50.)] = np.nan
    
    if ((hist.sum()>10) and plot_hist):
        fig, ax = plt.subplots()        
        hist.plot(x = x_name, y = y_name, 
                      norm = LogNorm(vmin = 1e1), cmap = 'jet', ax = ax,
                      cbar_kwargs = {'pad': 0.02, 'aspect': 30, 'label' : None},)
        
        ax.plot(MEAN, temp_c, label = 'mean',  color = 'k', lw = 3, ls = '--')
        
        ax.plot(dMEAN_dT*10, temp_c, c = '0.20', lw = 2, ls = '-')
        ax.plot(dMEAN_dT*10, temp_c, label = '10 $dDWR/dT$', 
                c = '0.80', lw = 2, ls = '--')
        
        ct = ax.contour(CDF[x_name],CDF[y_name], CDF.T, levels = [0.5,], 
                    colors = 'm', linewidths = 3)
        ct.collections[0].set_label('median')
        ax.grid()
        ax.legend()
        ax.set_title(title_str[:-2])
        ax.set_ylabel('Temperature [C]', fontsize = 16)
        ax.set_xlabel('$DWR_{Ku-Ka}$ [dB]', fontsize = 16)
        ax.set_xlim(-5,20)
        ax.set_ylim(-50,0)
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(fn_out)
    if return_mean:
        return MEAN

out_dir = os.path.join(home, 'Documents','Python3','OUTPUT','gpm', 'DWR_prof')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

fn_out = os.path.join(out_dir, 'histograms_DWR_T_rain_Z_DWR_year_all_v2.nc')

if not os.path.exists(fn_out):
    
    for year in range(2014, 2022):
        fn_in = os.path.join(out_dir, 
                    'histograms_DWR_T_rain_Z_DWR_year_%d_v2.nc' % year)
        if not os.path.exists(fn_in):
            continue
        tmp_dset = xr.open_dataset(fn_in, )
        # print([vn for vn in tmp_dset.variables if
        #                      vn not in tmp_dset.coords])
        
        if not 'dset_hist' in locals():
            print(year)
            dset_hist = tmp_dset*1
            hist_names = [vn for vn in tmp_dset.variables if
                             vn not in tmp_dset.coords]
        else:
            print(year)
            for vname in hist_names:
                dset_hist[vname] += tmp_dset[vname].values*1
                
            print(dset_hist[vname].sum())
        

    encoding = {key: {'dtype': 'int32','compression': 'gzip', } 
                for key in hist_names}
    # dset_hist.to_netcdf(fn_out, engine = 'h5netcdf', encoding= encoding)
else:
    dset_hist = xr.open_dataset(fn_out, )
    


    
coords = list(dset_hist.coords)
hist_names = [f for f in list(dset_hist.variables) if f not in coords]
coords_plots = ['DWR_bin', 'iceTemperature_bin']

# coords_mean = [f for f in hist_tmp.coords if f != 'DWR_bin']
dset_mean = xr.Dataset( coords = dset_hist.coords)

for hist_name in hist_names:
    hist_tmp = dset_hist[hist_name]
    coords_loops = [f for f in hist_tmp.coords if f not in coords_plots]
    if 'DWR_bin' not in hist_tmp.coords:
        continue
   
    for sum_set in powerset(coords_loops):
        print(sum_set)
        sel_set = [f for f in coords_loops if f not in sum_set]
        
        dset_sum = hist_tmp.sum(dim = sum_set)
        
        mean_name = 'histogram_DWR_iceTemperature'
        for var_n in sel_set:
            mean_name += '_' +var_n.replace('_bin','')
        

        
        if len(sel_set)>0:
            
            size_list = [dset_sum[dim_n].size for dim_n in sel_set]
            total_size = np.prod(np.array(size_list))
            
            new_list = [i for i in size_list]
            new_list.append(dset_sum['iceTemperature_bin'].size)
            dim_list = [i for i in sel_set]
            dim_list.append('iceTemperature_bin')
            
            dset_mean[mean_name] = xr.DataArray(np.zeros(new_list), 
                                                dims=dim_list)
            
            ind_lin = 0
            while ind_lin < total_size:
                ind_arr = np.unravel_index(ind_lin, size_list)
                ind_lin +=1
                
                isel_dict = {dim_n: ind_arr[kk] 
                             for kk,dim_n in enumerate(sel_set)}     
                hist_plot = dset_sum.isel(isel_dict)
                
                tmp_mean = plot_hist(hist_plot, return_mean=True,
                                     plot_hist = False)
                plt.close('all')
                
                loc_dict = {dim_n: hist_plot[dim_n].values
                             for dim_n in sel_set}
                dset_mean[mean_name].loc[loc_dict] = tmp_mean
                
        else:
            hist_plot = dset_sum*1.
            tmp_mean = plot_hist(hist_plot, return_mean=True,
                                 plot_hist = False)
            dset_mean[mean_name] = tmp_mean
            

    

for varn in ['histogram_DWR_iceTemperature_DWR_rain', 
             'histogram_DWR_iceTemperature_DWR_rain_SFC_type']:
    
    
    tmp_mean = dset_mean[varn]
    fn_out = os.path.join(out_dir, '{}_mean.png'.format(varn))
    if varn.endswith('SFC_type'):
        sft = 50.
        tmp_mean = dset_mean[varn].sel(SFC_type_bin = sft)
        fn_out = os.path.join(out_dir, '{1}_{0}_mean.png'.format(sft, varn))
        
    fig, ax = plt.subplots()       
    size_all =  dset_mean['DWR_rain_bin'].size
    sm = plt.cm.ScalarMappable(cmap='jet', 
            norm=plt.Normalize(vmin=0., vmax=20.))
    
    for ii in range(size_all):
        ax.plot(tmp_mean.isel(DWR_rain_bin = ii),
                dset_mean[varn]['iceTemperature_bin'],
                color= cm.jet(dset_mean.DWR_rain_bin[ii]/20),)
        
    
    plt.colorbar(sm, aspect = 30, pad =0.02, label = 'DWR rain [dB]')
    ax.grid()
    
    # ax.legend()
    # ax.set_title(title_str[:-2])
    ax.set_ylabel('Temperature [C]', fontsize = 16)
    ax.set_xlabel('$DWR_{Ku-Ka}$ [dB]', fontsize = 16)
    ax.set_xlim(-1,8)
    ax.set_ylim(-50,0)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(fn_out)

            
for varn in ['histogram_DWR_iceTemperature_Z_rain',
             'histogram_DWR_iceTemperature_SFC_type_Z_rain']:
             
    tmp_mean = dset_mean[varn]
    fn_out = os.path.join(out_dir, '{}_mean.png'.format(varn))
    if 'SFC_type' in varn:
        sft = 50.
        tmp_mean = dset_mean[varn].sel(SFC_type_bin = sft)
        fn_out = os.path.join(out_dir, '{1}_{0}_mean.png'.format(sft, varn))
    
    
    fig, ax = plt.subplots()       
    size_all =  dset_mean['Z_rain_bin'].size
    sm = plt.cm.ScalarMappable(cmap='jet', 
            norm=plt.Normalize(vmin=15., vmax=45.))
    for ii in range(size_all):
        ax.plot(tmp_mean.isel(Z_rain_bin = ii),
                dset_mean[varn]['iceTemperature_bin'],
                color= cm.jet((dset_mean.Z_rain_bin[ii]-15)/30),)
    plt.colorbar(sm, aspect = 30, pad =0.02, label = 'Z rain [dBZ]')
    ax.grid()
    # ax.legend()
    # ax.set_title(title_str[:-2])
    ax.set_ylabel('Temperature [C]', fontsize = 16)
    ax.set_xlabel('$DWR_{Ku-Ka}$ [dB]', fontsize = 16)
    ax.set_xlim(-1,8)
    ax.set_ylim(-50,0)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(fn_out)


for varn in hist_names:
    
    hist_tmp = dset_hist[varn]
    
    print(varn)
    # print(hist_tmp.sum().values)
    # print(hist_tmp.max().values)
    if 'Z_Ku_bin' not in hist_tmp.coords:
        continue
    if 'Z_rain' in varn: continue
    coords_plots = ['Z_Ka_bin', 'Z_Ku_bin']
    for t_ice in [-5, -35]:
        hist_plot = hist_tmp.sel(iceTemperature_bin = t_ice, 
                                 method = 'nearest')
        hist_plot = hist_plot.sum(dim = 'SFC_type_bin')
        
        coords_loops = [f for f in hist_plot.coords if f not in coords_plots]
        coords_loops.remove('iceTemperature_bin')
        coord_iter = coords_loops[0]
        for ii in range(hist_plot[coord_iter].size):
            
            h0 = hist_plot.isel({coord_iter:ii})
            if h0.sum()==0:
                continue
            fig, ax = plt.subplots()
            h0.plot(ax =ax, cmap = 'jet', vmin = 1, 
                    norm = LogNorm(vmin =1.),
                    y = 'Z_Ku_bin', cbar_kwargs={'label': ''})
            ax.plot(h0.Z_Ku_bin, h0.Z_Ku_bin, ':k')
            ax.grid()
            
            
            ax.vlines(21, h0.Z_Ku_bin[0], h0.Z_Ku_bin[-1], ls = '--')
            ax.hlines(15, h0.Z_Ku_bin[0], h0.Z_Ku_bin[-1], ls = '--')
            
            hist_ku = h0.sum(dim = 'Z_Ka_bin')
            hist_ka = h0.sum(dim = 'Z_Ku_bin')
            mean_zku = (hist_ku*h0.Z_Ku_bin).sum()/(hist_ku).sum()
            mean_zka = (hist_ka*h0.Z_Ka_bin).sum()/(hist_ka).sum()
            ax.plot(mean_zka, mean_zku, 'ok')
            ax.text(-8,45, 'DWR all: {:.2f} dB'.format(
                (mean_zku-mean_zka).values), fontsize = 14)
            
            
            
            h0 = h0.where(h0.Z_Ku_bin>15)
            h0 = h0.where(h0.Z_Ka_bin>21)
            
            hist_ku = h0.sum(dim = 'Z_Ka_bin')
            hist_ka = h0.sum(dim = 'Z_Ku_bin')
            mean_zku = (hist_ku*h0.Z_Ku_bin).sum()/(hist_ku).sum()
            mean_zka = (hist_ka*h0.Z_Ka_bin).sum()/(hist_ka).sum()
            
            
            ax.plot(mean_zka, mean_zku, 'xk')
            ax.text(-8,40, 'DWR high SNR: {:.2f} dB'.format(
                (mean_zku-mean_zka).values), fontsize = 14)
            
            h0 = h0.where(h0.Z_Ku_bin>21)
            h0 = h0.where(h0.Z_Ka_bin>21)
            
            hist_ku = h0.sum(dim = 'Z_Ka_bin')
            hist_ka = h0.sum(dim = 'Z_Ku_bin')
            mean_zku = (hist_ku*h0.Z_Ku_bin).sum()/(hist_ku).sum()
            mean_zka = (hist_ka*h0.Z_Ka_bin).sum()/(hist_ka).sum()
            
            
            ax.plot(mean_zka, mean_zku, 'dk')
            ax.text(-8,35, 'DWR Z>21: {:.2f} dB'.format(
                (mean_zku-mean_zka).values), fontsize = 14)
            
          
            ax.set_ylabel('Z Ku [dBZ]', fontsize = 16)
            ax.set_xlabel('Z Ka [dBZ]', fontsize = 16)
            tmp_t = ax.get_title()
            tmp_t = tmp_t.replace('_bin','').replace('_', ' ')
            ax.set_title(tmp_t, fontsize = 16)
            # ax.set_xlim(-5,35)
            # ax.set_ylim(-5,35)
            plt.tight_layout()
            
            tmp_t = tmp_t.replace(' ', '_',).replace('.','p').replace(
                '_=_','').replace(',','')
            fn_out = os.path.join(out_dir, 
                                  '{0}_{1}.png'.format(varn, tmp_t))
            fig.savefig(fn_out)
           
          
           
          
dwr_da = dset_hist.Z_Ku_bin - dset_hist.Z_Ka_bin 
dwr_da.name = 'dwr'
varn = 'histogram_Z_Ku_Z_Ka_iceTemperature_DWR_rain_SFC_type'

hist_tmp = dset_hist[varn]
hist_plot = hist_tmp.sum(dim = 'SFC_type_bin')
for Z_thr in range(12,24):
    hist_plot = hist_plot.where(dset_hist.Z_Ku_bin>Z_thr)
    hist_plot = hist_plot.where(dset_hist.Z_Ka_bin>Z_thr)
    
    tmp_dwr = (hist_plot*dwr_da).sum(dim = ['Z_Ka_bin', 'Z_Ku_bin'])/(
        hist_plot.sum(dim = ['Z_Ka_bin', 'Z_Ku_bin']))
    tmp_dwr.name = 'mean_DWR_iceTemperature_Z_larger21'
    
    fn_out = os.path.join(out_dir, '{}_{:.1f}.png'.format(tmp_dwr.name, Z_thr))
    
    h_m40 = hist_plot.sel(iceTemperature_bin = -30, method = 'nearest')
    counts_m40 = h_m40.sum(dim = ['Z_Ka_bin', 'Z_Ku_bin'])
    
    fig, ax = plt.subplots()       
    size_all =  tmp_dwr['DWR_rain_bin'].size
    sm = plt.cm.ScalarMappable(cmap='jet', 
            norm=plt.Normalize(vmin=0., vmax=20.))
    for ii in range(size_all):
        if counts_m40[ii]>1e3:
            ax.plot(tmp_dwr.isel(DWR_rain_bin = ii),
                tmp_dwr['iceTemperature_bin'],
                color= cm.jet(tmp_dwr.DWR_rain_bin[ii]/20),)
    ax.text(4, -40, 'Z>{:.0f} dBZ'.format(Z_thr), fontsize = 16, 
            fontweight = 'bold')
    
    plt.colorbar(sm, aspect = 30, pad =0.02, label = 'DWR rain [dB]')
    ax.grid()
    
    # ax.legend()
    # ax.set_title(title_str[:-2])
    ax.set_ylabel('Temperature [C]', fontsize = 16)
    ax.set_xlabel('$DWR_{Ku-Ka}$ [dB]', fontsize = 16)
    ax.set_xlim(-1,8)
    ax.set_ylim(-50,0)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(fn_out)



