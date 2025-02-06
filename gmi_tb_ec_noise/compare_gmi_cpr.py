import os
import numpy as np
import xarray as xr
from xhistogram.xarray import histogram as xhist
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



def hist_2_mean_var(xr_hist, dim = None):
    if dim is None:
        dim = list(xr_hist.coords)[0].replace('_bin', '')
    coord = dim + '_bin'
    weights = xr_hist.sum(dim = coord)
    weights_inv = 1/weights
    mean_val = (xr_hist * xr_hist[coord]).sum(dim = coord)
    mean_val *= weights_inv
    var_value = (xr_hist * xr_hist[coord]**2).sum(dim = coord)
    var_value *= weights_inv
    var_value += -mean_val**2
    return mean_val, var_value, weights

fig_dir = '/home/k/km357/Data/EarthCARE/figs/'

tb_files = [fn for fn in os.listdir(fig_dir) 
            if fn.startswith('EC_TB') and fn.endswith('ECMWF_sim.nc')]

tb_bins = np.arange(150,350,1)
tb_bins2 = np.arange(150,350,1)
noise_bins = np.arange(-147,-144,0.005)

delta_noise_bins = np.arange(-0.3,0.5,0.005)
std_noise_bins = np.arange(0,1.0,0.001)
delta_tb_bins = np.arange(-50,100,1)

tb_bias_bins = np.arange(-50,50,0.1)

dset_out = xr.Dataset()
for ind_file,tb_file in enumerate(tb_files):
    print(tb_file)
    tmp_ds = xr.open_dataset(os.path.join(fig_dir, tb_file))
    if 'naray' in tmp_ds.dims:
        tmp_ds = tmp_ds.swap_dims({'naray': 'nray'})

    tmp_ds['noise_l2_lin'] = ('nray', np.power(10.0, 0.1*tmp_ds['noisel2_dbm'].values))
    tmp_ds['noise_l2_smooth_dbm'] = 10*np.log10(
        tmp_ds['noise_l2_lin'].rolling(nray=15, center=True).mean())
    
    tmp_ds['noise_l2_dbm_smooth'] = tmp_ds['noisel2_dbm'].rolling(nray=15, center=True).mean()
    tmp_ds['noise_l2_dbm_dev'] = (tmp_ds['noisel2_dbm'] - tmp_ds['noise_l2_dbm_smooth'])**2
    tmp_ds['noise_l2_dbm_dloc_std'] = np.sqrt(tmp_ds['noise_l2_dbm_dev'].rolling(nray=15, center=True).mean())
    tmp_ds['clear_sky_flag_7km'] = tmp_ds['clear_sky_flag'].rolling(nray=15, center=False).mean()

    da1 = tmp_ds['clear_sky_flag'].rolling(nray=25, center=False).mean()
    da2 = (tmp_ds['clear_sky_flag'][::-1].rolling(nray=25, center=False).mean())[::-1]


    tmp_ds['clear_sky_flag_smooth'] = da1 * da2

    # plt.figure()
    # tmp_ds['clear_sky_flag_smooth'].plot()
    # tmp_ds['clear_sky_flag'].plot()
    # plt.savefig(os.path.join(fig_dir, 'tmp_plot.png'))


    ind_clear = np.where(tmp_ds["clear_sky_flag"]==1)[0]
    ind_not_clear = np.where(~(tmp_ds["clear_sky_flag"]==1))[0]

    tmp_ds['noise_l2_dbm_clear'] =  tmp_ds['noisel2_dbm']*1.0
    tmp_ds['noise_l2_dbm_clear'][ind_not_clear] = np.interp(ind_not_clear, 
                                                            ind_clear, 
                                                            tmp_ds['noise_l2_dbm_smooth'][ind_clear])

    tmp_ds['delta_noise_dbm'] = tmp_ds['noisel2_dbm'] -  tmp_ds['noise_l2_dbm_clear']
    tmp_ds['delta_noise_dbm_close'] =  tmp_ds['delta_noise_dbm'].where(tmp_ds['clear_sky_flag_smooth']>0.0)

    tmp_ds['delta_noise_dbm_smooth'] = tmp_ds['noise_l2_dbm_smooth'] -  tmp_ds['noise_l2_dbm_clear']
    tmp_ds['delta_noise_dbm_close_smooth'] =  tmp_ds['delta_noise_dbm_smooth'].where(tmp_ds['clear_sky_flag_smooth']>0.0)




    flag_good = (tmp_ds.dist_tb_km<10) & (tmp_ds.dist_tb_s<60*5)
    if flag_good.sum()==0:
        continue
    ind_good = np.where(flag_good)[0]
    tmp_ds = tmp_ds.isel(nray=ind_good)

    flag_clear = (tmp_ds["land_flag"]==1) & (tmp_ds["clear_sky_flag"]==1)


    for x_name in ['tb_89_v', 'tb_89_h',  ]:
        for y_name in ['noise_dbm', 'noisel2_dbm',]:
            if y_name in tmp_ds.variables:
                tb_hist = xhist(tmp_ds[x_name], tmp_ds[y_name], 
                                bins=[tb_bins, noise_bins],
                                block_size = 1000,) # type: ignore
                if tb_hist.name in dset_out:
                    dset_out[tb_hist.name] += tb_hist
                else:
                    dset_out[tb_hist.name] = tb_hist

    if flag_clear.sum()==0:
        continue    
    

    tb_calib = (tmp_ds["tb_89_v"]- tmp_ds["tbs_89_v"]).where(flag_clear).rolling(nray=501, center=True, min_periods = 50).median()
    # plt.figure()
    # (tmp_ds["tb_89_v"]- tmp_ds["tbs_89_v"]).plot()
    # (tmp_ds["tb_89_v"]- tmp_ds["tbs_89_v"]).where(flag_clear).plot()
    # tb_calib.plot()
    # plt.savefig(os.path.join(fig_dir, 'tmp_hist.png'))

    tmp_ds["tbs_calib_89_v"] = tmp_ds["tbs_89_v"] + tb_calib
    tmp_ds["delta_tb_89_v"] = tmp_ds["tb_89_v"] - tmp_ds["tbs_calib_89_v"]
    tmp_ds["delta_tb_89_v"] = tmp_ds["delta_tb_89_v"].where(tmp_ds["land_flag"]==1)


    tb_calib = (tmp_ds["tb_89_h"]- tmp_ds["tbs_89_h"]).where(flag_clear).rolling(nray=501, center=True, min_periods = 50).median()
    tmp_ds["tbs_calib_89_h"] = tmp_ds["tbs_89_h"] + tb_calib
    tmp_ds["delta_tb_89_h"] = tmp_ds["tb_89_h"] - tmp_ds["tbs_calib_89_h"]
    tmp_ds["delta_tb_89_h"] = tmp_ds["delta_tb_89_h"].where(tmp_ds["land_flag"]==1)

    tmp_ds['tbv_sim_bias'] = tmp_ds['tb_89_v'] - tmp_ds['tbs_89_v']
    tmp_ds['tbh_sim_bias'] = tmp_ds['tb_89_h'] - tmp_ds['tbs_89_h']

    
    

    for x_name, y_name  in zip([ 'tb_89_v', 'tb_89_h', 'tb_89_v', 'tb_89_h','tbs_89_v_edd', 'tbs_89_h_edd', ], 
                               [ 'tbs_89_v_edd', 'tbs_89_h_edd','tbs_89_v', 'tbs_89_h','tbs_89_v', 'tbs_89_h',]):
         
        tb_hist = xhist(tmp_ds[x_name][flag_clear], tmp_ds[y_name][flag_clear], 
                        bins=[tb_bins, tb_bins2],
                        block_size = 1000,) # type: ignore

        if tb_hist.name in dset_out:
            dset_out[tb_hist.name] += tb_hist
        else:
            dset_out[tb_hist.name] = tb_hist

    for x_name, y_name  in zip(['delta_tb_89_h', 'delta_tb_89_v','delta_tb_89_h', 'delta_tb_89_v', 
                                'delta_tb_89_h', 'delta_tb_89_v','delta_tb_89_h', 'delta_tb_89_v'], 
                               ['delta_noise_dbm', 'delta_noise_dbm', 'delta_noise_dbm_close', 'delta_noise_dbm_close',
                                'delta_noise_dbm_smooth', 'delta_noise_dbm_smooth', 'delta_noise_dbm_close_smooth', 'delta_noise_dbm_close_smooth']):
         
        tb_hist = xhist(tmp_ds[x_name][~(tmp_ds["clear_sky_flag"]==1)], tmp_ds[y_name][~(tmp_ds["clear_sky_flag"]==1)], 
                        bins=[delta_tb_bins, delta_noise_bins],
                        block_size = 1000,) # type: ignore

        if tb_hist.name in dset_out:
            dset_out[tb_hist.name] += tb_hist
        else:
            dset_out[tb_hist.name] = tb_hist

    for x_name   in [ 'tbv_sim_bias', 'tbh_sim_bias' ]:
         
        tb_hist = xhist(tmp_ds[x_name][flag_clear],
                        bins=[tb_bias_bins,],
                        block_size = 1000,) # type: ignore

        if tb_hist.name in dset_out:
            dset_out[tb_hist.name] += tb_hist
        else:
            dset_out[tb_hist.name] = tb_hist

    for x_name in [ 'noise_l2_dbm_dloc_std', ]:
        flag_all_clear = tmp_ds['clear_sky_flag_7km']>=0.999

        tb_hist = xhist(tmp_ds[x_name][flag_all_clear],
                        bins=[std_noise_bins,],
                        block_size = 1000,) # type: ignore

        if tb_hist.name in dset_out:
            dset_out[tb_hist.name] += tb_hist
        else:
            dset_out[tb_hist.name] = tb_hist

    # if ind_file>100:
    #     aaaa
    # plt.figure()
    # plt.scatter(tmp_ds.latitude, tmp_ds.delta_noise_dbm*50.0, label = r'50 x $\Delta$noise [dB]')
    # plt.scatter(tmp_ds.latitude, tmp_ds.delta_tb_89_h, label = r'$\Delta$ TB89H [K]')
    # plt.scatter(tmp_ds.latitude, tmp_ds.delta_tb_89_v, label = r'$\Delta$ TB89V [K]')
    # plt.xlabel('Latitude')
    # plt.grid()
    # plt.legend()
    # plt.savefig(os.path.join(fig_dir, tb_file.replace('.nc', '_delta_tb.png')))
    # plt.close('all')
   
dset_out.to_netcdf(os.path.join(fig_dir, 'histograms_EC_GMI_collocations.nc'))

dset_out = xr.open_dataset(os.path.join(fig_dir, 'histograms_EC_GMI_collocations.nc'))


ds_plot = dset_out['histogram_tbv_sim_bias']
noise_mean, noise_var,_ = hist_2_mean_var(ds_plot, dim = 'tbv_sim_bias')
print(noise_mean, noise_var)

ds_plot = dset_out['histogram_tbh_sim_bias']
noise_mean, noise_var,_ = hist_2_mean_var(ds_plot, dim = 'tbh_sim_bias')
print(noise_mean, noise_var)

ds_plot = dset_out['histogram_noise_l2_dbm_dloc_std']
noise_mean, noise_var,_ = hist_2_mean_var(ds_plot, dim = 'noise_l2_dbm_dloc_std')
print(noise_mean, noise_var)

plt.figure()
ds_plot.plot()
plt.savefig(os.path.join(fig_dir, 'tmp_hist.png'))
    
for hist_name in dset_out.data_vars:
    fig, ax = plt.subplots()
    dset_out[hist_name].plot(ax = ax) # type: ignore
    ax.plot(tb_bins, tb_bins, 'k--')
    fig.savefig(os.path.join(fig_dir, f'EC_TB_hist_{hist_name}.png'))

    fig, ax = plt.subplots()
    (np.log10(dset_out[hist_name])).plot(ax = ax) # type: ignore
    ax.plot(tb_bins, tb_bins, 'k--')
    fig.savefig(os.path.join(fig_dir, f'EC_TB_log_hist_{hist_name}.png'))
    plt.close('all')



hist_name = 'histogram_tb_89_v_noisel2_dbm'
ds_plot = dset_out[hist_name]
noise_mean, noise_var,_ = hist_2_mean_var(ds_plot, dim = 'noisel2_dbm')

fig, ax = plt.subplots()
(ds_plot).plot(ax = ax, cmap = 'plasma', cbar_kwargs = {"fraction": 0.05, "aspect": 40, 'label': '', 'pad': 0.02}, 
               norm = LogNorm(vmin = 5.)) # type: ignore
ax.grid()
ax.plot(noise_mean, noise_mean.tb_89_v_bin, lw =3, color = 'k')
ax.plot(noise_mean- np.sqrt(noise_var), noise_mean.tb_89_v_bin, lw =1, color = 'k')
ax.plot(noise_mean+ np.sqrt(noise_var), noise_mean.tb_89_v_bin, lw =1, color = 'k')
ax.set_xlabel('Noise level CPR [dBm]' )
ax.set_ylabel('GMI TB$_V$ 89 GHz [K]' )
ax.set_xlim(-146.5, -144.8)
ax.set_ylim(225, 290)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'EC_hist_TB89V_NoiseL2.png'))


tmp_bins = np.arange(-3.,3.,0.005)
tmp_hist = np.zeros_like(tmp_bins)

length_h = ds_plot.shape[1]
for ii in range(ds_plot.shape[0]):
    if np.isfinite(noise_mean[ii]):
        ind_0 = np.argmin(np.abs(tmp_bins- (ds_plot.noisel2_dbm_bin[0]- noise_mean[ii]).values))    
        tmp_hist[ind_0:ind_0+length_h]+= ds_plot[ii].values

tmp_hist_1D = xr.DataArray(data = tmp_hist, coords = {'noise_bin': tmp_bins})
noise_mean_reduced, noise_var_reduced,_ = hist_2_mean_var(tmp_hist_1D, dim = 'noise')

fig, ax = plt.subplots()
ds_1d = ds_plot.sum(dim = 'tb_89_v_bin')
noise_mean_1d, noise_var_1d,_ = hist_2_mean_var(ds_1d, dim = 'noisel2_dbm')
ax.plot(ds_1d.noisel2_dbm_bin-noise_mean_1d, ds_1d,  lw = 3, label = 'overall mean', color = 'k')
ax.plot(tmp_bins, tmp_hist, lw = 3, label = 'TB dependent mean')
ax.text(0.7, 0.9, "std = %.2f" % np.sqrt(noise_var_1d.values.item()), transform = ax.transAxes, color = 'k', fontsize = 16)
ax.text(0.7, 0.8, "std = %.2f" % np.sqrt(noise_var_reduced.values.item()),transform = ax.transAxes, color = 'C0', fontsize = 16)
ax.set_ylabel('counts' )
ax.set_xlabel('CPR noise variability [dB]' )
ax.set_xlim(-1,1)
ax.grid()
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'EC_hist_TB89V_1D.png'))



hist_name = 'histogram_tb_89_h_noisel2_dbm'
ds_plot = dset_out[hist_name]
noise_mean, noise_var,_ = hist_2_mean_var(ds_plot, dim = 'noisel2_dbm')

fig, ax = plt.subplots()
(ds_plot).plot(ax = ax, cmap = 'plasma', cbar_kwargs = {"fraction": 0.05, "aspect": 40, 'label': '', 'pad': 0.02}, 
               norm = LogNorm(vmin = 5.)) # type: ignore
ax.grid()
ax.plot(noise_mean, noise_mean.tb_89_h_bin, lw =3, color = 'k')
ax.plot(noise_mean- np.sqrt(noise_var), noise_mean.tb_89_h_bin, lw =1, color = 'k')
ax.plot(noise_mean+ np.sqrt(noise_var), noise_mean.tb_89_h_bin, lw =1, color = 'k')
ax.set_xlabel('Noise level CPR [dBm]' )
ax.set_ylabel('GMI TB$_H$ 89 GHz [K]' )
ax.set_xlim(-146.5, -144.8)
ax.set_ylim(175, 290)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'EC_hist_TB89H_NoiseL2.png'))


tmp_bins = np.arange(-3.,3.,0.005)
tmp_hist = np.zeros_like(tmp_bins)

length_h = ds_plot.shape[1]
for ii in range(ds_plot.shape[0]):
    if np.isfinite(noise_mean[ii]):
        ind_0 = np.argmin(np.abs(tmp_bins- (ds_plot.noisel2_dbm_bin[0]- noise_mean[ii]).values))    
        tmp_hist[ind_0:ind_0+length_h]+= ds_plot[ii].values

tmp_hist_1D = xr.DataArray(data = tmp_hist, coords = {'noise_bin': tmp_bins})
noise_mean_reduced, noise_var_reduced,_ = hist_2_mean_var(tmp_hist_1D, dim = 'noise')

fig, ax = plt.subplots()
ds_1d = ds_plot.sum(dim = 'tb_89_h_bin')
noise_mean_1d, noise_var_1d,_ = hist_2_mean_var(ds_1d, dim = 'noisel2_dbm')
ax.plot(ds_1d.noisel2_dbm_bin-noise_mean_1d, ds_1d,  lw = 3, label = 'overall mean', color = 'k')
ax.plot(tmp_bins, tmp_hist, lw = 3, label = 'TB dependent mean', color = 'C0')
ax.text(0.7, 0.9, "std = %.2f" % np.sqrt(noise_var_1d.values.item()), transform = ax.transAxes, color = 'k', fontsize = 16)
ax.text(0.7, 0.8, "std = %.2f" % np.sqrt(noise_var_reduced.values.item()),transform = ax.transAxes, color = 'C0', fontsize = 16)
ax.set_ylabel('counts' )
ax.set_xlabel('CPR noise variability [dB]' )
ax.set_xlim(-1,1)
ax.grid()
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'EC_hist_TB89H_1D.png'))


hist_name = 'histogram_delta_tb_89_h_delta_noise_dbm'
ds_plot = dset_out[hist_name]
noise_mean, noise_var,_ = hist_2_mean_var(ds_plot, dim = 'delta_noise_dbm')


fig, ax = plt.subplots()
(ds_plot).plot(ax = ax, cmap = 'plasma', cbar_kwargs = {"fraction": 0.05, "aspect": 40, 'label': '', 'pad': 0.02}, 
               norm = LogNorm(vmin = 5.)) # type: ignore
ax.grid()
ax.plot(noise_mean, noise_mean.delta_tb_89_h_bin, lw =3, color = 'k')
ax.plot(noise_mean- np.sqrt(noise_var), noise_mean.delta_tb_89_h_bin, lw =1, color = 'k')
ax.plot(noise_mean+ np.sqrt(noise_var), noise_mean.delta_tb_89_h_bin, lw =1, color = 'k')
ax.set_xlabel(r'$\Delta$ Noise 94 GHz[dB]' )
ax.set_ylabel(r'$\Delta TB_H$ 89 GHz [K]' )
ax.set_xlim(-0.3, 0.4)
ax.set_ylim(-10, 50)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'EC_hist_deltaTB89H_deltaNoiseL2.png'))



hist_name = 'histogram_delta_tb_89_v_delta_noise_dbm'
ds_plot = dset_out[hist_name]
noise_mean, noise_var,_ = hist_2_mean_var(ds_plot, dim = 'delta_noise_dbm')


fig, ax = plt.subplots()
(ds_plot).plot(ax = ax, cmap = 'plasma', cbar_kwargs = {"fraction": 0.05, "aspect": 40, 'label': '', 'pad': 0.02}, 
               norm = LogNorm(vmin = 5.)) # type: ignore
ax.grid()
ax.plot(noise_mean, noise_mean.delta_tb_89_v_bin, lw =3, color = 'k')
ax.plot(noise_mean- np.sqrt(noise_var), noise_mean.delta_tb_89_v_bin, lw =1, color = 'k')
ax.plot(noise_mean+ np.sqrt(noise_var), noise_mean.delta_tb_89_v_bin, lw =1, color = 'k')
ax.set_xlabel(r'$\Delta$ Noise 94 GHz[dB]' )
ax.set_ylabel(r'$\Delta TB_V$ 89 GHz [K]' )
ax.set_xlim(-0.3, 0.4)
ax.set_ylim(-10, 20)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'EC_hist_deltaTB89V_deltaNoiseL2.png'))


hist_name = 'histogram_delta_tb_89_v_delta_noise_dbm_close'
ds_plot = dset_out[hist_name]
noise_mean, noise_var,_ = hist_2_mean_var(ds_plot, dim = 'delta_noise_dbm_close')


fig, ax = plt.subplots()
(ds_plot).plot(ax = ax, cmap = 'plasma', cbar_kwargs = {"fraction": 0.05, "aspect": 40, 'label': '', 'pad': 0.02}, 
               norm = LogNorm(vmin = 5.)) # type: ignore
ax.grid()
ax.plot(noise_mean, noise_mean.delta_tb_89_v_bin, lw =3, color = 'k')
ax.plot(noise_mean- np.sqrt(noise_var), noise_mean.delta_tb_89_v_bin, lw =1, color = 'k')
ax.plot(noise_mean+ np.sqrt(noise_var), noise_mean.delta_tb_89_v_bin, lw =1, color = 'k')
ax.set_xlabel(r'$\Delta$ Noise 94 GHz[dB]' )
ax.set_ylabel(r'$\Delta TB_V$ 89 GHz [K]' )
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-10, 20)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'EC_hist_deltaTB89V_deltaNoiseL2_close.png'))


hist_name = 'histogram_delta_tb_89_h_delta_noise_dbm_close'
ds_plot = dset_out[hist_name]
noise_mean, noise_var,_ = hist_2_mean_var(ds_plot, dim = 'delta_noise_dbm_close')


fig, ax = plt.subplots()
(ds_plot).plot(ax = ax, cmap = 'plasma', cbar_kwargs = {"fraction": 0.05, "aspect": 40, 'label': '', 'pad': 0.02}, 
               norm = LogNorm(vmin = 5.)) # type: ignore
ax.grid()
ax.plot(noise_mean, noise_mean.delta_tb_89_h_bin, lw =3, color = 'k')
ax.plot(noise_mean- np.sqrt(noise_var), noise_mean.delta_tb_89_h_bin, lw =1, color = 'k')
ax.plot(noise_mean+ np.sqrt(noise_var), noise_mean.delta_tb_89_h_bin, lw =1, color = 'k')
ax.set_xlabel(r'$\Delta$ Noise 94 GHz[dB]' )
ax.set_ylabel(r'$\Delta TB_H$ 89 GHz [K]' )
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-10, 30)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'EC_hist_deltaTB89H_deltaNoiseL2_close.png'))


hist_name = 'histogram_delta_tb_89_v_delta_noise_dbm_close_smooth'
ds_plot = dset_out[hist_name]
noise_mean, noise_var,_ = hist_2_mean_var(ds_plot, dim = 'delta_noise_dbm_close_smooth')


fig, ax = plt.subplots()
(ds_plot).plot(ax = ax, cmap = 'plasma', cbar_kwargs = {"fraction": 0.05, "aspect": 40, 'label': '', 'pad': 0.02}, 
               norm = LogNorm(vmin = 5.)) # type: ignore
ax.grid()
ax.plot(noise_mean, noise_mean.delta_tb_89_v_bin, lw =3, color = 'k')
ax.plot(noise_mean- np.sqrt(noise_var), noise_mean.delta_tb_89_v_bin, lw =1, color = 'k')
ax.plot(noise_mean+ np.sqrt(noise_var), noise_mean.delta_tb_89_v_bin, lw =1, color = 'k')
ax.set_xlabel(r'$\Delta$ Noise 94 GHz[dB]' )
ax.set_ylabel(r'$\Delta TB_V$ 89 GHz [K]' )
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-10, 20)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'EC_hist_deltaTB89V_deltaNoiseL2_close_smooth.png'))


hist_name = 'histogram_delta_tb_89_h_delta_noise_dbm_close_smooth'
ds_plot = dset_out[hist_name]
noise_mean, noise_var,_ = hist_2_mean_var(ds_plot, dim = 'delta_noise_dbm_close_smooth')


fig, ax = plt.subplots()
(ds_plot).plot(ax = ax, cmap = 'plasma', cbar_kwargs = {"fraction": 0.05, "aspect": 40, 'label': '', 'pad': 0.02}, 
               norm = LogNorm(vmin = 5.)) # type: ignore
ax.grid()
ax.plot(noise_mean, noise_mean.delta_tb_89_h_bin, lw =3, color = 'k')
ax.plot(noise_mean- np.sqrt(noise_var), noise_mean.delta_tb_89_h_bin, lw =1, color = 'k')
ax.plot(noise_mean+ np.sqrt(noise_var), noise_mean.delta_tb_89_h_bin, lw =1, color = 'k')
ax.set_xlabel(r'$\Delta$ Noise 94 GHz[dB]' )
ax.set_ylabel(r'$\Delta TB_H$ 89 GHz [K]' )
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-10, 30)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'EC_hist_deltaTB89H_deltaNoiseL2_close_smooth.png'))