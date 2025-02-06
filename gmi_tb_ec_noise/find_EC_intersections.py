import sys
import os
home_dir = os.path.expanduser("~")
sys.path.append(os.path.join(home_dir, 'Scripts', 'Python3','gpym'))
from gpym.GPM import GPM

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import xarray as xr
import pandas as pd
import datetime as dt
import re
# from scipy.spatial import cKDTree 
from sklearn.neighbors import BallTree

def print_deg(deg, pretty_print='longitude', ndp=1):
    if pretty_print=='latitude':
        hemi = 'N' if deg>=0 else 'S'
    elif pretty_print=='longitude':
        hemi = 'E' if deg>=0 else 'W'
    else:
        hemi = '?'
    return '{d:.{ndp:d}f}°{hemi:1s}'.format(
                d=np.abs(deg), hemi=hemi, ndp=ndp)
    


Earth_R = 6372800  # Earth radius in meters


yy = 2024
gmi_data_dir = '/home/k/km357/Data/GPM/V07/%04d/%02d/%02d/1C'
cpr_data_dir = '/home/k/km357/Data/EarthCARE/CPR_NOM_1B/%04d/%02d/%02d'
tb_data_dir = "/home/k/km357/Data/EarthCARE/CPR_TB_2B/"

# cpr_l2data_dir = '/home/k/km357/Data/EarthCARE/CPR_FMR_2B'
# l2_dirs = [fn for fn in os.listdir(cpr_l2data_dir) if fn.startswith('ECA_EXAA_CPR_FMR_2A')]
# l2_dirs.sort()

fig_dir = "/home/k/km357/Data/EarthCARE/figs"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

ind_figure = 0
if len(sys.argv)>1:
    dd = int(sys.argv[1])
else:
    dd = 6

for mm in range(8,13):    
        # for dd in range(1,32):
        print('%04d-%02d-%02d' % (yy, mm, dd))
        try:
            date_ref = dt.datetime(yy,mm,dd,0,0,0)
        except:
            continue
        

        gmi_dir = gmi_data_dir % (yy, mm, dd)
        if not os.path.exists(gmi_dir):
            continue   

        gmi_files = [fn for fn in os.listdir(gmi_dir) if fn.startswith('1C.GPM.GMI.XCAL') and fn.endswith('.HDF5') ]
        gmi_files.sort()

        cpr_tmp_dirs = []
        cpr_s_time = []
        cpr_fns = []
        cpr_s_time_dt = []

        for date_shift in [-1, 0, 1]:
            tmp_date = date_ref + dt.timedelta(days = date_shift)

            _yy = tmp_date.year
            _mm = tmp_date.month
            _dd = tmp_date.day

            cpr_dir = cpr_data_dir % (_yy, _mm, _dd)
            if not os.path.exists(cpr_dir):
                continue      
            
            _cpr_tmp_dirs = [fn for fn in os.listdir(cpr_dir) if ('CPR_NOM_1B' in fn) and 
                                re.search(r'.%04d%02d%02dT(\d{6})Z' % (_yy, _mm, _dd), fn) and
                                not fn.endswith('.ZIP') ]
            
            _cpr_tmp_dirs.sort()
            for ii_cpr, cpr_tmp_dir in enumerate(_cpr_tmp_dirs):
                if ii_cpr<len(_cpr_tmp_dirs)-1:
                    if cpr_tmp_dir[-6:] == _cpr_tmp_dirs[ii_cpr+1][-6:]:
                        _cpr_tmp_dirs.remove(cpr_tmp_dir)

            cpr_tmp_dirs += _cpr_tmp_dirs
            cpr_fns += [os.path.join(cpr_dir, cpr_tmp_dir, cpr_tmp_dir +'.h5') for cpr_tmp_dir in _cpr_tmp_dirs]

           
            _cpr_s_time = [re.search(r'.%04d%02d%02dT(\d{6})Z' % (_yy, _mm, _dd), cpr_tmp_dir).group(1) # type: ignore
                      for cpr_tmp_dir in _cpr_tmp_dirs]
        
            cpr_s_time_dt += [dt.datetime.strptime('%04d%02d%02d' % (_yy, _mm, _dd) + tmp_time , '%Y%m%d%H%M%S')
                         for tmp_time in _cpr_s_time]
            

        cpr_duration = np.median(np.diff(np.array(cpr_s_time_dt)))
       
        for gmi_file in gmi_files:
            m = re.search(r'.%04d%02d%02d-S(\d{6})-E(\d{6}).' % (yy, mm, dd), gmi_file)            
            if m is None:
                continue

            

            gmi_orbit = gmi_file.split('.')[-3]

            if gmi_orbit not in ['059406']:
                continue

            gmi_s_time = m.group(1)
            gmi_e_time = m.group(2)
            gmi_s_time_dt = dt.datetime.strptime('%04d%02d%02d' % (yy, mm, dd) + gmi_s_time, '%Y%m%d%H%M%S')
            gmi_e_time_dt = dt.datetime.strptime('%04d%02d%02d' % (yy, mm, dd) + gmi_e_time, '%Y%m%d%H%M%S')
            if  gmi_s_time_dt > gmi_e_time_dt:
                gmi_e_time_dt += dt.timedelta(days = 1)

            ind_cpr = np.where((np.array(cpr_s_time_dt) + dt.timedelta(seconds = 5*60) + cpr_duration  >= gmi_s_time_dt) & 
                               (np.array(cpr_s_time_dt) - dt.timedelta(seconds = 5*60)<= gmi_e_time_dt))[0]
            
            if len(ind_cpr) == 0:
                continue
            
            gmi_ds = GPM(os.path.join(gmi_dir, gmi_file),['S1',])
            gmi_coords = np.deg2rad(np.column_stack((gmi_ds["S1"].SClatitude.values, gmi_ds["S1"].SClongitude.values, )))
            # tree = cKDTree(gmi_coords)
            btree = BallTree(gmi_coords, metric='haversine')
            
            
            ind_figure += 1
           
            
            
            for ii_cpr in ind_cpr:
                
                cpr_fn = cpr_fns[ii_cpr]
                tmp_fn =  cpr_fn.split('/')[-1]
                orbit_id = tmp_fn[-9:-3]
                # cpr_l2_dir = [fn for fn in l2_dirs if fn.endswith(orbit_id) ]
                # if len(cpr_l2_dir)>=1:
                #     cpr_l2_fn = os.path.join(cpr_l2data_dir, cpr_l2_dir[0], 
                #                              cpr_l2_dir[0] + '.h5')
                    
                    

                
                try:

                    cpr_ds = xr.open_dataset(cpr_fn, group = '/ScienceData/Geo', engine = 'h5netcdf', phony_dims='sort')
                    dir_tb = os.path.join('/', *cpr_fn.split('/')[:-2]).replace('CPR_NOM_1B', 'CPR_TB_2B')
                    fns_tb = [os.path.join(dir_tb, fn) for fn in os.listdir(dir_tb) if fn.endswith(orbit_id + '.h5') ]
                    if len(fns_tb)>=1:
                        # cpr_l2_ds = xr.open_dataset(
                        #     cpr_l2_fn, group = '/ScienceData', 
                        #     engine = 'h5netcdf', phony_dims='sort')
                        
                        tb_ds_data = xr.open_dataset(
                            fns_tb[-1], group = '/', 
                            engine = 'h5netcdf', phony_dims='sort')
                    else:
                        print('no tb data for %s' % orbit_id)
                        continue                  

                    
                    query = np.deg2rad(np.column_stack((cpr_ds.latitude.values, cpr_ds.longitude.values, )))

                    # dist, ind = tree.query(query, k = 1)
                    dist, ind = btree.query(query, k = 1)
                    dist = np.ravel(dist)
                    ind = np.ravel(ind)
                    

                except Exception as ex:
                    print(ex)
                    continue
                
                if np.min(dist) < np.deg2rad(4.0):
                    
                    ind_min = np.argmin(dist)

                    time_diff = ((cpr_ds.profileTime.values[ind_min] - gmi_ds["S1"].time.values[ind[ind_min]])/
                        np.timedelta64(1, 's'))
                    
                    if np.abs(time_diff) < 15*60:                          
                        
                        
                        sl = slice(np.maximum(0,ind[ind_min]-100), np.minimum(gmi_ds["S1"].time.size, ind[ind_min]+100))
                        
                        lat_mat = gmi_ds["S1"].Latitude.values[sl]
                        lon_mat = gmi_ds["S1"].Longitude.values[sl]
                        tb_89_mat = gmi_ds["S1"].Tc[sl,:,-2]

                        

                        time_mat = np.tile(np.reshape(gmi_ds["S1"].time.values[sl], (tb_89_mat.shape[0], 1)), (1, 221))
                      

                        tmp_shape = lat_mat.shape

                        gmi_coord_tb = np.deg2rad(np.column_stack(
                            (np.ravel(lat_mat), 
                             np.ravel(lon_mat),)))
                        
                        btree_tb = BallTree(gmi_coord_tb, metric='haversine')

                        dist_tb, ind_tb = btree_tb.query(query, k = 1)
                        dist_tb = np.ravel(dist_tb)
                        ind_tb = np.ravel(ind_tb)
                        
                        dist_tb_km = dist_tb*Earth_R*1e-3
                        dist_tb_s = ((np.ravel(time_mat))[ind_tb] - cpr_ds.profileTime.values)/np.timedelta64(1, 's')

                        tb_nearest = (np.ravel(tb_89_mat))[ind_tb]  
                        flag_good =  dist_tb_km<20.0                    
                        # tb_nearest[~flag_good] = np.nan

                        tb_89_mat_h = gmi_ds["S1"].Tc[sl,:,-1]
                        tb_nearest_h = (np.ravel(tb_89_mat_h))[ind_tb] 

                        
                        cpr_ds_data = xr.open_dataset(cpr_fn, group = '/ScienceData/Data', 
                                                      engine = 'h5netcdf', phony_dims='sort')
                        cpr_ds_geo= xr.open_dataset(cpr_fn, group = '/ScienceData/Geo', 
                                                      engine = 'h5netcdf', phony_dims='sort')                       

                        
            
                        if np.sum(flag_good)>0:     
                            if len(fns_tb)>=1:  


                                flag_clear_sky = (tb_ds_data.land_flag_L2 == 0) & (tb_ds_data.profile_class_L2 ==1) # clear sky ocean

                                fl_good_clear_sky_ocean = flag_good & flag_clear_sky.values
                                if sum(fl_good_clear_sky_ocean)>0:

                                    print(orbit_id)
                                    

                                    # fig2,ax2 = plt.subplots()                                   
                                    # ax2.scatter(cpr_ds.latitude.values[flag_good], tb_nearest[flag_good], label = 'TB 89 V')
                                    # ax2.scatter(cpr_ds.latitude.values[flag_good], tb_nearest_h[flag_good], label = 'TB 89 H')                                    
                                    # ax2.scatter(tb_ds_data.latitude.values[flag_good], tb_ds_data.TB_89_V_eddington[flag_good], 
                                    #             label = 'sim TB 89 V', s= 4)
                                    # ax2.scatter(tb_ds_data.latitude.values[flag_good], tb_ds_data.TB_89_H_eddington[flag_good], 
                                    #             label = 'sim TB 89 H', s =4 )       
                                    # ax2.scatter(cpr_ds.latitude.values[flag_good], 
                                    #                 ((tb_ds_data['noise_dbm_L2'][flag_good])+150)*50,
                                    #                 label = 'L2 noise', s = 4)                                                                        
                                    # ax2.scatter(tb_ds_data.latitude.values[flag_good], 
                                    #             tb_ds_data.TB_89_V_eddington[flag_good], c= 'r', s =1)
                                    # ax2.scatter(tb_ds_data.latitude.values[flag_good], 
                                    #             tb_ds_data.TB_89_H_eddington[flag_good], c = 'b', s = 1)                                    
                                    # ax2.scatter(tb_ds_data.latitude.values[fl_good_clear_sky_ocean], 
                                    #             tb_ds_data.TB_89_V_eddington[fl_good_clear_sky_ocean], c= 'k', s =1)
                                    # ax2.scatter(tb_ds_data.latitude.values[fl_good_clear_sky_ocean], 
                                    #             tb_ds_data.TB_89_H_eddington[fl_good_clear_sky_ocean], c = 'k', s = 1)     
                                    # ax2.set_xlabel('Latitude')
                                    # ax2.set_title(orbit_id)
                                    # ax2.legend()
                                    # ax2.grid()
                                    # fig2.tight_layout()
                                    # fig2.savefig('/home/k/km357/Data/EarthCARE/figs/EC_intersections_{}_ecmwf_sim_eddington.png'.format(orbit_id))
                                    
                                    fig = plt.figure(figsize=(7., 5.5))
                                    ax = plt.axes(projection = ccrs.PlateCarree())  
                                    ax.set_title('GPM orbit {}; EarthCARE frame {}'.format(gmi_orbit,orbit_id)) 
                                    pc = ax.pcolor(lon_mat, lat_mat, tb_89_mat_h, vmin = 180, vmax = 280, cmap = 'plasma')
                                    cbar0 = plt.colorbar(pc, location='right', pad = 0.15, aspect = 40, label = '', fraction = 0.05)
                                    cbar0.ax.set_title('TB$_H$ [K]')
                                    sc = ax.scatter(tb_ds_data.longitude.values[flag_good],
                                        tb_ds_data.latitude.values[flag_good], 
                                        c = tb_ds_data['noise_dbm_L2'].values[flag_good], 
                                        vmin = -146.5, vmax = -145.0, cmap = 'viridis',
                                        s = 4)
                                    cbar1 = plt.colorbar(sc, location='left', pad = 0.02, aspect = 40, label = '', fraction = 0.03)
                                    cbar1.ax.set_title('[dBm]')
                                    center_lon = np.mean(tb_ds_data.longitude.values[flag_good])
                                    center_lat = np.mean(tb_ds_data.latitude.values[flag_good])
                                    ax.set_xlim( center_lon - 5., center_lon + 5.)
                                    ax.set_ylim( center_lat - 5., center_lat + 5.)
                                    ax.coastlines() 
                                    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                        linewidth=1, color='gray', alpha=0.5, linestyle='--')
                                    gl.xlabels_top = False
                                    gl.ylabels_left = False
                                    fig.tight_layout()
                                    fig.savefig('/home/k/km357/Data/EarthCARE/figs/GMI_EC_intersections_map_{}.png'.format(orbit_id), dpi = 200)


                                    
                                    z_db = 10*np.log10(cpr_ds_data.radarReflectivityFactor.values[flag_good])
                                    tmp_y = cpr_ds_geo.binHeight.values[flag_good]*1e-3
                                    tmp_x = np.tile(cpr_ds_geo.longitude.values[flag_good], (z_db.shape[1], 1)).T


                                    tmp_da_lin = 10.0**(0.1*tb_ds_data['noise_dbm_L2'])
                                    tmp_da_smooth = tmp_da_lin.rolling(
                                        dim={tmp_da_lin.dims[0]: 14}, center = True, min_periods = 1).mean()
                                    tb_ds_data['noise_dbm_L2_smoothed'] = 10.0 * np.log10(tmp_da_smooth)

                                    
                                    fig, axes = plt.subplots(nrows=2, ncols=1, sharex = True, figsize=(10., 5.5))
                                    ax = axes[0]
                                    pc = ax.pcolor(tmp_x, tmp_y, z_db, vmin = -40, vmax = 20, cmap = 'plasma')
                                    cbar = plt.colorbar(pc, ax=ax, location='right', pad = 0.02, aspect = 30, 
                                        label = '', fraction = 0.02)
                                    cbar.ax.set_title('[dBZ]')
                                    ax.set_ylim(-1, None)
                                    if orbit_id == '01202H':
                                        ax.set_ylim(-1, 4)
                                    ax.set_ylabel('Height [km]')
                                    ax.grid()

                                    ax.set_title('GPM orbit {}; EarthCARE frame {}'.format(gmi_orbit,orbit_id))
                                    

                                    ax1= axes[1]     
                                    color = 'k'
                                    
                                    ax1.set_ylabel('dBm', color=color)                                   
                                    ax1.tick_params(axis='y', labelcolor=color)
                                    ax1.plot(cpr_ds_geo.longitude.values[flag_good], 
                                        tb_ds_data['noise_dbm_L1'].values[flag_good], 
                                        color = color, ls= '-', lw = 2, label = 'L1')
                                    ax1.plot(cpr_ds_geo.longitude.values[flag_good], 
                                        tb_ds_data['noise_dbm_L2_smoothed'].values[flag_good], 
                                        color = color, ls= '-', lw = 1, label = 'L2')
                                    ax1.legend()
                                    

                                    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

                                    color = 'tab:blue'
                                    ax2.set_ylabel('TB', color=color)  # we already handled the x-label with ax1
                                    ax2.plot(cpr_ds_geo.longitude.values[flag_good], tb_nearest[flag_good], 
                                        color=color, label = 'TB$_V$ 89 GHz', ls = '-', lw =2)
                                    ax2.plot(cpr_ds_geo.longitude.values[flag_good], tb_nearest_h[flag_good], 
                                        color=color, label = 'TB$_h$ 89 GHz', ls = '--', lw = 2)

                                    ax2.plot(cpr_ds_geo.longitude.values[flag_good], tb_ds_data.TB_89_V[flag_good].values, 
                                        color=color,  ls = '-', lw =1 )
                                    ax2.plot(cpr_ds_geo.longitude.values[flag_good], tb_ds_data.TB_89_H[flag_good].values, 
                                        color=color,  ls = '--', lw =1)
                                    ax2.set_ylim(170, None)
                                    
                                    ax2.tick_params(axis='y', labelcolor=color)
                                    ax2.legend()

                                    fig.tight_layout()
                                    r = ax.figure.subplotpars.right
                                    l = ax.figure.subplotpars.left

                                    b = ax2.figure.subplotpars.bottom
                                    t = ax2.figure.subplotpars.top

                                    ax1.set_position([l, b, (r-l)*0.96,  (t-b)*0.5])
                                    ax1.xaxis.grid()
                                    
                                    x_tick_labels = [print_deg(t, 'longitude',1).replace('.0°', '°') for t in ax1.get_xticks()]
                                    ax1.set_xticklabels(x_tick_labels)

                                    # ax2.set_position([l, b, (r-l)*0.9,  t-b])
                                    # fig.subplots_adjust(right=r-0.02, left=l+0.02, bottom = b+0.02, top = t-0.02)
                                    fig.savefig('/home/k/km357/Data/EarthCARE/figs/GMI_EC_intersections_reflectivity_{}.png'.format(orbit_id), dpi = 200)


                                    plt.close('all')
                        
                        
                                    ds_out = xr.Dataset()
                                    ds_out['time'] = ('nray', cpr_ds.profileTime.values[flag_good])

                                    ds_out['longitude'] = ('nray', cpr_ds.longitude.values[flag_good] )
                                    ds_out['latitude'] = ('nray', cpr_ds.latitude.values[flag_good])                                    

                                    ds_out['dist_tb_s'] = ('nray', dist_tb_s[flag_good])
                                    ds_out['dist_tb_km'] = ('nray', dist_tb_km[flag_good])

                                    ds_out['tb_89_v'] = ('nray', tb_nearest[flag_good]) 
                                    ds_out['tb_89_h'] = ('nray', tb_nearest_h[flag_good]) 

                                    ds_out['noise_dbm'] = ('nray', tb_ds_data['noise_dbm_L1'].values[flag_good])                                    
                                    ds_out['noisel2_dbm'] = ('nray', tb_ds_data['noise_dbm_L2'].values[flag_good])

                                    ds_out['tbs_89_v'] = ('nray', tb_ds_data.TB_89_V[flag_good].values) 
                                    ds_out['tbs_89_h'] = ('nray', tb_ds_data.TB_89_H[flag_good].values) 

                                    ds_out['tbs_89_v_edd'] = ('nray', tb_ds_data.TB_89_V_eddington[flag_good].values) 
                                    ds_out['tbs_89_h_edd'] = ('nray', tb_ds_data.TB_89_H_eddington[flag_good].values) 


                                    ds_out['tbs_94'] = ('nray', tb_ds_data.TB_94[flag_good].values) 
                                    ds_out['SST'] = ('nray', tb_ds_data.SST[flag_good].values)
                                    ds_out['emmis_94'] = ('nray', tb_ds_data.emmis_94[flag_good].values)
                                    
                                    ds_out['emmis_89h'] = ('nray', tb_ds_data.emmis_89h[flag_good].values)
                                    ds_out['emmis_89v'] = ('nray', tb_ds_data.emmis_89v[flag_good].values)

                                    tmp_clear =(tb_ds_data.profile_class_L2 ==1)                                        
                                    ds_out['clear_sky_flag'] = ('nray', np.int16(tmp_clear[flag_good]) )
                                    
                                    tmp_clear =(tb_ds_data.land_flag_L2 == 0)                                        
                                    ds_out['land_flag'] = ('nray', np.int16(tmp_clear[flag_good]))


                                
                                ds_out=ds_out.set_coords(['time', 'longitude', 'latitude'])
                                ds_out.to_netcdf('/home/k/km357/Data/EarthCARE/figs/EC_TB_{}_ECMWF_sim.nc'.format(
                                    cpr_fn[-9:-3]))
                cpr_ds.close()
            
            # plt.close(fig)