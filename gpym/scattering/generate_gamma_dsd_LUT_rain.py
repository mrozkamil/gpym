import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline
import socket
from datetime import datetime
import xarray as xr
import os, sys
import matplotlib.pyplot as plt
home = os.path.expanduser('~')

home = os.path.expanduser("~")
sys.path.append(
    os.path.join(home, 'Documents','Python3', 'gpym'))
from gpym.psd import NormalizedGammaPSD
from gpym.env import rain_terminal_velocity
from gpym.refractive import refractive_index

V_tmp = rain_terminal_velocity()
V = lambda D: V_tmp(D*1e3)

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

freqs = {'S' : 2.7, 
      'C': 5.6, 
      'X': 9.4,
      'Ku': 13.6,
      'K': 24.,
      'Ka': 35.5, 
      'W': 94.0,
      'G': 200}
bands = list(freqs.keys())

# bands = ['Ku','Ka']

def weighted_integral(func, weight_function, xmin,xmax, normalize = True):
    if normalize:
        norm_const = integrate.quad(weight_function, xmin, xmax)[0]
    else:
        norm_const = 1
    tmp_func = lambda D: func(D) * weight_function(D)
    tmp_v = integrate.quad(tmp_func, xmin, xmax)[0]
    return tmp_v/norm_const

D0 = np.arange(0, 5, 0.05)[1:]*1e-3
MU = np.arange(-3, 20, 0.25)[1:]

current_dir = os.path.dirname(os.path.abspath(__file__))

out_path = os.path.join(current_dir, 'rain', 'PSD_integrated', 'gamma')
in_path = os.path.join(current_dir, 'rain', 'SSP', 'Tmatrix_dbase')

for band in bands:
    print(band)
    freq_str = ('%.2f' % freqs[band]).replace('.','p')
    
    f_name_in = os.path.join(in_path,'Tmatrix_wobbly_rain_%s_GHz.nc' % freq_str)
    f_name_out = os.path.join(out_path,
                    'Tmatrix_wobbly_rain_%s_GHz_gamma_dsd.nc' % freq_str) 
    
    dset_in = xr.open_dataset(f_name_in)
    
    d = dset_in['D'].values
    temp = dset_in['T'].values
    
    K_sq = np.array([refractive_index(freq = freqs[band],temp = tt-273.15)[1] 
                     for tt in temp])
    refl = dset_in['bsca'].values*1e6*dset_in.attrs['wave length']**4 / (
        K_sq*np.pi**5)
    att = dset_in['ext'].values
    
    plt.figure()
    plt.plot(d,refl)
    plt.yscale('log')    
    plt.grid()
   
        
    Dm = np.zeros((D0.size,MU.size))   
    Sm = np.zeros((D0.size,MU.size))   
    PR = np.zeros((D0.size,MU.size))   
    
    Refl = np.zeros((D0.size,MU.size,temp.size))
    Att = np.zeros((D0.size,MU.size,temp.size))
    Vel = np.zeros((D0.size,MU.size,temp.size))
    
    for it, tt in enumerate(temp):
        
        refl_fun = UnivariateSpline(d, refl[:,it], w=None, k=1, 
            s=0, ext= 'const', check_finite=False)
        att_fun = UnivariateSpline(d, att[:,it], w=None, k=1, 
            s=0, ext= 'const', check_finite=False)
            
        print('temperature of %1.0f C' % (tt-273.15))
        # if np.abs(tt-283)>1: continue
        
        for im,mu in enumerate(MU):
            print('shape parameter mu of %1.2f' % mu)
            # if mu not in [-2,0,3,6,12]: continue;
            for id,d0 in enumerate(D0):
                D_max = np.minimum(d[-1], d0*3)
                D_min = d[0]
                rain_psd = NormalizedGammaPSD(Dm=d0,mu = mu, 
                            D_max = D_max, D_min = D_min) 
                
                Dm[id,im] = rain_psd.D_m()
                Sm[id,im] = rain_psd.Sigma_m()
                WC = rain_psd.WaterContent()
                
                WC_g = WC*1e3
                
                tmp_refl = weighted_integral(rain_psd.func,refl_fun, 
                            xmin = D_min,xmax = D_max, normalize = False)
                
                Refl[id,im,it] =  tmp_refl/WC_g
                tmp_att = weighted_integral(rain_psd.func,att_fun, 
                            xmin = D_min, xmax = D_max, normalize = False)
                
                Att[id,im,it] =  tmp_att/WC_g*1e3*4.343 #dB per km per g m-3 
                
                
                tmp_fun = lambda D : rain_psd.func(D) * refl_fun(D)
                tmp_vel = weighted_integral(V,tmp_fun, 
                            xmin = D_min, xmax = D_max, normalize = False)                
                Vel[id,im,it] = tmp_vel/tmp_refl
                
                tmp_fun = lambda D : rain_psd.func(D) * rain_psd.mass_function(D)
                tmp_vel = weighted_integral(V,tmp_fun, 
                            xmin = D_min, xmax = D_max, normalize = False)
                PR[id,im] = tmp_vel/WC_g*3.6*1e3
                
    
    
    # plt.figure()
    # plt.pcolormesh(D0,MU, 10*np.log10(Refl[:,:,-1].T))
    # plt.colorbar()
    
    # plt.figure()
    # plt.pcolormesh(D0,MU, (Att[:,:,-1].T))
    # plt.colorbar()
    
    
    # plt.figure()
    # plt.pcolormesh(D0,MU, (Vel[:,:,-1].T))
    # plt.colorbar()
    
    # plt.figure()
    # plt.pcolormesh(D0,MU, (PR[:,:].T))
    # plt.colorbar()
    
    # plt.figure()
    # plt.scatter(PR.ravel(), Att[:,:,-1].ravel())
    # plt.scatter(PR.ravel(), 0.28*PR.ravel())
    
    Refl = xr.DataArray(
        dims = ['Dm_tmp','mu','T'], 
        coords = [D0,MU, temp ],
        data = 10*np.log10(Refl),
        attrs={'long_name':'%s-band Radar Reflectivity' % band,
                'description': 'Gamma PSD simulation for WC = 1 g m-3',
                'units':'dBZ', }, ) 
    
    MDV = xr.DataArray(
        dims = ['Dm_tmp','mu','T'], 
        coords = [D0,MU, temp ],
        data = Vel,
        attrs={'long_name':'%s-band Mean Doppler Velocity' % band,
                'description': 'Gamma PSD simulation for WC = 1 g m-3',
                'units':'m/s', }, ) 
    
    Ext = xr.DataArray(
        dims = ['Dm_tmp','mu','T'], 
        coords = [D0,MU, temp ],
        data = Att,
        attrs={'long_name':'%s-band specific attenuation' % band,
                'description': 'Gamma PSD simulation for WC = 1 g m-3',
                'units':'dB km-1', }, ) 
    Precip = xr.DataArray(
        dims = ['Dm_tmp','mu'], 
        coords = [D0,MU ],
        data = PR,
        attrs={'long_name':'Precipitation Rate',
                'description': 'Gamma PSD simulation for WC = 1 g m-3, standard pressure',
                'units':'mm h-1', }, ) 
    
    Diam = xr.DataArray(
        dims = ['Dm_tmp','mu'], 
        coords = [D0,MU ],
        data = Dm*1e3,
        attrs={'long_name':'Mass-weighted mean diameter',
                'description': 'Gamma PSD simulation for WC = 1 g m-3',
                'units':'mm', }, ) 
    Sigma = xr.DataArray(
        dims = ['Dm_tmp','mu'], 
        coords = [D0,MU ],
        data = Sm*1e3,
        attrs={'long_name':'Mass-weighted PSD std',
                'description': 'Gamma PSD simulation for WC = 1 g m-3',
                'units':'mm', }, ) 
    
    variables = {'Dm': Diam, 'Sm': Sigma, 'RR': Precip, 'Z': Refl,
                 'k': Ext, 'MDV': MDV }
    dset_out = xr.Dataset(data_vars=variables,)    
    
    encoding =  {key: {'zlib': True, 'complevel': 6,
            '_FillValue': -999.,} for key in variables.keys()}

    global_attributes = {'created_by': 'Kamil Mroz (NCEO UoL)',
                        'host_machine':socket.gethostname(),                           
                        'created_on': str(datetime.now()),}
    
    dset_out.attrs = global_attributes       

    
    dset_out.to_netcdf(f_name_out, engine = 'h5netcdf',encoding =  encoding)
    
bands = list(freqs.keys())

    
    
# bands = list(dopp.keys())
# mus =[-2,0,3,12]
# it =np.argmin(np.abs(temp-283))
# for ib,band1 in enumerate(bands[:-1]):
#     for band2 in bands[ib+1:]:
#         plt.figure()
#         for m in mus:
#             im = np.argmin(np.abs(m-MU))
#             plt.plot(dm[band1][:,im]*1e3,dopp[band1][:,im,it] - dopp[band2][:,im,it], 
#             label = '$\mu$ = %1.0f' % (MU[im],), lw =2)
        
#         plt.grid()
#         plt.legend()
#         f_name = os.path.join( data_dir,
#             'ddv_%s_%s_std_beta_15deg.png' % ( band1,band2))
#         plt.xlabel('$D_m$ [mm]')
#         plt.ylabel('$DDV_{%s-%s}$ [m/s]' % (band1, band2))
#         plt.tight_layout()
#         plt.xlim(0,3.5)
#         plt.savefig(f_name)
#         plt.show(block = False)
        

 
# for band in bands:
#     plt.figure()
#     for m in mus:
#         im = np.argmin(np.abs(m-MU))
#         plt.plot(dm[band1][:,im]*1e3,dopp[band][:,im,it], 
#         label = '$\mu$ = %1.2f' % (MU[im],))
    
#     plt.grid()
#     plt.legend()
#     f_name = os.path.join( data_dir,
#         'vel_%s_band_std_beta_15deg.png' % ( band,))
#     plt.xlabel('Dm [mm]')
#     plt.ylabel('$V_{%s}$ [m/s]' % (band, ))
#     plt.tight_layout()
#     plt.savefig(f_name)
#     plt.show(block = False)
    
        

    
    
    
    
    
    