import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline, PchipInterpolator
import h5py
from radar.env import rain_terminal_velocity
from radar.psd import NormalizedGammaPSD
import os
import matplotlib.pyplot as plt
home = os.path.expanduser('~')
data_dir = os.path.join(home, 'Documents','Python3', 't_matrix')

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
      'Ku': 13.63,
      'K': 24.23,
      'Ka': 35.55, 
      'W': 94.0,
      'G': 200}
bands = list(freqs.keys())

bands = ['G','W']

def weighted_integral(func, weight_function, xmin,xmax, normalize = True):
    if normalize:
        norm_const = integrate.quad(weight_function, xmin, xmax)[0]
    else:
        norm_const = 1
    tmp_func = lambda D: func(D) * weight_function(D)
    tmp_v = integrate.quad(tmp_func, xmin, xmax)[0]
    return tmp_v/norm_const

D0 = np.arange(0, 5, 0.1)[1:]*1e-3
MU = np.arange(-3, 20, 0.25)[1:]



for band in bands:
    print(band)
    h5_name = ('rain_tmat_freq_%2.3f_std_beta_15deg' % (
            freqs[band],)).replace('.','p')
    h5_name2 = ('rain_tmat_freq_psd_integrated_%2.3f_std_beta_15deg' % (
            freqs[band],)).replace('.','p')
    if not os.path.exists(os.path.join( data_dir, h5_name+'.h5')):
        continue
    hf1 = h5py.File(os.path.join( data_dir, h5_name+'.h5'), 'r') 
     
    d = hf1['D_eq'][:]*1e-3
    temp = hf1['tem'][:]
    refl = hf1['/radar/refl'][:]
    refl[np.isnan(refl)]=0
    att = hf1['/radar/Ai'][:]
    att[np.isnan(att)]=0
    hf1.close()
    
    
    Dm = np.zeros((D0.size,MU.size))
    
    
    Refl = np.zeros((D0.size,MU.size,temp.size))
    Att = np.zeros((D0.size,MU.size,temp.size))
    Vel = np.zeros((D0.size,MU.size,temp.size))
    
    for it, tt in enumerate(temp):
        
        refl_fun = UnivariateSpline(d, refl[it,:], w=None, k=1, 
            s=0, ext= 'const', check_finite=False)
        att_fun = UnivariateSpline(d, att[it,:], w=None, k=1, 
            s=0, ext= 'const', check_finite=False)
            
        print('temperature of %1.0f K' % tt)
        if np.abs(tt-283)>1: continue
        
        for im,mu in enumerate(MU):
            print('shape parameter mu of %1.2f' % mu)
            if mu not in [-2,0,3,6,12]: continue;
            for id,d0 in enumerate(D0):
                D_max = np.minimum(d[-1], d0*3)
                rain_psd = NormalizedGammaPSD(Dm=d0,mu = mu, D_max = D_max) 
                
                Dm[id,im] = rain_psd.D_m()
                WC = rain_psd.WaterContent()
                
                tmp_refl = weighted_integral(rain_psd.func,refl_fun, 
                            xmin = 0,xmax = D_max, normalize = False)
                
                Refl[id,im,it] =  tmp_refl/WC*1e-3
                tmp_att = weighted_integral(rain_psd.func,att_fun, 
                            xmin = 0,xmax = D_max, normalize = False)
                
                Att[id,im,it] =  tmp_att/WC*1e-3
                
                
                tmp_fun = lambda D : rain_psd.func(D) * refl_fun(D)
                tmp_vel = weighted_integral(V,tmp_fun, 
                            xmin = 0,xmax = D_max, normalize = False)
                Vel[id,im,it] = tmp_vel/tmp_refl
    
    
    hf2 = h5py.File(os.path.join( data_dir, h5_name2+'.h5'), 'w') 
    hf2.create_dataset('Refl', data = Refl)
    hf2.create_dataset('Att', data = Att)
    hf2.create_dataset('Vel', data = Vel)
    hf2.create_dataset('Dm', data = Dm)
    hf2.create_dataset('mu', data = MU)
    hf2.close()
    
bands = list(freqs.keys())

dopp = {}
dm   = {}
refl = {}
for band in bands:
    h5_name2 = ('rain_tmat_freq_psd_integrated_%2.3f_std_beta_15deg' % (
            freqs[band],)).replace('.','p')
    if not os.path.exists(os.path.join( data_dir, h5_name2+'.h5')):
        continue
    hf2 = h5py.File(os.path.join( data_dir, h5_name2+'.h5'), 'r') 
    dopp[band] = hf2['Vel'][:]
    refl[band] = hf2['Refl'][:]
    dm[band] = hf2['Dm'][:]
    hf2.close()
    
    
bands = list(dopp.keys())
mus =[-2,0,3,12]
it =np.argmin(np.abs(temp-283))
for ib,band1 in enumerate(bands[:-1]):
    for band2 in bands[ib+1:]:
        plt.figure()
        for m in mus:
            im = np.argmin(np.abs(m-MU))
            plt.plot(dm[band1][:,im]*1e3,dopp[band1][:,im,it] - dopp[band2][:,im,it], 
            label = '$\mu$ = %1.0f' % (MU[im],), lw =2)
        
        plt.grid()
        plt.legend()
        f_name = os.path.join( data_dir,
            'ddv_%s_%s_std_beta_15deg.png' % ( band1,band2))
        plt.xlabel('$D_m$ [mm]')
        plt.ylabel('$DDV_{%s-%s}$ [m/s]' % (band1, band2))
        plt.tight_layout()
        plt.xlim(0,3.5)
        plt.savefig(f_name)
        plt.show(block = False)
        

 
for band in bands:
    plt.figure()
    for m in mus:
        im = np.argmin(np.abs(m-MU))
        plt.plot(dm[band1][:,im]*1e3,dopp[band][:,im,it], 
        label = '$\mu$ = %1.2f' % (MU[im],))
    
    plt.grid()
    plt.legend()
    f_name = os.path.join( data_dir,
        'vel_%s_band_std_beta_15deg.png' % ( band,))
    plt.xlabel('Dm [mm]')
    plt.ylabel('$V_{%s}$ [m/s]' % (band, ))
    plt.tight_layout()
    plt.savefig(f_name)
    plt.show(block = False)
    
        

    
    
    
    
    
    