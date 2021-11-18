import numpy as np
def refractive_index(freq, temp,):
    '''
    Abstract:
    The "Turner-Kneifel-Cadeddu" liquid water absorption model (JTECH 2016).
    It was built using both laboratory observations (primarily at warm temperatures) and
    field data observed by MWRs at multiple frequencies at supercool temperatures. The field
    data were published in Kneifel et al. JAMC 2014.  The strength of the TKC model is the
    use of an optimal estimation framework to determine the empirical coefficients of the
    double-Debye model.  A full description of this model is given in

        Turner, D.D., S. Kneifel, and M.P. Cadeddu, 2016: An improved liquid
        water absorption model in the microwave for supercooled liquid clouds.
        J. Atmos. Oceanic Technol., 33(1), pp.33-44, doi:10.1175/JTECH-D-15-0074.1.

    Note that the model is designed to operate over the frequency range from 0.5 to 500
    GHz, and temperatures from -40 degC to +50 degC only for freshwater (no salinity)



    idl code authors:
    Dave Turner, National Severe Storms Laborotory / NOAA
    Stefan Kneifel, University of Cologne
    Maria Cadeddu, Argonne National Laboratory
    python3 version:
    Kamil Mroz, National Centre for Earth Observation, University of Leicester, UK

    Call:
    refractive_index		    ; Returns the complex refractive index of water,
                            ; dielectric constant and liquid mass absorption [m2 kg-1]
    freq, 				    ; Input frequency (scalar float), in GHz
    temp, 				    ; Input cloud temperature (scalar float), in degC
    '''

    # Convert the frequency from GHz to Hz
    frq = freq * 1e9

    # Some constants
    cl = 299.792458e6 #speed of light in vacuum
    pi = np.pi

    # Empirical coefficients for the TKC model. The first 4 are a1, b1, c1, & d1,
    # the next four are a2, b2, c2, & d2, & the last one is tc.
    coef = [8.111e1, 4.434e-3, 1.302e-13, 6.627e2,
        2.025, 1.073e-2, 1.012e-14, 6.089e2,
        1.342e+2]

    # This helps to understand how things work below
    a_1 = coef[0]
    b_1 = coef[1]
    c_1 = coef[2]
    d_1 = coef[3]

    a_2 = coef[4]
    b_2 = coef[5]
    c_2 = coef[6]
    d_2 = coef[7]

    t_c = coef[8]


    # Compute the static dielectric permittivity (Eq 6)
    eps_s = 87.9144 - 0.404399 * temp + 9.58726e-4 * temp**2. - 1.32802e-6 * temp**3.

    # Compute the components of the relaxation terms (Eqs 9 & 10)
        # First Debye component
    delta_1 = a_1 * np.exp(-b_1 * temp)
    tau_1   = c_1 * np.exp(d_1 / (temp + t_c))
        # Second Debye component
    delta_2 = a_2 * np.exp(-b_2 * temp)
    tau_2   = c_2 * np.exp(d_2 / (temp + t_c))

    # Compute the relaxation terms (Eq 7) for the two Debye components
    term1_p1 = (tau_1**2 * delta_1) / (1. + (2.*pi*frq*tau_1)**2)
    term2_p1 = (tau_2**2 * delta_2) / (1. + (2.*pi*frq*tau_2)**2)

    # Compute the real permittivitity coefficient (Eq 4)
    eps1 = eps_s - ((2.*pi*frq)**2)*(term1_p1 + term2_p1)


    # Compute the relaxation terms (Eq 8) for the two Debye components
    term1_p1 = (tau_1 * delta_1) / (1. + (2.*pi*frq*tau_1)**2)
    term2_p1 = (tau_2 * delta_2) / (1. + (2.*pi*frq*tau_2)**2)

    # Compute the imaginary permittivitity coefficient (Eq 5)
    eps2 = 2.*pi*frq * (term1_p1 + term2_p1)

    n_sq = np.complex(eps1, eps2)
    n = np.sqrt(n_sq)
    if np.imag(n)<0:
        n = n.conjugate()
    # Compute the mass absorption coefficient (Eq 1)
    K = (n_sq-1)/(n_sq+2)
    K_sq = np.absolute(K**2)

    alpha = 6.*np.pi*np.imag(K)*frq*1e-3/cl

    return n, K_sq, alpha


# test
def test():
    freq = np.linspace(1,200,1000)
    temp = 15.
    perm1 = np.zeros_like(freq)
    perm2 = np.zeros_like(freq)
    Ksq = np.zeros_like(freq)
    for indf,ff in enumerate(freq):
        eps,ksq,_ = refractive_index(ff,temp)
        eps_sq = eps**2
        perm1[indf] = np.real(eps_sq)
        perm2[indf] = np.imag(eps_sq)
        Ksq[indf] = ksq

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Freq (GHz)')
    ax1.set_ylabel("$\epsilon$'", color=color)
    ax1.plot(freq,perm1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0,100)
    plt.xscale('log')
    plt.text(20, 80, 'Temperature of 15C' )
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel("$\epsilon$''", color=color)  # we already handled the x-label with ax1
    ax2.plot(freq, perm2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0,50)
    plt.grid()
    plt.title('real and imaginary parts of permitivity ')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('perm_Turner_Kneifel_Cadeddu.png')


    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Freq (GHz)')
    ax1.set_ylabel("$|K|^2$", color=color)
    ax1.plot(freq,Ksq, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.xscale('log')
    plt.text(20, 0.8, 'Temperature of 15C' )
    plt.title('dielectric constant')
    plt.grid()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('dielectric_Turner_Kneifel_Cadeddu.png')
    plt.show()









