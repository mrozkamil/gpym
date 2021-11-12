import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.special import gamma
import matplotlib.pyplot as plt

class mass_snow():    
    def mass_size(self, Dmax, alpha, beta ):
        return alpha*np.power(Dmax,beta)
    
    def __init__(self, alpha_sn = 0.015,beta_sn = 2.05,
                 alpha_gr = 469, beta_gr = 3.36):
        Dmax = np.logspace(-2, 2, 401)*1e-3
        
        self.rho_ice = 916.7
        self.Dmax = Dmax
        self.alpha_ice = np.pi/6*self.rho_ice
        self.beta_ice = 3.
        
        self.alpha_sn = alpha_sn
        self.beta_sn = beta_sn 
        
        self.alpha_gr = alpha_gr
        self.beta_gr = beta_gr 
        
        self.snow_min_mass = lambda Dmax: self.mass_size(Dmax,
                           alpha = self.alpha_ice*1e-3, 
                           beta = self.beta_ice)
        
        self.snow_mass = lambda Dmax: self.mass_size(Dmax,
                           alpha = self.alpha_sn, 
                           beta = self.beta_sn)
        
        self.graup_mass = lambda Dmax: self.mass_size(Dmax,
                           alpha = self.alpha_gr, 
                           beta = self.beta_gr) 
        
        self.solid_mass = lambda Dmax: self.mass_size(Dmax,
                           alpha = self.alpha_ice, 
                           beta = self.beta_ice) 
        
        self.poly_b_loga = np.array([ 3.52093999, -8.57494282]) #based on the HIWC dataset
        self.poly_loga_b = np.array([1/self.poly_b_loga[0], 
                            -self.poly_b_loga[1]/self.poly_b_loga[0]])
        
        self.alpha = alpha_sn 
        self.beta = beta_sn
        self.riming_model = 'MG'
        self.mass_function = lambda Dmax: np.maximum(
            np.minimum(self.snow_mass(Dmax), self.solid_mass(Dmax)),
            self.snow_min_mass(Dmax))
        
    def _construct_mass_function(self, alpha = None, beta = None, 
                 riming_model = 'MG'):
        if alpha is None:
            alpha = self.alpha_sn            
        if beta is None:
            if riming_model == 'MG':
                beta = self.beta_sn 
            elif riming_model == 'coupled':
                beta = np.polyval(self.poly_loga_b, np.log10(alpha))
                
        if not((alpha == self.alpha) and 
               (beta==self.beta) and
               (self.riming_model == riming_model)): 
        
            m_snowflake = lambda Dmax: self.mass_size(Dmax,
                                alpha = alpha, beta = beta)  
           
            if (riming_model == 'MG') and (alpha>self.alpha_sn):  
                tmp_fun = lambda Dmax: self.mass_size(Dmax,
                                alpha = alpha, beta = beta)
                m_snowflake = lambda Dmax: np.maximum(
                    np.minimum(self.graup_mass(Dmax),tmp_fun(Dmax)),
                    self.snow_mass(Dmax))
            else:
                m_snowflake = lambda Dmax: self.mass_size(Dmax,
                                alpha = alpha, beta = beta)
            
                            
            self.mass_function = lambda Dmax: np.maximum(
                np.minimum(m_snowflake(Dmax), self.solid_mass(Dmax)),
                self.snow_min_mass(Dmax))
           
            self.alpha = alpha 
            self.beta = beta
            self.riming_model = riming_model
            
        
    
    def __call__(self, Dmax, alpha = None, beta = None, 
                 riming_model = 'MG'):
        self._construct_mass_function(alpha = alpha, beta = beta, 
                 riming_model = riming_model)
        return self.mass_function(Dmax)
        
       
    

class PSD():
    """ Constructor of a particle size distribution (PSD)
        D is assumed to be in [m]
        we assume the following mass-diameter relation:
            m [kg] = a D^b
            where a = pi/6*1e3, b = 3 (i.e. water, D is the equivalent volume diameter)
        please modify the atributes mD_a and mD_b for any changes, 
        alternatively a handel to the mass-diamter function can be implicitely provided
    Attributes:
        D_min: the minimum diameter where the PSD>0 
        D_max: the maximum diameter where the PSD>0 
        mD_a: a prefactor of the mass-diameter relation
        mD_b: an exponent of the mass-diameter relation
        mass_function: the mass-diameter relation function, if provided mD_a,
            mD_b will be ignored
    """
    def __init__(self, D_min = 0, D_max = 8*1e-3, 
                 mD_a= None, mD_b = None, mass_function = None
    ):
        self.func = lambda D: np.zeros_like(D)
        self.D_min = D_min
        self.D_max = D_max
        mD_a = np.pi/6*1e3 if mD_a is None else mD_a
        mD_b = 3 if mD_b is None else mD_b
        self.mass_function = lambda D: mD_a* np.power(D, mD_b)
        
    def __call__(self, D):
        v = self.func(D)
        v[(D>self.D_max) & (D<self.D_min)] = 0
        return v

    def _get_weighted_moment(self, moment = 1, center=None, 
                weight_function = None, normalize = True):
        if weight_function is None:
            weight_function = lambda D: np.ones_like(D) 
        # print(weight_function)
        # print(weight_function(np.linspace(1,5)))
        
        center = 0. if center is None else center
        tmp_func = lambda D: self.func(D) * np.power(D-center,moment) * weight_function(D)
        tmp_v = integrate.quad(tmp_func, self.D_min, self.D_max)[0]
        if normalize:
            tmp_func = lambda D: self.func(D) *  weight_function(D)
            norm_const = integrate.quad(tmp_func, self.D_min, self.D_max)[0]
            tmp_v /= norm_const
        return tmp_v
    def Nt(self):
        return self._get_weighted_moment(moment = 0, normalize = False)
    def D_m(self):
        return self._get_weighted_moment(moment = 1, 
            weight_function=self.mass_function )
    def Sigma_m(self):
        return np.sqrt(self._get_weighted_moment( moment = 2, 
        center=self.D_m(), weight_function=self.mass_function ))
    def Skewness_m(self):
        return (self._get_weighted_moment(moment = 3, center=self.D_m(),
        weight_function=self.mass_function )/self.Sigma_m()**3)
    def Kurtosis_m(self):
        return (self._get_weighted_moment(moment = 4, center=self.D_m(), 
        weight_function=self.mass_function )/self.Sigma_m()**4)
    def WaterContent(self):
        return self._get_weighted_moment(moment = 0, center=None, 
                weight_function = self.mass_function, normalize = False)
    def show(self):
        plt.figure()
        tmp_d = np.linspace(0,self.D_max,101)[1:]
        tmp_n = self.__call__(tmp_d)
        plt.plot(tmp_d*1e3,tmp_n)
        plt.grid()
        plt.yscale('log')
        plt.ylabel('N [m$^{-4}$]')
        plt.xlabel('D [mm]')
        plt.tight_layout()
        plt.show(block = False)
 

class GammaPSD(PSD):
    """ Constructor of the Gamma particle size distribution (PSD) of the form:
    N(D) = N0 * D**mu * exp(-Lambda*D)
    D is assumed to be in [m]

    Attributes:
        N0: the intercept parameter, defaults to 8e6
        Lambda: the slope parameter, defaults to 4e3
        mu: the shape parameter, defaults to 0. 
        D_max: the maximum diameter where the PSD>0 (defaults to 11/Lambda, 
            if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """
    
    def __init__(self, N0=8e6, Lambda=4e3, mu=0.0, D_min = 0, D_max=None,
         mD_a= None, mD_b = None, mass_function = None):
        D_max = 11.0/Lambda if D_max is None else D_max
        super().__init__(D_min, D_max, mD_a= mD_a, mD_b = mD_b, 
                mass_function = mass_function)
        self.mu = mu
        self.N0 = N0
        self.Lambda = Lambda
        self.func = lambda D: self.N0 * np.power(D, self.mu)* np.exp(-self.Lambda*D)
        
class GeneralizedGammaPSD(PSD):
    """ Constructor of the Gamma particle size distribution (PSD) of the form:
    N(D) = M0 * c*Lambda/(gamma(mu))* (Lambda*D)**(c*mu-1) * exp(-(Lambda*D)**c)
    D is assumed to be in [m]

    Attributes:
        M0: the zero moment of the PSD, defaults to 8e6
        Lambda: the slope parameter, defaults to 4e3
        mu: the shape parameter, defaults to 0. 
        c: the second shape parameter, defaults to 1. 
        D_max: the maximum diameter where the PSD>0 (defaults to 11/Lambda, 
            if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """
    
    def __init__(self, M0=8e6, Lambda=4e3, mu=0.0, c = 1., D_min = 0, D_max=None,
         mD_a= None, mD_b = None, mass_function = None):
        D_max = 11.0/Lambda if D_max is None else D_max
        super().__init__(D_min, D_max, mD_a= mD_a, mD_b = mD_b, 
                mass_function = mass_function)
        self.mu = mu
        self.M0 = M0
        self.Lambda = Lambda
        self.c = c
        
        self.func = lambda D: self.M0*self.c*self.Lambda/gamma(self.mu)* (self.Lambda*D)**(self.c*self.mu-1) * np.exp(-(self.Lambda*D)**self.c)

class NormalizedGammaPSD(PSD):
    """Constructor of the normalized gamma particle size distribution (PSD)
    of the form:
    N(D) = N0 * f(mu) * (D/D0)**mu * exp(-(3.67+mu)*D/D0)
    f(mu) = 6/(4**4) * (4+mu)**(mu+4)/Gamma(mu+4)

    Attributes:
        Dm: the mean volume diameter, defaults to 0.001 m
        N0: the intercept parameter, defaults to 8e6 m^-4
        mu: the shape parameter, defaults to 0
        D_max: the maximum diameter to consider (defaults to 3*D0 when
            if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """

    def __init__(self, Dm=1.0*1e-3, N0=8e6, mu=0.0, D_min = 0, D_max=None,
         mD_a= None, mD_b = None, mass_function = None):
        D_max = Dm*3 if D_max is None else D_max
        super().__init__(D_min, D_max, mD_a= mD_a, mD_b = mD_b, 
                mass_function = mass_function)
        self.Dm = Dm
        self.mu = mu
        self.N0 = N0
        fmu =  6.0/(4**4) * (4+mu)**(mu+4)/gamma(mu+4)
        h = lambda x: fmu * x**mu * np.exp(-(mu+4)*x)
        self.func = lambda D: N0*h(D/Dm)
class NormalizedGeneralizedGammaPSD(PSD):
    """ Constructor of the Gamma particle size distribution (PSD) of the form:
    N(D) = N0 * h_gg(mu,c,x)
    where N0 = M_i^((j+1)/(j-1))M_j^((i+1)/(i-j))
    Dm = (M_j/M_i)^(1/(j-i))
    h(x) = c*Gamma_i^((j+c*mu)/(i-j))*Gamma_j^((-i-c*mu)/(i-j))*x^(c*mu-1)*exp(-(Gamma_i/Gamma_j)^(c/(i-j))*x^c
    x = D/D_m
    Gamma_i = Gamma(mu+i/c)
    D is assumed to be in [m]
    i =3, j= 4
    

    Attributes:
        N0: the median volume diameter, defaults to 0.001 m
        Dm: the intercept parameter, defaults to 8e6 m^-4
        mu: the shape parameter, defaults to 1
        c: the other shape parameter, defaults to 1
        D_max: the maximum diameter to consider (defaults to 3*D0 when
            if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """

    def __init__(self, Dm=1.0*1e-3, N0=8e6, mu=1.0, c=1., D_min = 0, D_max=None,
         mD_a= None, mD_b = None, mass_function = None):
        D_max = Dm*3 if D_max is None else D_max
        super().__init__(D_min, D_max, mD_a= mD_a, mD_b = mD_b, 
                mass_function = mass_function)
        self.Dm = Dm
        self.mu = mu
        self.N0 = N0
        self.c = c
        i, j = 3, 4
        G_i = gamma(mu+i/c)
        G_j = gamma(mu+j/c)
        h = lambda x: (c * G_i**((j+c*mu)/(i-j)) * G_j**((-i-c*mu)/(i-j)) * 
                x**(c*mu-1) * np.exp(-(G_i/G_j)**(c/(i-j)) * x**c))
        self.func = lambda D: N0*h(D/Dm)



class BinnedPSD(PSD):
    """Binned gamma particle size distribution (PSD).
    
    Callable class to provide a binned PSD with the given bin edges and PSD
    values.

    Args (constructor):
        The first argument to the constructor should specify n+1 bin edges, 
        and the second should specify n bin_psd values.        
        
    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters outside the bins.
    """
    def __init__(self, D_bin_edges, N,  
        mD_a= None, mD_b = None, mass_function = None):
        D_max = np.nanmax(D_bin_edges)
        D_min = np.nanmin(D_bin_edges)
        super().__init__(D_min, D_max, mD_a= mD_a, mD_b = mD_b, 
                mass_function = mass_function)
        self.func = interp1d(D_bin_edges,np.r_[N,0.], kind = 'next',
        bounds_error = False, fill_value = 0.)
        
        


        
        
        