'''
Functions to compute the velocity reconstruction
from a CMB map and a halo density field
'''

import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline

import nbodykit
from nbodykit.lab import *

import lightcone
import nbody


def CheckSource(Source):
    if type(Source) not in [nbody.NBodySim, lightcone.LightCone]:
        raise Exception("Unsupported Source: need to create an nbodysim or lightcone instance")

    # TODO: Read off from pixell or healpix formats
    if type(Source) == lightcone.LightCone and (CMBMap.shape[0] != Nmesh or CMBMap.shape[1] != Nmesh) and (RA==None or DEC==None):
        raise Exception("Incomplete CMB Map: need to specify RA and DEC arrays")

    # TODO: check to make sure Pk are computed
    if not (hasattr(Source, "Phh") or hasattr(Source, "Phh_kmu")) and type(Source) == nbody.NBodySim:
        raise Exception("Need to compute either the 1D or 2D power spectrum in nbodysim to run reconstruction")

    return

def CreateTGrid(Source, CMBMap, RA=None, DEC=None):
    '''
    '''
    
    Nmesh = Source.Nmesh
    T_grid = np.zeros((Nmesh, Nmesh, Nmesh))

    if CMBMap.shape[0] != Nmesh or CMBMap.shape[1] != Nmesh:
        CMBMap_interp = interp2d(RA, DEC, CMBMap)
        RA_mesh = np.linspace(np.min(RA), np.max(RA), Nmesh)
        DEC_mesh = np.linspace(np.min(DEC), np.max(DEC), Nmesh)
        CMBMap = CMBMap_interp(RA_mesh, DEC_mesh)

    # Assumes the plane-parallel approximation
    # Copies over the CMB map on a grid
    for i in range(Nmesh): T_grid[i] = CMBMap
    
    return T_grid

def CreateFilters(Source, Iso=True):
    '''
    Iso: bool, use isotropic filtering with k = |k| or (if False) anisotropic using k and mu
    '''
    
    filter_dict = {}

    if type(Source) == nbody.NBodySim and Iso:
        k     = Source.Pmh['k']
        Pk_mh = Source.Pmh['power'].real
        Pk_hh = Source.Phh['power'].real

        indexes = (~np.isnan(Pk_mh)) & (~np.isnan(Pk_hh)) & (k>0)
        k       = k[indexes]
        Pk_mh   = Pk_mh[indexes]
        Pk_hh   = Pk_hh[indexes]
    
        log_P_ge_logk = interp1d(np.log10(k), np.log10(Pk_mh), bounds_error=False, fill_value=0, kind='cubic')
        log_P_gg_logk = interp1d(np.log10(k), np.log10(Pk_hh), bounds_error=False, fill_value=0, kind='cubic')
        
        def pge_pgg_filter(k, v):
            kk = sum(ki ** 2 for ki in k) # k^2 on the mesh
            kk[kk == 0] = 1
            num = 10**log_P_ge_logk(np.log10(np.sqrt(kk)))
            den = 10**log_P_gg_logk(np.log10(np.sqrt(kk)))
            return v * num/den # apply delta_e filter
            
        def pge2_pgg_filter(k, v):
            kk = sum(ki ** 2 for ki in k) # k^2 on the mesh
            kk[kk == 0] = 1
            num = 10**log_P_ge_logk(np.log10(np.sqrt(kk)))
            den = 10**log_P_gg_logk(np.log10(np.sqrt(kk)))
            return v * num**2 / den # apply delta_e filter

    elif type(Source) == nbody.NBodySim and Iso == False:
        k     = Source.Pmh_kmu.coords['k']
        mu    = Source.Pmh_kmu.coords['mu']
        Pk_mh = Source.Pmh_kmu['power'].real
        Pk_hh = Source.Phh_kmu['power'].real
        muind = mu > 0
        kind  = k > 0
        k     = k[kind]
        mu    = mu[muind]
        Pk_mh = Pk_mh[kind].T[muind].T
        Pk_hh = Pk_hh[kind].T[muind].T
        
        log_P_ge_logk = RectBivariateSpline(np.log10(k), mu, np.log10(Pk_mh), kx=1, ky=1)
        log_P_gg_logk = RectBivariateSpline(np.log10(k), mu, np.log10(Pk_hh), kx=1, ky=1)
        
        def pge_pgg_filter(k, v):
            kk = sum(ki ** 2 for ki in k) # k^2 on the mesh
            kk[kk == 0] = 1
            mu = abs(k[2] / kk)
            num = 10**log_P_ge_logk(np.log10(np.sqrt(kk)), mu)
            den = 10**log_P_gg_logk(np.log10(np.sqrt(kk)), mu)
            fil = num / den
            fil[np.isnan(fil)] = 1.
            return v * fil # apply delta_e filter
        
        def pge2_pgg_filter(k, v):
            kk = sum(ki ** 2 for ki in k) # k^2 on the mesh
            kk[kk == 0] = 1
            mu = abs(k[2] / kk)
            num = 10**log_P_ge_logk(np.log10(np.sqrt(kk)), mu, grid=False)
            den = 10**log_P_gg_logk(np.log10(np.sqrt(kk)), mu, grid=False)
            fil = num**2 / den
            fil[np.isnan(fil)] = 1.
            return v * fil # apply delta_e filter

    elif type(Source) == lightcone.LightCone:
        pass # TODO


    filter_dict['pge_pgg'] = pge_pgg_filter
    filter_dict['pge2_pgg'] = pge2_pgg_filter

    return filter_dict

def Delta_eField(Source, FilterDictionary):
    '''
    Filter delta_g to get delta_e using Pge/Pgg
    For nbody data, the Pge and Pgg are known
    For lightcone data, the Pge and Pgg have to be modeled
    '''
    
    # Estimate delta_e on grid
    pge_pgg_filter = FilterDictionary['pge_pgg']
    mesh_delta_e  = Source.halo_mesh.apply(pge_pgg_filter, mode='complex', kind='wavenumber')
    delta_e_field = mesh_delta_e.to_field() # get data in array form

    return delta_e_field

def CalculateNoise(Source, Field, FilterDictionary):
    '''
    Calculation of the noise for the quadratic estimator
    '''

    # TODO: CMB Noise f_2_of_k
    
    ones_field_k = Field.copy().r2c() # delta_e or delta_e_times_T work to make sure they have the same shape
    ones_field_k /= ones_field_k
    ones_field_k[np.isnan(ones_field_k)] = 0.
    
    pge2_pgg_filter = FilterDictionary['pge2_pgg']
    ones_field_k = ones_field_k.apply(pge2_pgg_filter, kind='wavenumber')
    
    f_1_of_x = ones_field_k.c2r()
    
    noise_of_k = f_1_of_x.r2c()

    return noise_of_k


def RunReconstruction(Source, CMBMap, RA=None, DEC=None, ComputePower=True, dk=5e-3, Iso=True, Nmu=5):
    '''
    Input: Source - Either nbodysim or lightcone object
    '''
    
    CheckSource(Source)
    
    T_grid          = CreateTGrid(Source, CMBMap, RA=RA, DEC=DEC)
    filter_dict     = CreateFilters(Source, Iso=Iso)
    delta_e_field   = Delta_eField(Source, filter_dict)
    delta_e_times_T = delta_e_field * T_grid
    noise_of_k      = CalculateNoise(Source, delta_e_times_T, filter_dict)
    vhat_of_k       = (noise_of_k)**-1 * delta_e_times_T.r2c()
    vhat            = vhat_of_k.c2r()
    
    if type(Source) == nbody.NBodySim and ComputePower and Iso:
        Pk_vv = FFTPower(vhat_of_k, mode='1d', dk=dk, kmax=0.3, kmin=0,).power
        Pk_vq = FFTPower(vhat_of_k, second=Source.particle_momentum_mesh, mode='1d', dk=dk, kmax=0.3, kmin=0).power
        Pk_qq = FFTPower(Source.particle_momentum_mesh, mode='1d', dk=dk, kmax=0.3, kmin=0).power
        
        return vhat, Pk_vv, Pk_vq, Pk_qq

    if type(Source) == nbody.NBodySim and ComputePower and Iso==False:
        Pk_vv = FFTPower(vhat_of_k, mode='2d', dk=dk, kmax=0.3, kmin=0, Nmu=Nmu).power
        Pk_vq = FFTPower(vhat_of_k, second=Source.particle_momentum_mesh, mode='2d', dk=dk, kmax=0.3, kmin=0, Nmu=Nmu).power
        Pk_qq = FFTPower(Source.particle_momentum_mesh, mode='2d', dk=dk, kmax=0.3, kmin=0, Nmu=Nmu).power

        return vhat, Pk_vv, Pk_vq, Pk_qq
    
    elif type(Source) == lightcone.LightCone and ComputePower:
        # TODO: add 3d pk mu
        Pk_vv = ConvolvedFFTPower(vhat_of_k, mode='1d', poles=[0, 2], dk=dk, kmax=0.3, kmin=0,).power
        
        return vhat, Pk_vv #TODO Pk_vq
    
    else:
        return vhat
