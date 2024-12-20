####
# Functions to compute the velocity reconstruction
# from a CMB map and a halo density field
###

import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline, RegularGridInterpolator, CubicSpline
from scipy import constants

import nbodykit
from nbodykit.lab import *
import pmesh
try:
    from nbodykit.algorithms.convpower.catalog import FKPVelocityCatalog
except:
    pass

from pypower import CatalogFFTPower

import healpy as hp
from pixell import enmap

import lightcone
import nbody
import cmb


def CheckSource(Source):
    if type(Source) not in [nbody.NBodySim, lightcone.LightCone]:
        raise Exception("Unsupported Source: need to create an nbodysim or lightcone instance")

    # TODO: Read off from pixell or healpix formats
    #if type(Source) == lightcone.LightCone and (Map.shape[0] != Nmesh or Map.shape[1] != Nmesh) and (RA==None or DEC==None):
    #    raise Exception("Incomplete CMB Map: need to specify RA and DEC arrays")

    # TODO: check to make sure Pk are computed
    if not (hasattr(Source, "Phh") or hasattr(Source, "Phh_kmu")) and type(Source) == nbody.NBodySim:
        raise Exception("Need to compute either the 1D or 2D power spectrum in nbodysim to run reconstruction")

    if not hasattr(Source, "Pgg_kmu") and type(Source) == lightcone.LightCone:
        raise Exception("Need to compute fiducial Pge/Pgg spectra from model before running reconstruction")

    return

def CreateTGrid(Source, CMBMap, RA=None, DEC=None, NSIDE=None):
    '''
    '''
    
    Nmesh = Source.Nmesh
    T_grid = np.zeros((Nmesh, Nmesh, Nmesh))
    
    if hasattr(Source, "minRA") is True:
        RA_mesh = np.linspace(Source.minRA, Source.maxRA, Nmesh)
        DEC_mesh = np.linspace(Source.minDEC, Source.maxDEC, Nmesh)
    
    if isinstance(Source, nbody.NBodySim):
        if (CMBMap.shape[0] != Nmesh or CMBMap.shape[1] != Nmesh) and hasattr(Source, "minRA") is True:
            if type(CMBMap) == cmb.CMBMap:
                RAs = np.linspace(Source.minRA, Source.maxRA, CMBMap.shape[0])
                DECs = np.linspace(Source.minDEC, Source.maxDEC, CMBMap.shape[1])
                CMBMap_interp = RectBivariateSpline(RAs, DECs, CMBMap.to_array())
            else:
                print(np.array(CMBMap).shape)
                RAs = np.linspace(Source.minRA, Source.maxRA, np.array(CMBMap).shape[0])
                DECs = np.linspace(Source.minDEC, Source.maxDEC, np.array(CMBMap).shape[1])
                CMBMap_interp = RectBivariateSpline(RAs, DECs, np.array(CMBMap))
        
            CMBMap = CMBMap_interp(RA_mesh, DEC_mesh)
    
        elif (CMBMap.shape[0] != Nmesh or CMBMap.shape[1] != Nmesh) and hasattr(Source, "minRA") is False:
            raise Exception("The shape of the CMB map does not match the shape of the density grid; specify RA and DEC range")

        # Assumes the plane-parallel approximation
        # Copies over the CMB map on a grid
        for i in range(Nmesh): T_grid[i] = CMBMap

    else:
        # for lightcone, the picture is a bit more complicated
        # scale the angles as function of los distance
        # use RectGridInterpolator to set fill_value to 0. outside RA/DEC range
        # TODO: use regular grid interp everywhere
       
        # T grid procedure
        ras = np.array(Source.data['ra'])
        decs = np.array(Source.data['dec']) 
        #print(np.min(ras), np.max(ras), np.min(decs), np.max(decs), "0.43", "0.7")
        
        #ras = np.array(Source.randoms['ra'])
        #decs = np.array(Source.randoms['dec'])
        #data_pos = np.array(Source.randoms['Position']) - np.min(np.array(Source.randoms['Position']), axis=0)
        #data_pos = data_pos[::10]
        data_pos = np.array(Source.data['Position'])-np.min(np.array(Source.data['Position']), axis=0)
        
        trial_grid = False # debugging option
        zshuffle = False
        cart_grid = False # using cartesian grid of points
        sph_grid = False # using spherical (ra, dec, z) grid of points 
        if trial_grid:
            N = 768 #10_000_000
            
            if cart_grid:
                data_pos = np.array(Source.data['Position']) # don't remove min
                xmin, ymin, zmin = np.min(data_pos, axis=0)
                xmax, ymax, zmax = np.max(data_pos, axis=0)
                xc = np.linspace(xmin, xmax, N)
                yc = np.linspace(ymin, ymax, N)
                #zc = np.linspace(zmin, zmax, Source.Nmesh)
                zc = np.linspace(Source.cosmo.comoving_distance(0.40), Source.cosmo.comoving_distance(0.73), N)
                #print("Edges: ", xmin, xmax, ymin, ymax, zmin, zmax)
                
                xc, yc, zc = np.meshgrid(xc, yc, zc)
                #xc +=abs(np.min(np.array(Source.data['Position']), axis=0))[0]
                #yc +=abs(np.min(np.array(Source.data['Position']), axis=0))[1]
                #zc +=abs(np.min(np.array(Source.data['Position']), axis=0))[2]
                data_pos = np.array([xc.ravel(), yc.ravel(), zc.ravel()]).T
                #obs = -np.min(np.array(Source.data['Position']), axis=0)
                ras, decs, redshifts = transform.CartesianToSky(data_pos, cosmo=Source.cosmo) #observer=obs)
                ras = np.array(ras)
                decs = np.array(decs)
                #redshifts = np.array(redshifts)
                #print(np.min(ras), np.max(ras),np.min(decs), np.max(decs), np.min(redshifts), np.max(redshifts))
                del xc
                del yc
                del zc
                del redshifts

            elif sph_grid:                
                ras = np.linspace(ras.min(), ras.max(), N)
                decs = np.linspace(decs.min(), decs.max(), N)
                zs = np.linspace(0.43, 0.7, N)
                #redshifts = np.array(Source.data['z'])
                #zs_interp = np.linspace(0.3, 0.8, 5000)
                #z_from_chis = CubicSpline(Source.cosmo.comoving_distance(zs_interp), zs_interp)
                #zs = z_from_chis(np.linspace(Source.cosmo.comoving_distance(0.43), Source.cosmo.comoving_distance(0.7), N)) # uniform in chi
            
                ras, decs, zs = np.meshgrid(ras, decs, zs)
                ras = ras.ravel()
                decs = decs.ravel()
                zs = zs.ravel()
                data_pos = transform.SkyToCartesian(ras, decs, zs, Source.cosmo)
        
        if zshuffle:
            Nc = 512
            zs = np.array([np.linspace(0.43, 0.7, len(ras)) for _ in range(Nc)]).ravel()
            ras = np.array([ras for _ in range(Nc)]).ravel()
            decs = np.array([decs for _ in range(Nc)]).ravel()
            data_pos = transform.SkyToCartesian(ras, decs, zs, Source.cosmo)
            data_pos = np.array(data_pos) - np.min(data_pos, axis=0)

        if CMBMap.ndim == 2:
            RAs = np.linspace(Source.minRA, Source.maxRA, CMBMap.shape[0])
            DECs = np.linspace(Source.minDEC,Source.maxDEC, CMBMap.shape[1])
            
            CMBMap_interp = RectBivariateSpline(RAs, DECs, CMBMap)
            
            
            T_vals = CMBMap_interp(ras, decs, grid=False)
            

        elif CMBMap.ndim == 1:
            
            # Read pixel values
            pixels = hp.ang2pix(NSIDE, ras, decs, lonlat=True)  
            T_vals = CMBMap[pixels]
            

        Source.T_vals = T_vals

    return



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

        # Load Pgg and Pge from model
        Pge = Source.Pge_kmu
        Pgg = Source.Pgg_kmu
        h = Source.cosmo.h

        def pge_pgg_filter(k, v):
            kk = sum(ki ** 2 for ki in k) # k^2 on the mesh
            kk[kk == 0] = 1
            mu = k[2] / kk #DEBUG
            num = Pge(np.sqrt(kk), mu)#*h**3
            den = Pgg(np.sqrt(kk), mu)#*h**3
            fil = num / den
            fil[np.isnan(fil)] = 1.
            return v * fil
        
        def pge2_pgg_filter(k, v):
            kk = sum(ki ** 2 for ki in k) # k^2 on the mesh
            kk[kk == 0] = 1
            mu = k[2] / kk #DEBUG k[0] or k[2]?
            num = Pge(np.sqrt(kk), mu)#*h**3
            den = Pgg(np.sqrt(kk), mu)#*h**3
            fil = num**2 / den
            fil[np.isnan(fil)] = 1.
            return v * fil


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
    
    if isinstance(Source.halo_mesh, pmesh.pm.RealField):
        delta_e_field  = Source.halo_mesh.r2c().apply(pge_pgg_filter, kind='wavenumber').c2r()
        
    else:
        mesh_delta_e  = Source.halo_mesh.apply(pge_pgg_filter, mode='complex', kind='wavenumber')
        delta_e_field = mesh_delta_e.to_field() - 1 # get data in array form

    return delta_e_field

def CalculateNoise(Source, Field, FilterDictionary, ClMap=None, RA=None, DEC=None):
    '''
    Calculation of the noise for the quadratic estimator
    '''

    # TODO: CMB Noise f_2_of_k
    
    ones_field_k = Field.copy().r2c() # delta_e or delta_e_times_T work to make sure they have the same shape
    ones_field_k /= ones_field_k
    ones_field_k[np.isnan(ones_field_k)] = 0.
    
    pge2_pgg_filter = FilterDictionary['pge2_pgg']
    ones_field_k    = ones_field_k.apply(pge2_pgg_filter, kind='wavenumber')

    f_1_of_x = ones_field_k.c2r() #.paint(mode='real') #ones_field_k.c2r()
    
    ## DEBUG
    #if type(ClMap) == np.ndarray or type(ClMap) == enmap.ndmap:
    #    f_2_of_x = CreateTGrid(Source, ClMap, RA=RA, DEC=DEC)
    #    noise_of_k = (f_1_of_x * f_2_of_x).r2c()
    #    
    #else:
    noise_of_k = f_1_of_x.r2c()
    
    #ClTT not passed from lightcone object...
    #if type(Source) == lightcone.LightCone:
    #    Source.cltt
    #    one_over_cl_interp = interp1d(np.arange(len(one_over_cl)), one_over_cl, bounds_error=False, fill_value=0.)
    #    chi_star = Source.comsmo.comoving_distance(Source.zeff) / Source.cosmo.h # to Mpc CHECK kh units!
    #    ell_for_int = chi_star*model_full[2]['ks']
    #    integrand = model_full[2]['ks']*model_full[2]['sPge'][2][:,51]**2/model_full[2]['sPggtot'][2][:,51]
    #    integrand *= one_over_cl_interp(ell_for_int) /2/np.pi
    #    noise_of_k = np.trapz(integrand, x=model_full[2]['ks'])
    
    return noise_of_k


def CalculateNoiseFromFilter(Source, CMBFilterPath=None):
    '''
    Follows the calculation of 1810.13423 Eq. (55)-(56)
    '''

    if hasattr(Source, "K_star") == False:
        Source.CalculatePrefactors()

    #Cls = Cls[2:] # remove monopole and dipole
    #one_over_cl = 1. / Cls
    #one_over_cl_interp = interp1d(np.arange(2, len(one_over_cl)-2), one_over_cl, 
    #                          bounds_error=False, 
    #                          fill_value=one_over_cl[-1])
    
    # Load filter
    fil = np.loadtxt(CMBFilterPath)
    one_over_cl_interp = interp1d(fil[:,0], fil[:,1], bounds_error=False, fill_value=(0., fil[:,1][-1])) #fil[:,1][-1])
    
    Pge = Source.Pge_kmu
    Pgg = Source.Pgg_kmu
    
    # Assume isotropic noise with mu = 0
    k_samples = np.geomspace(1e-3, 100, 2000)
    Pge_samples = Pge(k_samples) # now removed mu
    Pgg_samples = Pgg(k_samples)

    # Compute integral
    ell_for_int = k_samples * Source.Chi_star
    integrand = k_samples * Pge_samples**2 / Pgg_samples
    integrand[np.isnan(integrand)] = 0.
    integrand *= one_over_cl_interp(ell_for_int) / 2 / np.pi
    integral = np.trapz(integrand, x=k_samples)
    
    # Nvr
    noise = Source.Chi_star**2 / Source.K_star**2 * integral**-1

    # prefactor
    full_prefactor = Source.K_star / Source.Chi_star**2 * noise
    
    return full_prefactor, noise

def PaintedVelocities(Source, vhat):
    '''
    '''
    
    positions = np.array(Source.data['Position'])
    xgrid = np.linspace(np.min(positions[:,0]), np.max(positions[:,0]), Source.Nmesh)
    ygrid = np.linspace(np.min(positions[:,1]), np.max(positions[:,1]), Source.Nmesh)
    zgrid = np.linspace(np.min(positions[:,2]), np.max(positions[:,2]), Source.Nmesh)
    
    # Undo reshuffling done by nbodykit before interpolation
    vhat_rolled = vhat
    for i in range(3):
        vhat_rolled = np.roll(vhat_rolled, int(Source.Nmesh/2), axis=i)

    velocity_interp = RegularGridInterpolator((xgrid, ygrid, zgrid), vhat_rolled)
    
    return velocity_interp(positions)

def ReconstructedVelocityMesh(Source, painted_velocities):
    '''
    '''
    #halo_cat_with_vhat = Source.
    #halo_cat_with_vhat['Position'] = np.array(Source.data['Position'])
    #halo_cat_with_vhat['Vz'] = painted_velocities
    vhat_fkp_cat = FKPVelocityCatalog(Source.data)
    vhat_fkp_cat['data/NZ'] = Source.nofz(Source.data['z'])
    vhat_fkp_cat['data/Vz'] = painted_velocities / (np.var(painted_velocities) + 1e8*vhat_fkp_cat['data/NZ']) # replace true velocities with recon
    vhat_fkp_mesh = vhat_fkp_cat.to_mesh(Nmesh=Source.Nmesh, fkp_weight='Vz', resampler='tsc', compensated=False)
    
    return  vhat_fkp_mesh

def RunReconstruction(Source, CMBMap, ClMap=None, RA=None, DEC=None, NSIDE=None, ComputePower=True, dk=5e-3, Iso=True, Nmu=5, dk_poles=1e-2, kmax=0.3, use_T_grid=True):
    '''
    Input: Source - Either nbodysim or lightcone object
    '''
    
    CheckSource(Source)
    print(type(Source.halo_mesh))
    
    T_grid          = CreateTGrid(Source, CMBMap, RA=RA, DEC=DEC, NSIDE=NSIDE)
    filter_dict     = CreateFilters(Source, Iso=Iso)
    delta_e_field   = Delta_eField(Source, filter_dict)
    if use_T_grid:
        delta_e_times_T = delta_e_field * T_grid
    else:
        delta_e_times_T = delta_e_field
    #noise_of_k      = CalculateNoise(Source, delta_e_times_T, filter_dict, ClMap=ClMap, RA=RA, DEC=DEC)
    #vhat_of_k       = (noise_of_k)**-1 * delta_e_times_T.r2c()
    
    #vhat_of_k[~np.isfinite(vhat_of_k)] = 0. + 0. * 1j
    #vhat            = vhat_of_k.c2r()
    
    ## DEBUG
    vhat = delta_e_times_T

    vhat[delta_e_times_T==0] = 0. # electron velocity undefined where there are no electrons
    #vhat[Source.halo_mesh.to_field()==0.] = 0.
    print(np.max(T_grid))

    #vhat_at_halos   = PaintedVelocities(Source, vhat)
    #vhat_fkp_mesh   = ReconstructedVelocityMesh(Source, vhat_at_halos)

    #print(type(delta_e_field))
    
    if type(Source) == nbody.NBodySim and ComputePower and Iso:
        Pk_vv = FFTPower(vhat_of_k, mode='1d', dk=dk, kmax=kmax, kmin=0,).power
        Pk_vq = FFTPower(vhat_of_k, second=Source.particle_momentum_mesh, mode='1d', dk=dk, kmax=kmax, kmin=0).power
        Pk_qq = FFTPower(Source.particle_momentum_mesh, mode='1d', dk=dk, kmax=kmax, kmin=0).power
        
        return vhat, Pk_vv, Pk_vq, Pk_qq

    elif type(Source) == nbody.NBodySim and ComputePower and Iso==False:
        Pk_vv = FFTPower(vhat_of_k, mode='2d', dk=dk, kmax=kmax, kmin=0, Nmu=Nmu).power
        Pk_vq = FFTPower(vhat_of_k, second=Source.particle_momentum_mesh, mode='2d', dk=dk, kmax=kmax, kmin=0, Nmu=Nmu).power
        Pk_qq = FFTPower(Source.particle_momentum_mesh, mode='2d', dk=dk, kmax=kmax, kmin=0, Nmu=Nmu).power

        return vhat, Pk_vv, Pk_vq, Pk_qq
    
    elif type(Source) == lightcone.LightCone and ComputePower:
        # TODO: add 3d pk mu
        #Pk_vv = ConvolvedFFTPower(vhat_of_k, poles=[0, 2], dk=dk, kmax=0.3, kmin=0,).power
        #Pk_vg = ConvolvedFFTPower(vhat_of_k, second=Source.halo_mesh, poles=[0, 2], dk=dk, kmax=0.3, kmin=0,).power
        Pk_vv = FFTPower(vhat, mode='1d', dk=dk, kmin=0, kmax=kmax)
        Pk_vq = FFTPower(vhat, second=Source.halo_momentum_mesh, mode='1d', dk=dk, kmin=0, kmax=kmax)#, BoxSize=Pk_vv.attrs['BoxSize'])
        Pk_qq = FFTPower(Source.halo_momentum_mesh, mode='1d', dk=dk, kmin=0, kmax=kmax)
        #Pk_gg = FFTPower(Source.halo_mesh, mode='1d', dk=dk, kmin=0, kmax=0.3)
        
        #Pk_ell_vq = ConvolvedFFTPower(Source.halo_mesh, second=Source.halo_momentum_mesh, poles=[0,], dk=dk_poles, kmin=0.) # halo mesh must be first
        #Pk_ell_vq_hat = ConvolvedFFTPower(Source.halo_mesh, second=vhat_fkp_mesh, poles=[0,], dk=dk_poles, kmin=0.)

        
        return vhat_at_halos, Pk_vq, Pk_vv, Pk_qq, vhat, Source.halo_mesh, Source.halo_momentum_mesh #Pk_ell_vq, Pk_ell_vv, Pk_ell_qq 
    
    else:
        return vhat
