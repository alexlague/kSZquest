'''
Class to host lightcone loaded from data
or generated using an Nbody class
'''

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RegularGridInterpolator
from scipy.optimize import minimize

import nbodykit
from nbodykit.lab import *

from hmvec.ksz import get_ksz_auto_squeezed

class LightCone:
    '''
    Lightcone class to store RA, DEC, z and compute multipole moments 
    '''

    def __init__(self, FSKY, Nmesh=64, CosmoParams=None):
        if CosmoParams == None:
            print("No cosmological parameters specified: assuming Planck15")
            self.cosmo = cosmology.Planck15

            self.CosmoParams           = {}
            self.CosmoParams['omch2']  = self.cosmo.Odm0 * self.cosmo.h**2
            self.CosmoParams['ombh2']  = self.cosmo.Ob0 * self.cosmo.h**2
            self.CosmoParams['hubble'] = self.cosmo.h * 100
            self.CosmoParams['As']     = self.cosmo.A_s
            self.CosmoParams['ns']     = self.cosmo.n_s
            self.CosmoParams['mnu']    = self.cosmo.m_ncdm
            self.CosmoParams['tau']    = self.cosmo.tau_reio
        
        else:
            pass #TODO

        self.FSKY = FSKY
        self.Nmesh = Nmesh
        
        return

    def LoadGalaxies(self, SimDir, SimType, GenerateRand=False, zmin=0.0, zmax=5.0, alpha=0.1):
        if SimType=='Magneticum':
            halos = np.loadtxt(SimDir)
            flag = halos[:,17]
            halos = halos[flag==0]
            xpix     = halos[:,1]
            ypix     = halos[:,2]
            z_obs    = halos[:,7]

            ind = (z_obs <= zmax) & (z_obs >= zmin)
            
            self.data = {}
            self.data['ra'] = (xpix[ind]-.5) * 35
            self.data['dec'] = (ypix[ind]-.5) * 35
            self.data['z'] = z_obs[ind]
        else:
            raise Exception("Simulation type not implemented")
        
        self.minRA = self.data['ra'].min()
        self.maxRA = self.data['ra'].max()
        self.minDEC = self.data['dec'].min()
        self.maxDEC = self.data['dec'].max()
        self.minZ = self.data['z'].min()
        self.maxZ = self.data['z'].max()

        # Store in ArrayCatalog object
        self.data = ArrayCatalog(self.data)

        # Include Cartesian coords
        self.data['Position'] = transform.SkyToCartesian(self.data['ra'], self.data['dec'], self.data['z'], cosmo=self.cosmo)
        
        if GenerateRand:
            # Calculate nofz from data
            self.CalculateNofZ(UseData=True)
            self.GenerateRandoms(alpha=alpha)

        # TODO: add warning if alpha changed but not generating randoms
        
        return
    
    def LoadRandoms(self, SimDir, SimType):
        return

    def GenerateFKPCatalog(self):
        
        self.fkp_catalog = FKPCatalog(self.data, self.randoms)
        self.fkp_catalog['randoms/NZ'] = self.nofz(self.randoms['z'])
        self.fkp_catalog['data/NZ'] = self.nofz(self.data['z'])
        
        # TODO: function in case FKP/completeness included in data
        self.fkp_catalog['data/FKPWeight'] = 1.0 / (1 + self.fkp_catalog['data/NZ'] * 1e4)
        self.fkp_catalog['randoms/FKPWeight'] = 1.0 / (1 + self.fkp_catalog['randoms/NZ'] * 1e4)
        
        return

    def LoadCMBMap(self, SimDir, SimType):
        return

    def CalculateWindowFunction(self):
        return

    def DeconvolveWindowFunction(self):
        return

    def GenerateRandoms(self, alpha=0.1):
        
        rand = RandomCatalog(int(2*len(self.data)/alpha))
        rand['z']   = rand.rng.uniform(low=self.minZ, high=self.maxZ)
        rand['ra']  = rand.rng.uniform(low=self.minRA, high=self.maxRA)
        rand['dec'] = rand.rng.uniform(low=self.minDEC, high=self.maxDEC)
        
        # Subselect z to match nofz from data
        h_indexes = np.arange(len(rand['z']))
        hzs = rand['z']
        dist = self.nofz(hzs) # distribution
        dist /= np.sum(dist) # normalized
        Nrand =  self.data.csize / alpha

        his = np.random.choice(h_indexes, size=int(Nrand), p=dist, replace=False) # indexes of selected halos

        self.randoms = {}
        self.randoms['z'] = rand['z'][his]
        self.randoms['ra'] = rand['ra'][his]
        self.randoms['dec'] = rand['dec'][his]

        self.randoms = ArrayCatalog(self.randoms)
        self.randoms['Position'] = transform.SkyToCartesian(self.randoms['ra'], self.randoms['dec'], self.randoms['z'], cosmo=self.cosmo)
        self.randoms['NZ'] = self.nofz(self.randoms['z'])
        
        return
    
    def CalculateNofZ(self, UseData=False):
        
        if UseData:
            zhist = RedshiftHistogram(self.data, self.FSKY, self.cosmo, redshift='z')
            self.nofz = InterpolatedUnivariateSpline(zhist.bin_centers, zhist.nbar)
            self.data['NZ'] = self.nofz(self.data['z'])
        else:
            zhist = RedshiftHistogram(self.randoms, self.FSKY, self.cosmo, redshift='z')
            self.nofz = InterpolatedUnivariateSpline(zhist.bin_centers, self.alpha*zhist.nbar)

        return

    def ComputeZeff(self):
        return
    
    def PaintMesh(self):
        # TODO: include completeness weights
        self.halo_mesh = self.fkp_catalog.to_mesh(Nmesh=self.Nmesh, nbar='NZ', fkp_weight='FKPWeight')
        
        return
    
    def CalculateMultipoles(self, poles=[0, 2, 4], kmin=0.0, kmax=0.3, dk=5e-3):
        self.r = ConvolvedFFTPower(self.halo_mesh, poles=poles, dk=dk, kmin=kmin, kmax=kmax).poles
        return

    def GetPowerSpectraModel(self):
        nzbins = 5
        ls = np.arange(1000)
        zs = np.linspace(self.minZ, self.maxZ, nzbins)
        #Delta_chi = self.cosmo.comoving_distance(self.maxZ) - self.cosmo.comoving_distance(self.minZ)
        #vol = 4 * np.pi/3 * self.FSKY * (Delta_chi/self.cosmo.h)**3
        #vol /= 1000**3 # to Gpc^3
        vol = 100 # low enough kmin for interpolation
        ngals = self.nofz(zs) * self.cosmo.h**3 # to 1/Mpc^3

        if hasattr(self, "bg"):
            bgs = self.bg * np.ones(len(zs)) #* self.cosmo.h**3 # TODO moe h^3 correction to bg
        else:
            bgs = None
        
        model = get_ksz_auto_squeezed(ells=ls, volume_gpc3=vol, zs=zs, ngals_mpc3=ngals, params=self.CosmoParams, template=True, rsd=True, bgs=bgs)
        
        self.model = model[2]

        print(self.model.keys())
        # Getting Pkmu at zeff
        mus = np.linspace(-1, 1, len(self.model['lPggtot'][0, 0, :]))
        Pge_kmu_interp = RegularGridInterpolator((zs, self.model['ks'], mus), self.model['sPge'], bounds_error=False, fill_value=0.)
        Pgg_kmu_interp = RegularGridInterpolator((zs, self.model['ks'], mus), self.model['lPggtot'], bounds_error=False, fill_value=0.)


        if hasattr(self, "zeff"):
            self.Pge_kmu = lambda k, mu: Pge_kmu_interp((zeff, k, mu))
            self.Pgg_kmu = lambda k, mu: Pgg_kmu_interp((zeff, k, mu))
        else:
            self.Pge_kmu = lambda k, mu: Pge_kmu_interp(((self.maxZ+self.minZ)/2, k, mu))
            self.Pgg_kmu = lambda k, mu: Pgg_kmu_interp(((self.maxZ+self.minZ)/2, k, mu))
            
        return
        
    def FitPgg(self):
        
        #nzbins = len(self.model['lPggtot'])
        #zs = np.linspace(self.minZ, self.maxZ, nzbins)

        # TODO: check ell normaliztion of missing 1/2
        #Pell0   = np.trapz(self.model['lPggtot'], np.linspace(-1, 1, 102), axis=-1).T # integrate over mu
        #Pell0_interp = RegularGridInterpolator((self.model['ks'], zs), Pell0)
        
        # TODO: check units of plin
        if hasattr(self, "zeff"):
            plin = cosmology.LinearPower(self.cosmo, self.zeff, transfer='EisensteinHu')
            f = self.cosmo.scale_independent_growth_factor(self.zeff)
        else:
            plin = cosmology.LinearPower(self.cosmo, (self.maxZ+self.minZ)/2, transfer='EisensteinHu')
            f = self.cosmo.scale_independent_growth_factor((self.maxZ+self.minZ)/2)

        P0 = lambda k, b: (b**2 + b*f/3 + f**2/5) * plin(k)
        P0_data = self.r['power_0'].real - self.r.attrs['shotnoise']
        k_data  = self.r['k']
        
        # take difference in power on large scales to correct galaxy bias
        to_min  = lambda b: np.sum((P0(k_data[k_data<=0.07], b) - P0_data[k_data<=0.07])**2)
        
        best_fit_bg = minimize(to_min, x0=1.)['x']
        
        self.bg = best_fit_bg
        
        return
        
