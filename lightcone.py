'''
Class to host lightcone loaded from data
or generated using an Nbody class
'''

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RegularGridInterpolator
from scipy.optimize import minimize

import nbodykit
from nbodykit.lab import *

class LightCone:
    '''
    Lightcone class to store RA, DEC, z and compute multipole moments 
    '''

    def __init__(self, FSKY, Nmesh=64, CosmoParams=None):
        if CosmoParams == None:
            print("No cosmological parameters specified: assuming Planck15")
            self.cosmo = cosmology.Planck15
        else:
            pass #TODO

        self.FSKY = FSKY
        self.Nmesh = Nmesh
        
        return

    def LoadGalaxies(self, SimDir, SimType, GenerateRand=False, zmin=0.0, zmax=5.0):
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
            self.GenerateRandoms()

        return
    
    def LoadRandoms(self, SimDir, SimType):
        return

    def GenerateFKPCatalogs(self):
        self.fkp = {}
        self.fkp['randoms/NZ'] = self.nofz(randoms['z'])
        self.fkp['data/NZ'] = self.nofz(data['z'])
        self.fkp_catalog = FKPCatalog(self.data, self.randoms)
        
        # TODO: function in case FKP/completeness included in data
        fkp['data/FKPWeight'] = 1.0 / (1 + fkp['data/NZ'] * 1e4)
        fkp['randoms/FKPWeight'] = 1.0 / (1 + fkp['randoms/NZ'] * 1e4)
        
        return

    def LoadCMBMap(self, SimDir, SimType):
        return

    def CalculateWindowFunction(self):
        return

    def DeconvolveWindowFunction(self):
        return

    def GenerateRandoms(self, alpha=0.1):
        
        self.randoms = RandomCatalog(int(2*len(self.data)/alpha))
        self.randoms['z']   = self.randoms.rng.uniform(low=self.minZ, high=self.maxZ)
        self.randoms['ra']  = self.randoms.rng.uniform(low=self.minRA, high=self.maxRA)
        self.randoms['dec'] = self.randoms.rng.uniform(low=self.minDEC, high=self.maxDEC)
        
        # Subselect z to match nofz from data
        h_indexes = np.arange(len(self.randoms['z']))
        hzs = self.randoms['z']
        dist = self.nofz(hzs) # distribution
        dist /= np.sum(dist) # normalized
        Nrand =  self.data.csize / alpha

        his = np.random.choice(h_indexes, size=int(Nrand), p=dist, replace=False) # indexes of selected halos

        # TODO: fix waaayy too slow
        ind = np.zeros(len(self.randoms['z']))
        for i in range(len(ind)):
            if i in his:
                ind[i] = True
            else:
                ind[i] = False

        self.randoms['z'] = self.randoms['z'][ind]
        self.randoms['ra'] = self.randoms['ra'][ind]
        self.randoms['dec'] = self.randoms['dec'][ind]

        self.randoms['Position'] = transform.SkyToCartesian(self.randoms['ra'], self.randoms['dec'], self.randoms['z'], cosmo=self.cosmo)
        
        return
    
    def CalculateNofZ(self, UseData=False):
        
        if UseData:
            zhist = RedshiftHistogram(self.data, self.FSKY, self.cosmo, redshift='z')
            self.nofz = InterpolatedUnivariateSpline(zhist.bin_centers, zhist.nbar)
        else:
            zhist = RedshiftHistogram(self.randoms, self.FSKY, self.cosmo, redshift='z')
            self.nofz = InterpolatedUnivariateSpline(zhist.bin_centers, self.alpha*zhist.nbar)

        return

    def ComputeZeff(self):
        return
    
    def PaintMesh(self):
        self.galaxy_mesh = self.fkp.to_mesh(Nmesh=self.Nmesh, nbar='NZ', fkp_weight='FKPWeight')
        
        return
    
    def CalculateMultipoles(self, poles=[0, 2, 4], kmin=0.0, kmax=0.3, dk=5e-3):
        self.r = ConvolvedFFTPower(self.galaxy_mesh, poles=poles, dk=dk, kmin=kmin).poles
        return

    def GetPowerSpectraModel(self):
        nzbins = 10
        ls = np.arange(1000)
        zs = np.linspace(self.minZ, self.maxZ, nzbins)
        zs_edges = np.linspace(self.minZ, self.maxZ, nzbins+1)
        chis = self.cosmo.comoving_distance(zs_edges) / self.cosmo.h # switch to Mpc
        volumes = 4 * np.pi/3 * self.FSKY * np.array([chis[i+1]-chis[i] for i in range(nzbins)])**3
        volumes /= 1000**3 # to Gpc^3
        ngals = self.nofz(zs) * self.cosmo.h**3 # to 1/Mpc^3
        
        model = get_ksz_auto_squeezed(np.arange(1000), volumes, zs, ngals_mpc3=ngals, params=self.CosmoParams, template=True, rsd=True)
        
        self.model = model[2]
        
        # Getting Pkmu at zeff
        mus = np.linspace(-1, 1, len(self.model['sPgg'][0, 0, :]))
        Pge_kmu_interp = RegularGridInterpolator((zs, self.model['ks'], mus), self.model['sPge'])
        Pgg_kmu_interp = RegularGridInterpolator((zs, self.model['ks'], mus), self.model['sPggtot'])

        if hasattr(self, "zeff"):
            self.Pge_kmu = lambda k, mu: Pge_kmu_interp(zeff, k, mu)
            self.Pgg_kmu = lambda k, mu: Pgg_kmu_interp(zeff, k, mu)
        else:
            self.Pge_kmu = lambda k, mu: Pge_kmu_interp((self.maxZ+self.minZ)/2, k, mu)
            self.Pgg_kmu = lambda k, mu: Pgg_kmu_interp((self.maxZ+self.minZ)/2, k, mu)
            
        return
        
    def FitPgg(self):
        
        nzbins = len(self.model['lPggtot'])
        zs = np.linspace(self.minZ, self.maxZ, nzbins)

        # TODO: check ell normaliztion of missing 1/2
        Pell0   = np.trapz(self.model['lPggtot'], np.linspace(-1, 1, 102), axis=-1).T # integrate over mu
        Pell0_interp = RegularGridInterpolator((self.model['ks'], zs), Pell0)
        
        if hasattr(self, "zeff"):
            P0 = lambda k: Pell0_interp(k, self.zeff, grid=False)
        else:
            P0 = lambda k: Pell0_interp(k, (self.maxZ+self.minZ)/2, grid=False)
            
        P0_data = self.r['power_0'].real
        k_data  = self.r['k']
        
        # take difference in power on large scales to correct galaxy bias
        to_min  = lambda bg_over_bg_fid: (bg_over_bgfid * P0(k_data[k_data<=0.07]) - P0_data[k_data<=0.07])**2
        
        best_fit_bg = minimize(to_min, x0=1.)['x']
        
        self.bg_correction = best_fit_bg
        
        return
        
