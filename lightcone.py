'''
Class to host lightcone loaded from data
or generated using an Nbody class
'''

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import nbodykit
from nbodykit.lab import *

class LightCone:
    '''
    Lightcone class to store RA, DEC, z and compute multipole moments 
    '''

    def __init__(self, FSKY, CosmoParams=None):
        if CosmoParams == None:
            print("No cosmological parameters specified: assuming Planck15")
            self.cosmo = cosmology.Planck15
        else:
            pass #TODO

        self.FSKY = FSKY
        
        return

    def LoadGalaxies(self, SimDir, SimType, GenerateRand=False):
        if SimType=='Magneticum':
            halos = np.loadtxt(SimDir)
        else:
            raise Exception("Simulation type not implemented")
        
        self.minRA = self.data['ra'].min()
        self.maxRA = self.data['ra'].max()
        self.minDEC = self.data['dec'].min()
        self.maxDEC = self.data['dec'].max()
        self.minZ = self.data['z'].min()
        self.maxZ = self.data['z'].max()

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
        Nrand = alpha * self.data.csize

        his = np.random.choice(h_indexes, size=int(Nrand), p=dist, replace=False) # indexes of selected halos

        self.randoms['z'] = self.randoms['z'][his]
        self.randoms['ra'] = self.randoms['ra'][his]
        self.randoms['dec'] = self.randoms['dec'][his]

        self.randoms['Position'] = transform.SkyToCartesian(self.randoms['ra'], self.randoms['dec'], self.randoms['z'], cosmo=self.cosmo)
        
        return
    
    def CalculateNofZ(self, UseData=False):
        
        if UseData:
            zhist = RedshiftHistogram(self.data, FSKY, self.cosmo, redshift='z')
            self.nofz = InterpolatedUnivariateSpline(zhist.bin_centers, zhist.nbar)
        else:
            zhist = RedshiftHistogram(self.randoms, FSKY, self.cosmo, redshift='z')
            self.nofz = InterpolatedUnivariateSpline(zhist.bin_centers, self.alpha*zhist.nbar)

        return
    
    def PaintMesh(self):
        self.mesh = self.fkp.to_mesh(Nmesh=self.Nmesh, nbar='NZ', fkp_weight='FKPWeight')
        
        return
    
    def CalculateMultipoles(self, poles=[0, 2, 4], kmin=0.0, kmax=0.3, dk=5e-3):
        self.r = ConvolvedFFTPower(self.mesh, poles=poles, dk=dk, kmin=kmin)
        return

    def FitPgg(self):
        self.Phh = Pggtot
        return
