#####
# Class to host lightcone loaded from data
# or generated using an Nbody class
###

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RegularGridInterpolator, CubicSpline
from scipy.optimize import minimize
from scipy import constants

import dask.dataframe as dd
import dask.array as da
import h5py
import pandas as pd
import pickle
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import nbodykit
from nbodykit.lab import *
from nbodykit.algorithms.convpower.catalog import FKPVelocityCatalog

from pypower import CatalogFFTPower, smooth_window

import hmvec as hm
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
        self.T_vals = None
        
        return

    def LoadGalaxies(self, SimDir, SimType, GenerateRand=False, z_min=0.0, z_max=5.0, ra_min=-180, ra_max=180, dec_min=-90, dec_max=90, alpha=0.1, Offset=0, Downsample=1):
        '''
        Implemented SimTypes: Magneticum, Sehgal, WebSky, numpy, and Nseries
        '''
        
        self.alpha = alpha

        if SimType=='Magneticum':
            halos = np.loadtxt(SimDir)
            flag = halos[:,17]
            halos = halos[flag==0]
            xpix     = halos[:,1]
            ypix     = halos[:,2]
            z_true   = halos[:,6]
            z_obs    = halos[:,7]

            ras = (xpix-.5) * 35
            decs = (ypix-.5) * 35

            ind = (z_obs <= z_max) & (z_obs >= z_min) & (ras >= ra_min) & (ras <= ra_max) & (decs >= dec_min) & (decs <= dec_max)
            
            self.data = {}
            self.data['ra'] = ras[ind]
            self.data['dec'] = decs[ind]
            self.data['z'] = z_obs[ind]

            # more accurate way of solving for vlos which doesn't require vlos/c << 1
            z_sample = np.linspace(0, 2.5, 1000)
            z_of_dist = InterpolatedUnivariateSpline(self.cosmo.comoving_distance(z_sample), z_sample)

            def rsd(vz_over_aH, zobs, ztrue):
                return zobs-z_of_dist(self.cosmo.comoving_distance(ztrue)+vz_over_aH)
            
            from scipy.optimize import fsolve
            vlos = np.zeros(xpix[ind].shape)
            for i in range(len(xpix[ind])):
                func = lambda vz: rsd(vz, z_true[i], z_obs[i])
                vlos[i] = fsolve(func, x0=0.)[0]
            
            speed_of_light = 299792.458 # km/s 
            a = 1/(1+z_true[ind])
            H = self.cosmo.hubble_function(z_true[ind]) # H(z) / c in Mpc^{-1}
            H *= speed_of_light # now in km/s/Mpc 
            
            vlos *= a * H
            '''
            
            speed_of_light = 299792.458 # km/s
            vlos = (1 + z_obs[ind]) / (1 + z_true[ind]) - 1
            vlos *= speed_of_light
            '''
            self.data['Vz'] = vlos
            
        elif SimType == 'Sehgal':
            halos = np.loadtxt(SimDir)
            z_true = halos[:,0]
            ras = halos[:,1]
            decs = halos[:,2]
            
            ind = (z_true <= z_max) & (z_true >= z_min) & (ras >= ra_min) & (ras <= ra_max) & (decs >= dec_min) & (decs <= dec_max)
            pos = halos[:,3:6][ind]
            vel = halos[:,6:9][ind]
            
            # LOS velocity
            n_hat = (pos.T/np.linalg.norm(pos, axis=1)).T
            vel_los = np.array([np.dot(vel[i], n_hat[i]) for i in range(len(vel))])
            
            z_interp = np.linspace(z_min, z_max, 1000)
            z_of_chi = InterpolatedUnivariateSpline(self.cosmo.comoving_distance(z_interp)/self.cosmo.h, z_interp) # TODO: Fix error in RSD calc
            aH = 1/(1+z_true[ind]) * self.cosmo.hubble_function(z_true[ind])
            chi_obs = self.cosmo.comoving_distance(z_true[ind]) / self.cosmo.h + vel_los / 3e5 / aH
            z_obs = z_of_chi(chi_obs)

            self.data = {}
            self.data['ra'] = ras[ind]
            self.data['dec'] = decs[ind]
            self.data['z'] = z_true[ind] # selection already applied
            self.data['Vz'] = vel[:,2] #vel_los
            #self.data['V'] = vel

        elif SimType == 'WebSky':
            #from astropy.cosmology import FlatLambdaCDM, z_at_value
            #import astropy.units as u

            #z_at_value = z_at_value
            #u = u
            #astropy_cosmo = FlatLambdaCDM(H0=self.websky_cosmo['h']*100, Om0=self.websky_cosmo['Omega_M'])
            omegab = 0.049
            omegac = 0.261
            omegam = omegab + omegac
            h      = 0.68
            ns     = 0.965
            sigma8 = 0.81
            cosmo = cosmology.Cosmology(h=h, Omega0_b=omegab, Omega0_cdm=omegac)
            cosmo = cosmo.match(sigma8=sigma8)

            halo_catalogue_file = SimDir# + 'halos-light.pksc'
            rho = 2.775e11*omegam*h**2 # Msun/Mpc^3
            
            with open(SimDir) as f:
                N=np.fromfile(f,count=3,dtype=np.int32)[0]
            
                # only take first five entries for testing (there are ~8e8 halos total...)
                # comment the following line to read in all halos
                #N = int(4e8)
                
                catalog=np.fromfile(f,count=N*10,dtype=np.float32)
            
            catalog=np.reshape(catalog,(N,10))
            
            x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # Mpc (comoving)
            vx = catalog[:,3]; vy = catalog[:,4]; vz = catalog[:,5] # km/sec
            R  = catalog[:,6] # Mpc 
         
            # convert to mass, comoving distance, radial velocity, redshfit, RA and DEc
            M200m    = 4*np.pi/3.*rho*R**3        # this is M200m (mean density 200 times mean) in Msun 
            chi      = np.sqrt(x**2+y**2+z**2)    # Mpc
            vrad     = (x*vx + y*vy + z*vz) / chi # km/sec
            #z_interp = np.linspace(z_min, z_max, 1000)
            #z_of_chi = InterpolatedUnivariateSpline(self.cosmo.comoving_distance(z_interp)/self.cosmo.h, z_interp)
            #redshift = z_of_chi(chi)
            
            #self.data['z'] = halodata[:,7]
            self.data = {}

            ra, dec, z = transform.CartesianToSky(catalog[:, 0:3]*self.cosmo.h, self.cosmo, velocity=catalog[:, 3:6])
            
            ind = (z >= z_min) & (z <= z_max) & (ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max)
            self.data['ra'] = np.array(ra)[ind]
            self.data['dec'] = np.array(dec)[ind]
            self.data['z'] = np.array(z)[ind] # redshift
            self.data['Vz'] = vrad[ind]
            self.data['M200m'] = M200m[ind]
            
            print(self.data['z'].shape)
            
        elif SimType == 'numpy':
            # Need to have x, y, z, vrad, ra, dec, redshift columns in npy file
            catalog = np.load(SimDir)
            x, y, z, vrad, ra, dec, redshift, M200m = catalog[:, Offset::Downsample]
            ind = (redshift <= z_max) & (redshift >= z_min) & (ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max)
            print("Loaded catalog with " + str(len(x[ind])) + " objects") 
            self.data = {}
            self.data['ra'] = ra[ind]
            self.data['dec'] = dec[ind]
            self.data['z'] = redshift[ind]
            self.data['Position'] = np.array([x[ind], y[ind], z[ind]]).T
            self.data['Vz'] = vrad[ind]
            self.data['M200m'] = M200m[ind]
        
        elif SimType == 'Nseries':
            pass
            
        else:
            raise Exception("Simulation type not implemented")
        
        self.minRA = self.data['ra'].min()
        self.maxRA = self.data['ra'].max()
        self.minDEC = self.data['dec'].min()
        self.maxDEC = self.data['dec'].max()
        self.minZ = self.data['z'].min()
        self.maxZ = self.data['z'].max()

        # Include Cartesian coords
        if 'Position' not in self.data.keys():
            self.data = ArrayCatalog(self.data)  # Store in ArrayCatalog object
            self.data['Position'] = transform.SkyToCartesian(self.data['ra'], self.data['dec'], self.data['z'], cosmo=self.cosmo)
        

        else:
            self.data = ArrayCatalog(self.data) # Store in ArrayCatalog object
        
        #box = np.max(np.array(self.data['Position'])) - np.min(np.array(self.data['Position']))
        box = [np.max(np.array(self.data['Position'])[:,i]) - np.min(np.array(self.data['Position'])[:,i]) for i in range(3)]
        print(box)
        self.BoxSize = box
        
        if GenerateRand:
            # Calculate nofz from data
            self.CalculateNofZ(UseData=True)
            self.GenerateRandoms(alpha=alpha)
            
        # TODO: add warning if alpha changed but not generating randoms
        
        return
    
    def LoadRandoms(self, SimDir, FileType, Nfiles=1):
        
        self.randoms = {}
        tmp = {}
        
        frames = []
        for i in range(Nfiles):
            fn = SimDir + str(i) + '.' + FileType
            #frames.append(pd.DataFrame(np.array(h5py.File(fn)['Position'])))
            frames.append(np.array(h5py.File(fn)['Position']))
        tmp['Position'] = np.concatenate(frames) #pd.concat(frames)
        
        tmp = ArrayCatalog(tmp)
        
        ra, dec, redshift = transform.CartesianToSky(tmp['Position'], cosmo=self.cosmo)
        ind = (redshift <= self.maxZ) & (redshift >= self.minZ) & (ra >= self.minRA) & (ra <= self.maxRA) & (dec >= self.minDEC) & (dec <= self.maxDEC)
        
        self.randoms['Position'] = tmp['Position'][ind]
        self.randoms['ra'], self.randoms['dec'], self.randoms['z'] = np.array(ra)[ind], np.array(dec)[ind], np.array(redshift)[ind]
        
        print("Loaded randoms with shape ", self.randoms['Position'].shape)
        self.randoms = ArrayCatalog(self.randoms)
        
        self.alpha = self.data.csize / self.randoms.csize
        self.CalculateNofZ()
        
        return

    def GenerateFKPCatalog(self):
        
        self.data['NZ'] = self.nofz(self.data['z'])
        self.randoms['NZ'] = self.nofz(self.randoms['z'])
        self.fkp_catalog = FKPCatalog(self.data, self.randoms)

        #self.fkp_catalog['randoms/NZ'] = self.nofz(self.randoms['z'])
        #self.fkp_catalog['data/NZ'] = self.nofz(self.data['z'])
        
        # TODO: function in case FKP/completeness included in data
        # DEBUG DENSITY FIELD WEIGHTS
        self.fkp_catalog['data/FKPWeight'] = np.ones(len(self.fkp_catalog['data/NZ'])) #/ (1 + self.fkp_catalog['data/NZ'] * 1e4)
        self.fkp_catalog['randoms/FKPWeight'] = np.ones(len(self.fkp_catalog['randoms/NZ'])) #/ (1 + self.fkp_catalog['randoms/NZ'] * 1e4)

        # Create a velocity catalog (without randoms)
        self.fkp_velocity_catalog = FKPVelocityCatalog(self.data)
        self.fkp_velocity_catalog['data/NZ'] = self.nofz(self.data['z'])
        self.fkp_velocity_catalog['data/FKPWeight'] = self.data['Vz'] #/ (np.var(self.data['Vz']) + self.fkp_velocity_catalog['data/NZ'] *1e8) 
        #/ () # TODO: Implement Howlett's weights

        return


    def GenerateRandoms(self, alpha=0.1, nbins=100):
        
        
        total_rand_cat_pos = []
    
        zbins = np.linspace(self.minZ*0.95, self.maxZ*1.05, nbins) # slightly higher range to capture endpoints well
        pos = np.array(self.data['Position'])
        
        #Loop over redshift bins and generate sample s.t. nbar_data = alpha * nbar_rand
        for i in range(nbins-1):
            
            z_range_halos = (self.data['z'] >= zbins[i]) & (self.data['z'] <= zbins[i+1])
            n_in_range = len(self.data['z'][z_range_halos])
            
            # if bin non-empty, create uniform sample and add to random catalog
            if n_in_range > 0:
                min_pos = np.min(pos[z_range_halos], axis=0)
                max_pos = np.max(pos[z_range_halos], axis=0)
    
                unif_cat_x = np.random.uniform(min_pos[0], max_pos[0], size=int(n_in_range/alpha))
                unif_cat_y = np.random.uniform(min_pos[1], max_pos[1], size=int(n_in_range/alpha))
                unif_cat_z = np.random.uniform(min_pos[2], max_pos[2], size=int(n_in_range/alpha))

                total_rand_cat_pos.extend(np.array([unif_cat_x, unif_cat_y, unif_cat_z]).T)
        
        # store random catalog as ArrayCatalog
        self.randoms = ArrayCatalog({'Position':np.array(total_rand_cat_pos)})
        self.randoms['ra'], self.randoms['dec'], self.randoms['z'] = nbodykit.transform.CartesianToSky(self.randoms['Position'], self.cosmo)
        
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
        self.halo_mesh = self.fkp_catalog.to_mesh(Nmesh=self.Nmesh, BoxSize=self.BoxSize, resampler='tsc', compensated=True)
        
        if hasattr(self, "fkp_velocity_catalog"):
            self.halo_momentum_mesh = self.fkp_velocity_catalog.to_mesh(Nmesh=self.Nmesh, BoxSize=self.BoxSize, resampler='tsc', compensated=True)
        
        use_cart_grid = False
        if use_cart_grid:
            array_cat = ArrayCatalog({'Position': self.data['Position'], 'Vz': self.data['Vz'] }, BoxSize=self.BoxSize, Nmesh=self.Nmesh)
            self.halo_mesh = array_cat.to_mesh()
            self.halo_momentum_mesh = array_cat.to_mesh(value='Vz')
        return
    
    def CalculateMultipoles(self, poles=[0, 2, 4], kmin=0.0, kmax=0.3, dk=5e-3):
        self.r = ConvolvedFFTPower(self.halo_mesh, poles=poles, dk=dk, kmin=kmin, kmax=kmax).poles
        return

    def GetPowerSpectraModel(self, LoadFile=None):
        nzbins = 5
        zs = np.linspace(self.minZ, self.maxZ, nzbins)

        if LoadFile is None:
            
            ks = np.geomspace(1e-5,100,1000)

            minz,maxz,zeff  = 0.43, 0.7, 0.55
            ngal = 1e-4 # rough CMASS number density Mpc^-3
            ms = np.geomspace(2e10,1e17,40)
            hcos = hm.HaloModel([zeff],ks,ms=ms)
            hcos.add_battaglia_profile("electron",family="AGN")
            hcos.add_hod(name="g",ngal=np.asarray([ngal]))
            bg = hcos.hods['g']['bg'][0]  # Note that bg is calculated given the ngal
            hpge = hcos.get_power_1halo("g","electron") + hcos.get_power_2halo("g","electron")
            hpggtot = hcos.get_power_1halo("g","g") + hcos.get_power_2halo("g","g") + 1./ngal
            self.model = {'ks': ks, 'sPge':hpge[0], 'lPggtot': hpggtot[0]} # DEBUG: check factors of h

        else:
            # load pre-computed model
            with open(LoadFile, 'rb') as handle:
                self.model = pickle.load(handle)

        # Getting Pk at zeff
        Pge_kmu_interp = CubicSpline(self.model['ks'], self.model['sPge'])
        Pgg_kmu_interp = CubicSpline(self.model['ks'], self.model['lPggtot'])

        self.Pge_kmu = lambda k: Pge_kmu_interp(k*self.cosmo.h)
        self.Pgg_kmu = lambda k: Pgg_kmu_interp(k*self.cosmo.h)



            
        return
        
    def FitPgg(self):
        '''
        '''
        

        # TODO: check ell normaliztion of missing 1/2
        
        
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
        
    def CalculatePrefactors(self):
        '''
        Calculate the K_star and Chi_star which enter the noise and amplitude integrals
        (adapted from U. Giri's code)
        '''
        
        if hasattr(self, "zeff"):
            redshift = self.zeff
        else:
            redshift = (self.minZ + self.maxZ) / 2
            
        rho_critical = rho_critical = 2.775e11 #self.cosmo.rho_crit(0) # 10^{10} (M_\odot/h) (\mathrm{Mpc}/h)^{-3}
        omega_b = self.cosmo.Omega_b(0)
        h = self.cosmo.h
        Tcmb = 2.7255e6 #self.cosmo.T0_cmb * 1e6 #muK
        kgtoMsun = 5.0e-31
        n_e0 = (rho_critical * omega_b) / (1.14*constants.proton_mass*kgtoMsun)
        mtompc = 3.241e-23
        sigmaT = constants.physical_constants['Thomson cross section'][0]*mtompc**2
        x_e = 1.0; exptau = 1.0
        speed_of_light = constants.speed_of_light/1e3 # km/s
        
    
        self.K_star = Tcmb * n_e0 * sigmaT * x_e * exptau #/ speed_of_light not necessary because of units of q field?
        self.Chi_star = self.cosmo.comoving_distance(redshift) # Mpc/h
        print(f"K_star = {self.K_star}")
        print(f"Chi_star = {self.Chi_star}")
        return
