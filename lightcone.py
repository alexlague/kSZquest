'''
Class to host lightcone loaded from data
or generated using an Nbody class
'''

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RegularGridInterpolator
from scipy.optimize import minimize

import nbodykit
from nbodykit.lab import *
from nbodykit.algorithms.convpower.catalog import FKPVelocityCatalog

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

    def LoadGalaxies(self, SimDir, SimType, GenerateRand=False, z_min=0.0, z_max=5.0, ra_min=-180, ra_max=180, dec_min=-90, dec_max=90, alpha=0.1):
        '''
        Implemented SimTypes: Magneticum, Sehgal
        '''
        if SimType=='Magneticum':
            halos = np.loadtxt(SimDir)
            flag = halos[:,17]
            halos = halos[flag==0]
            xpix     = halos[:,1]
            ypix     = halos[:,2]
            z_obs    = halos[:,7]

            ras = (xpix[ind]-.5) * 35
            decs = (ypix[ind]-.5) * 35
            
            ind = (z_obs <= z_max) & (z_obs >= z_min) & (ras >= ra_min) & (ras <= ra_max) & (decs >= dec_min) & (decs <= dec_max) 
            self.data = {}
            self.data['ra'] = ras[ind]
            self.data['dec'] = decs[ind]
            self.data['z'] = z_obs[ind]

            
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

            halo_catalogue_file = SimDir# + 'halos-light.pksc'
            # load catalogue header
            Nhalo            = np.fromfile(halo_catalogue_file, dtype=np.int32, count=1)[0]
            #if not(Nmax is None):
            #    Nhalo = int(Nmax)
            RTHMAXin         = np.fromfile(halo_catalogue_file, dtype=np.float32, count=1)
            redshiftbox      = np.fromfile(halo_catalogue_file, dtype=np.float32, count=1)
            print("\nNumber of Halos in full catalogue %d \n " % Nhalo)

            nfloats_perhalo = 10 #4 instad of 10 since using light
            npkdata         = nfloats_perhalo*Nhalo
            print(npkdata)

            # load catalogue data
            halodata        = np.fromfile(halo_catalogue_file, dtype=np.float32, count=npkdata)
            halodata        = np.reshape(halodata,(Nhalo,nfloats_perhalo))

            # change from R_th to halo mass (M_200,M)
            rho_mean = 2.775e11 * self.cosmo.Om0 * self.cosmo.h**2
            halodata[:,6] = 4.0/3*np.pi * halodata[:,6]**3 * rho_mean        
        
            # cut mass range
            #if mmin > 0 or mmax < np.inf:
            #    dm = (halodata[:,6] > mmin) & (halodata[:,6] < mmax) 
            #    halodata = halodata[dm]

            # cut redshift range
            if z_min > 0 or z_max < np.inf:
                #self.import_astropy()
                rofzmin = self.cosmo.comoving_distance(zmin)
                rofzmax = self.cosmo.comoving_distance(zmax)

                rpp =  np.sqrt( np.sum(halodata[:,:3]**2, axis=1))

                dm = (rpp > rofzmin) & (rpp < rofzmax) 
                halodata = halodata[dm]

            # cut distance range
            #if rmin > 0 or rmax < np.inf:
            #    rpp =  np.sqrt(np.sum(halodata[:,:3]**2, axis=1))

           #     dm = (rpp > rmin) & (rpp < rmax) 
           #     halodata = halodata[dm]


            # get halo redshifts and crop all non practical information
            #self.import_astropy()

            # set up comoving distance to redshift interpolation table
            rpp =  np.sqrt( np.sum(halodata[:,:3]**2, axis=1))

            z_interp = np.linspace(z_min, z_max, 1000)
            z_of_chi = InterpolatedUnivariateSpline(self.cosmo.comoving_distance(z_interp)/self.cosmo.h, z_interp) 
            #zminh = self.z_at_value(self.astropy_cosmo.comoving_distance, rpp.min()*self.u.Mpc)
            #zmaxh = self.z_at_value(self.astropy_cosmo.comoving_distance, rpp.max()*self.u.Mpc)
            zminh = z_of_chi(rpp.min())
            zmaxh = z_of_chi(rpp.max())
            zgrid = np.linspace(zminh, zmaxh, 10000)
            dgrid = self.cosmo.comoving_distance(zgrid)
            
            # first replace 7th column with redshift
            halodata[:,7] = np.interp(rpp, dgrid, zgrid)
            
            # crop un-practical halo information
            halodata = halodata[:,:8]
            
            Nhalo = halodata.shape[0] 
            Nfloat_perhalo = halodata.shape[1]

            # write out halo catalogue information
            #if self.verbose: 
            print("Halo catalogue after cuts: np.array((Nhalo=%d, floats_per_halo=%d)), containing:\n" % (Nhalo, Nfloat_perhalo))
            print("0:x [Mpc], 1:y [Mpc], 2:z [Mpc], 3:vx [km/s], 4:vy [km/s], 5:vz [km/s],\n 6:M [M_sun (M_200,m)], 7:redshift(chi_halo) \n")
            #else:
            #    print("0:x [Mpc], 1:y [Mpc], 2:z [Mpc], 3:vx [km/s], 4:vy [km/s], 5:vz [km/s],\n"+ 
            #          "6:M [M_sun (M_200,m)], 7:x_lag [Mpc], 8:y_lag [Mpc], 9:z_lag [Mpc]\n")

            #self.data['z'] = halodata[:,7]
            self.data['ra'], self.data['dec'], self.data['z'] = transform.CartesianToSky(halodata[:, 0:3])
        
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
        box = np.max(np.array(self.data['Position'])) - np.min(np.array(self.data['Position']))
        print(box)
        self.BoxSize = box
        
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

        # Create a velocity catalog (without randoms)
        self.fkp_velocity_catalog = FKPVelocityCatalog(self.data)
        self.fkp_velocity_catalog['data/NZ'] = self.nofz(self.data['z'])
        self.fkp_velocity_catalog['data/FKPWeight'] = self.data['Vz'] #/ () # TODO: Implement Howlett's weights

        return

    def LoadCMBMap(self, SimDir, SimType):
        return

    def CalculateWindowFunction(self):
        return

    def DeconvolveWindowFunction(self):
        return

    def GenerateRandoms(self, alpha=0.1):
        
        rand = RandomCatalog(int(4*len(self.data)/alpha))
        rand['z']   = rand.rng.uniform(low=self.minZ, high=self.maxZ)
        rand['ra']  = rand.rng.uniform(low=self.minRA, high=self.maxRA)
        rand['dec'] = rand.rng.uniform(low=self.minDEC, high=self.maxDEC)
        
        # Subselect z to match nofz from data
        h_indexes = np.arange(len(rand['z']))
        hzs = rand['z']
        dist = abs(self.nofz(hzs)) # distribution
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
        self.halo_mesh = self.fkp_catalog.to_mesh(Nmesh=self.Nmesh, nbar='NZ', fkp_weight='FKPWeight', BoxSize=self.BoxSize)
        self.halo_momentum_mesh = self.fkp_velocity_catalog.to_mesh(Nmesh=self.Nmesh, nbar='NZ', fkp_weight='FKPWeight', BoxSize=self.BoxSize)
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
        
        #model = get_ksz_auto_squeezed(ells=ls, volume_gpc3=vol, zs=zs, ngals_mpc3=ngals, params=self.CosmoParams, template=True, rsd=True, bgs=bgs)
        model = get_ksz_auto_squeezed(ells=ls, volume_gpc3=vol, zs=zs, ngals_mpc3=None, params=self.CosmoParams, template=True, rsd=True, bgs=bgs)
       
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
        
