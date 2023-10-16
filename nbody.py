'''
Class for N-Body simulation data and painting the
momentum field using particles as tracers of electrons
Can switch between a box configuration or a lightcone
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

import nbodykit
from nbodykit.lab import *

import sys
sys.path.append('/home/r/rbond/alague/scratch/ksz-pipeline/abacus_data/AbacusCosmos')
from AbacusCosmos import Halos

class NBodySim:
    

    def __init__(self, Redshift, BoxSize, CosmoParams=None, LOS=[0, 0, 1]):
        if CosmoParams == None:
            print("No cosmological parameters specified: assuming Planck15")
            self.cosmo = cosmology.Planck15
        else:
            pass #TODO

        self.Redshift = Redshift
        self.BoxSize = BoxSize
        self.LOS = np.array(LOS)
        
        return
    

    def _check_mesh_(self):
        if hasattr(self, "particle_momentum_mesh")==False and Nmesh==None:
            raise Exception("Need to run mesh painting first, please specify Nmesh")
        elif hasattr(self, "particle_momentum_mesh")==False and Nmesh!=None:
            self.Nmesh=Nmesh
            self.PaintBoxMeshes(Nmesh)

        return
    
    def LoadData(self, SimDir, SimType, DownSample=100, HaloParticles=True, FieldParticles=True):
        '''
        '''

        speed_of_light = 299792.458 # km/s

        if SimType == 'Abacus':
            
            # Load halos     
            cat = Halos.make_catalog_from_dir(dirname=SimDir, load_subsamples=True, load_pids=False)
        
            # Load particles found in halos
            if HaloParticles:
                subsamples = cat.subsamples

            
            # Generate halo and particle catalogs
            halo_dict = {}
            halo_dict['Position'] = cat.halos[:]['pos']
            halo_dict['Velocity'] = cat.halos[:]['vel']
        
            part_dict = {}
            part_pos_x, part_pos_y, part_pos_z = [], [], []
            part_vel_x, part_vel_y, part_vel_z = [], [], []

            if HaloParticles:
                part_pos_x.extend(subsamples['pos'][:,0][::DownSample])
                part_pos_y.extend(subsamples['pos'][:,1][::DownSample])
                part_pos_z.extend(subsamples['pos'][:,2][::DownSample])
        
                part_vel_x.extend(subsamples['vel'][:,0][::DownSample])
                part_vel_y.extend(subsamples['vel'][:,1][::DownSample])
                part_vel_z.extend(subsamples['vel'][:,2][::DownSample])

            if FieldParticles:
                for i in range(4):
                    part_pos = np.fromfile(SimDir+'/field_particles_'+str(i), dtype=np.float32).reshape((-1,6))
                    part_pos_x.extend(part_pos[:,0][::DownSample])
                    part_pos_y.extend(part_pos[:,1][::DownSample])
                    part_pos_z.extend(part_pos[:,2][::DownSample])
                    part_vel_x.extend(part_pos[:,3][::DownSample])
                    part_vel_y.extend(part_pos[:,4][::DownSample])
                    part_vel_z.extend(part_pos[:,5][::DownSample])
        
            part_dict['Position'] = np.array([part_pos_x, part_pos_y, part_pos_z]).T
            part_dict['Velocity'] = np.array([part_vel_x, part_vel_y, part_vel_z]).T

        else:
            raise Exception("Only Abacus simulations are supported for now")
        
        # Store in ArrayCatalog object 
        self.halo_catalog = ArrayCatalog(halo_dict)
        
        # Apply RSD
        # Delta chi = v/aH (converting to Mpc/h units)
        one_over_aH = (1+self.Redshift) / (self.cosmo.hubble_function(self.Redshift)*speed_of_light/self.cosmo.h)
        self.halo_catalog['VelocityOffset'] = self.halo_catalog['Velocity'] * one_over_aH # 3D vector to add to positions
        self.halo_catalog['RSDPosition']    = self.halo_catalog['Position'] + self.halo_catalog['VelocityOffset'] * self.LOS
        self.halo_catalog['VelocityOffset'] = np.dot(self.halo_catalog['VelocityOffset'], self.LOS) # Now scalar value dep. on LOS

        print("Loaded halo catalog at redshift {}".format(self.Redshift))

        # Note if snapshot is loaded with some type of particles 
        if HaloParticles or FieldParticles:
            part_cat = ArrayCatalog(part_dict)
            self.includes_particles = True
            self.particle_catalog = part_cat
            self.particle_catalog['VelocityOffset'] = self.particle_catalog['Velocity'] * one_over_aH
            self.particle_catalog['RSDPosition'] = self.particle_catalog['Position'] + self.particle_catalog['VelocityOffset'] * self.LOS
            self.particle_catalog['VelocityOffset'] = np.dot(self.particle_catalog['VelocityOffset'], self.LOS) # Now scalar value dep. on LOS

            print("Loaded particle catalog downsampled by a factor of {}".format(DownSample))
        else:
            self.includes_particles = False
        

        return
    
    # From U. Giri
    def prefactor(self)-> float:
        """The coefficient which appears before the integral term in kSZ
        Returns:
        float -- The coefficient
        """
        #mtompc = 3.24078e-23
        #rho_critical = 1e-26/mtompc**3
        rho_critical = self.cosmo.rho_crit(0) # 10^{10} (M_\odot/h) (\mathrm{Mpc}/h)^{-3}
        omega_b = self.cosmo.Omega_b(0)
        h = self.cosmo.h
        Tcmb = self.cosmo.T0_cmb * 1e6 #K
        n_e0 = (rho_critical * omega_b) / (1.14*constants.proton_mass)
        sigmaT = constants.physical_constants['Thomson cross section'][0]*mtompc**2
        x_e = 1.0; exptau = 1.0
        speed_of_light = constants.speed_of_light/1e3 # km/s
        K_star = self.Tcmb * n_e0 * sigmaT * x_e * exptau #/ speed_of_light not necessary because of units of q field?
    
        Chi = self.cosmo.comoving_distance(self.Redshift) # Mpc/h
        
        return K_star / Chi**2 * (1+self.Redshift)**2
    
    def to_LightCone(zBins, NzBins, nbar):
        '''
        Transform the cubic box to a lightcone with a mask and N(z)
        '''
        
        nz_interp = interp1d(zBins, Nzbins, bounds_error=False, fill_value=0., kind='cubic')
        
        # TODO: Add mask

        # Select halos based on N(z) and nbar
        h_indexes = np.arange(len(halo_dict['Position']))
        
        
        
        
        return

    def PaintBoxMeshes(self, Nmesh, RSD=False):
        '''
        '''
        self.Nmesh = Nmesh
        
        if RSD:
            pos = 'RSDPosition'
        else:
            pos = 'Position'
        
        #self.particle_catalog['Vz'] = self.particle_catalog['Velocity'][:,2]
        self.particle_mesh = self.particle_catalog.to_mesh(Nmesh=Nmesh, BoxSize=self.BoxSize, position=pos)
        self.halo_mesh = self.halo_catalog.to_mesh(Nmesh=Nmesh, BoxSize=self.BoxSize, position=pos)
        self.particle_momentum_mesh = self.particle_catalog.to_mesh(Nmesh=Nmesh, BoxSize=self.BoxSize, position=pos, value='VelocityOffset')
        
        
        return
    
    def GeneratekSZMap(self, Nmesh=None):
        '''
        Generate a mock kSZ map using the particles as electrons following the
        procedure of Giri and Smith
        '''
        
        # TODO: include K(z)/chi_star(z)
        
        if self.includes_particles == False:
            raise Exception("Need to load particles in snapshot to generate kSZ map")
        
        self._check_mesh_()
        
        if Nmesh != None:
            if Nmesh!=self.Nmesh:
                raise Exception("Nmesh value provided for the kSZ map does not match existing Nmesh")
        
        # Convert to array and sum over z-axis
        q_field = self.particle_momentum_mesh.to_field()
        self.kSZ_map = np.sum(q_field, axis=0)
        
        return

    def ComputePower1D(self, Nmesh=None, Fields=['mm', 'hh', 'mh', 'qh'], kmin=0.0, kmax=0.3, dk=5e-3):
        '''
        '''
        
        self._check_mesh_()
        
        if 'hh' in Fields:
            self.Phh = FFTPower(self.halo_mesh, mode='1d', kmin=kmin, kmax=kmax, dk=dk).power
        if 'mm'in Fields:
            self.Pmm = FFTPower(self.particle_mesh, mode='1d', kmin=kmin, kmax=kmax, dk=dk).power
        if 'mh' in Fields or 'hm' in Fields:
            self.Pmh = FFTPower(self.halo_mesh, second=self.particle_mesh, mode='1d', kmin=kmin, kmax=kmax, dk=dk).power
        if 'qq' in Fields:
            self.Pqq = FFTPower(self.particle_momentum_mesh, mode='1d', kmin=kmin, kmax=kmax, dk=dk).power
        if 'qh' in Fields or 'hq' in Fields:
            self.Pqh = FFTPower(self.particle_momentum_mesh, second=self.halo_mesh, mode='1d', kmin=kmin, kmax=kmax, dk=dk).power
        return


    def ComputeMultipoles(self, Nmesh=None, Fields=['mm', 'hh', 'mh', 'qh'], poles=[0, 2], kmin=0.0, kmax=0.3, dk=5e-3):
        '''
        '''
        
        self._check_mesh_()

        if Nmesh != None:
            if Nmesh!=self.Nmesh:
                raise Exception("Nmesh value provided for the kSZ map does not match existing Nmesh")
        
        if 'hh' in Fields:
            self.Phh_ell = FFTPower(self.halo_mesh, poles=poles, mode='1d', los=self.LOS, kmin=kmin, kmax=kmax, dk=dk)
        if 'mm'in Fields:
            self.Pmm_ell = FFTPower(self.particle_mesh, poles=poles, mode='1d', los=self.LOS, kmin=kmin, kmax=kmax, dk=dk)
        if 'mh' in Fields or 'hm' in Fields:
            self.Pmh_ell = FFTPower(self.halo_mesh, second=self.particle_mesh, poles=poles, mode='1d', los=self.LOS, kmin=kmin, kmax=kmax, dk=dk)
        if 'qq' in Fields:
            self.Pqq_ell = FFTPower(self.particle_momentum_mesh, poles=poles, mode='1d', los=self.LOS, kmin=kmin, kmax=kmax, dk=dk)
        if 'qh' in Fields or 'hq' in Fields:
            self.Pqh_ell = FFTPower(self.particle_momentum_mesh, second=self.halo_mesh, poles=poles, mode='1d', los=self.LOS, kmin=kmin, kmax=kmax, dk=dk)
        
        print("Computed requested power spectrum multipoles")
        
        return

    def ComputePower2D(self, Nmesh=None, Fields=['mm', 'hh', 'mh', 'qh'], kmin=0.0, kmax=0.3, dk=5e-3, Nmu=10):
        '''
        '''

        self._check_mesh_()

        if 'hh' in Fields:
            self.Phh_kmu = FFTPower(self.halo_mesh, mode='2d', kmin=kmin, kmax=kmax, dk=dk, los=self.LOS, Nmu=Nmu).power
        if 'mm'in Fields:
            self.Pmm_kmu = FFTPower(self.particle_mesh, mode='2d', kmin=kmin, kmax=kmax, dk=dk, los=self.LOS, Nmu=Nmu).power
        if 'mh' in Fields or 'hm' in Fields:
            self.Pmh_kmu = FFTPower(self.halo_mesh, second=self.particle_mesh, mode='2d', kmin=kmin, kmax=kmax, dk=dk, los=self.LOS, Nmu=Nmu).power
        if 'qq' in Fields:
            self.Pqq_kmu = FFTPower(self.particle_momentum_mesh, mode='2d', kmin=kmin, kmax=kmax, dk=dk, los=self.LOS, Nmu=Nmu).power
        if 'qh' in Fields or 'hq' in Fields:
            self.Pqh_kmu = FFTPower(self.particle_momentum_mesh, second=self.halo_mesh, mode='2d', kmin=kmin, kmax=kmax, dk=dk, los=self.LOS, Nmu=Nmu).power
        return
