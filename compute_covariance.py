# Compute covariance matrix for the reconstruction Pgv
# using mock galaxy catalogs and simulated CMB maps

import numpy as np
from nbodykit.lab import *
import nbodykit
import healpy as hp
from pypower import CatalogFFTPower
from pyrecon import MultiGridReconstruction
from pypower.fft_power import normalization_from_nbar
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from pixell import enmap, utils, powspec, curvedsky, reproject
import random
import sys
sys.path.append('../repo/kSZquest/')

import lightcone
import cmb
import recon

import utils as kutils
args = kutils.jobargs
paths = kutils.paths

# the fiducial BOSS DR12 cosmology
cosmo = cosmology.Cosmology(h=0.676).match(Omega0_m=0.31)
freq = 'f090'
NSIDE = 2048
Nmesh = 256
Nkeep = 10 # keep only the N points in the largest scales (to avoid nans)

# rand catalog the same for every mock

mock_dir = '/home/r/rbond/alague/scratch/ksz-pipeline/QPM_mock_data/'
rand1 = np.loadtxt(mock_dir+'mock_random_DR12_CMASS_N_50x1.rdzw')
#rand2 = np.loadtxt(mock_dir+'mock_random_DR12_CMASS_N_50x2.rdzw')
rand = rand1 #np.concatenate((rand1, rand2))
rand_pos = np.array([rand[:,0], rand[:,1], cosmo.comoving_distance(rand[:,2])])

def load_galaxy_mock(imock):
    
    ID = str(imock)
    if len(ID) == 1:
        ID = '000' + ID
    elif len(ID) == 2:
        ID = '00' + ID
    elif len(ID) == 3:
        ID = '0' + ID

    data = np.loadtxt(mock_dir+'mock_galaxy_DR12_CMASS_N_QPM_' + ID + '.rdzw')
    data_pos = np.array([data[:,0], data[:,1], cosmo.comoving_distance(data[:,2])])

    data_cat = ArrayCatalog({'ra': data[:,0], 'dec': data[:,1], 'redshift': data[:,2]})
    data_cat['Position'] = nbodykit.transform.SkyToCartesian(data_cat['ra'],
                                                             data_cat['dec'],
                                                             data_cat['redshift'], cosmo)

    rand_cat = ArrayCatalog({'ra': rand[:,0], 'dec': rand[:,1], 'redshift': rand[:,2]})
    rand_cat['Position'] = nbodykit.transform.SkyToCartesian(rand_cat['ra'],
                                                             rand_cat['dec'],
                                                             rand_cat['redshift'], cosmo)


    ACT_SLICE = data_cat['dec'] <= 23
    ACT_SLICE_RAND = rand_cat['dec'] <= 23


    # extract velocities from reconstructed catalog
    vel = np.loadtxt(mock_dir + 'recon/bao_reconstructed_velocities_N' + ID + '.dat')

    # extract n(z) by inverting FKP weights (assuming P0 = 2e4)
    nz = (1. / data[:,3] - 1) / 2e4

    # smooth n(z) and <v^2(z)>
    nbins = 200
    zbin_edges = np.linspace(data[:,2].min(), data[:,2].max(), nbins+1)
    zbins = np.linspace(data[:,2].min(), data[:,2].max(), nbins)
    var_vel_z = np.zeros(nbins)
    nz_z = np.zeros(nbins)
    for i in range(nbins):
        ind = (data[:,2] >= zbin_edges[i]) & (data[:,2] <= zbin_edges[i+1])
        var_vel_z[i] = np.var(vel[ind])
        nz_z[i] = np.mean(nz[ind])

    # velocity weights
    P0 = 1e11
    v2 = CubicSpline(zbins, var_vel_z)(data[:,2])
    n = CubicSpline(zbins, nz_z)(data[:,2])
    vel_weights = 1./(v2+n*P0)

    # Store data in lightcone object
    
    FSKY = 0.1


    # Create lightcone
    lc = lightcone.LightCone(FSKY, Nmesh=Nmesh)

    # Use smoothed n(z) function
    data_cat['NZ'] = CubicSpline(zbins, nz_z)(data[:,2])
    rand_cat['NZ'] = CubicSpline(zbins, nz_z)(rand[:,2])

    # Use already loaded data
    lc.data = data_cat[ACT_SLICE]
    lc.randoms = rand_cat[ACT_SLICE_RAND]
    lc.data['NZ'] = data_cat['NZ'][ACT_SLICE]
    lc.randoms['NZ'] = rand_cat['NZ'][ACT_SLICE_RAND]
    lc.randoms['Position'] = rand_cat['Position'][ACT_SLICE_RAND]

    lc.data['ra'] = data_cat['ra'][ACT_SLICE]
    lc.data['dec'] = data_cat['dec'][ACT_SLICE]
    lc.data['z'] = data_cat['redshift'][ACT_SLICE]
    lc.randoms['z'] = rand_cat['redshift'][ACT_SLICE_RAND]
    lc.data['Position'] = data_cat['Position'][ACT_SLICE]

    lc.fkp_catalog = FKPCatalog(lc.data, lc.randoms)
    lc.fkp_catalog['data/FKPWeight'] = data[:,3][ACT_SLICE]
    lc.fkp_catalog['randoms/FKPWeight'] = rand[:,3][ACT_SLICE_RAND]

    vel_weights = vel_weights[ACT_SLICE]

    lc.minZ = 0.43
    lc.maxZ = 0.7
    lc.nofz = CubicSpline(zbins, nz_z)
    lc.BoxSize = np.max(np.array(data_cat['Position']), axis=0) - np.min(np.array(data_cat['Position']), axis=0)
    print("Box: ", lc.BoxSize)

    # Catalog to mesh
    lc.PaintMesh()

    # Compute model
    fid_model = '/home/r/rbond/alague/scratch/ksz-pipeline/ksz-analysis/quadratic_estimator/development_code/prepared_maps/cmass_fiducial_model.pkl'
    lc.GetPowerSpectraModel(LoadFile=fid_model)


    # Pre-compute velocity weights
    wnorm_v = normalization_from_nbar(lc.data['NZ'])
    wnorm_gv = (normalization_from_nbar(lc.data['NZ']) * normalization_from_nbar(lc.data['NZ'], data_weights=lc.fkp_catalog['data/FKPWeight']))**0.5
    
    return lc, vel_weights, wnorm_v, wnorm_gv #data_cat, rand_cat, vel_weights
    
def load_cmb_mock(jcmbsim):
    
    args.freq = freq
    sstr = kutils.save_string(args)
    filtered_alms = hp.fitsfunc.read_alm(paths.out_dir + 'sims/filtered_alms_' + sstr + '_simid_'  + str(jcmbsim)  + '.fits')
    alms = np.zeros((3, len(filtered_alms)), dtype=complex)
    alms[0] = filtered_alms
    healpix_filtered = hp.alm2map(alms, NSIDE, pol=False)[0]
    LMAX = 5000
    ksz_lc = cmb.CMBMap(healpix_filtered, {}, LMAX,
                        noise_lvl=None,
                        theta_FWHM=None,
                        NSIDE=NSIDE,
                        do_beam=False)
    
    DoFilter = False
    AddNoise = False
    AddPrimary = False

    ksz_lc.PrepareRecon(AddPrimary=AddPrimary, AddNoise=AddNoise, DoFilter=DoFilter)
    
    return ksz_lc.kSZ_map


def run_pipeline(imock):

    kedges = np.linspace(0, 0.2, 26)
    #Ncmbsims = 1
    #poles_array = np.zeros((Nkeep, Ncmbsims))

    # collect lss and cmb data
    lc, vel_weights, wnorm_v, wnorm_gv = load_galaxy_mock(imock)
    
    # iterate over cmb realizations
    #for jcmbsim in [imock]: #range(1, Ncmbsims+1):
    jcmbsim = imock
    filtered_cmb_map = load_cmb_mock(jcmbsim)

    # main reconstruction step
    fil_dir = '/home/r/rbond/alague/scratch/ksz-pipeline/ksz-analysis/quadratic_estimator/'
    fil_dir += 'development_code/prepared_maps/'
    fil_path = fil_dir+"theory_filter_daynight_night_fit_lmax_7000_fit_lmin_1200_freq_f090_freqs_['f090']_lmax_8000_lmin_100.txt"
    prefactor, recon_noise = recon.CalculateNoiseFromFilter(lc, CMBFilterPath=fil_path)

    vhat = recon.RunReconstruction(lc, filtered_cmb_map, ComputePower=False, NSIDE=NSIDE)
    vhat *= prefactor * 3e5 # to units of km/s by multiplying by c
    
    # reverse the nbodykit shuffle
    vhat = np.roll(np.roll(np.roll(vhat, Nmesh//2, axis=0), Nmesh//2, axis=1), Nmesh//2, axis=2)
    X = np.linspace(0, 1, Nmesh)
    vhat_interp = RegularGridInterpolator((X, X, X), vhat)
    
    pos_array = np.array(lc.data['Position'])
    box = np.max(pos_array, axis=0) - np.min(pos_array, axis=0)
    pos_grid = (pos_array - np.min(pos_array, axis=0)) / box # between 0 and 1
    vel_grid = vhat_interp(pos_grid) # interpolated velocities
    
    # compute Pgv
    poles_vgr = CatalogFFTPower(data_positions1=pos_array,
                                data_weights1=vel_grid-np.mean(vel_grid),
                                data_positions2=pos_array,
                                randoms_positions2=np.array(lc.randoms['Position']),
                                randoms_weights2=np.array(lc.fkp_catalog['randoms/FKPWeight']), #rand[:,3],
                                data_weights2=np.array(lc.fkp_catalog['data/FKPWeight']), #data[:,3],
                                nmesh=256,
                                resampler='tsc',
                                interlacing=2,
                                ells=(0, 1, 2, 4),
                                los='firstpoint',
                                edges=kedges,
                                position_type='pos',
                                dtype='f4', wnorm=np.array(wnorm_gv)).poles

    pk = poles_vgr(ell=1, complex=False)
    #poles_array[jcmbsim-1] =  pk[:Nkeep] # keep only large scales
    
    poles = pk[:Nkeep]
    
    return poles


## RUN MAIN CODE ##

from multiprocessing import Pool

with Pool(16) as p:
    pgv_array = p.map(run_pipeline, range(1, 30))

print(f"Array shape after parallel computation is {np.array(pgv_array).shape}")
cov_matrix = np.cov(np.array(pgv_array).T)
np.savetxt(paths.out_dir + "Pgv_ell_1_NGC_f090_cov_mat.dat", cov_matrix)

#print(np.cov(np.array(pgv_array, dtype=np.float32).reshape(Nkeep, -1)).shape)


