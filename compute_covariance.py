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
#freq = 'f090'
NSIDE = 2048
Nmesh = 512
Nkeep = 20 # keep only the N points in the largest scales (to avoid nans)

# If use data then always supply the ACT data CMB map to run null test with
# mock galaxy catalogs
use_act_data = False
use_boss_data = True
# Use sims with imprinted ksz signal
# store in different directory named after the Tgrid painting procedure
with_ksz = False
with_ksz_dir = "Tgrid_tests/filter_only_cmb/"

if with_ksz:
    assert use_act_data == False and use_boss_data == False

# rand catalog the same for every mock
if use_boss_data:
    BOSS_dir = '/home/r/rbond/alague/scratch/ksz-pipeline/BOSS_data/CMASS/'
    BOSS_rand = 'pre-recon/cmass/random0_DR12v5_CMASS_North_t1.txt'
    rand1 = np.loadtxt(BOSS_dir+BOSS_rand)
else:
    mock_dir = '/home/r/rbond/alague/scratch/ksz-pipeline/QPM_mock_data/'
    rand1 = np.loadtxt(mock_dir+'mock_random_DR12_CMASS_N_50x1.rdzw')

def load_galaxy_mock(imock, freq):
    
    ID = str(imock)
    if len(ID) == 1:
        ID = '000' + ID
    elif len(ID) == 2:
        ID = '00' + ID
    elif len(ID) == 3:
        ID = '0' + ID

    if use_boss_data:
        BOSS_dir = '/home/r/rbond/alague/scratch/ksz-pipeline/BOSS_data/CMASS/'
        BOSS_file = 'pre-recon/cmass/galaxy_DR12v5_CMASS_North_t2.txt'
        data = np.loadtxt(BOSS_dir+BOSS_file)
    else:
        data = np.loadtxt(mock_dir+'mock_galaxy_DR12_CMASS_N_QPM_' + ID + '.rdzw')

    # Remove galaxies outside of redshift range
    z_min = 0.43
    z_max = 0.7
    zsel = (data[:,2] >= z_min) & (data[:,2] <= z_max)
    data = data[zsel]
    
    rand = rand1
    zsel_rand = (rand[:,2] >= z_min) & (rand[:,2] <= z_max)
    rand = rand[zsel_rand]

    data_cat = ArrayCatalog({'ra': data[:,0], 'dec': data[:,1], 'redshift': data[:,2]})
    data_cat['Position'] = nbodykit.transform.SkyToCartesian(data_cat['ra'],
                                                             data_cat['dec'],
                                                             data_cat['redshift'], cosmo)

    rand_cat = ArrayCatalog({'ra': rand[:,0], 'dec': rand[:,1], 'redshift': rand[:,2]})
    rand_cat['Position'] = nbodykit.transform.SkyToCartesian(rand_cat['ra'],
                                                             rand_cat['dec'],
                                                             rand_cat['redshift'], cosmo)

    # Remove galaxies not in ACT footprint

    ACT_SLICE = data_cat['dec'] <= 21.6166
    ACT_SLICE_RAND = rand_cat['dec'] <= 21.6166



    # extract n(z) by inverting FKP weights (assuming P0 = 2e4)
    # for sims
    if use_boss_data:
        nz= data[:, -1]
    else:
        nz = (1. / data[:,3] - 1) / 2e4

    # smooth n(z) and <v^2(z)>
    nbins = 200
    zbin_edges = np.linspace(data[:,2].min(), data[:,2].max(), nbins+1)
    zbins = np.linspace(data[:,2].min(), data[:,2].max(), nbins)
    var_vel_z = np.zeros(nbins)
    nz_z = np.zeros(nbins)
    for i in range(nbins):
        ind = (data[:,2] >= zbin_edges[i]) & (data[:,2] <= zbin_edges[i+1])
        nz_z[i] = np.mean(nz[ind])

    # velocity weights
    P0 = 5e9 
    v2 = 1500**2
    n = CubicSpline(zbins, nz_z)(data[:,2])
    vel_weights = 1/(v2+n*P0)
    vel_weights /= np.mean(vel_weights)

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

    lc.minZ = 0.43
    lc.maxZ = 0.7
    lc.nofz = CubicSpline(zbins, nz_z)
    lc.BoxSize = np.max(np.array(data_cat['Position']), axis=0) - np.min(np.array(data_cat['Position']), axis=0)
    print("Box: ", lc.BoxSize)

    # Catalog to mesh
    lc.PaintMesh()

    # Compute/load model
    lc.GetPowerSpectraModel()

    # FKP & Completeness weights
    fkp_weights = data[:,3][ACT_SLICE]
    if use_boss_data:
        comp_weights = (data[:,6] * (data[:,4] + data[:,5] - 1))[ACT_SLICE]  # WEIGHT_SYSTOT * (WEIGHT_CP+WEIGHT_NOZ-1)
    else:
        comp_weights = data[:,4][ACT_SLICE]

    # Pre-compute velocity weights
    wnorm_v = normalization_from_nbar(nz[ACT_SLICE], data_weights=comp_weights*vel_weights[ACT_SLICE])
    wnorm_gv = (wnorm_v * normalization_from_nbar(nz[ACT_SLICE], data_weights=fkp_weights*comp_weights))**0.5
    wnorm_g = normalization_from_nbar(np.array(rand_cat['NZ'][ACT_SLICE_RAND]), data_weights=fkp_weights*comp_weights, weights=rand[:,3][ACT_SLICE_RAND])

    print(f"gv weight norm is: {wnorm_gv}")

    return lc, data_cat['Position'][ACT_SLICE], rand_cat['Position'][ACT_SLICE_RAND], wnorm_v, wnorm_gv, wnorm_g, fkp_weights, comp_weights, rand[:,3][ACT_SLICE_RAND], vel_weights[ACT_SLICE]
    
def load_cmb_mock(jcmbsim, freq):
        
    if use_act_data:
        prefix = "filtered_alms_daynight_daynight_fit_lmax_7980_fit_lmin_1000_freq_"
        suffix = "_lmax_8000_lmin_100.fits"
        filtered_alms = hp.fitsfunc.read_alm(paths.out_dir + prefix + freq + suffix)
        
    else:
        prefix = "sims/filtered_alms_daynight_daynight_fit_lmax_7980_fit_lmin_1000_freq_"
        suffix = "_lmax_8000_lmin_100"
        if with_ksz:
            
            ksz_dir = '/home/r/rbond/alague/scratch/ksz-pipeline/ksz-analysis/quadratic_estimator/development_code/QPM_maps/'
            index = jcmbsim
            ksz_sim = hp.read_map(ksz_dir + f'ksz_map_from_BAO_recon_{index}.fits', dtype=np.float32)
            ksz_sim *= 1e6 
            LMAX = 8000 #hp.sphtfunc.Alm.getlmax(len(alm))
            filtered_alms = hp.map2alm(ksz_sim, lmax=LMAX)

        else:
            filtered_alms = hp.fitsfunc.read_alm(paths.out_dir + prefix + freq + suffix + '_simid_'  + str(jcmbsim)  + '.fits')

    alms = np.zeros((3, len(filtered_alms)), dtype=complex)
    alms[0] = filtered_alms
    healpix_filtered = hp.alm2map(alms, NSIDE, pol=False)[0] # ksz sims
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

    del alms # free memory

    return ksz_lc.kSZ_map


def run_pipeline(imock, freq):
    '''
    Run the analysis pipeline on the mock catalogs and CMB maps
    imock: index of the mock catalog/cmb sim
    freq: frequency of 'f090' or 'f150' to compute cross-covariance
    '''

    kedges = np.linspace(0, 0.2, 51)

    # collect lss and cmb data
    lc, data_pos, rand_pos, wnorm_v, wnorm_gv, wnorm_g, fkp_weights, comp_weights, rand_fkp_weights, vel_weights = load_galaxy_mock(imock, freq)
    
    # iterate over cmb realizations
    jcmbsim = imock
    filtered_cmb_map = load_cmb_mock(jcmbsim, freq)

    ## main reconstruction step ##
    
    # load Cl filter for normalization
    if use_act_data:
        fil_dir = '/home/r/rbond/alague/scratch/ksz-pipeline/ksz-analysis/quadratic_estimator/'
        fil_dir += 'development_code/prepared_maps/'
        fil_path = fil_dir+"theory_filter_daynight_daynight_fit_lmax_7980_fit_lmin_1000_freq_" + freq + "_lmax_8000_lmin_100.txt"
    else:
        fil_dir = '/home/r/rbond/alague/scratch/ksz-pipeline/ksz-analysis/quadratic_estimator/'
        fil_dir += 'development_code/prepared_maps/sims/'
        prefix = "theory_filter_daynight_daynight_fit_lmax_7980_fit_lmin_1000_freq_"
        suffix = "_lmax_8000_lmin_100_simid_"
        if with_ksz:
            pass
        else:
            fil_path = fil_dir + prefix + freq + suffix + str(jcmbsim) + ".txt"

    #if the simulations are the ksz component, then we normalize after to match the bv of the data 
    if with_ksz == False:        
        prefactor, recon_noise = recon.CalculateNoiseFromFilter(lc, CMBFilterPath=fil_path)
    else:
        prefactor, recon_noise = 1., 1.
    print(f"prefactor and recon noise: {prefactor}, {recon_noise}")

    
    vhat = recon.RunReconstruction(lc, filtered_cmb_map, ComputePower=False, NSIDE=NSIDE, use_T_grid=False)
    
    pos_array = np.array(data_pos)

    
    ## TRYING KENDRICKS METHOD -> USING ONLY (NORMALIZED) TVALS ##

    print("T values std: ", np.std(lc.T_vals))
    vel_grid =  lc.T_vals * prefactor * 3e5
    
    vel_grid -= np.mean(vel_grid)
    dp = pos_array
    dw = (fkp_weights * comp_weights)
    rp = np.array(rand_pos)
    rw = rand_fkp_weights
    vel_grid = vel_grid * comp_weights * vel_weights


    # compute Pgv
    poles_vgr = CatalogFFTPower(data_positions1=dp,
                                data_weights1=vel_grid,
                               data_positions2=dp,
                               randoms_positions2=rp,
                               randoms_weights2=rw,
                               data_weights2=dw,
                               nmesh=256,
                               resampler='tsc',
                               interlacing=2,
                               ells=(0, 1, 2, 3, 4),
                               los='endpoint',
                               edges=kedges,
                               position_type='pos',
                                dtype='f4', wnorm=wnorm_gv).poles

    poles_gg = CatalogFFTPower(data_positions1=dp,
                                data_weights1=dw,
                               randoms_positions1=rp,
                               randoms_weights1=rw,
                               nmesh=256,
                               resampler='tsc',
                               interlacing=2,
                               ells=(0, 1, 2, 3, 4),
                               los='endpoint',
                               edges=kedges,
                               position_type='pos',
                                dtype='f4', shotnoise=0).poles


    poles_vvr = CatalogFFTPower(data_positions1=dp,
                                data_weights1=vel_grid,
                                nmesh=256,
                                resampler='tsc',
                                interlacing=2,
                                ells=(0, 1, 2, 3, 4),
                                los='endpoint',
                                edges=kedges,
                                position_type='pos',
                                dtype='f4', wnorm=wnorm_v, shotnoise=0).poles

    pgv = poles_vgr(ell=1, complex=False)
    poles_gv = pgv[:Nkeep]

    pvv = poles_vgr(ell=0, complex=False)
    poles_vv = pvv[:Nkeep]

    pgg = poles_gg(ell=0, complex=False)
    poles_gg = pgg[:Nkeep]

    return np.concatenate((poles_gv, poles_gg, poles_vv))


## RUN MAIN CODE ##

from multiprocessing import Pool
from functools import partial

# Run over set of 99 sims in parallel
with Pool(10) as p:
    run_pipeline_f090 = partial(run_pipeline, freq='f090')
    p_array_f090 = np.array(p.map(run_pipeline_f090, range(1, 100))) #100)))

pgv_array_f090 = p_array_f090[:, :20]
pgg_array_f090 = p_array_f090[:, 20:40]
pvv_array_f090 = p_array_f090[:, 40:]

with Pool(10) as p:
    run_pipeline_f150 = partial(run_pipeline, freq='f150')
    p_array_f150 = np.array(p.map(run_pipeline_f150, range(1, 100)))

pgv_array_f150 = p_array_f150[:, :20]
pgg_array_f150 = p_array_f150[:, 20:40]
pvv_array_f150 = p_array_f150[:, 40:]

print(f"Array shape after parallel computation is {pgv_array_f090.shape}, {pvv_array_f090.shape}")

cov_matrix_f090 = np.cov(np.array(pgv_array_f090).T)
cov_matrix_f150 = np.cov(np.array(pgv_array_f150).T)

out_dir = paths.out_dir

if use_act_data:
    out_dir += "null_test_act_maps_mock_galaxies/"
elif use_boss_data:
    out_dir += "null_test_boss_data_mock_cmb/"
elif with_ksz:
    out_dir += with_ksz_dir

# Save spectra and covmat to file
# The file extension may need to be editied if computing for SGC (deprecated)

#np.savetxt(out_dir + "Pgv_ell_1_NGC_f090_array.dat", np.array(pgv_array_f090).T)
np.savetxt(out_dir + "Pgv_ell_1_NGC_f090_array.dat", np.array(pgv_array_f090).T)
np.savetxt(out_dir + "Pgv_ell_1_NGC_f150_array.dat", np.array(pgv_array_f150).T)

np.savetxt(out_dir + "Pvv_ell_0_NGC_f090_array.dat", np.array(pvv_array_f090).T)
np.savetxt(out_dir + "Pvv_ell_0_NGC_f150_array.dat", np.array(pvv_array_f150).T)

np.savetxt(out_dir + "Pgg_ell_0_NGC_f090_array.dat", np.array(pgg_array_f090).T)
np.savetxt(out_dir + "Pgg_ell_0_NGC_f150_array.dat", np.array(pgg_array_f150).T)

np.savetxt(out_dir + "Pgv_ell_1_NGC_f090_cov_mat.dat", cov_matrix_f090)
np.savetxt(out_dir + "Pgv_ell_1_NGC_f150_cov_mat.dat", cov_matrix_f150)

pgv_array_full = np.concatenate((np.array(pgv_array_f090).T, np.array(pgv_array_f150).T))
print(f"Array shape after concatenation is {np.array(pgv_array_full).shape}")

cov_matrix_full = np.cov(np.array(pgv_array_full))
#np.savetxt(out_dir + "Pgv_ell_1_NGC_full_cov_mat.dat", cov_matrix_full)
np.savetxt(out_dir + "Pgv_ell_1_NGC_full_cov_mat.dat", cov_matrix_full)

print(f"Final covariance matrix shape is {cov_matrix_full.shape}")
