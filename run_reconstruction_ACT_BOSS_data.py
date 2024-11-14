##############################################
# MAIN RECONSTRUCTION MODULE TO USE ON DATA
#
# Specify frequency map and galactic cap
# (SGC and BOTH are there but deprecated)
#
# Can also run null test by shuffling
# the velocities by setting null_test=True
# 
#
##############################################

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
import os
import time
sys.path.append('../repo/kSZquest/')

import lightcone
import cmb
import recon

# Fiducial values and constants
speed_of_light = 299792.458 # km/s
f = 0.762
b = 1.92


# Select which datasets to use and whether to run null test
freq = "f090"
cap = "NGC"
null_test = False

## LOAD LSS DATA ##
BOSS_dir = '/home/r/rbond/alague/scratch/ksz-pipeline/BOSS_data/CMASS/'
BOSS_data = 'CMASS'

if cap == "SGC":
    BOSS_file = 'pre-recon/cmass/galaxy_DR12v5_CMASS_South_t2.txt'
    BOSS_recon = 'post-recon/cmass/galaxy_DR12v5_CMASS_South_t2_recon.txt'
    BOSS_rand = 'pre-recon/cmass/random0_DR12v5_CMASS_South_t1.txt'
    data = np.loadtxt(BOSS_dir+BOSS_file)
    rand = np.loadtxt(BOSS_dir+BOSS_rand)
    data_recon = np.loadtxt(BOSS_dir+BOSS_recon)

elif cap == "NGC":
    BOSS_file = 'pre-recon/cmass/galaxy_DR12v5_CMASS_North_t2.txt'
    BOSS_recon = 'post-recon/cmass/galaxy_DR12v5_CMASS_North_t2_recon.txt'
    BOSS_rand = 'pre-recon/cmass/random0_DR12v5_CMASS_North_t1.txt'
    data = np.loadtxt(BOSS_dir+BOSS_file)
    rand = np.loadtxt(BOSS_dir+BOSS_rand)
    data_recon = np.loadtxt(BOSS_dir+BOSS_recon)

elif cap == "BOTH":
    BOSS_file = 'pre-recon/cmass/galaxy_DR12v5_CMASS_North_t2.txt'
    BOSS_recon = 'post-recon/cmass/galaxy_DR12v5_CMASS_North_t2_recon.txt'
    BOSS_rand = 'pre-recon/cmass/random0_DR12v5_CMASS_North_t1.txt'

    data = np.loadtxt(BOSS_dir+BOSS_file)
    rand = np.loadtxt(BOSS_dir+BOSS_rand)
    data_recon = np.loadtxt(BOSS_dir+BOSS_recon)

    BOSS_file = 'pre-recon/cmass/galaxy_DR12v5_CMASS_South_t2.txt'
    BOSS_recon = 'post-recon/cmass/galaxy_DR12v5_CMASS_South_t2_recon.txt'
    BOSS_rand = 'pre-recon/cmass/random0_DR12v5_CMASS_South_t1.txt'
    data = np.concatenate((data, np.loadtxt(BOSS_dir+BOSS_file)))
    rand = np.concatenate((rand, np.loadtxt(BOSS_dir+BOSS_rand)))
    data_recon = np.concatenate((data_recon, np.loadtxt(BOSS_dir+BOSS_recon)))

z_min = 0.43
z_max = 0.7
z_eff = 0.55

# Can remove massive objects for tSZ contamination
# list of objects is pre-generated through the prepare_galaxy_catalog.py file
remove_massive = False
if remove_massive:
    
    prep_dir = '/home/r/rbond/alague/scratch/ksz-pipeline/ksz-analysis/quadratic_estimator/development_code/prepared_maps/'
    m_gal_ras, m_gal_decs, m_gal_zs = np.loadtxt(prep_dir + 'massive_clusters_to_remove_CMASS_'+cap+'.dat')
    keep = np.ones(len(data))
    for gi in range(len(data)):
        if np.any(np.isclose(m_gal_ras, data[:,0][gi]) * np.isclose(m_gal_decs, data[:,1][gi]) * np.isclose(m_gal_zs, data[:,2][gi])):
            keep[gi] = 0
    keep[list(massive_galaxies_indexes)] = 0
    data = data[keep==1]
    data_recon = data_recon[keep==1]
    print("Number of data points removed by cluster mask: ", len(keep[keep!=1]))
    
zsel = (data[:,2] >= z_min) & (data[:,2] <= z_max)
data = data[zsel]

data_FKP = data[:,3]
data_comp = data[:,6] * (data[:,4] + data[:,5] - 1)  # WEIGHT_SYSTOT * (WEIGHT_CP+WEIGHT_NOZ-1) 

data_recon = data_recon[zsel]

zsel_rand = (rand[:,2] >= z_min) & (rand[:,2] <= z_max)
rand = rand[zsel_rand]
rand_FKP = rand[:,3]

# the fiducial BOSS DR12 cosmology
cosmo = cosmology.Cosmology(h=0.676).match(Omega0_m=0.31)

# export to ArrayCatalog
data_cat = ArrayCatalog({'RA':data[:,0], 'DEC':data[:,1], 'Z':data[:,2]})
rand_cat = ArrayCatalog({'RA':rand[:,0], 'DEC':rand[:,1], 'Z':rand[:,2]})
data_recon_cat = ArrayCatalog({'RA':data_recon[:,0], 'DEC':data_recon[:,1], 'Z':data_recon[:,2]})

# add Cartesian position column
data_cat['Position'] = transform.SkyToCartesian(data_cat['RA'], data_cat['DEC'], data_cat['Z'], cosmo=cosmo)
data_recon_cat['Position'] = transform.SkyToCartesian(data_recon_cat['RA'], data_recon_cat['DEC'], data_recon_cat['Z'], cosmo=cosmo)
rand_cat['Position'] = transform.SkyToCartesian(rand_cat['RA'], rand_cat['DEC'], rand_cat['Z'], cosmo=cosmo)

# LOAD CMB DATA

# alms
NSIDE = 2048
prefix = "filtered_alms_daynight_daynight_fit_lmax_7980_fit_lmin_1000_freq_"
suffix = "_lmax_8000_lmin_100.fits"
filtered_alms = hp.fitsfunc.read_alm('prepared_maps/' + prefix + freq + suffix)
alms = np.zeros((3, len(filtered_alms)), dtype=complex)
alms[0] = filtered_alms
healpix_filtered = hp.alm2map(alms, NSIDE, pol=False)[0]

def run_recon(seed=0):

    data_positions = np.array(data_cat['Position'])
    data_positions_rec = np.array(data_recon_cat['Position'])
    
    Delta_pos = data_positions_rec - data_positions
    vel = np.sum(Delta_pos*data_positions_rec, axis=1)
    vel /= np.linalg.norm(data_positions_rec, axis=1) # in Mpc/h
    vel /= cosmo.h # in Mpc                                                                     

    H = cosmo.hubble_function(z_eff)*299792.458 # H(zeff)/c * speed_of_light gives km/s/Mpc
    vel *= H / (1+z_eff) # in km/s
    
    # galaxy density already stored in galaxy catalog data file
    nz = data[:,7]
    
    # smooth n(z) and <v^2(z)>
    nbins = 200
    zbin_edges = np.linspace(z_min, z_max, nbins+1)
    zbins = np.linspace(z_min, z_max, nbins)
    var_vel_z = np.zeros(nbins)
    nz_z = np.zeros(nbins)
    redshifts = data[:,2]
    for i in range(nbins):
        ind = (redshifts >= zbin_edges[i]) & (redshifts <= zbin_edges[i+1])
        var_vel_z[i] = np.var(vel[ind])
        nz_z[i] = np.mean(nz[ind])

    # velocity weights
    P0 = 5e9 
    v2 = 1500**2 
    n = CubicSpline(zbins, nz_z)(redshifts)
    vel_weights = 1./(v2+n*P0) # include completeness weights
    
    kedges = np.linspace(0, 0.2, 51)

    
    #========== QE starts here ==========#
    
    ## STORE DATA IN LIGHTCONE OBJECT ##
    
    FSKY = 0.1 # does not matter in calculation, but needs to be set TODO:remove
    Nmesh = 512 

    # Create lightcone
    lc = lightcone.LightCone(FSKY, Nmesh=Nmesh)

    # ACT footprint overlaps with BOSS below dec of 21.61666 deg TODO: not hard-coded
    ACT_SLICE = data_cat['DEC'] <= 21.61666
    ACT_SLICE_RAND = rand_cat['DEC'] <= 21.61666

    # slice removes data outside of act footprint
    vel_weights = vel_weights[ACT_SLICE]/np.mean(vel_weights[ACT_SLICE]) * data_comp[ACT_SLICE] # normalize by the mean for numerical stability
    wnorm_v = normalization_from_nbar(nz[ACT_SLICE], data_weights=vel_weights)
    wnorm_gv = (wnorm_v * normalization_from_nbar(nz[ACT_SLICE], data_weights=data[:,3][ACT_SLICE]*data_comp[ACT_SLICE]))**0.5
    
    print(f"gv weight norm is: {wnorm_gv}")

    # Use already loaded data
    lc.data = data_cat[ACT_SLICE]
    lc.randoms = rand_cat[ACT_SLICE_RAND]
    lc.data['NZ'] = data[:,-1][ACT_SLICE]
    lc.randoms['NZ'] = rand[:,-1][ACT_SLICE_RAND]

    lc.data['ra'] = data_cat['RA'][ACT_SLICE]
    lc.data['dec'] = data_cat['DEC'][ACT_SLICE]
    lc.data['z'] = data_cat['Z'][ACT_SLICE]

    # if computing Tgrid from randoms
    lc.randoms['ra'] = rand_cat['RA'][ACT_SLICE_RAND]
    lc.randoms['dec'] = rand_cat['DEC'][ACT_SLICE_RAND]
    lc.randoms['z'] = rand_cat['Z'][ACT_SLICE_RAND]

    lc.fkp_catalog = FKPCatalog(lc.data, lc.randoms)
    lc.fkp_catalog['data/FKPWeight'] = data[:,3][ACT_SLICE]
    lc.fkp_catalog['randoms/FKPWeight'] = rand[:,3][ACT_SLICE_RAND]

    lc.minZ = z_min
    lc.maxZ = z_max
    lc.nofz = CubicSpline(zbins, nz_z)
    lc.BoxSize = np.max(data_positions, axis=0) - np.min(data_positions, axis=0)
    print("Box: ", lc.BoxSize)
    
    # Catalog to mesh
    lc.PaintMesh()

    # Compute/load model
    lc.GetPowerSpectraModel()

    LMAX = 5000
    ksz_lc = cmb.CMBMap(healpix_filtered, {}, LMAX,
                        noise_lvl=None,
                        theta_FWHM=None,
                        NSIDE=NSIDE, 
                        do_beam=False)
    ## RUN RECON ##
    DoFilter = False
    AddNoise = False
    AddPrimary = False
    ksz_lc.PrepareRecon(AddPrimary=AddPrimary, AddNoise=AddNoise, DoFilter=DoFilter)
    
    fil_dir = '/home/r/rbond/alague/scratch/ksz-pipeline/ksz-analysis/quadratic_estimator/'
    fil_dir += 'development_code/prepared_maps/'
    fil_path = fil_dir+"theory_filter_daynight_daynight_fit_lmax_7980_fit_lmin_1000_freq_" + freq + "_lmax_8000_lmin_100.txt"
    prefactor, recon_noise = recon.CalculateNoiseFromFilter(lc, CMBFilterPath=fil_path)

    print(prefactor, recon_noise)

    # reconstruction output does not include N^0 normalization yet: stored in prefactor
    vhat = recon.RunReconstruction(lc, ksz_lc.kSZ_map, ComputePower=False, NSIDE=NSIDE, use_T_grid=False)

    # normalize with N0 and convert to km/s
    vel_grid =  lc.T_vals * prefactor * 3e5

    if null_test:
        np.random.seed((os.getpid() * int(time.time())) % 123456789) # trying this to get different seeds
        vel_grid = vel_grid[np.random.permutation(len(vel_grid))]
    else:
        np.savetxt('BOSS_ACT_QE_velocities_'+cap+'_'+freq+'.dat', vel_grid.real)

    print("Recon vel std: ", np.std(vel_grid))
    print("BAO vel std: ", np.std(vel))

    # Remove velocity offset
    vel_grid -= np.mean(vel_grid)

    ## COMPUTE SPECTRA ##

    # Remove outliers 
    NO_OUT = abs(vel_grid) <= 10 * vel_grid.std()
    
    # positions and weights
    dp = data_positions[ACT_SLICE][NO_OUT]
    dw = (data[:,3] * data_comp)[ACT_SLICE][NO_OUT]
    rp = np.array(rand_cat['Position'])[ACT_SLICE_RAND]
    rw = rand[:,3][ACT_SLICE_RAND]
    vel = (vel * data_comp)[ACT_SLICE][NO_OUT] #data[:,4]
    vel_grid = (vel_grid * vel_weights)[NO_OUT] # DEBUG: check completness weights for vel

    poles_vgr = CatalogFFTPower(data_positions1=dp,
                                data_weights1=vel_grid,
                               data_positions2=dp,
                               randoms_positions2=rp,
                               randoms_weights2=rw,
                               data_weights2=dw,
                               nmesh=256,
                               resampler='tsc',
                               interlacing=2,
                               ells=(0, 1, 2, 4),
                               los='endpoint',
                               edges=kedges,
                               position_type='pos',
                                dtype='f4', wnorm=wnorm_gv).poles

    poles_vvr = CatalogFFTPower(data_positions1=dp,
                                    data_weights1=vel_grid,
                                    data_positions2=dp,
                                    data_weights2=vel,
                                    nmesh=256,
                                    resampler='tsc',
                                    interlacing=2,
                                    ells=(0, 1, 2, 4),
                                    los='endpoint',
                                    edges=kedges,
                                    position_type='pos',
                                    dtype='f4', wnorm=wnorm_v, shotnoise=0).poles
    
    poles_v = CatalogFFTPower(data_positions1=dp,
                              data_weights1=vel,
                              nmesh=256,
                              resampler='tsc',
                              interlacing=2,
                              ells=(0, 2, 4),
                              los='endpoint',
                              edges=kedges,
                              position_type='pos',
                              dtype='f4', wnorm=wnorm_v, shotnoise=0).poles
    

    poles_vr = CatalogFFTPower(data_positions1=dp,
                               data_weights1=vel_grid,
                               nmesh=256,
                               resampler='tsc',
                               interlacing=2,
                               ells=(0, 2, 4),
                               los='endpoint',
                               edges=kedges,
                               position_type='pos',
                               dtype='f4', wnorm=wnorm_v, shotnoise=0).poles


    # gg
    poles = CatalogFFTPower(data_positions1=dp,
                        randoms_positions1=rp,
                            randoms_weights1=rw,
                            data_weights1=dw,
                            nmesh=256, 
                            resampler='tsc', 
                            interlacing=2, 
                            ells=(0, 2, 4), 
                            los='endpoint', 
                            edges=kedges, 
                            position_type='pos', 
                            dtype='f4').poles

        
        
    poles_vg = CatalogFFTPower(data_positions1=dp,
                               data_weights1=vel,
                               data_positions2=dp,
                               randoms_positions2=rp,
                               randoms_weights2=rw,
                               data_weights2=dw,
                               nmesh=256,
                               resampler='tsc',
                               interlacing=2,
                               ells=(0, 1, 2, 4),
                               los='endpoint',
                               edges=kedges,
                               position_type='pos',
                               dtype='f4', wnorm=wnorm_gv).poles
    
    
    
    print("Saving to file!")
    
    ## SAVE ALL SPECTRA TO FILE ##
    # If null test, return the array to avoid file writing conflicts between processes
    
    spectra_dir = BOSS_data + '_spectra/' + freq + '/'  #f150/ #f150/# _minus_f150/'

    if null_test:
        
        return poles_vgr(ell=1, complex=False)
    
    else:
        
        np.savetxt(spectra_dir + 'gg_monopole_' + cap + '.dat', poles(ell=0, complex=False))
        np.savetxt(spectra_dir + 'vv_monopole_'+ cap  + '.dat', poles_v(ell=0, complex=False))
        np.savetxt(spectra_dir + 'vv_recon_monopole_'+ cap  + '.dat', poles_vr(ell=0, complex=False))
        np.savetxt(spectra_dir + 'gv_dipole_'+ cap + '.dat', poles_vg(ell=1, complex=False))
        np.savetxt(spectra_dir + 'gv_recon_dipole_'+ cap  + '.dat', poles_vgr(ell=1, complex=False))
        np.savetxt(spectra_dir + 'gvv_recon_monopole_'+ cap  + '.dat', poles_vvr(ell=0, complex=False))

        return

## RUN CODE ##
if null_test:

    spectra_dir = BOSS_data + '_spectra/' + freq + '/'

    from multiprocessing import Pool
    with Pool(16) as p:
        Pgv_array = p.map(run_recon, range(1, 128))

    np.savetxt(spectra_dir + 'null/shuffles/gv_recon_dipole_null_NGC_array.dat', Pgv_array)

else:
    run_recon()

