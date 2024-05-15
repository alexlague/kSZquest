# Compute hmvec model for Pge
# and fit Pggtot with fiducial values
# to avoid recomputing at every step

import numpy as np
from orphics import io
from nbodykit.lab import *
import nbodykit
from scipy.interpolate import CubicSpline, RegularGridInterpolator
import pickle
import sys
sys.path.append('../repo/kSZquest/')

import lightcone

import utils as kutils
args = kutils.jobargs
paths = kutils.paths


## LOAD LSS DATA ##
BOSS_file = 'pre-recon/cmass/galaxy_DR12v5_CMASS_North_t2.txt'

f = 0.762
b = 1.92
z_min = 0.43
z_max = 0.7
z_eff = 0.55

data = np.loadtxt(paths.boss_root+BOSS_file)
zsel = (data[:,2] >= z_min) & (data[:,2] <= z_max)
data = data[zsel]
nz = data[:,7]

# smooth n(z)
nbins = 200
zbin_edges = np.linspace(z_min, z_max, nbins+1)
zbins = np.linspace(z_min, z_max, nbins)
nz_z = np.zeros(nbins)
redshifts = data[:,2]
for i in range(nbins):
    ind = (redshifts >= zbin_edges[i]) & (redshifts <= zbin_edges[i+1])
    nz_z[i] = np.mean(nz[ind])
    

FSKY = 0.1 # shouldn't matter CHECK
Nmesh = 256

# Initialize lightcone object
lc = lightcone.LightCone(FSKY, Nmesh=Nmesh)

lc.minZ = z_min
lc.maxZ = z_max
lc.nofz = CubicSpline(zbins, nz_z)

# Compute model
lc.bg = b
lc.GetPowerSpectraModel()

# Save output
fiducial = lc.model

with open(paths.out_dir + 'cmass_fiducial_model.pkl', 'wb') as fp:
    pickle.dump(fiducial, fp, protocol=pickle.HIGHEST_PROTOCOL)
