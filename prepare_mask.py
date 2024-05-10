from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,curvedsky as cs,reproject
import numpy as np
import os,sys
import healpy as hp
import utils as kutils

paths = kutils.paths

ivar_file = f"{paths.act_root}/act_dr5.01_s08s18_AA_f090_night_ivar.fits"
mask_file = f"{paths.planck_root}HFI_Mask_GalPlane-apo0_2048_R2.00.fits"

shape,wcs = enmap.read_map_geometry(ivar_file)

# Mask
hmask = hp.read_map(mask_file,field=3)
gmask = reproject.healpix2map(hmask,shape,wcs,method='spline',order=0,rot='gal,equ')

enmap.write_map(f"{paths.out_dir}/planck_galactic_mask_070.fits",gmask)
