from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,curvedsky as cs,reproject,bunch
import numpy as np
import os,sys
import healpy as hp
import paths
import utils as kutils

args = bunch.Bunch({})

args = kutils.defaults


# This is relevant for making the mask.
# Inspect the output masked_map.png to make sure this is ok
rms_threshold = 70.0 # uK-arcmin

mask_file = f"{paths.out_dir}/planck_galactic_mask_070.fits" # the 070 here refers to 70% fsky
gmask = enmap.read_map(mask_file)

for freq in kutils.freqs:
    args.freq = freq
    act_file = kutils.act_file(freq)
    ivar_file = kutils.ivar_file(freq)

    # Load and mask map
    imap = enmap.read_map(act_file,sel=np.s_[0,...])
    ivar = enmap.read_map(ivar_file,sel=np.s_[0,...])
    rms = maps.rms_from_ivar(ivar)
    gmask[rms>=rms_threshold] = 0

    enmap.write_map(f'mask_{args.freq}.fits',gmask)


    ls,bells = kutils.get_beam(args.freq)


    alm, falm, ells, theory_filter = kutils.get_single_frequency_alms(imap, gmask,ls,bells,args.fit_lmin,args.fit_lmax,args.lmin,args.lmax,kutils.wfid(args.freq),debug=False)
    sstr = kutils.save_string(args)
    hp.write_alm(f'filtered_alms_{sstr}.fits',falm,overwrite=True)
    io.save_cols(f'theory_filter_{sstr}.txt',(ells,theory_filter))
