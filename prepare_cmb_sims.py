from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,curvedsky as cs,reproject,bunch
import numpy as np
import os,sys
import healpy as hp
import paths
import utils as kutils

Nsims = 100

args = kutils.defaults

gmask = enmap.read_map(f'mask_{args.freq}.fits')
ls,bells = kutils.get_beam(args.freq)
sstr = kutils.save_string(args)

comm,rank,my_tasks = mpi.distribute(Nsims)

ivars = {}
for freq in kutils.freqs:
    ivars[freq] = enmap.read_map(kutils.ivar_file(freq))

for task in my_tasks:

    alm = hp.read_alm(f"{paths.act_sim_root}fullskyLensedUnabberatedCMB_alm_set00_{(task+1):05d}.fits",hdu=1)

    for freq in kutils.freqs:
        args.freq = freq
        ls,bells = kutils.get_beam(freq)
        oalm = cs.almxfl(alm,bells)
        cmap = cs.alm2map(oalm,enmap.empty(gmask.shape,gmask.wcs,dtype=np.float32)) # beam convolved signal
        
        alm, falm, ells, theory_filter = kutils.get_single_frequency_alms(imap, gmask,ls,bells,args.fit_lmin,args.fit_lmax,args.lmin,args.lmax,kutils.wfid(args.freq),debug=False)
        hp.write_alm(f'filtered_alms_{sstr}_simid_{task}.fits',falm,overwrite=True)
        io.save_cols(f'theory_filter_{sstr}_simid_{task}.txt',(ells,theory_filter))

