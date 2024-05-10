from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,curvedsky as cs,reproject,bunch
import numpy as np
import os,sys
import healpy as hp
import utils as kutils

Nsims = 100

args = kutils.jobargs # these will be added to filenames
paths = kutils.paths

comm,rank,my_tasks = mpi.distribute(Nsims)
io.mkdir(f'{paths.out_dir}/sims/',comm=comm)

ivars = {}
gmasks = {}
beams = {}
pareas = {}
for freq in args.freqs:
    print(freq)
    ivars[freq] = enmap.read_map(kutils.ivar_file(freq),sel=np.s_[0,...])
    gmasks[freq] = enmap.read_map(f'{paths.out_dir}/mask_{freq}.fits')
    beams[freq] = kutils.get_beam(freq)
    pareas[freq] = enmap.pixsizemap(gmasks[freq].shape,gmasks[freq].wcs)

for task in my_tasks:

    index = task+1
    alm = hp.read_alm(f"{paths.act_sim_root}fullskyLensedUnabberatedCMB_alm_set00_{index:05d}.fits",hdu=1)

    for freq in args.freqs:
        print(f"Rank {rank} starting {freq} for task {index} / {len(my_tasks)}...")
        args.freq = freq
        sstr = kutils.save_string(args)
        ls = beams[freq][0]
        bells = beams[freq][1]
        oalm = cs.almxfl(alm,bells)
        cmap = cs.alm2map(oalm,enmap.empty(gmasks[freq].shape,gmasks[freq].wcs,dtype=np.float32)) # beam convolved signal
        wfit = np.loadtxt(f'{paths.out_dir}/wfit_{sstr}.txt').flatten()[0]

        nmap = maps.white_noise(gmasks[freq].shape,gmasks[freq].wcs,wfit)
        imap = cmap + nmap
        
        alm, falm, ells, theory_filter, wfit, fcls = kutils.get_single_frequency_alms(imap, gmasks[freq],ls,bells,args.fit_lmin,args.fit_lmax,args.lmin,args.lmax,kutils.wfid(args.freqs),debug=False)
        hp.write_alm(f'{paths.out_dir}/sims/filtered_alms_{sstr}_simid_{index}.fits',falm,overwrite=True)
        io.save_cols(f'{paths.out_dir}/sims/theory_filter_{sstr}_simid_{index}.txt',(ells,theory_filter))
        np.savetxt(f'{paths.out_dir}/sims/fcls_{sstr}_simid_{index}.txt',fcls)

    
