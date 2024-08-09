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
for freq in kutils.defaults.freqs:
    ivars[freq] = enmap.read_map(kutils.ivar_file(freq),sel=np.s_[0,...])
    gmasks[freq] = enmap.read_map(f'{paths.out_dir}/mask_{freq}.fits')
    beams[freq] = kutils.get_beam(freq)
    pareas[freq] = enmap.pixsizemap(gmasks[freq].shape,gmasks[freq].wcs)

for task in my_tasks:

    index = task+1
    alm = hp.read_alm(f"{paths.act_sim_root}fullskyLensedUnabberatedCMB_alm_set00_{index:05d}.fits",hdu=1)
    
    # AL Modif
    # add correlated ksz
    add_ksz = True
    if add_ksz:
        ksz_dir = '/home/r/rbond/alague/scratch/ksz-pipeline/ksz-analysis/quadratic_estimator/development_code/QPM_maps/'
        ksz_sim = hp.read_map(ksz_dir + f'ksz_map_from_BAO_recon_{index}.fits', dtype=np.float32)
        ksz_sim *= 1e6 # to match units of ACT sims
        #cmb = hp.alm2map(alm, 1024) # nside of simulated ksz maps
        LMAX = hp.sphtfunc.Alm.getlmax(len(alm))
        ksz_alm = hp.map2alm(ksz_sim, lmax=LMAX)
        alm = ksz_alm + alm # DEBUG TO SEE IF KSZ MAP MATCHES INPUT MOCK CATALOG
        #hp.map2alm(ksz_sim)
        #alm = map2alm(ksz_plus_cmb, lmax=hp.sphtfunc.Alm.getlmax(alm))
    
    for freq in kutils.defaults.freqs:
        print(f"Rank {rank} starting {freq} for task {index} / {len(my_tasks)}...")
        args.freq = freq
        sstr = kutils.save_string(args)
        ls = beams[freq][0]
        bells = beams[freq][1]
        oalm = cs.almxfl(alm,bells)
        cmap = cs.alm2map(oalm,enmap.empty(gmasks[freq].shape,gmasks[freq].wcs,dtype=np.float32)) # beam convolved signal
        wfit = np.loadtxt(f'{paths.out_dir}/wfit_{sstr}.txt').flatten()[0]
        lkneefit = np.loadtxt(f'{paths.out_dir}/lkneefit_{sstr}.txt').flatten()[0]
        mfact = np.loadtxt(f'{paths.out_dir}/mfact_{sstr}.txt').flatten()[0]

        nmap = maps.modulated_noise_map(ivars[freq],lknee=lkneefit,alpha=-3,lmin=args.lmin,lmax=args.lmax)
        nmap = nmap * mfact

        imap = cmap + nmap

        gls,gfls = kutils.get_galaxy_filter()
        alm, falm, ells, theory_filter, wfit, fcls,_,_ = kutils.get_single_frequency_alms(imap, gmasks[freq],ls,bells,
                                                                                          args.fit_lmin,args.fit_lmax,args.lmin,args.lmax,
                                                                                          kutils.wfid(args.freq),debug=False,is_sim=True,
                                                                                          gls=gls,gfls=gfls)
        if add_ksz:
            hp.write_alm(f'{paths.out_dir}/sims/alms_{sstr}_simid_{index}_with_ksz.fits',alm,overwrite=True)
            hp.write_alm(f'{paths.out_dir}/sims/filtered_alms_{sstr}_simid_{index}_with_ksz.fits',falm,overwrite=True)
            io.save_cols(f'{paths.out_dir}/sims/theory_filter_{sstr}_simid_{index}_with_ksz.txt',(ells,theory_filter))
            np.savetxt(f'{paths.out_dir}/sims/fcls_{sstr}_simid_{index}_with_ksz.txt',fcls)
        else:
            hp.write_alm(f'{paths.out_dir}/sims/alms_{sstr}_simid_{index}.fits',alm,overwrite=True)
            hp.write_alm(f'{paths.out_dir}/sims/filtered_alms_{sstr}_simid_{index}.fits',falm,overwrite=True)
            io.save_cols(f'{paths.out_dir}/sims/theory_filter_{sstr}_simid_{index}.txt',(ells,theory_filter))
            np.savetxt(f'{paths.out_dir}/sims/fcls_{sstr}_simid_{index}.txt',fcls)
