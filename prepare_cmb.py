from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,curvedsky as cs,reproject,bunch
import numpy as np
import os,sys
import healpy as hp
import utils as kutils

args = kutils.jobargs # these will be added to filenames
paths = kutils.paths

mask_file = f"{paths.out_dir}/planck_galactic_mask_070.fits" # the 070 here refers to 70% fsky
gmask = enmap.read_map(mask_file)

for freq in kutils.defaults.freqs:
    args.freq = freq # add this to the filename
    act_file = kutils.act_file(freq)
    ivar_file = kutils.ivar_file(freq)

    # Load and mask map
    imap = enmap.read_map(act_file,sel=np.s_[0,...])
    ivar = enmap.read_map(ivar_file,sel=np.s_[0,...])
    rms = maps.rms_from_ivar(ivar)
    gmask[ivar<=0] = 0
    gmask[rms>=kutils.defaults.rms_threshold] = 0

    enmap.write_map(f'{paths.out_dir}/mask_{args.freq}.fits',gmask)


    ls,bells = kutils.get_beam(args.freq)


    gls,gfls = kutils.get_galaxy_filter()
    alm, falm, ells, theory_filter, wfit, fcls,mfact, lkneefit = kutils.get_single_frequency_alms(imap, gmask,ls,bells,
                                                                                                  args.fit_lmin,args.fit_lmax,args.lmin,
                                                                                                  args.lmax,kutils.wfid(args.freq),debug=freq,
                                                                                                  is_sim=False,ivar=ivar,
                                                                                                  gls=gls,gfls=gfls)
    sstr = kutils.save_string(args)
    hp.write_alm(f'{paths.out_dir}/alms_{sstr}.fits',alm,overwrite=True)
    hp.write_alm(f'{paths.out_dir}/filtered_alms_{sstr}.fits',falm,overwrite=True)
    io.save_cols(f'{paths.out_dir}/theory_filter_{sstr}.txt',(ells,theory_filter))
    np.savetxt(f'{paths.out_dir}/wfit_{sstr}.txt',np.asarray((wfit,)))
    np.savetxt(f'{paths.out_dir}/lkneefit_{sstr}.txt',np.asarray((lkneefit,)))
    np.savetxt(f'{paths.out_dir}/mfact_{sstr}.txt',np.asarray((mfact,)))
    np.savetxt(f'{paths.out_dir}/fcls_{sstr}.txt',fcls)

    
