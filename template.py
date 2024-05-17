from __future__ import print_function
from orphics import maps,io,cosmology,stats,catalogs
from pixell import enmap
import numpy as np
import os,sys
import healpy as hp

root = "/data5/sdss/"

def get_vrecon(fnames,tags):
    cols = {}
    for fname,tag in zip(fnames,tags):
        cols[tag] = catalogs.load_fits(f"{root}{fname}",['RA','DEC','Z','WEIGHT_FKP'])

    print("Starting velocity recon...")
    catalogs.reconstruct_velocities(cols['data']['RA'],cols['data']['DEC'],cols['data']['Z'],
                                    cols['random']['RA'],cols['random']['DEC'],cols['random']['Z'],
                                    zeff=0.55,bg=1.92, # cmass defaults
                                    h = 0.676, omegam=0.31,
                                    fkp_weights=cols['data']['WEIGHT_FKP'],
                                    fkp_weights_rand=cols['random']['WEIGHT_FKP'])
    
get_vrecon([f"boss_dr12/galaxy_DR12v5_CMASS_North.fits.gz",
        f"boss_dr12/random0_DR12v5_CMASS_North.fits.gz",],
       ['data','random'])
