from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,curvedsky as cs,reproject
import numpy as np
import os,sys
import healpy as hp
import paths as kutils
from scipy.optimize import curve_fit

# Minimum and maxmimum multipoles of the CMB map used
lmin = 100
lmax = 8000

# Change this to f150 to run on the 150 GHz map
freq = "f090"

# This is only relevant for fitting the white noise level. Probably no need to change this.
fit_lmin = 1200
fit_lmax = 7000

# This is relevant for making the mask.
# Inspect the output masked_map.png to make 
rms_threshold = 70.0

mask_file = f"{kutils.out_dir}/planck_galactic_mask_070.fits"
act_file = f"{kutils.act_root}act_planck_dr5.01_s08s18_AA_{freq}_night_map_srcfree.fits"
ivar_file = f"{kutils.act_root}act_planck_dr5.01_s08s18_AA_{freq}_night_ivar.fits"
beam_file = f"{kutils.act_root}beams/act_planck_dr5.01_s08s18_{freq}_night_beam.txt"


# Load and mask map
imap = enmap.read_map(act_file,sel=np.s_[0,...])
ivar = enmap.read_map(ivar_file,sel=np.s_[0,...])
rms = maps.rms_from_ivar(ivar)
gmask = enmap.read_map(mask_file)
gmask[rms>=rms_threshold] = 0
w2 = maps.wfactor(2,gmask)
# Mask power correction factor
print(f"W factor: {w2}")

imap[gmask<=0] = 0
io.hplot(imap,"masked_map",downgrade=8,mask=0)
alm = cs.map2alm(imap,lmax=lmax)

# Empirical power spectrum of map
dcltt = cs.alm2cl(alm)/w2
ells = np.arange(dcltt.size)



# Load beam and normalize it
ls,bells = np.loadtxt(beam_file,unpack=True)
if ls[0]<=0:
    bells = bells/bells[0]
else:
    raise ValueError

# Fit white noise
bls = maps.interp(ls,bells)(ells)
theory = cosmology.default_theory()
cltt = theory.lCl('TT',ells)
bin_edges = np.arange(fit_lmin,fit_lmax,40)
binner = stats.bin1D(bin_edges)
cents,ecl_binned = binner.bin(ells,dcltt)
cents,tcl_binned = binner.bin(ells,cltt*bls**2.)
fitfunc = lambda x,a,w: a*tcl_binned+(w*np.pi/180./60.)**2.
w_fid = {'f090':50.0,'f150':30.}[freq]
popt,_ = curve_fit(fitfunc,ells,ecl_binned,p0=[1.,w_fid])
wfit = popt[1]
print(f"Fit white noise {wfit} uK-arcmin. Use this value for simulations.")

pl = io.Plotter('Cell')
pl.add(cents,ecl_binned,label='binned data')
pl.add(cents,tcl_binned,ls=':',label='theory template')
pl.add(cents,fitfunc(ells,*popt),ls='--',label='best fit')
pl.hline((w_fid*np.pi/180./60.)**2.,label='white noise guess')
pl.done('fitcls.png')


# Construct filters
wnoise = (wfit * np.pi/180./60.)**2.
decon_filter = bls / ((bls**2)*cltt + wnoise)  # filter to apply to map
decon_filter[ells<lmin] = 0
theory_filter = 1./(cltt + wnoise/bls**2) # filter to use in theory normalization calculations
theory_filter[ells<lmin] = 0

if np.any(~np.isfinite(decon_filter)): raise ValueError
if np.any(~np.isfinite(theory_filter)): raise ValueError

# Filter
falm = cs.almxfl(alm,decon_filter)
hp.write_alm(f'filtered_alms_{freq}.fits',falm,overwrite=True)
io.save_cols(f'theory_filter.txt',(ells,theory_filter))
fcls = cs.alm2cl(falm)

pl = io.Plotter('Cell')
pl.add(ells,dcltt,label='empirical power')
pl.add(ells,(theory_filter**2.) * (cltt*bls**2+wnoise)/bls**2.,label='expected filtered power')
pl.add(ells,fcls/w2,label='filtered power')
pl.done('empcls.png')
