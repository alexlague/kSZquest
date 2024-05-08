import paths
from orphics import maps,io,cosmology,stats
from pixell import enmap,curvedsky as cs,reproject,bunch
import numpy as np
from scipy.optimize import curve_fit

defaults = bunch.Bunch({})

# Minimum and maxmimum multipoles of the CMB map used
defaults.lmin = 100
defaults.lmax = 8000

# This is only relevant for fitting the white noise level. Probably no need to change this.
defaults.fit_lmin = 1200
defaults.fit_lmax = 7000

# TODO: change to list of both frequencies
#freqs = ['f090','f150']
freqs = ['f090']

def act_file(freq):
    # TODO: change to daynight
    return f"{paths.act_root}act_planck_dr5.01_s08s18_AA_{freq}_night_map_srcfree.fits"

def ivar_file(freq):
    return f"{paths.act_root}act_planck_dr5.01_s08s18_AA_{freq}_night_ivar.fits"

def wfid(freq):
    return {'f090':50.0,'f150':30.}[freq]

def get_beam(freq):
    # Warning: this doesn't check that delta_ell=1, which is assumed elsewhere
    # TODO: change to daynight
    beam_file = f"{paths.act_root}beams/act_planck_dr5.01_s08s18_{freq}_night_beam.txt"

    # Load beam and normalize it
    ls,bells = np.loadtxt(beam_file,unpack=True)
    if ls[0]==0:
        bells = bells/bells[0]
    else:
        raise ValueError

    return ls,bells


def save_string(args):
    sstr = ""
    for key in sorted(args.keys()):
        sstr = sstr + f"{key}_{args._dict[key]}_"
    return sstr[:-1]

def get_single_frequency_alms(imap, gmask,ls,bells,fit_lmin,fit_lmax,lmin,lmax,w_fid,debug=False):
    imap[gmask<=0] = 0
    if debug:
        io.hplot(imap,"masked_map",downgrade=8,mask=0)
    w2 = maps.wfactor(2,gmask)
    # Mask power correction factor
    print(f"W factor: {w2}")
                              
    alm = cs.map2alm(imap,lmax=lmax)

    # Empirical power spectrum of map
    dcltt = cs.alm2cl(alm)/w2
    ells = np.arange(dcltt.size)


    # Fit white noise
    bls = maps.interp(ls,bells)(ells)
    theory = cosmology.default_theory()
    cltt = theory.lCl('TT',ells)
    bin_edges = np.arange(fit_lmin,fit_lmax,40)
    binner = stats.bin1D(bin_edges)
    cents,ecl_binned = binner.bin(ells,dcltt)
    cents,tcl_binned = binner.bin(ells,cltt*bls**2.)
    fitfunc = lambda x,a,w: a*tcl_binned+(w*np.pi/180./60.)**2.
    popt,_ = curve_fit(fitfunc,ells,ecl_binned,p0=[1.,w_fid])
    wfit = popt[1]
    print(f"Fit white noise {wfit} uK-arcmin. Use this value for simulations.")

    if debug:
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

    if debug:
        fcls = cs.alm2cl(falm)
        pl = io.Plotter('Cell')
        pl.add(ells,dcltt,label='empirical power')
        pl.add(ells,(theory_filter**2.) * (cltt*bls**2+wnoise)/bls**2.,label='expected filtered power')
        pl.add(ells,fcls/w2,label='filtered power')
        pl.done('empcls.png')

    return alm, falm, ells, theory_filter
