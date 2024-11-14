from orphics import maps,io,cosmology,stats
from pixell import enmap,curvedsky as cs,reproject,bunch
import numpy as np
from scipy.optimize import curve_fit

defaults = bunch.Bunch(io.config_from_yaml("defaults_local.yaml"))
jobargs = bunch.Bunch(defaults.jobargs)
paths = bunch.Bunch(defaults.paths)

def act_file(freq):
    return f"{paths.act_root}act_planck_dr5.01_s08s18_AA_{freq}_{jobargs.daynight}_map_srcfree.fits"

def ivar_file(freq):
    return f"{paths.act_root}act_planck_dr5.01_s08s18_AA_{freq}_{jobargs.daynight}_ivar.fits"

def wfid(freq):
    return {'f090':30.0,'f150':20.}[freq]

def get_beam(freq):
    # Warning: this doesn't check that delta_ell=1, which is assumed elsewhere
    beam_file = f"{paths.act_root}../auxilliary/beams/act_planck_dr5.01_s08s18_{freq}_{jobargs.daynight}_beam.txt"

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

def get_single_frequency_alms(imap, gmask,ls,bells,fit_lmin,fit_lmax,lmin,lmax,w_fid,debug=False,is_sim=False,ivar=None,gls=None,gfls=None):
    """
    If gfls is not None, it is assumed to be an additional filter to apply
    """
    
    imap[gmask<=0] = 0
    if debug:
        io.hplot(imap,f"masked_map_{debug}",downgrade=8,mask=0)
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
    fitfunc = lambda x,a,w,lknee: a*tcl_binned+ binner.bin(ells,maps.rednoise(ells,w,lknee,-3))[1]
    popt,_ = curve_fit(fitfunc,cents,ecl_binned,p0=[1.,w_fid,3000.])
    wfit = popt[1]
    lkneefit = popt[2]
    print(f"Fit white noise {wfit} uK-arcmin and lknee {lkneefit}. Use these values for simulations.")

    if not(is_sim):
        nmap = maps.modulated_noise_map(ivar,lknee=lkneefit,alpha=-3,lmax=lmax,lmin=lmin)
        nmap[gmask<=0] = 0
        nalm = cs.map2alm(nmap,lmax=lmax)
        # Empirical power spectrum of map
        ncltt = cs.alm2cl(nalm)/w2
        nmean = ncltt[-500:].mean()
        wsim = np.sqrt(nmean)/(np.pi/180./60.)
        nmean_dat = dcltt[-500:].mean()
        wdat = np.sqrt(nmean_dat)/(np.pi/180./60.)
        mfact = wdat/wsim
        print(f"White noise rescale factor is {mfact}. Use this value for simulations.")

    else:
        mfact = None

    if debug:
        pl = io.Plotter('Cell')
        pl.add(cents,ecl_binned,label='binned data')
        # pl.add(cents,tcl_binned,ls=':',label='theory template')
        pl.add(cents,fitfunc(ells,*popt),ls='--',label='best fit')
        pl.hline((w_fid*np.pi/180./60.)**2.,label='white noise guess')
        pl.done(f'fitcls_{debug}.png')


    # Construct filters

    if not(gfls is None):
        ogfls = maps.interp(gls,gfls)(ells)
    else:
        ogfls = 1.0
    
    wnoise = maps.rednoise(ells,wfit,lkneefit,alpha=-3)
    decon_filter = ogfls * bls / ((bls**2)*cltt + wnoise)  # filter to apply to map
    decon_filter[ells<lmin] = 0
    theory_filter = ogfls/(cltt + wnoise/bls**2) # filter to use in theory normalization calculations
    theory_filter[ells<lmin] = 0

    if np.any(~np.isfinite(decon_filter)): raise ValueError
    if np.any(~np.isfinite(theory_filter)): raise ValueError

    # Filter
    falm = cs.almxfl(alm,decon_filter)
    fcls = cs.alm2cl(falm)

    if debug:
        pl = io.Plotter('Cell')
        pl.add(ells,dcltt,label='empirical power')
        pl.add(ells,(theory_filter**2.) * (cltt*bls**2+wnoise)/bls**2.,label='expected filtered power')
        pl.add(ells,fcls/w2,label='filtered power')
        pl.done(f'empcls_{debug}.png')

    return alm, falm, ells, theory_filter, wfit, fcls/w2, mfact, lkneefit


    
        
        
def get_galaxy_filter():
    if defaults['apply_gal_filter']:
        gls,gfls = np.loadtxt(f'{paths.out_dir}galaxy_ell_filter.txt',unpack=True)
    else:
        gls = None
        gfls = None
    return gls, gfls

