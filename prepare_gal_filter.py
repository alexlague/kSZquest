import numpy as np
from orphics import io,maps,cosmology
import utils
# import pickle
from scipy.interpolate import InterpolatedUnivariateSpline, RegularGridInterpolator
import hmvec as hm
import sys

paths = kutils.paths

# I've commented out comparisons with Alex's filter

# Load Alex's pickle file
# with open(utils.defaults.gal_filter_location, 'rb') as f:
#     model = pickle.load(f)

# ks = model['ks']
ks = np.geomspace(1e-5,100,1000)

minz,maxz,zeff  = 0.43, 0.7, 0.55
# nzbins = 5
# h = 0.68
# zs = np.linspace(minz, maxz, nzbins)
# mus = np.linspace(-1, 1, len(model['lPggtot'][0, 0, :]))
# Pge_kmu_interp = RegularGridInterpolator((zs, model['ks'], mus), model['sPge'], bounds_error=False, fill_value=0.)
# Pgg_kmu_interp = RegularGridInterpolator((zs, model['ks'], mus), model['sPggtot'], bounds_error=False, fill_value=0.)

# Pgg = Pgg_kmu_interp((zeff, ks*h, 1)) # mu=1
# Pge = Pge_kmu_interp((zeff, ks*h, 1)) # mu=1


# Make my own Pgg and Pge
ngal = 1e-4 # rough CMASS number density Mpc^-3
ms = np.geomspace(2e10,1e17,40)
hcos = hm.HaloModel([zeff],ks,ms=ms)
hcos.add_battaglia_profile("electron",family="AGN")
hcos.add_hod(name="g",ngal=np.asarray([ngal]))
bg = hcos.hods['g']['bg'][0]  # Note that bg is calculated given the ngal
hpge = hcos.get_power_1halo("g","electron") + hcos.get_power_2halo("g","electron")
hpggtot = hcos.get_power_1halo("g","g") + hcos.get_power_2halo("g","g") + 1./ngal

pl = io.Plotter(xyscale='loglog',xlabel='$k$ (Mpc$^{-1}$)', ylabel='$P(k)$ (Mpc$^3$)')
# pl.add(ks,Pge,label=r'Alex $sP_{ge}$')
pl.add(ks,hpge[0],ls=':',label=f'$P_{{ge}}$ hmvec $b_g={bg:.2f}$')
#pl.legend('outside')
#pl.done('pge.png')


#pl = io.Plotter(xyscale='loglog',xlabel='$k$ (Mpc$^{-1}$)', ylabel='$P_{gg}$ (Mpc$^3$)')
# pl.add(ks,Pgg,label=r'Alex $sP_{gg}^{\rm tot}$')
pl.add(ks,hpggtot[0],ls=':',label=f'$P_{{gg}}^{{\\rm tot}}$ hmvec $b_g={bg:.2f}$')
pl.add(ks,1./ngal+ks*0,ls='--',label=f'$n_{{\\rm gal}} = {ngal:.4f}$ Mpc${{^3}}$')
pl.legend('outside')
pl._ax.set_ylim(1e2,3.5e5)
pl.done('pggtot.png')


lmax = utils.jobargs.lmax
ells = np.arange(lmax)
chieff = hcos.comoving_radial_distance(zeff)
kevals = ells/chieff

kfilter = hpge[0]/hpggtot[0]

cfilter = maps.interp(ks,kfilter)(kevals)
cfilter[ells<2] = 0
theory = cosmology.default_theory()
cltt = theory.lCl('TT',ells)
cltttot = cltt + (10.*np.pi/180./60./maps.gauss_beam(ells,1.4))**2.
icfilter = 1./cltttot
icfilter[ells<2] = 0

finfilter = cfilter*icfilter
finfilter[ells<2] = 0

pl = io.Plotter('Cell',xyscale='linlin')
pl.add(ells,cfilter/cfilter.max(),label='$P_{ge}/P_{gg}^{\\rm tot}(k=\\ell/\\chi(z_{\\rm eff}))$')
pl.add(ells,icfilter/icfilter.max(),ls=':',label='$1/C_{\\rm ell}^{\\rm tot}$')
pl.add(ells,finfilter/finfilter.max(),ls='--',label='Final filter')
pl.legend('outside')
pl.done('cfilter.png')

pl = io.Plotter('Pk',xyscale='loglin',ylabel='$F(k)$')
pl.add(ks,kfilter,label='$P_{ge}/P_{gg}^{\\rm tot}$')
pl.done('kfilter.png')


io.save_cols(f'{paths.out_dir}/galaxy_ell_filter.txt',(ells,cfilter))
