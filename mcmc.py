import numpy as np
import zeus
from nbodykit.lab import *
from multiprocessing import Pool
from scipy.interpolate import CubicSpline
from hmvec import cosmology as hcos

cosmo = cosmology.Cosmology(h=0.676).match(Omega0_m=0.31)
h = hcos.Cosmology(engine='camb')
z_eff = 0.55
Plin = cosmology.LinearPower(cosmo, z_eff, transfer='CLASS')
H = cosmo.hubble_function(0.55)*299792.458 # H(zeff)/c * speed_of_light gives km/s/Mpc

# Fiducial linear pk
b_g = 1.92
f = 0.762
faH = 0.762/(1+z_eff)* H

ks = np.geomspace(1e-4, 20, 1000) #h/Mpc
mus = np.linspace(-1, 1, 500)
k_mu = np.meshgrid(ks, mus)
Tk = h.Tk(k_mu[0]*cosmo.h, type ='eisenhu_osc') # k in Mpc^-1
Omega_m = 0.31
deltac = 1.42
growth =  h.D_growth(1/(1+z_eff))
alpha = (2. * (k_mu[0]*cosmo.h)**2. * Tk) * growth # k in Mpc^-1
alpha /= (3.* Omega_m * h.h_of_z(0)**2.)
beta = 2. * deltac * (b_g-1.)

# Load data and covariance matrix
Nbins = 30
Pgv_ksz = {'NGC':{}}

spectra_dir = '/home/r/rbond/alague/scratch/ksz-pipeline/ksz-analysis/'
spectra_dir += 'quadratic_estimator/development_code/CMASS_spectra/'

caps = ['NGC']
freqs = ['f090', 'f150']

Npts = 10
N = Npts + 1

for cap in caps:
    for freq in freqs:  
        Pgv_ksz[cap][freq] = np.loadtxt(spectra_dir + freq + '/gv_recon_dipole_'+cap+'.dat')[1:N]
                
# k-binning
kedges = np.linspace(0, 0.2, 51)
k = np.array([(kedges[i+1]+kedges[i])/2 for i in range(len(kedges)-1)])[1:N]

# Correct covariance
cov_dir = spectra_dir + '../prepared_maps/'
cov = {'NGC':{}}
Nsims = 99
hartlap = (Nsims - Npts - 2) / (Nsims - 1)

# Load samples from the map + NGC QPM mocks (noise only)
for freq in freqs:
    cov["NGC"][freq] = np.cov(np.loadtxt(cov_dir+'null_test_act_maps_mock_galaxies/Pgv_ell_1_NGC_'+freq+'_array.dat')[1:Npts+1,:])

# Add signal part to covariance
# Load signal-only sims
sims_dir = '/home/r/rbond/alague/scratch/ksz-pipeline/ksz-analysis/'
sims_dir += 'quadratic_estimator/development_code/prepared_maps/Tgrid_tests/filter_only_cmb/'

Pgv_sims = {"NGC":{"f090":np.loadtxt(sims_dir+'Pgv_ell_1_NGC_f090_array.dat')[1:Npts+1,:],
                   "f150":np.loadtxt(sims_dir+'Pgv_ell_1_NGC_f150_array.dat')[1:Npts+1,:]}}

# Add covariance of signal (rescaled to bv~1)
# for cap in caps:
for freq in freqs:
    cov["NGC"][freq] += np.cov(Pgv_sims["NGC"]["f090"]/200)
    cov["NGC"][freq] = np.linalg.inv(hartlap * np.linalg.inv(cov["NGC"][freq]))


# Inverse variance combination
inv_v_cov = np.linalg.inv(np.linalg.inv(cov['NGC']['f090'])
                          + np.linalg.inv(cov['NGC']['f150']))
inv_v_comb = (np.dot(np.linalg.inv(cov['NGC']['f090']), Pgv_ksz['NGC']['f090'] ) 
              + np.dot(np.linalg.inv(cov['NGC']['f150']), Pgv_ksz['NGC']['f150'] ))
inv_v_comb = np.dot(inv_v_cov, inv_v_comb)

Pgv_data = -inv_v_comb
Pgv_cov = inv_v_cov

inv_Pgv_cov = np.linalg.inv(Pgv_cov)

# Define model and likelihood
import sys
sys.path.append(spectra_dir + '../')
from wide_angle_corrections import compute_wa_correction, compute_wa_correction_pypower


P = Plin(k_mu[0])

def log_prob(params):
    if len(params) > 1:
        bv, hundred_fnl = params
    else:
        hundred_fnl = params
        bv = 1.
    fnl = hundred_fnl * 100
    bg_fnl = b_g + fnl * (beta/alpha)
    Pgv_model_fnl =(bg_fnl + f*k_mu[1]**2) * faH/k_mu[0] * k_mu[1] * P
    model_gv_wa = abs(compute_wa_correction(ks, mus, Pgv_model_fnl, 1).real)
    model_gv_at_k_data = np.interp(k, ks, model_gv_wa)
    Delta = bv * model_gv_at_k_data - Pgv_data
    chi2 = np.dot(np.dot(Delta, inv_Pgv_cov), Delta)

    return -0.5 * chi2


# Run MCMC
nsteps, nwalkers, ndim = 2500, 4, 2 # ndim is 1 for keeping bv=1

if ndim > 1:
    start = np.random.normal([1, 0], [0.5, 4], size=(nwalkers,ndim))
else:
    start = np.random.normal([1], [0.5], size=(nwalkers,ndim))

cb0 = zeus.callbacks.AutocorrelationCallback(ncheck=100, dact=0.01, nact=50, discard=0.3)
cb1 = zeus.callbacks.SplitRCallback(ncheck=100, epsilon=0.02, nsplits=2, discard=0.3)
cb2 = zeus.callbacks.MinIterCallback(nmin=500)

sampler = zeus.EnsembleSampler(nwalkers, ndim, log_prob)
sampler.run_mcmc(start, nsteps, callbacks=[cb0, cb1, cb2])

chain = sampler.get_chain(flat=True)

np.savetxt("Pgv_test_chain_with_fnl.dat", chain)

