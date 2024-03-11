# File for CMB manipulation functions

import numpy as np
from pixell import enmap, utils
from scipy.interpolate import interp1d
import camb

class CMBMap:
    
    def __init__(self, 
                 kSZMap, 
                 CosmoParams, 
                 LMAX,
                 minRA=0., 
                 maxRA=0., 
                 minDEC=0., 
                 maxDEC=0.,
                 NSIDE=None,
                 noise_lvl=None, 
                 theta_FWHM=None, 
                 beam=None):
        '''
        CMB Map storing data about microwave sky in
        pixell format
        
        Inputs:
        kSZMap: 2d numpy array of temperature fluctuations in K
        CosmoParams: dictionary of cosmological parameters with 6 LCDM params
        LMAX: maximum ell to at which to generate the primary CMB and noise maps
        min/max RA/DEC: corners of the map in degrees
        noise_lvl: noise of CMB instrument in muK arcmin (needed only if generating noise map)
        theta_FWHM: in arcmin (optional for noise computation)
        beam: beam of the CMB instrument (needed only if convolving with beam)
        '''
        
        
        self.CosmoParams = CosmoParams
        self.LMAX = LMAX
        
        self.minRA = minRA
        self.maxRA = maxRA
        self.minDEC = minDEC
        self.maxDEC = maxDEC

        if noise_lvl != None:
            self.noise_lvl = noise_lvl
        if theta_FWHM != None:
            self.theta_FWHM = theta_FWHM
        if beam != None:
            self.beam

        # transform numpy array to enmap
        if kSZMap.ndim == 2:
            self.Nmesh_x = kSZMap.shape[0]
            self.Nmesh_y = kSZMap.shape[1]
        
            shape, wcs = self.GenerateMapTemplate()
            kSZ_map_pixell = enmap.empty(shape, wcs)
            kSZ_map_pixell[:,:] = kSZMap
            self.kSZ_map = kSZ_map_pixell
        
            assert np.allclose(self.kSZ_map-kSZMap, 0)
        
        # if healpy map, specify NSIDE 
        else:
            assert NSIDE is not None
            self.kSZ_map = kSZMap
            self.NSIDE = NSIDE
 
        return
    
    def CalculateTheoryCls(self):
        '''
        Calculate the power spectrum of the CMB anisotropies
        given cosmological model assuming flat LCDM
        (monopole and dipole not removed)
        
        Outputs:
        self.ls: \ell
        self.cltt: C_\ell^{TT}
        '''
        #Set up a new set of parameters for CAMB
        # TODO: unpack cosmo params
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.5, 
                           ombh2=0.022, 
                           omch2=0.122, 
                           mnu=0.06, 
                           omk=0, 
                           tau=0.06)
        pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        pars.set_for_lmax(self.LMAX, lens_potential_accuracy=0);
        results = camb.get_results(pars)
        
        # Rescale to match definition for pixell
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
        totCL  = powers['total']
        ls     = np.arange(totCL.shape[0])
        cltt = totCL[:,0] / 1e12 / (ls*(ls+1)/2/np.pi)
        
        cltt = cltt[ls<=self.LMAX]
        ls = ls[ls<=self.LMAX]
        
        assert np.max(ls) == self.LMAX
        
        self.ls = ls
        self.cltt = cltt
        
        return
    
    def GenerateMapTemplate(self):
        '''
        Create shape and wcs needed to create pixell maps
        from numpy 2d arrays
        
        Outputs:
        shape: tuple
        wcs: pixell world coordinate system
        '''
        
        RA_range = self.maxRA - self.minRA # deg
        DEC_range = self.maxDEC - self.minDEC # deg
        
        if RA_range == 0. or DEC_range == 0.:
            raise Exception("Invalid angular min/max")
        
        center = (self.minRA + RA_range/2, self.minDEC + DEC_range/2)
        
        Nmesh_x, Nmesh_y = self.Nmesh_x, self.Nmesh_y
        
        # check resolution
        #assert np.isclose(RA_range/Nmesh_x, DEC_range/Nmesh_y) # check if not reversed
        
        res = RA_range/Nmesh_x
        
        shape, wcs = enmap.geometry(shape=(Nmesh_x, Nmesh_y), 
                                    res=np.deg2rad(res), 
                                    pos=np.deg2rad(center))
        return shape, wcs
    
    def GeneratePrimaryCMB(self):
        '''
        Create a realization of the primary CMB from the
        choice of cosmological parameters
        The shape of the map matches that of the input kSZ map

        Outputs:
        self.primary_cmb_map in units of K
        '''
        
        if hasattr(self, "cltt"):
            ls = self.ls
            cltt = self.cltt
        else:
            self.CalculateTheoryCls()
        
        shape, wcs = self.GenerateMapTemplate()
        
        ps = self.cltt
        ps[:2] = 0.
        primary_cmb = enmap.rand_map(shape, wcs, ps[None,None])
        
        self.primary_cmb_map = primary_cmb
        
        return
    
    def GenerateCMBNoise(self):
        '''
        '''
        if (hasattr(self, "noise_lvl") and hasattr(self, "theta_FWHM")):
        
            self.theta_FWHM *= np.pi / (60*180) # to radians
        
            ls = np.arange(self.LMAX+1)
        
            nltt = (self.noise_lvl* np.pi / (60*180))**2  
            nltt *= np.exp(ls*(ls+1) * self.theta_FWHM**2/8/np.log(2)) # muK
            nltt /= 1e12 # K
            nltt[np.isnan(nltt)] = 0.
            nltt[np.isinf(nltt)] = 0.
        
            self.nltt = nltt
            ps = nltt
            shape, wcs = self.GenerateMapTemplate()
            cmb_noise_map = enmap.rand_map(shape, wcs, ps[None,None])
            
            self.cmb_noise_map = cmb_noise_map
        
        else:
            raise Exception("Please specify noise parameters if generating noise map")
        
        return
    
    def CombineMaps(self):
        '''
        '''
        if not hasattr(self, "cltt"):
            self.CalculateTheoryCls()

        # Combine maps+cls computed so far
        self.combined_cmb_map = self.kSZ_map.copy()
        
        if hasattr(self, "cmb_noise_map") and hasattr(self, "primary_cmb_map"):
            self.cltotal = self.cltt + self.nltt
            self.combined_cmb_map += self.cmb_noise_map + self.primary_cmb_map
        elif hasattr(self, "primary_cmb_map"):
            self.cltotal = self.cltt
            self.combined_cmb_map += self.primary_cmb_map
        elif hasattr(self, "cmb_noise_map"):
            self.cltotal = self.nltt
            self.combined_cmb_map += self.cmb_noise_map
        
        return
        
    def FilterCMB(self):
        '''
        '''
        self.CombineMaps()
        
        # Convolve with beam
        if hasattr(self, "beam"):
            self.combined_cmb_map = enmap.smooth_gauss(self.combined_cmb_map, self.beam)
            
        ps = self.cltotal
        fl = 1./ps
        fl[np.isnan(fl)] = 0.
        fl[np.isinf(fl)] = 0.
        fl = fl / np.max(fl) # easier to filter if normalized
        
        imap = self.combined_cmb_map
        kmap = enmap.fft(imap, normalize="phys")

        modlmap = enmap.modlmap(imap.shape, imap.wcs)
        modlmap = (imap).modlmap()

        fl2d = interp1d(self.ls, fl, bounds_error=False, fill_value=0)(modlmap)
        kfiltered = kmap*fl2d
        filtered = enmap.ifft(kfiltered, normalize="phys").real
        
        self.filtered_cmb_map = filtered
        
        
        # integrand for noise computation
        one_over_cl_map = fl2d*kmap/kmap
        one_over_cl_map[np.isnan(one_over_cl_map)] = 0
        one_over_cl_map_ifft = enmap.ifft(one_over_cl_map, normalize="phys").real
        self.one_over_cl_map = fl2d * one_over_cl_map_ifft

        return
    
    @property
    def shape(self):
        return np.array(self.kSZ_map).shape

    def to_array(self):
        return np.array(self.kSZ_map)

    def PrepareRecon(self, AddPrimary=True, AddNoise=True, DoFilter=True):
        '''
        '''
        
        if AddPrimary:
            self.CalculateTheoryCls()
            self.GeneratePrimaryCMB()
        
        if AddNoise:
            self.GenerateCMBNoise()
        
        if DoFilter:
            self.FilterCMB()
            outmap = [self.filtered_cmb_map, self.one_over_cl_map]
        
        else:
            self.CombineMaps()
            outmap = self.combined_cmb_map

        return outmap
