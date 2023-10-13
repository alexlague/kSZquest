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
                 NSIDE,
                 minRA=0., 
                 maxRA=0., 
                 minDEC=0., 
                 maxDEC=0.,
                 noise_lvl=None, 
                 theta_FWHM=None, 
                 beam=None):
        '''
        noise in muK arcmin
        theta_FWHM in arcmin
        '''
        
        
        self.CosmoParams = CosmoParams
        self.LMAX = LMAX
        self.NSIDE = NSIDE
        
        self.minRA = minRA
        self.maxRA = maxRA
        self.minDEC = minDEC
        self.maxDEC = maxDEC
        
        self.Nmesh_x = kSZMap.shape[0]
        self.Nmesh_y = kSZMap.shape[1]
        
        if not (noise_lvl == None):
            self.noise_lvl = noise_lvl
        if not (noise_lvl == None):
            self.theta_FWHM = theta_FWHM

        shape, wcs = self.GenerateMapTemplate()
        kSZ_map_pixell = enmap.empty(shape, wcs)
        kSZ_map_pixell[:,:] = kSZMap
        self.kSZ_map = kSZ_map_pixell
        
        assert np.allclose(self.kSZ_map-kSZMap, 0)
        
        return
    
    def CalculateTheoryCls(self):
        '''
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
        RA_range = self.maxRA - self.minRA # deg
        DEC_range = self.maxDEC - self.minDEC # deg
        
        if RA_range == 0. or DEC_range == 0.:
            raise Exception("Invalid angular min/max")
        
        center = (self.minRA + RA_range/2, self.minDEC + DEC_range/2)
        
        Nmesh_x, Nmesh_y = self.Nmesh_x, self.Nmesh_y
        
        # check resolution
        assert RA_range/Nmesh_x == DEC_range/Nmesh_y # check if not reversed
        
        res = RA_range/Nmesh_x
        
        shape, wcs = enmap.geometry(shape=(Nmesh_x, Nmesh_y), 
                                    res=np.deg2rad(res), 
                                    pos=np.deg2rad(center))
        return shape, wcs
    
    def GeneratePrimaryCMB(self):
        '''
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
    
    def FilterCMB(self):
        '''
        '''
        if hasattr(self, "cltt"):
            ls = self.ls
            cltt = self.cltt
        else:
            self.CalculateTheoryCls()
            ls = self.ls
            cltt = self.cltt
        
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
        
        return
