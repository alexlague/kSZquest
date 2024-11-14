##
## Remove most massive galaxies from catalogs to
## avoid tSZ contamination
##

import numpy as np
from astropy.io import fits
from nbodykit.lab import *
import utils as kutils
args = kutils.jobargs
paths = kutils.paths

BOSS_dir = '/home/r/rbond/alague/scratch/ksz-pipeline/BOSS_data/CMASS/'
BOSS_file = 'pre-recon/cmass/galaxy_DR12v5_CMASS_North_t2.txt'
data = np.loadtxt(BOSS_dir+BOSS_file) # catalog of galxies with ra, dec, z as first columns

use_both_caps = False
if use_both_caps:
    BOSS_file = 'pre-recon/cmass/galaxy_DR12v5_CMASS_South_t2.txt'
    data = np.concatenate((data, np.loadtxt(BOSS_dir+BOSS_file)))

# boundaries of survey
max_ra = np.max(data[:,0])
min_ra = np.min(data[:,0])
max_dec = np.max(data[:,1]) 
min_dec = np.min(data[:,1])
min_z = 0.43
max_z = 0.7

ra, dec, z = data[:,:3].T
ind = (dec<21.61666) & (data[:,2]>min_z) & (data[:,2]<max_z)
ra = ra[ind]
dec = dec[ind]
z = z[ind]

# FROM
#https://www.sdss4.org/dr17/spectro/galaxy_portsmouth/#fitting

#mass_file = 'portsmouth_stellarmass_starforming_salp-DR12-boss.fits'
#mass_cat = FITSCatalog(BOSS_dir+mass_file)

#z_fits_array = np.array(mass_cat['Z'])
#dec_fits_array = np.array(mass_cat['DEC'])
#ra_fits_array = np.array(mass_cat['RA'])
#log_mass_array = np.array(mass_cat['LOGMASS'])

# NOW USING THE ACT CLUSTER CATALOG
mask_dir = "/home/r/rbond/alague/scratch/ksz-pipeline/act_data_and_sims/"
fits_image_filename =mask_dir+ "DR5_cluster-catalog_v1.1.fits"
with fits.open(fits_image_filename) as hdul:
    cluster_masses = hdul[1].data['M500c   ']
    cluster_ra = hdul[1].data['RADeg   ']
    cluster_dec = hdul[1].data['decDeg   ']

'''
def find_galaxy_from_mass_index(index):
    
    candidate_z = z_fits_array[index]
    candidate_dec = dec_fits_array[index]
    candidate_ra = ra_fits_array[index]

    ind_z = abs(data[:,2]-candidate_z) < 1e-4
    ind_dec = abs(data[:,1]-candidate_dec) < 1e-4
    ind_ra = abs(data[:,0]-candidate_ra) < 1e-4
    
    gal_ind = np.arange(len(data[:,0]))[ind_z*ind_dec*ind_ra]
    
    if candidate_z >= min_z and candidate_z <= max_z:
        if candidate_dec >= min_dec and candidate_dec <= max_dec:
            if candidate_ra >= min_ra and candidate_ra <= max_ra:
                assert len(gal_ind) < 2
    
    return gal_ind
'''
massive_galaxies_indexes = []
#massive_galaxies_ras = []
#massive_galaxies_decs = []
#massive_galaxies_zs = []
'''
for M in range(len(log_mass_array)):
    if log_mass_array[M] >= 12.1: #12.0: #12.1: #11.74: # Schaan + value
        gal = find_galaxy_from_mass_index(M)
        if len(gal) > 0:
            if data[:,1][gal[0]] < 21.61666:
                massive_galaxies_indexes.append(gal[0])
                massive_galaxies_ras.append(data[:,0][gal[0]])
                massive_galaxies_decs.append(data[:,1][gal[0]])
                massive_galaxies_zs.append(data[:,2][gal[0]])
'''

thresh = 4/60 # 4 arcmins
most_massive = cluster_masses*1e14 > 2.5e14
print(len(cluster_ra[most_massive]))

from scipy.spatial import KDTree
tree = KDTree(np.array([ra, dec]).T)

massive_galaxies_indexes = []
for x in tree.query_ball_point(np.array([cluster_ra[most_massive], cluster_dec[most_massive]]).T, thresh):
    if x != []:
        massive_galaxies_indexes.extend(x)

massive_galaxies_indexes = np.unique(massive_galaxies_indexes) # remove duplicates
print(len(massive_galaxies_indexes))

massive_galaxies_ras = ra[massive_galaxies_indexes]
massive_galaxies_decs = dec[massive_galaxies_indexes]
massive_galaxies_zs = z[massive_galaxies_indexes]

np.savetxt(paths.out_dir + "massive_clusters_to_remove_CMASS_NGC.dat", np.array([massive_galaxies_ras, massive_galaxies_decs, massive_galaxies_zs]))
