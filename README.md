# kSZQuEst

## CMB preparation scripts

These scripts can be used to produce filtered CMB maps and associated simulations.
Before running these scripts, copy `defaults.yaml` to `defaults_local.yaml` to reflect your local system.
(You should not edit `defaults.yaml` unless you are making changes to the way the scripts here work and
want that under git version control.)

Set `apply_gal_filter` to False if you do not want the galaxy box filter to be applied to the CMB map.
If this is true, the theory filter that is saved (for norm calculation) will also contain the galaxy
filter.

### Galaxy filter preparation: `prepare_gal_filter.py`

Run this script if you plan to use `apply_gal_filter` to move the galaxy filtering into the CMB
map (following Kendrick's suggestion).  This will save a `galaxy_ell_filter.txt` to the
output directory, which will be used in the `prepare_cmb.py` and `prepare_cmb_sims.py`
stages.

### Mask generation: `prepare_mask.py`

This converts a Planck 70% galactic mask (on GPC, or PLA) into CAR pixelization in Equatorial coordinates.

### CMB pre-processing: `prepare_cmb.py`

For each CMB single-frequency map, this will prepare a mask based on an RMS noise threshold and the mask
from the previous script. It will then produce spherical harmonics of the original map as well as
a beam-deconvolved filtered map, fit white noise levels, the filter to use in theory calculations and the power spectrum of the filtered map.
In the YAML config, you will likely want to specify `daynight` to use day+night maps, and also uncomment both 90 and 150 GHz. I didn't do this
because I didn't have the relevant ACT maps on disk.

### CMB simulations : `prepare_cmb_sims.py`

This is the same as the previous script, but loops over `Nsims` number of lensed CMB simulations, adding beam and white noise as fit previously.
These simulations are then saved in the same format as above.  You can edit `Nsims` in this script to be up to 1999.
This script is MPI-enabled, so you can wrap it in `mpirun` or `srun` to distribute `Nsims` over multiple jobs.