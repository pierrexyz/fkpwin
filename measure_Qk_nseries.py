from nbodykit.lab import *
from nbodykit import setup_logging, style
import sys, os
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import interp1d

# Read command line arguments
Nmesh = 2**4 # 2**10 # 2^10 = 1024: this might require a lot of memory, (2^10)^3 * 6 * 16 bits ~ 100Gb! 
n_wide = int(sys.argv[1]) # wide-angle expansion order
boxside = float(sys.argv[2]) # 100000, 35000, 10000, 3500
boxsize = [boxside, boxside, boxside]

ZMIN = 0.43
ZMAX = 0.7

# adjust paths
catalog_dir = os.path.join('/', 'cluster', 'work', 'senatore', 'lssdata', 'nseries')
out_dir = os.path.join('/', 'cluster', 'work', 'senatore', 'fkpwin', 'Qk_', 'Qk_Lbox%.0f') % boxside

# NOTE: change this path if you downloaded the data somewhere else!
data_path = os.path.join(catalog_dir, 'CutskyN1.rdzw') # I put box 1 here
randoms_path = os.path.join(catalog_dir, 'Nseries_cutsky_randoms_50x_redshifts.dat')
print (data_path)
print (randoms_path)

# initialize the catalog objects for data and randoms
data_names = ['RA', 'DEC', 'Z', 'WEIGHT_FKP', 'WEIGHT']
data = CSVCatalog(data_path, data_names)
randoms_names = ['RA', 'DEC', 'Z', 'WEIGHT_FKP', 'WEIGHT']
randoms = CSVCatalog(randoms_path, randoms_names)

# slice the data
valid = (data['Z'] > ZMIN)&(data['Z'] < ZMAX)
data = data[valid]

valid = (randoms['Z'] > ZMIN)&(randoms['Z'] < ZMAX)
randoms = randoms[valid]

print ('catalogs sliced and loaded')

# the fiducial BOSS DR12 cosmology
cosmo = cosmology.Cosmology(h=0.676).match(Omega0_m=0.31)
# add Cartesian position column
data['Position'] = transform.SkyToCartesian(data['RA'], data['DEC'], data['Z'], cosmo=cosmo)
randoms['Position'] = transform.SkyToCartesian(randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo)

# add density from file
zz, nz = np.loadtxt(os.path.join(catalog_dir, 'nbar_DR12v5_CMASS_North_om0p31_Pfkp10000.dat'), usecols=(0,3), unpack=True)
data['NZ'] = interp1d(zz, nz, kind='cubic')(data['Z'])
randoms['NZ'] = interp1d(zz, nz, kind='cubic')(randoms['Z'])

if n_wide > 0: # wide-angle expansion of order n: add a factor r_2^{-n} in the integral over d^3r_2 r_2^{-n} e^{-i k r_2} F(r_2) L_\ell(\hat k \cdot \hat r_2) 
    randoms['WEIGHT_FKP_2'] = randoms['WEIGHT_FKP'] * norm(randoms['Position'].compute(), axis=-1)**-n_wide 

print ('coordinates transformed to distances')

# combine the data and randoms into a single catalog
fkp = FKPCatalog(randoms, None, BoxPad=0.) #, randoms)

print ('FKP catalogs instanciated')

# dtype='c16' or 'c8' to get correct odd multipoles, see https://nbodykit.readthedocs.io/en/latest/api/_autosummary/nbodykit.algorithms.convpower.catalog.html#nbodykit.algorithms.convpower.catalog.FKPCatalog.to_mesh
mesh = fkp.to_mesh(Nmesh=Nmesh, BoxSize=boxsize, nbar='NZ', dtype='c8', # 'c8' for lower memory usage
    fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT', resampler='tsc', interlaced=True) 

if n_wide == 0: 
    mesh2 = None
elif n_wide > 0: 
    mesh2 = fkp.to_mesh(Nmesh=Nmesh, BoxSize=boxsize, nbar='NZ', dtype='c8', 
        fkp_weight='WEIGHT_FKP_2', comp_weight='WEIGHT', resampler='tsc', interlaced=True) 

print ('mesh grid created')

# compute the multipoles
r = ConvolvedFFTPower(mesh, poles=[0,1,2,3,4], second=mesh2, kmin=0.)

print ('multipole computed')

# alpha and norm
alpha = np.sum(data['WEIGHT'].compute()) / np.sum(randoms['WEIGHT'].compute())                     # ratio of weighted number of data objects over the number of randoms
norm = np.sum(data['NZ'].compute() * data['WEIGHT'].compute() * data['WEIGHT_FKP'].compute()**2)   # BOSS normalization to be compared with

# window
win = np.stack([r.poles['k'], r.poles['power_0'].real, r.poles['power_1'].imag, r.poles['power_2'].real, r.poles['power_3'].imag, r.poles['power_4'].real]).T

# saving file
if not os.path.exists(out_dir): os.makedirs(out_dir)
np.savetxt(os.path.join(out_dir, 'Qk_n%s_nseries.dat') % (n_wide), 
    win, fmt='%.6e', header='k, q0, q1/i, q2, q3/i, q4, norm, alpha = %.5e, %.5e' % (norm, alpha))

print ('file saved')


