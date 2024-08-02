from nbodykit.lab import *
from nbodykit import setup_logging, style
import sys, os
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import interp1d

# Read command line arguments
Nmesh = 2**9 # 2**10 # 2^10 = 1024: this might require a lot of memory, (2^10)^3 * 6 * 16 bits ~ 100Gb! 
n_w = int(sys.argv[1]) # n = 1 or 2: powers of weight in one of the F(k,n) = w^n e^{ikr} to get either the power spectrum F(k,1) F(-k,1) or the semistochastic contributions to the bispectrum F(k,1) F(-k,2)
nbox = int(sys.argv[2])
boxside = float(sys.argv[3]) # 3500
boxsize = [boxside, boxside, boxside]

ZMIN = 0.43
ZMAX = 0.7

# adjust paths
catalog_dir = os.path.join('/', 'cluster', 'work', 'senatore', 'lssdata', 'nseries')
out_dir = os.path.join('/', 'cluster', 'work', 'senatore', 'fkpwin', 'Pk_', 'nseries') 

# NOTE: change this path if you downloaded the data somewhere else!
data_path = os.path.join(catalog_dir, 'CutskyN%s.rdzw') % nbox
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

if n_w == 2: # FKP-weight squared in one of the F(k,n) = w^n e^{ikr} to get either the the semistochastic contributions to the bispectrum F(k,1) F(-k,2)
    data['WEIGHT_FKP_2'] = data['WEIGHT_FKP']**2
    randoms['WEIGHT_FKP_2'] = randoms['WEIGHT_FKP']**2

print ('coordinates transformed to distances')

# combine the data and randoms into a single catalog
fkp = FKPCatalog(data, randoms)

print ('FKP catalogs instanciated')

# dtype='c16' or 'c8' to get correct odd multipoles, see https://nbodykit.readthedocs.io/en/latest/api/_autosummary/nbodykit.algorithms.convpower.catalog.html#nbodykit.algorithms.convpower.catalog.FKPCatalog.to_mesh
mesh = fkp.to_mesh(Nmesh=Nmesh, BoxSize=boxsize, nbar='NZ', dtype='c8', # 'c8' for lower memory usage
    fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT', resampler='tsc', interlaced=True) 

if n_w == 1: 
    mesh2 = None
elif n_w == 2: 
    mesh2 = fkp.to_mesh(Nmesh=Nmesh, BoxSize=boxsize, nbar='NZ', dtype='c8', 
        fkp_weight='WEIGHT_FKP_2', comp_weight='WEIGHT', compensated=True, resampler='tsc', interlaced=True) 

print ('mesh grid created')

# compute the multipoles
r = ConvolvedFFTPower(mesh, poles=[0,1,2,3,4], second=mesh2)

print ('multipole computed')

# alpha, norm, and shot noise 
n_d, w_d, wfkp_d = data['NZ'].compute(), data['WEIGHT'].compute(), data['WEIGHT_FKP'].compute() 
n_r, w_r, wfkp_r = randoms['NZ'].compute(), randoms['WEIGHT'].compute(), randoms['WEIGHT_FKP'].compute() # note that n_r already carries one power of alpha!!!
alpha = np.sum(w_d) / np.sum(w_r) 

def Iab(a, b, n, w, wfkp, alpha=1., beta=1.):
    return alpha * np.sum((beta * n)**(a-1.) * w * wfkp**b)

I22_d, I33_d, I12_d, I13_d = Iab(2, 2, n_d, w_d, wfkp_d), Iab(3, 3, n_d, w_d, wfkp_d), Iab(1, 2, n_d, w_d, wfkp_d), Iab(1, 3, n_d, w_d, wfkp_d), 
I22_r, I33_r, I12_r, I13_r = Iab(2, 2, n_r, w_r, wfkp_r, alpha=alpha), Iab(3, 3, n_r, w_r, wfkp_r, alpha=alpha), Iab(1, 2, n_r, w_r, wfkp_r, alpha=alpha), Iab(1, 3, n_r, w_r, wfkp_r, alpha=alpha) 

# power spectrum
pk = np.stack([r.poles['k'], r.poles['power_0'].real, r.poles['power_1'].imag, r.poles['power_2'].real, r.poles['power_3'].imag, r.poles['power_4'].real]).T

# unormalizing the power spectrum
pk[:,1:] *= r.attrs['randoms.norm'] 

# saving file
if not os.path.exists(out_dir): os.makedirs(out_dir)
np.savetxt(os.path.join(out_dir, 'pk_nw%s_nseries_box%s.dat') % (n_w, nbox), 
    pk, fmt='%.6e', header='k, q0, q1/i, q2, q3/i, q4 | ' +
    'alpha, I22_d, I22_r, I33_d, I33_r, I12_d, I12_r, I13_d, I13_r = %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e' % (alpha, I22_d, I22_r, I33_d, I33_r, I12_d, I12_r, I13_d, I13_r)
) 

np.savetxt(os.path.join(out_dir, 'Iab_nseries_box%s.dat') % (nbox), 
    np.array([alpha, I22_d, I22_r, I33_d, I33_r, I12_d, I12_r, I13_d, I13_r]), fmt='%.6e', 
    header='alpha, I22_d, I22_r, I33_d, I33_r, I12_d, I12_r, I13_d, I13_r'
) 

print ('file saved')


