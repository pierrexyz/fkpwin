from nbodykit.lab import *
from nbodykit import setup_logging, style
import sys, os
import numpy as np
from numpy.linalg import norm

# Read command line arguments
Nmesh = 2**4 # 2^10 = 1024: this might require a lot of memory, (2^10)^3 * 4 (TSC) * 2 (aliasing) * 16 bits ~ 137Gb! 
n_wide = int(sys.argv[1]) # wide-angle expansion order
red = str(sys.argv[2]) # cmass or lowz
sky = str(sys.argv[3]) # ngc or sgc
boxside = float(sys.argv[4]) # 100000, 35000, 10000, 3500
boxsize = [boxside, boxside, boxside]

if 'lowz' in red:
    ZMIN = 0.2
    ZMAX = 0.43
elif 'cmass' in red or 'nseries' in red:
    ZMIN = 0.43
    ZMAX = 0.7

if 'ngc' in sky:
    skybin = 'NGC'
elif 'sgc' in sky:
    skybin = 'SGC'

box = '0001' # unused, we are measuring the window on the randoms

# adjust paths
catalog_dir = os.path.join('/', 'cluster', 'work', 'senatore', 'lssdata', 'patchy')
out_dir = os.path.join('/', 'cluster', 'work', 'senatore', 'fkpwin', 'Qk_', 'Qk_Lbox%.0f') % boxside

# NOTE: change this path if you downloaded the data somewhere else!
data_path = os.path.join(catalog_dir, sky, 'Patchy-Mocks-DR12%s-COMPSAM_V6C_%s.dat') % (skybin, box)
randoms_path = os.path.join(catalog_dir, 'Patchy-Mocks-Randoms-DR12%s-COMPSAM_V6C_x50.dat') % (skybin)

print (data_path)
print (randoms_path)

# initialize the catalog objects for data and randoms
data_names = ['RA', 'DEC', 'Z', 'MSTAR', 'NZ', 'BIAS', 'VETO_FLAG', 'FIBER_COLLISION']
data = CSVCatalog(data_path, data_names)
randoms_names = ['RA', 'DEC', 'Z', 'NZ', 'BIAS', 'VETO_FLAG', 'FIBER_COLLISION']
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

print ('coordinates transformed to distances')

# add completeness weight
data['WEIGHT'] = 1.0 * data['VETO_FLAG'] * data['FIBER_COLLISION']
randoms['WEIGHT'] = 1.0 * randoms['VETO_FLAG'] * randoms['FIBER_COLLISION']

# add FKP weights
P0 = 10000.
data['WEIGHT_FKP'] = 1.0 / (1. + P0 * data['NZ'])
randoms['WEIGHT_FKP'] = 1.0 / (1. + P0 * randoms['NZ'])

if n_wide > 0: # wide-angle expansion of order n: add a factor r_2^{-n} in the integral over d^3r_2 r_2^{-n} e^{-i k r_2} F(r_2) L_\ell(\hat k \cdot \hat r_2) 
    randoms['WEIGHT_FKP_2'] = randoms['WEIGHT_FKP'] * norm(randoms['Position'].compute(), axis=-1)**-n_wide 

# combine the data and randoms into a single catalog
fkp = FKPCatalog(randoms, None, BoxPad=0.) #, randoms)

print ('FKP catalogs instanciated')

# dtype='c16' or 'c8' to get correct odd multipoles, see https://nbodykit.readthedocs.io/en/latest/api/_autosummary/nbodykit.algorithms.convpower.catalog.html#nbodykit.algorithms.convpower.catalog.FKPCatalog.to_mesh
mesh = fkp.to_mesh(Nmesh=Nmesh, BoxSize=boxsize, nbar='NZ', dtype='c8', # 'c8' for lower memory usage
    fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT', compensated=True, resampler='tsc', interlaced=True) 

if n_wide == 0: 
    mesh2 = None
elif n_wide > 0: 
    mesh2 = fkp.to_mesh(Nmesh=Nmesh, BoxSize=boxsize, nbar='NZ', dtype='c8', 
        fkp_weight='WEIGHT_FKP_2', comp_weight='WEIGHT', compensated=True, resampler='tsc', interlaced=True) 

print ('mesh grid created')

# compute the multipoles
r = ConvolvedFFTPower(mesh, poles=[0,1,2,3,4], second=mesh2, kmin=0.)

print ('multipole computed')

# alpha, norm, and shot noise
n_d, w_d, wfkp_d = data['NZ'].compute(), data['WEIGHT'].compute(), data['WEIGHT_FKP'].compute() 
n_r, w_r, wfkp_r = randoms['NZ'].compute(), randoms['WEIGHT'].compute(), randoms['WEIGHT_FKP'].compute() # note that n_r already carries one power of alpha!!!
alpha = np.sum(w_d) / np.sum(w_r) 

def Iab(a, b, n, w, wfkp, alpha=1., beta=1.):
    return alpha * np.sum((beta * n)**(a-1.) * w * wfkp**b)

I22_d, I33_d, I12_d, I13_d = Iab(2, 2, n_d, w_d, wfkp_d), Iab(3, 3, n_d, w_d, wfkp_d), Iab(1, 2, n_d, w_d, wfkp_d), Iab(1, 3, n_d, w_d, wfkp_d), 
I22_r, I33_r, I12_r, I13_r = Iab(2, 2, n_r, w_r, wfkp_r, alpha=alpha), Iab(3, 3, n_r, w_r, wfkp_r, alpha=alpha), Iab(1, 2, n_r, w_r, wfkp_r, alpha=alpha), Iab(1, 3, n_r, w_r, wfkp_r, alpha=alpha) 

# window
win = np.stack([r.poles['k'], r.poles['power_0'].real, r.poles['power_1'].imag, r.poles['power_2'].real, r.poles['power_3'].imag, r.poles['power_4'].real]).T

# saving file
if not os.path.exists(out_dir): os.makedirs(out_dir)
np.savetxt(os.path.join(out_dir, 'Qk_n%s_patchy_%s_%s.dat') % (n_wide, red, sky), 
    win, fmt='%.6e', header='k, q0, q1/i, q2, q3/i, q4 | ' +
    'alpha, I22_d, I22_r, I33_d, I33_r, I12_d, I12_r, I13_d, I13_r = %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e' % (alpha, I22_d, I22_r, I33_d, I33_r, I12_d, I12_r, I13_d, I13_r)
) 

print ('file saved')


