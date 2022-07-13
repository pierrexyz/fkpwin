from nbodykit.lab import *
from nbodykit import setup_logging, style
import sys, os
import numpy as np

# catalog_dir = 'catalogs' 
catalog_dir = os.path.join('/', 'exports', 'pierre', 'lssmap', 'catalog')

# Read command line arguments
red = str(sys.argv[1]) # cmass or lowz
sky = str(sys.argv[2]) # ngc or sgc
boxside = float(sys.argv[3]) # 100000, 30000, 3500, 1000
boxsize = [boxside, boxside, boxside]

if 'lowz' in red:
    ZMIN = 0.2
    ZMAX = 0.43
elif 'cmass' in red:
    ZMIN = 0.43
    ZMAX = 0.7

if 'ngc' in sky:
    skybin = 'North'
elif 'sgc' in sky:
    skybin = 'South'

# NOTE: change this path if you downloaded the data somewhere else!
data_path = os.path.join(catalog_dir, 'galaxy_DR12v5_CMASSLOWZTOT_%s.fits') % (skybin)
randoms_path = os.path.join(catalog_dir, 'random0_DR12v5_CMASSLOWZTOT_%s.fits') % (skybin)

print (data_path)
print (randoms_path)

# initialize the FITS catalog objects for data and randoms
data = FITSCatalog(data_path)
randoms = FITSCatalog(randoms_path)

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

# add completeness weight: no systematic weights for randoms by construction
randoms['WEIGHT'] = 1.0
data['WEIGHT'] = data['WEIGHT_SYSTOT'] * (data['WEIGHT_NOZ'] + data['WEIGHT_CP'] - 1.0)

print ('coordinates transformed to distances')

# combine the data and randoms into a single catalog
fkp = FKPCatalog(randoms, None, BoxPad=0.) #, randoms)

print ('FKP catalogs instanciated')

mesh = fkp.to_mesh(Nmesh=512, BoxSize=boxsize, nbar='NZ', 
    fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT', 
                   resampler='pcs', interlaced=True) 

print ('mesh grid created')

# compute the multipoles
r = ConvolvedFFTPower(mesh, poles=[0,2,4], kmin=0.)

print ('multipole computed')

# alpha
alpha = 1.0 * data.csize / randoms.csize

# window
win = np.stack([r.poles['k'], r.poles['power_0'].real, r.poles['power_2'].real, r.poles['power_4'].real]).T

# np.savetxt(os.path.join('window_boss_%s_%s.dat') % (red, sky), 
np.savetxt(os.path.join('/', 'exports', 'pierre', 'lssmap', 'spec', 'window_boss_%s_%s.dat') % (red, sky), 
    win, fmt='%.6e', header='k, w0, w2, w4, norm, alpha = %s, %s' % (r.attrs["data.norm"], alpha))

# r.save(os.path.join('/', 'exports', 'pierre', 'lssmap', 'spec', "window_boss_%s_%s.json" % (red, sky)))

print ('file saved')


