from nbodykit.lab import *
from nbodykit import setup_logging, style
import sys, os
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import interp1d

# Read command line arguments
Nmesh = 2**9 # 2**10 # 2^10 = 1024: this might require a lot of memory, (2^10)^3 * 6 * 16 bits ~ 100Gb! 
boxside = float(sys.argv[1]) # 100000, 35000, 10000, 3500
boxsize = [boxside, boxside, boxside]

# adjust paths
catalog_dir = os.path.join('/', 'cluster', 'work', 'senatore', 'lssdata', 'sphere')
out_dir = os.path.join('/', 'cluster', 'work', 'senatore', 'fkpwin', 'Qk_', 'Qk_Lbox%.0f') % boxside

# NOTE: change this path if you downloaded the data somewhere else!
randoms_path = os.path.join(catalog_dir, 'hitomi_sphere_R50_N8.txt')
print (randoms_path)

# initialize the catalog objects for data and randoms
randoms_names = ['X', 'Y', 'Z', 'NZ'] 
randoms = CSVCatalog(randoms_path, randoms_names) 

# add Cartesian position column
randoms['Position'] = transform.StackColumns(randoms['X'], randoms['Y'], randoms['Z'])

print ('coordinates transformed to distances')

# combine the data and randoms into a single catalog
fkp = FKPCatalog(randoms, None, BoxPad=0.) #, randoms)

print ('FKP catalogs instanciated')

# dtype='c16' or 'c8' to get correct odd multipoles, see https://nbodykit.readthedocs.io/en/latest/api/_autosummary/nbodykit.algorithms.convpower.catalog.html#nbodykit.algorithms.convpower.catalog.FKPCatalog.to_mesh
mesh = fkp.to_mesh(Nmesh=Nmesh, BoxSize=boxsize, nbar='NZ', dtype='c8', resampler='tsc', interlaced=True) 

print ('mesh grid created')

# compute the multipoles
r = ConvolvedFFTPower(mesh, poles=[0,2,4], kmin=0.)

print ('multipole computed')

# window
win = np.stack([r.poles['k'], r.poles['power_0'].real, r.poles['power_2'].real, r.poles['power_4'].real]).T

# saving file
if not os.path.exists(out_dir): os.makedirs(out_dir)
np.savetxt(os.path.join(out_dir, 'Qk_sphere.dat') , win, fmt='%.6e', header='k, q0, q2, q4 | ') 

print ('file saved')


