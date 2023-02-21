from nbodykit.lab import *
from nbodykit import setup_logging, style
import sys, os
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import interp1d

# Read command line arguments
Nmesh = 2**6 # 2^10 = 1024: this might require a lot of memory, (2^10)^3 * 6 * 16 bits ~ 100Gb! 
boxsize = [3500, 3500, 3500]

ZMIN = 0.43
ZMAX = 0.7

# adjust paths
catalog_dir = os.path.join('/', 'cluster', 'work', 'senatore', 'lssdata', 'nseries')

alpha, norm = [], []

for i in range(84):
    print (i)
    # NOTE: change this path if you downloaded the data somewhere else!
    data_path = os.path.join(catalog_dir, 'CutskyN%s.rdzw') % (i+1)
    # print (data_path)

    # initialize the catalog objects for data and randoms
    data_names = ['RA', 'DEC', 'Z', 'WEIGHT_FKP', 'WEIGHT']
    data = CSVCatalog(data_path, data_names)

    # slice the data
    valid = (data['Z'] > ZMIN)&(data['Z'] < ZMAX)
    data = data[valid]

    # print ('catalogs sliced and loaded')

    # the fiducial BOSS DR12 cosmology
    cosmo = cosmology.Cosmology(h=0.676).match(Omega0_m=0.31)
    # add Cartesian position column
    data['Position'] = transform.SkyToCartesian(data['RA'], data['DEC'], data['Z'], cosmo=cosmo)

    # add density from file
    zz, nz = np.loadtxt(os.path.join(catalog_dir, 'nbar_DR12v5_CMASS_North_om0p31_Pfkp10000.dat'), usecols=(0,3), unpack=True)
    data['NZ'] = interp1d(zz, nz, kind='cubic')(data['Z'])

    # print ('coordinates transformed to distances')

    # combine the data and randoms into a single catalog
    fkp = FKPCatalog(data, None, BoxPad=0.) #, randoms)

    # print ('FKP catalogs instanciated')
    
    mesh = fkp.to_mesh(Nmesh=Nmesh, BoxSize=boxsize, nbar='NZ', 
        fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT', resampler='tsc', interlaced=True) 

    # print ('mesh grid created')

    # compute the multipoles
    r = ConvolvedFFTPower(mesh, poles=[0], kmin=0.)

    # print ('multipole computed')

    # alpha
    alpha.append ( 1. * data.csize )
    norm.append ( 1. * r.attrs["data.norm"] )

randoms_path = os.path.join(catalog_dir, 'Nseries_cutsky_randoms_50x_redshifts.dat')
randoms_names = ['RA', 'DEC', 'Z', 'WEIGHT_FKP', 'WEIGHT']
randoms = CSVCatalog(randoms_path, randoms_names)

alpha = np.array(alpha) / randoms.csize
norm = np.array(norm)

print ('alpha = %s +/- %s' % (np.mean(alpha), np.std(alpha)))
print ('norm = %s +/- %s' % (np.mean(norm), np.std(norm)))





