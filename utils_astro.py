import numpy as np
import os, sys
from collections import OrderedDict
from astropy.table import Table, vstack
import astropy.table
import scipy.stats

# constants
logr21 = np.log10(0.65)
fidlogalpha = np.log10(4.35)
helium = np.log10(1.36)
logpi = np.log10(np.pi)
hiappfac = np.log10(0.06)

# read some tabular data
aco_data_dir = '~/python/aklpyutils/data_files/'
aco_sdtab = Table.read(aco_data_dir+'helpers_acosdtab.csv')
aco_sdx = np.array(aco_sdtab['sdtothresh_in_re'])
aco_sdy = np.array(aco_sdtab['aco_sdterm_integration'])
 
aco_ztab = Table.read(aco_data_dir+'helpers_acoztab.csv')
aco_zx = np.array(aco_ztab['z_at_re'])
aco_zy = np.array(aco_ztab['aco_zterm_integration'])

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Generic forms
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def broken_law(xval=np.nan,
               xmin=np.nan, xmax=np.nan, 
               minval=np.nan, maxval=np.nan):

    yval = \
           (xval <= xmax)*minval + \
           (xval > xmin)*(xval < xmax)*(xval - xmin)* \
           (maxval-minval)/(xmax - xmin) + \
           (xval > xmax)*maxval
        
    return(yval)

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Conversion routines
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

# useful for plotting secondary axes or other simple calculations

def sfrgas2tdep(x):
    return(-1.0*x)

def tdep2sfrgas(x):
    return(-1.*x)

def sfratom2tdep(x):
    return(-1.0*x)

def tdep2sfratom(x):
    return(-1.*x)

def sfrco2tdep(x):
    return(-1.0*x+fidlogalpha)

def tdep2sfrco(x):
    return(-1.*x-fidlogalpha)

def lco2mmol(x):
    return(x+fidlogalpha)

def mmol2lco(x):
    return(x-fidlogalpha)

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Metallicity and conversion factors
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def logaco_mass_corr(x, xmin=9.0, xmax=11.0):
    x0 = 10.0
    b = -8.31
    m = -0.434
    
    xfid = 10.6
    val_at_xfid = m*(xfid-x0)+b
    
    x = (x < xmin)*xmin + (x > xmax)*xmax + x*(x >= xmin)*(x < xmax)
    val_at_x = m*(x-x0)+b
    
    aco_corr = val_at_x - val_at_xfid
    return(aco_corr)

def logaco_bwl13(z_zsun=1.0, sd_mol_nominal=0.0, sd_other=100.
                 , fidaco=4.35, zc=0.4, sdthresh=100., sdpow=0.5, acomin=0.435
                 , iters=1, values_reavg=False):
    """Implement BWL13 alpha prescription. Optionally flip the
    values_at_re flag to indicate that the values are at Reff and the
    results for whole-galaxy integrals should be used instead of the
    formula itself. Tuning parameters are the product of metallicity
    and the cloud surface density to use in the exponential and the
    surface density threshold to turn on the transition to the
    starburst regime, as well as the fiducial and minimum allowed alphaCO.
    """

    # Calculate the metallicity term
    if values_reavg:
        zterm = np.interp(z_zsun,aco_zx,aco_zy)
    else:
        zterm = np.exp(zc/z_zsun)
        
    # Iterate on surface density
    current_sd = sd_mol_nominal + sd_other
    for ii in range(iters):  
        if values_reavg:
            sd_to_thresh = current_sd / sdthresh
            sdterm = np.interp(sd_to_thresh,aco_sdx,aco_sdy)
        else:
            sdterm = (current_sd > sdthresh)* \
                     (current_sd/sdthresh)**(-1.0*sdpow) + \
                     (current_sd <= sdthresh)*1.0

        aco_vec = fidaco*zterm*sdterm

        # Rescale the surface densities
        current_sd = (aco_vec/fidaco)*sd_mol_nominal + sd_other

    # Implement floor
    aco_vec = np.maximum(aco_vec, acomin)

    # combine the two terms
    return(np.log10(aco_vec))

def massmetal_chiang(mass, reff=1.0):
    
    # line
    a = 9.02
    b = 0.017
    
    # polynomial
    p0 = 8.647
    p1 = -0.718
    p2 = 0.682
    p3 = -0.133
    offset = -0.31

    # grad
    grad_per_reff = -0.1

    # at reff
    x = mass
    x = (x < 9.0)*9.0 + (x > 10.7)*10.7 + (x >= 9.0)*(x < 10.7)*x
    x = x - 8.0
    pred = (p0 + offset) + \
           (p1 * x) + \
           (p2 * x**2) + \
           (p3 * x**3) + \
           grad_per_reff*(reff - 1.0)
    
    return(pred)
    
def massmetal_amart(mass, reff=1.0):
    
    logoh_asym = 8.798
    mto = 8.901
    gamma = 0.64

    pred = logoh_asym - \
           np.log10(1.0+(10.**mto/10.**mass)**gamma)

    grad_per_reff = -0.1

    pred += grad_per_reff*(reff - 1.0)

    return(pred)

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Dust and the balmer decrement
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def cc89_axtoav(lam_microns,rv=3.1):
    """
    Cardelli, Clayton, and Mathis 89 optical/uv curve
    """
    x = 1./lam_microns
    y = x-1.82
    a_x = 1. + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + \
          0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b_x = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 + \
          - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    axtoav = a_x + b_x/rv
    return(axtoav)

def decrement_from_ebv(ebv, 
    intrinsic=2.86,knum=2.53, kdenom=3.61,model='screen'):
    """
    Calculate the change to an expected ratio from extinction.
    """
    
    if model == 'screen':
        factor = 10.**(-1.0*ebv/2.5*(knum-kdenom))
    elif model == 'mixture':
        factor = kdenom/knum* \
            (1.-10.**(-1.*ebv/2.5*knum))/ \
            (1.-10.**(-1.*ebv/2.5*kdenom))
    else:
        return(np.nan)

    return(factor*intrinsic)
