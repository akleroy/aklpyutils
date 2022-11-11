# Functions and numbers related to properties of nearby galaxies.

from astropy.table import Table, vstack
import numpy as np
import os, sys
import scipy.stats

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# GENERIC DEFINITION OF BROKEN POWER LAW
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def broken_law(xval=np.nan,
               xmin=np.nan, xmax=np.nan, 
               minval=np.nan, maxval=np.nan):

    yval = \
           (xval <= xmax)*minval + \
           (xval > xmin)*(xval < xmax)*(xval - xmin)* \
           (maxval-minval)/(xmax - xmin) + \
           (xval > xmax)*maxval
        
    return(yval)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# STAR FORMING MAIN SEQUENCE
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def broken_ms(xval=np.nan,
              mt=np.nan, a=np.nan, b=np.nan, c=np.nan):

    yval = \
       (xval <= mt)*(b*(xval-mt)+c) + \
       (xval > mt)*(a*(xval-mt)+c)
        
    return(yval)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# DEFINITIONS OF WAVELENGTHS AND CONSTANTS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Salpeter to Chabrier
s55_to_c03 = np.log10(0.58)

# Constants
sol = 2.99792458E10
sol_kms = sol/1E5
pc = 3.0857E18
lsun_3p4 = 1.83E18

# Wavelengths and SFR terms
nu_fuv = sol/(154.E-9*1E2)
nu_nuv = sol/(231.E-9*1E2)
nu_w1 = sol/(3.4E-6*1E2)
nu_w2 = sol/(4.5E-6*1E2)
nu_w3 = sol/(12.E-6*1E2)
nu_w4 = sol/(22.E-6*1E2)

# Fiducial coefficients to convert to SFR
coeff_fuv = float(-43.42)
coeff_nuv = float(-43.24)
coeff_fuv_to_nuv = -0.06

coeff_w3only = -42.70
coeff_w4only = -42.63

coeff_nuvw3 = -42.86
coeff_nuvw4 = -42.79

coeff_fuvw3 = -42.79
coeff_fuvw4 = -42.73

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# COLOR BASED SFR CORRECTIONS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

xmin_w4w1 = -0.85
xmax_w4w1 = -0.10
minval_w4w1 = -42.95
maxval_w4w1 = -42.63

xmin_nuvw1 = -2.85
xmax_nuvw1 = -1.32
minval_nuvw1 = -43.05
maxval_nuvw1 = -42.68

xmin_mtol = -11.0
xmax_mtol = -10.2
minval_mtol = 0.5
maxval_mtol = 0.2

def w4w1_corr(xval=np.nan):
    yval = broken_law(xval=xval,
                      xmin = xmin_w4w1, xmax=xmax_w4w1,
                      minval = minval_w4w1, maxval = maxval_w4w1,
                  )
    return(yval)

def nuvw1_corr(xval=np.nan):
    yval = broken_law(xval=xval,
                      xmin = xmin_nuvw1, xmax=xmax_nuvw1,
                      minval = minval_nuvw1, maxval = maxval_nuvw1,
                  )
    return(yval)

def sfrw1_mtol(xval=np.nan):
    yval = broken_law(xval=xval,
                      xmin = xmin_mtol, xmax=xmax_mtol,
                      minval = minval_mtol, maxval = maxval_mtol,
                  )
    return(yval)    

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# LUMINOSITY TO SFR
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def sfr_from_w4(lum_w4=None, w4w1_color=None):
    if w4w1_color is None:
        sfr = lum_to_sfr(
            lum=lum_w4, nu=nu_w4,
            logcoeff=coeff_w4only)
    else:
        this_coeff = w4w1_corr(xval=w4w1_color)
        sfr = lum_to_sfr(
            lum=lum_w4, nu=nu_w4,
            logcoeff=this_coeff)
    return(sfr)

def sfr_from_fuvw4(
        lum_fuv=None, lum_w4=None, 
        nuvw1_color=None):
    sfr_uv =  lum_to_sfr(
        lum=lum_fuv, nu=nu_fuv,
        logcoeff=coeff_fuv)
    if nuvw1_color is None:
        sfr_w4 = lum_to_sfr(
            lum=lum_w4, nu=nu_w4,
            logcoeff=coeff_fuvw4)
    else:
        this_coeff = nuvw1_corr(xval=nuvw1_color)
        sfr_w4 = lum_to_sfr(
            lum=lum_w4, nu=nu_w4,
            logcoeff=this_coeff)
    sfr = sfr_uv + sfr_w4
    return(sfr)

def sfr_from_nuvw4(
        lum_nuv=None, lum_w4=None, 
        nuvw1_color=None):
    sfr_uv =  lum_to_sfr(
        lum=lum_nuv, nu=nu_nuv,
        logcoeff=coeff_nuv)
    if nuvw1_color is None:
        sfr_w4 = lum_to_sfr(
            lum=lum_w4, nu=nu_w4,
            logcoeff=coeff_nuvw4)
    else:
        this_coeff = nuvw1_corr(xval=nuvw1_color)
        sfr_w4 = lum_to_sfr(
            lum=lum_w4, nu=nu_w4,
            logcoeff=this_coeff) + \
            coeff_fuv_to_nuv
    sfr = sfr_uv + sfr_w4
    return(sfr)

def lum_to_sfr(lum=None, nu=None, logcoeff=None):
    sfr = 10.**(logcoeff+np.log10(nu)+np.log10(lum))
    return(sfr)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# LUMINOSITY TO MSTAR
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def lum_to_mass(lum=None, sfrw1=None):

    nu=nu_w1

    if sfrw1 is None:
        mtol=0.4
    else:
        mtol = sfrw1_mtol(sfrw1)

    #+np.log10(nu)
    mass = np.log10(lum)-np.log10(lsun_3p4)+np.log10(mtol)

    return(mass)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# R25 to Reff
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#print("Slope: ", (maxval_rer25 - minval_rer25)/(xmax_rer25-xmin_rer25))

# Median in S4g
#r25_to_reff = 1./2.54

xmin_rer25 = 8.5
xmax_rer25 = 10.30
minval_rer25 = -0.25
maxval_rer25 = -0.575

def pred_rer25_mstar(
    mstar, 
    xmin=xmin_rer25, xmax=xmax_rer25, 
    minval=minval_rer25, maxval=maxval_rer25):
    """
    Predict the log ratio of re to r25 as a function of r25.
    """

    rer25 = broken_law(mstar, xmin=xmin, xmax=xmax,
                       minval=minval, maxval=maxval)
    return(rer25)
