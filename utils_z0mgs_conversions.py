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

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Galametz et al. 2013 Sigma_TIR
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def calc_sigtir(
        i24=None, i70=None, i100=None,
        i160=None, i250=None):
    """
    Implements Table 3 of Galametz+ 2013.
    """

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Setup
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    
    # Figure out what bands we have
    have24 = False
    have70 = False
    have100 = False
    have160 = False
    have250 = False    
    
    if i24 is not None:
        have24 = True
    if i70 is not None:
        have70 = True
    if i100 is not None:
        have100 = True
    if i160 is not None:
        have160 = True
    if i250 is not None:
        have250 = True

    # Constants
    c = 2.99792458e10
    pc = 3.0857e18
    nu24 = c/24.0e-4
    nu70 = c/70.0e-4
    nu100 = c/100.0e-4
    nu160 = c/160.0e-4
    nu250 = c/250.0e-4
    
    # MJy/sr -> W/kpc^2
    fac24 = nu24*1e-17*1e-7*4.0*np.pi*(pc*1e3)^2
    fac70 = nu70*1e-17*1e-7*4.0*np.pi*(pc*1e3)^2
    fac100 = nu100*1e-17*1e-7*4.0*np.pi*(pc*1e3)^2
    fac160 = nu160*1e-17*1e-7*4.0*np.pi*(pc*1e3)^2
    fac250 = nu250*1e-17*1e-7*4.0*np.pi*(pc*1e3)^2

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Monochromatic conversions
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

    # 24
    if have24==True and \
       have70==False and \
       have100==False and \
       have160==False and \
       have250==False:
        pass
    
    # 70
    if have24==False and \
       have70==True and \
       have100==False and \
       have160==False and \
       have250==False:
        pass
    
    # 100
    if have24==False and \
       have70==False and \
       have100==True and \
       have160==False and \
       have250==False:
        pass
    
    # 160
    if have24==False and \
       have70==False and \
       have100==False and \
       have160==True and \
       have250==False:
        pass
    
    # 250
    if have24==False and \
       have70==False and \
       have100==False and \
       have160==False and \
       have250==True:
        pass
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Dual-band conversions
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # 24 + 70
    if have24==True and \
       have70==True and \
       have100==False and \
       have160==False and \
       have250==False:

        s_tir = \
            (3.925)*i24*fac24 + \
            (1.551)*i70*fac70
        return(s_tir)

    # 24 + 100
    if have24==True and \
       have70==False and \
       have100==True and \
       have160==False and \
       have250==False:

        s_tir = \
            (2.421)*i24*fac24 + \
            (1.410)*i100*fac100
        return(s_tir)
        
    # 24 + 160    
    if have24==True and \
       have70==False and \
       have100==False and \
       have160==True and \
       have250==False:

        s_tir = \
            (3.854)*i24*fac24 + \
            (1.373)*i160*fac160
        return(s_tir)

    # 24 + 250
    if have24==True and \
       have70==False and \
       have100==False and \
       have160==False and \
       have250==True:

        s_tir = \
            (5.179)*i24*fac24 + \
            (3.196)*i250*fac250
        return(s_tir)

    # 70 + 100
    if have24==False and \
       have70==True and \
       have100==True and \
       have160==False and \
       have250==False:

        s_tir = \
            (0.458)*i70*fac70 + \
            (1.444)*i100*fac100
        return(s_tir)

    # 70 + 160
    if have24==False and \
       have70==True and \
       have100==False and \
       have160==True and \
       have250==False:

        s_tir = \
            (0.999)*i70*fac70 + \
            (1.226)*i160*fac160
        return(s_tir)

    # 70 + 250
    if have24==False and \
       have70==True and \
       have100==False and \
       have160==False and \
       have250==True:

        s_tir = \
            (1.306)*i70*fac70 + \
            (2.752)*i250*fac250
        return(s_tir)

    # 100 + 160
    if have24==False and \
       have70==False and \
       have100==True and \
       have160==True and \
       have250==False:

        s_tir = \
            (1.239)*i100*fac100 + \
            (0.620)*i160*fac160
        return(s_tir)

    # 100 + 250
    if have24==False and \
       have70==False and \
       have100==True and \
       have160==False and \
       have250==True:
        
        s_tir = \
            (1.403)*i100*fac100 + \
            (1.242)*i250*fac250
        return(s_tir)

    # 160 + 250
    if have24==False and \
       have70==False and \
       have100==False and \
       have160==True and \
       have250==True:
        
        s_tir = \
            (2.342)*i160*fac160 + \
            (-0.944)*i250*fac250
        return(s_tir)
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Three-band conversions
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    if have24==True and \
       have70==True and \
       have100==True and \
       have160==False and \
       have250==False:

        s_tir = \
            (2.162)*i24*fac24 + \
            (0.185)*i70*fac70 + \
            (1.319)*i100*fac100
        return(s_tir)

    if have24==False and \
       have70==True and \
       have100==True and \
       have160==True and \
       have250==False:

        s_tir = \
            (0.789)*i70*fac70 + \
            (0.387)*i100*fac100 + \
            (0.960)*i160*fac160
        return(s_tir)

    if have24==False and \
       have70==False and \
       have100==True and \
       have160==True and \
       have250==True:

        s_tir = \
            (1.363)*i100*fac100 + \
            (0.097)*i160*fac160 + \
            (1.090)*i250*fac250
        return(s_tir)
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Four-band conversions
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    if have24==True and \
       have70==True and \
       have100==True and \
       have160==True and \
       have250==False:

        s_tir = \
            (2.051)*i24*fac24 + \            
            (0.521)*i70*fac70 + \            
            (0.294)*i100*fac100 + \
            (0.934)*i160*fac160
        return(s_tir)

    if have24==True and \
       have70==True and \
       have100==True and \
       have160==False and \
       have250==True:

        s_tir = \
            (1.983)*i24*fac24 + \            
            (0.427)*i70*fac70 + \            
            (0.708)*i100*fac100 + \
            (1.561)*i250*fac250
        return(s_tir)

    if have24==True and \
       have70==True and \
       have100==False and \
       have160==True and \
       have250==True:

        s_tir = \
            (2.119)*i24*fac24 + \            
            (0.688)*i70*fac70 + \            
            (0.995)*i160*fac160 + \
            (0.354)*i250*fac250
        return(s_tir)

    if have24==True and \
       have70==False and \
       have100==True and \
       have160==True and \
       have250==True:

        s_tir = \
            (2.643)*i24*fac24 + \            
            (0.836)*i100*fac100 + \            
            (0.357)*i160*fac160 + \
            (0.791)*i250*fac250
        return(s_tir)

    if have24==False and \
       have70==True and \
       have100==True and \
       have160==True and \
       have250==True:

        s_tir = \
            (2.643)*i24*fac24 + \            
            (0.836)*i100*fac100 + \            
            (0.357)*i160*fac160 + \
            (0.791)*i250*fac250
        return(s_tir)
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Five-band conversions
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    if have24==True and \
       have70==True and \
       have100==True and \
       have160==True and \
       have250==True:

        s_tir = \
            (2.013)*i24*fac24 + \
            (0.508)*i70*fac70 + \                        
            (0.393)*i100*fac100 + \            
            (0.599)*i160*fac160 + \
            (0.680)*i250*fac250
        return(s_tir)
