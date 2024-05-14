import numpy as np

def calc_ew(xaxis, spec):
    deltax = np.abs(xaxis[1]-xaxis[0])
    tpeak = np.nanmax(spec)
    spec_sum = np.nansum(spec*deltax)
    ew = spec_sum/tpeak
    return(ew)

def calc_mom2(xaxis, spec):
    mom1 = np.nansum(xaxis*spec)/np.nansum(spec)
    offset = xaxis-mom1
    mom2 = np.sqrt(np.nansum(offset**2*spec)/np.nansum(spec))
    return(mom2)

def mom2_corr_fac(peak=4.0,edge_thresh=0.0):
    fid_x = np.arange(-10.0,10.0,0.01)
    fid_sig = 1.0
    fid_y = peak*np.exp(-1.0*(fid_x/fid_sig)**2/2.0)
    real_mom2 = calc_mom2(fid_x,fid_y)
    clipped_mom2 = calc_mom2(fid_x,fid_y*(fid_y >= edge_thresh))    
    return(clipped_mom2/real_mom2)

def ew_corr_fac(peak=4.0,edge_thresh=0.0):
    fid_x = np.arange(-10.0,10.0,0.01)
    fid_sig = 1.0
    fid_y = peak*np.exp(-1.0*(fid_x/fid_sig)**2/2.0)
    real_ew = calc_ew(fid_x,fid_y)
    clipped_ew = calc_ew(fid_x,fid_y*(fid_y >= edge_thresh))    
    return(clipped_ew/real_ew)
    

    


