from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
import re
import warnings

from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

from reproject import reproject_interp, reproject_exact

# ------------------------------------------------------------------------
# Make a clean new simple header to spec
# ------------------------------------------------------------------------

def make_simple_header(ra_ctr, dec_ctr, pix_scale,
                       extent_x = None, extent_y = None,
                       nx = None, ny = None, beam_deg = None):

    # Deal with center, skycoords, units, etc. better
    
    if nx is not None and ny is not None:
        extent_x = pix_scale * nx
        extent_y = pix_scale * ny
    elif extent_x is not None and extent_y is not None:
        nx = int(np.ceil(extent_x*0.5 / pix_scale) * 2 + 1)
        ny = int(np.ceil(extent_y*0.5 / pix_scale) * 2 + 1)

    new_hdr = fits.Header()
    new_hdr['NAXIS'] = 2
    new_hdr['NAXIS1'] = nx
    new_hdr['NAXIS2'] = ny
    
    new_hdr['CTYPE1'] = 'RA---SIN'
    new_hdr['CRVAL1'] = ra_ctr
    new_hdr['CRPIX1'] = np.float16((nx / 2) * 1 - 0.5)
    new_hdr['CDELT1'] = -1.0 * pix_scale
    
    new_hdr['CTYPE2'] = 'DEC--SIN'
    new_hdr['CRVAL2'] = dec_ctr
    new_hdr['CRPIX2'] = np.float16((ny / 2) * 1 - 0.5)
    new_hdr['CDELT2'] = 1.0 * pix_scale
    
    new_hdr['EQUINOX'] = 2000.0
    new_hdr['RADESYS'] = 'FK5'

    if beam_deg is not None:
        new_hdr['BMAJ'] = beam_deg
        new_hdr['BMIN'] = beam_deg
        new_hdr['BPA'] = 0.0

    return (new_hdr)
    
# ------------------------------------------------------------------------
# Reprojection
# ------------------------------------------------------------------------

def align_image(hdu_to_align, target_header, hdu_in=0,
                order='bilinear', missing_value=np.nan,
                use_exact=False, outfile=None, overwrite=True):
    """
    Aligns an image to a target header and handles reattaching the
    header to the file with updated WCS keywords.
    """

    # Run the reprojection
    if use_exact:
        reprojected_image, footprint = reproject_exact(
            hdu_to_align, target_header, hdu_in=hdu_in,
            order=order, return_footprint=True)
    else:
        reprojected_image, footprint = reproject_interp(
            hdu_to_align, target_header, hdu_in=hdu_in,
            order=order, return_footprint=True)

    # Blank missing locations outside the footprint    
    missing_mask = (footprint == 0)
    reprojected_image[missing_mask] = missing_value
    
    # Get the WCS of the target header
    target_wcs = WCS(target_header)
    target_wcs_keywords = target_wcs.to_header()
    
    # Get the WCS of the original header
    orig_header = hdu_to_align.header
    orig_wcs = WCS(orig_header)
    orig_wcs_keywords = orig_wcs.to_header()

    # Create a reprojected header using the target WCS but keeping
    # other keywords the same.
    reprojected_header = hdu_to_align.header
    for this_keyword in orig_wcs_keywords:
        if this_keyword in reprojected_header:
            del reprojected_header[this_keyword]

    for this_keyword in target_wcs_keywords:
        reprojected_header[this_keyword] = target_wcs_keywords[this_keyword]

    # Make a combined HDU merging the image and new header 
    reprojected_hdu = fits.PrimaryHDU(
        reprojected_image, reprojected_header)

    # Write or return
    if outfile is not None:
        reprojected_hdu.writeto(outfile, overwrite=overwrite)
    
    return(reprojected_hdu)

# ------------------------------------------------------------------------
# Create coordinate grids from headers
# ------------------------------------------------------------------------

def make_vaxis(header):
    """
    Docs forthcoming.
    """

    vpix = np.arange(naxis3)
    vdelt = vpix - (header['CRPIX3'] - 1)
    vaxis = vdelt * header['CDELT3'] + header['CRVAL3']

    return(vaxis)
        
def make_axes(header=None, wcs=None, naxis=None
              , ra_axis=None, dec_axis=None):
    """
    Docs forthcoming

    Python adapted from Jiayi Sun
    """

    # Figure out our approach
    use_hdr = False
    use_wcs = False
    use_axes = False

    # Deal with a common radio astrometry issue
    #if 'GLS' in wcs.wcs.ctype[0]:
    #    wcs.wcs.ctype[0] = wcs.wcs.ctype[0].replace('GLS', 'SFL')
    #    print(f"Replaced GLS with SFL; ctype[0] now = {wcs.wcs.ctype[0]}")
    
    if header is not None:

        # Extract WCS
        wcs_cel = WCS(header).celestial
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2']
        
    elif wcs is not None and naxis is not None:

        # Get the relevant data from the WCS
        wcs_cel = wcs.celestial
        naxis1, naxis2 = naxis
        
    elif ra_axis is not None and dec_axies is not None:
        
        if ra.ndim == 1:
            ra_deg, dec_deg = np.broadcast_arrays(ra, dec)
        else:
            ra_deg, dec_deg = ra, dec
            if verbose:
                print("ra ndim != 1")
        if hasattr(ra, 'unit'):
            ra_deg = ra.to(u.deg).value
            dec_deg = dec.to(u.deg).value
        
    else:
        # Throw error message
        return(None)

    # If we get here we have naxis and a WCS, proceed ...
        
    ix = np.arange(naxis1)
    iy = np.arange(naxis2).reshape(-1, 1)
    ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)
    
    return(ra_deg, dec_deg)

def axes_from_image(ra_deg, dec_deg):
    """
    Extract central row and column for use as axes.
    """
    
    naxis2, naxis1 = ra_deg.shape()
    
    # Extract the axes from the central row and column through the image
    raxis = ra_deg[naxis2 // 2, :]
    daxis = dec_deg[:, naxis1 // 2]

    return(raxis, daxis)

# ------------------------------------------------------------------------
# Clean up headers, stripping bad keywords from Jiayi Sun
# https://github.com/astrojysun/Sun_Astro_Tools/blob/master/sun_astro_tools/fits.py
# ------------------------------------------------------------------------

def clean_header(
        hdr, auto=None, remove_keys=[], keep_keys=[],
        simplify_scale_matrix=True):
    """
    Clean a FITS header and retain only the necessary keywords.
    Parameters
    ----------
    hdr : FITS header object
        Header to be cleaned
    auto : {'image', 'cube'}, optional
        'image' - retain only WCS keywords relevant to 2D images
        'cube' - retain only WCS keywords relevant to 3D cubes
    remove_keys : iterable, optional
        Keywords to manually remove
    keep_keys : iterable, optional
        Keywords to manually keep
    simplify_scale_matrix : bool, optional
        Whether to reduce CD or PC matrix if possible (default: True)
    Returns
    -------
    newhdr : FITS header object
        Cleaned header

    Author: Jiayi Sun
    Origin: https://github.com/astrojysun/Sun_Astro_Tools/blob/master/sun_astro_tools/fits.py
    """
    newhdr = hdr.copy()

    # remove keywords
    for key in remove_keys + [
            'OBSGEO-X', 'OBSGEO-Y', 'OBSGEO-Z', 'OBS-RA', 'OBS-DEC',
            'WCSAXES']:
        newhdr.remove(key, ignore_missing=True, remove_all=True)

    # make sure the number of NAXISi is consistent with WCSAXES
    newwcs = WCS(newhdr)
    if newwcs.pixel_shape is not None:
        naxis_missing = newwcs.pixel_n_dim - len(newwcs.pixel_shape)
        for i in range(naxis_missing):
            newhdr[f"NAXIS{len(newwcs.pixel_shape)+1+i}"] = 1

    # auto clean
    if auto == 'image':
        newwcs = WCS(newhdr).reorient_celestial_first().sub(2)
    elif auto == 'cube':
        newwcs = WCS(newhdr).reorient_celestial_first().sub(3)
    else:
        newwcs = WCS(newhdr).reorient_celestial_first()

    # simplify pixel scale matrix
    if simplify_scale_matrix:
        mask = ~np.eye(newwcs.pixel_scale_matrix.shape[0]).astype('?')
        if not any(newwcs.pixel_scale_matrix[mask]):
            cdelt = newwcs.pixel_scale_matrix.diagonal()
            del newwcs.wcs.pc
            del newwcs.wcs.cd
            newwcs.wcs.cdelt = cdelt
        else:
            warnings.warn(
                "WCS pixel scale matrix has non-zero "
                "off-diagonal elements. 'CDELT' keywords "
                "might not reflect actual pixel size.")

    # construct new header
    newhdr = newwcs.to_header()
    # insert mandatory keywords
    if newwcs.pixel_shape is not None:
        for i in range(newwcs.pixel_n_dim):
            newhdr.insert(
                i, ('NAXIS{}'.format(i+1), newwcs.pixel_shape[i]))
    newhdr.insert(0, ('NAXIS', newhdr['WCSAXES']))
    newhdr.remove('WCSAXES')
    for key in ['BITPIX', 'SIMPLE']:
        if key in hdr:
            newhdr.insert(0, (key, hdr[key]))
    # retain old keywords
    for key in keep_keys:
        if key not in hdr:
            continue
        newhdr[key] = hdr[key]

    return newhdr

# ------------------------------------------------------------------------
# Deproject
# ------------------------------------------------------------------------

def make_offset_image():
    print("TBD")
    return(None)

def deproject(center_coord=None, incl=0*u.deg, pa=0*u.deg,
              header=None, wcs=None, naxis=None, ra=None, dec=None,
              return_offset=False, verbose=False):

    """
    Calculate deprojected radii and projected angles in a disk.

    This function deals with projected images of astronomical objects
    with an intrinsic disk geometry. Given sky coordinates of the
    disk center, disk inclination and position angle, this function
    calculates deprojected radii and projected angles based on
    (1) a FITS header (`header`), or
    (2) a WCS object with specified axis sizes (`wcs` + `naxis`), or
    (3) RA and DEC coodinates (`ra` + `dec`).
    Both deprojected radii and projected angles are defined relative
    to the center in the inclined disk frame. For (1) and (2), the
    outputs are 2D images; for (3), the outputs are arrays with shapes
    matching the broadcasted shape of `ra` and `dec`.

    Parameters
    ----------
    center_coord : `~astropy.coordinates.SkyCoord` object or array-like
        Sky coordinates of the disk center
    incl : `~astropy.units.Quantity` object or number, optional
        Inclination angle of the disk (0 degree means face-on)
        Default is 0 degree.
    pa : `~astropy.units.Quantity` object or number, optional
        Position angle of the disk (red/receding side, North->East)
        Default is 0 degree.
    header : `~astropy.io.fits.Header` object, optional
        FITS header specifying the WCS and size of the output 2D maps
    wcs : `~astropy.wcs.WCS` object, optional
        WCS of the output 2D maps
    naxis : array-like (with two elements), optional
        Size of the output 2D maps
    ra : array-like, optional
        RA coordinate of the sky locations of interest
    dec : array-like, optional
        DEC coordinate of the sky locations of interest
    return_offset : bool, optional
        Whether to return the angular offset coordinates together with
        deprojected radii and angles. Default is to not return.

    Returns
    -------
    deprojected coordinates : list of arrays
        If `return_offset` is set to True, the returned arrays include
        deprojected radii, projected angles, as well as angular offset
        coordinates along East-West and North-South direction;
        otherwise only the former two arrays will be returned.

    Notes
    -----
    This is the Python version of an IDL function `deproject` included
    in the `cpropstoo` package. See URL below:
    https://github.com/akleroy/cpropstoo/blob/master/cubes/deproject.pro

    Python routine from Jiayi Sun.
    """

    if isinstance(center_coord, SkyCoord):
        x0_deg = center_coord.ra.degree
        y0_deg = center_coord.dec.degree
    else:
        x0_deg, y0_deg = center_coord
        if hasattr(x0_deg, 'unit'):
            x0_deg = x0_deg.to(u.deg).value
            y0_deg = y0_deg.to(u.deg).value
    if hasattr(incl, 'unit'):
        incl_deg = incl.to(u.deg).value
    else:
        incl_deg = incl
    if hasattr(pa, 'unit'):
        pa_deg = pa.to(u.deg).value
    else:
        pa_deg = pa

    if header is not None:
        wcs_cel = WCS(header).celestial
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2']
        # create ra and dec grids
        ix = np.arange(naxis1)
        iy = np.arange(naxis2).reshape(-1, 1)
        ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)
    elif (wcs is not None) and (naxis is not None):
        wcs_cel = wcs.celestial
        naxis1, naxis2 = naxis
        # create ra and dec grids
        ix = np.arange(naxis1)
        iy = np.arange(naxis2).reshape(-1, 1)
        ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)
    else:
        if ra.ndim == 1:
            ra_deg, dec_deg = np.broadcast_arrays(ra, dec)
        else:
            ra_deg, dec_deg = ra, dec
            if verbose:
                print("ra ndim != 1")
        if hasattr(ra, 'unit'):
            ra_deg = ra.to(u.deg).value
            dec_deg = dec.to(u.deg).value
    
    
    #else:
        #ra_deg, dec_deg = np.broadcast_arrays(ra, dec)
        #if hasattr(ra_deg, 'unit'):
            #ra_deg = ra_deg.to(u.deg).value
            #dec_deg = dec_deg.to(u.deg).value

    # recast the ra and dec arrays in term of the center coordinates
    # arrays are now in degrees from the center
    dx_deg = (ra_deg - x0_deg) * np.cos(np.deg2rad(y0_deg))
    dy_deg = dec_deg - y0_deg

    # rotation angle (rotate x-axis up to the major axis)
    rotangle = np.pi/2 - np.deg2rad(pa_deg)

    # create deprojected coordinate grids
    deprojdx_deg = (dx_deg * np.cos(rotangle) +
                    dy_deg * np.sin(rotangle))
    deprojdy_deg = (dy_deg * np.cos(rotangle) -
                    dx_deg * np.sin(rotangle))
    deprojdy_deg /= np.cos(np.deg2rad(incl_deg))

    # make map of deprojected distance from the center
    radius_deg = np.sqrt(deprojdx_deg**2 + deprojdy_deg**2)

    # make map of angle w.r.t. position angle
    projang_deg = np.rad2deg(np.arctan2(deprojdy_deg, deprojdx_deg))

    if return_offset:
        return radius_deg, projang_deg, deprojdx_deg , deprojdy_deg
    else:
        return radius_deg, projang_deg
