#!/usr/bin/env python

import glob
from os import path
from time import perf_counter as clock
import os
from astropy.io import fits
from scipy import ndimage
import numpy as np

import ppxf as ppxf_package
from ppxf import ppxf
import ppxf_util as util

from pdb import set_trace
#import der_snr
from der_snr import*
from scipy.signal import find_peaks, peak_widths
#f1 = open("coadd-0-66003-20200315-fibermap.txt","w")

def get_RMS_from_resolution_data(wave, resdata, fwhm=False):
    """
    Calculate the Root Mean Square (RMS) or Full Width at Half Maximum (FWHM) values from resolution data.

    Parameters:
    wave (array): Vector or array representing the wavelength values.
    resdata (2D array): 2D array of resolution data.
                        First dimension represents the spatial dimension,
                        second dimension represents the wavelength dimension.
    fwhm (bool, optional): Flag to calculate and return FWHM values instead of RMS values. Default is False.

    Returns:
    RMS (array): Root Mean Square values representing the line width dispersion for each wavelength point.
                (or)
    FWHM (array): Full Width at Half Maximum values representing the line profile width for each wavelength point.
    """
    N = np.shape(resdata)[1] #number of wavelength points
    peaks=[5] #centre of 11-sized line profile
    
    FWHM = np.zeros(N)
    RMS = np.zeros(N)
    for l in range(5,N-6): #loop through wavelength vector
        results_half = peak_widths(resdata[:,l], peaks, rel_height=0.5)
        dlam = (wave[l+6] - wave[l-5])/11 #mean delta_lambda around this wavelength pixel
        FWHM[l] = results_half[0]*dlam
        RMS[l] = FWHM[l]/(2*np.sqrt(2*np.log(2)))
    
    if fwhm:
        return FWHM
    else:
        return RMS

def read_data(input_file, index=None, invalid_pixel_noise=1.e10):
    """
    Read data from a FITS file and return the flux, noise, wavelength, and resolution data.

    Parameters:
    input_file (str): Path to the input FITS file.
    index (int or None, optional): Index of the data to extract. Default is None.
    invalid_pixel_noise (float, optional): Value assigned to invalid spaxels. Default is 1.e10.

    Returns:
    flux (2D array): Flux data extracted from the FITS file.
    nois (2D array): Noise data extracted from the FITS file.
    wave (1D array): Wavelength data extracted from the FITS file.
    resolution (2D array): Resolution data extracted from the FITS file.
    """
    with fits.open(input_file) as hdu:
        flux = hdu['B_FLUX'].data
        nois = hdu['B_IVAR'].data
        wave = hdu['B_WAVELENGTH'].data
        resolution = hdu['B_RESOLUTION'].data
        #print ("resolution b=", resolution)
        fibermap = hdu[1].data
        #print ("fibermap",fibermap[1])
        #f1.write("%s\n" % (fibermap))
        #print (np.sum(flux)/np.sum(np.sqrt(flux+nois)),sum(flux),sum(nois))
        #print(DER_SNR(flux))
        

    if index is not None:
        flux = flux[index, :]
        nois = nois[index, :]
        resolution = resolution[index, :]
    with np.errstate(divide='ignore'):
        nois = np.sqrt(1/nois)

    # Assign `invalid_pixel_noise` to invalid spaxels.
    invalid_pixel = np.logical_or.reduce((
        nois==0, flux<=0, ~np.isfinite(nois), ~np.isfinite(flux)))
    nois[invalid_pixel] = invalid_pixel_noise

    return flux, nois, wave, resolution

    
        
def valid_pixels(nois, invalid_pixel_noise=1.e10):
    """
    Find the indices of valid pixels based on the noise data.

    Parameters:
    nois (array): Array of noise values.
    invalid_pixel_noise (float, optional): Value assigned to invalid pixels. Default is 1.e10.

    Returns:
    indices (array): Array of indices corresponding to the valid pixels.
    """

    indices = np.arange(len(nois), dtype=int)

    return indices[nois!=invalid_pixel_noise]



def ppxf_desi_example(input_file, index):
    """
    The function ppxf_desi_example takes two parameters: input_file 
    (the path to the input FITS file containing the galaxy spectrum) 
    and index (the index of the data to extract from the FITS file). 
    It does not return any value; instead, it performs the pPXF fitting on the given galaxy spectrum.
    
    The function essentially performs the pPXF fitting procedure on a DESI galaxy spectrum,
    including preprocessing steps, logarithmic rebinning, and fitting with a template library.
    It incorporates several helper functions to handle data reading, preprocessing, 
    and calculations specific to the pPXF fitting process.
    
    Example function to perform pPXF fitting on a DESI galaxy spectrum.

    Parameters:
    input_file (str): Path to the input FITS file containing the galaxy spectrum.
    index (int): Index of the data to extract from the FITS file.

    Returns:
    None
    """
    ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))

    # Read a galaxy spectrum and define the wavelength range
    #
    """
    Reading and preprocessing the galaxy spectrum:

    1. The function calls the read_data function to read the flux, noise, wavelength, and resolution data from the FITS file for the given index.
    2. The function also calculates and assigns the wavelength range (waverange) based on the first and last elements of the wavelength array.
    3. The function calculates the Root Mean Square (RMS) of the resolution data (RMS_b_lin) using the get_RMS_from_resolution_data function.
    4. The variable full_res is set to the median value of RMS_b_lin. If full_res is less than or equal to 0.2, it is set to 0.6.
    5. The function prints the length, shape, and dimensions of the resolution and wavelength data for debugging purposes.
    6. The variable FWHM_gal is calculated as full_res * 2.355, assuming a Nyquist sampling.
    7. The redshift (z) is set to 0 (assumed to be close to the rest frame).
    8. The wavelength array (wave) and waverange are divided by (1 + z), FWHM_gal is also divided by (1 + z).
    """
    flux, nois, wave, resolution = read_data(input_file, index=index, invalid_pixel_noise=1.e10)
    waverange = wave[[0, -1]] # Shorthand notation.
    RMS_b_lin = get_RMS_from_resolution_data(wave, resolution)
    full_res = np.median(RMS_b_lin)
    if full_res <= 0.2:
        full_res = 0.6
    print ("resolution", len(resolution),np.shape(resolution),np.shape(wave))
    print ("1D resolution", full_res)
    #res01 = np.sqrt(np.mean(resolution**2.))
    #print ("resolution",res01)
    FWHM_gal = full_res * 2.355 # No idea of value here, assuming Nyquist sampling. # [\AA]
    #print ("resolution",np.sqrt(np.median(np.diff(wave))/2.))
    z = 0. # Seems close to rest-frame.
    wave /= (1 + z)
    waverange /= (1 + z)
    FWHM_gal /= (1 + z)
    
    """
    Logarithmic rebinning of the galaxy spectrum:

    1. The function calls the log_rebin function to perform logarithmic rebinning on the flux and noise data, 
    using the waverange and the specified velscale.
    2. The logarithmically rebinned flux data (lrflux) is then normalized by its median value.
    3. The function also logarithmically rebins the squared noise data (nois**2) and calculates the square root (lrnois), 
    which is also normalized by the median flux value.
    """
    lrflux, lrwave, velscale = util.log_rebin(waverange, flux)
    median_lrflux = np.median(lrflux)
    lrflux /= median_lrflux  # Normalize spectrum to avoid numerical issues
    lrnois, _     , _        = util.log_rebin(waverange, nois**2.) 
    lrnois = np.sqrt(lrnois)
    lrnois /= median_lrflux

    # Read the list of filenames from the IndoUS Stellar Template library.
    """
    Reading and preprocessing the template library:

    1. The function retrieves the list of filenames from the IndoUS Stellar Template library.
    2. The IndoUS template spectra are convolved with the quadratic difference between the DESI and IndoUS instrumental resolutions.
    3. The function uses the log_rebin function to logarithmically rebin the template spectra to 
    a velocity scale 2x smaller than the DESI galaxy spectrum.
    4. The rebinned templates are stored in the templates array.
    """
    indous = glob.glob('indous_lite/*.fits')
    indous.sort()
    FWHM_tem = 1.35  # I think this is the IndoUS constant resolution FWHM, in \AA.
    velscale_ratio = 2  # adopts 2x higher spectral sampling for templates than for galaxy

    # Extract the wavelength range and logarithmically rebin one spectrum
    # to a velocity scale 2x smaller than the DESI galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    #
    hdu = fits.open(indous[0])
    ssp_flux = hdu[0].data
    h2 = hdu[0].header
    ssp_waverange = h2['CRVAL1'] + np.array([0., h2['CDELT1']*(h2['NAXIS1'] - 1)])
    ssp_lrflux, ssp_lrwave, velscale_temp = util.log_rebin(
        ssp_waverange, ssp_flux, velscale=velscale/velscale_ratio)
    templates = np.empty((ssp_lrflux.size, len(indous)))

    # Convolve the whole IndoUS library of spectral templates
    # with the quadratic difference between the DESI and the
    # IndoUS instrumental resolution. Logarithmically rebin
    # and store each template as a column in the array TEMPLATES.

    # Quadratic sigma difference in pixels IndoUS --> DESI
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    #
    FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
    sigma = FWHM_dif/2.355/h2['CDELT1']  # Sigma difference in pixels

    if not np.isfinite(sigma): raise ValueError(
        'No, the templates must have better spectral resolution than the '
        'spectrum to be fit')

    for j, file in enumerate(indous):
        hdu = fits.open(file)
        ssp_flux = hdu[0].data
        ssp_flux = ndimage.gaussian_filter1d(ssp_flux, sigma)
        ssp_lrflux, ssp_lrwave, velscale_temp = util.log_rebin(
            ssp_waverange, ssp_flux, velscale=velscale/velscale_ratio)
        templates[:, j] = ssp_lrflux/np.median(ssp_lrflux)  # Normalizes templates

    # The galaxy and the template spectra do not have the same starting wavelength.
    # For this reason an extra velocity shift DV has to be applied to the template
    # to fit the galaxy spectrum. We remove this artificial shift by using the
    # keyword VSYST in the call to PPXF below, so that all velocities are
    # measured with respect to DV. This assume the redshift is negligible.
    # In the case of a high-redshift galaxy one should de-redshift its
    # wavelength to the rest frame before using the line below (see above).
    #
    """
    Velocity shift and redshift calculations:

    1. The function calculates the velocity shift (dv) between the starting wavelength 
    of the template spectra and the first wavelength element of the galaxy spectrum.
    2. The redshift (z) is set to a value specified earlier.
    3. The function determines the good pixels for fitting by calling the determine_goodpixels function
    with the logarithmically rebinned wavelength array (lrwave), 
    the wavelength range of the template spectra (ssp_waverange), and the redshift (z).
    4. The function also determines the valid pixels using the valid_pixels function with 
    the logarithmically rebinned noise data (lrnois) and a specified invalid pixel noise value.
    """
    c = 299792.458
    dv = (np.mean(ssp_lrwave[:velscale_ratio]) - lrwave[0])*c  # km/s

    z = redshift00 # Initial redshift estimate of the galaxy
    goodPixels = util.determine_goodpixels(lrwave, ssp_waverange, z)
    validPixels = valid_pixels(lrnois, invalid_pixel_noise=1.e8) # Notice lower value b/c of rebinning.
    goodPixels = np.intersect1d(goodPixels, validPixels)

    # Here the actual fit starts. The best fit is plotted on the screen.
    # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
    #
    """
    pPXF fitting:

    1. The function performs the pPXF fitting using the ppxf function.
    2. The fitting is done with the template library (templates), the logarithmically rebinned flux data (lrflux), 
    the logarithmically rebinned noise data (lrnois), the velocity scale (velscale), 
    a starting guess for velocity and sigma (start), the indices of good pixels (goodPixels), and other specified parameters.
    3. The function prints the formal errors (dV and dsigma) calculated from the pPXF fit.
    4. The fitting results, such as the best-fitting velocity (pp.sol[0]) and sigma (pp.sol[1]), are written to a file (f1).
    5. The elapsed time for the pPXF fitting is printed, and the signal-to-noise ratio (DER_SNR) of 
    the logarithmically rebinned flux data is also printed.
    """
    vel = c*np.log(1 + z)   # eq.(8) of Cappellari (2017)
    start = [vel, 200.]  # (km/s), starting guess for [V, sigma]
    t = clock()
    print(name00)
    if DER_SNR(lrflux) > 1. and redshift00 < 0.3:
        pp = ppxf(
            templates, lrflux, lrnois, velscale, start,
            goodpixels=goodPixels, plot=True, moments=2,
            degree=4, vsyst=dv, velscale_ratio=velscale_ratio,
            lam=np.exp(lrwave))
    
        print("Formal errors:")
        print("     dV    dsigma")
        print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))
    #print('Best-fitting redshift:', (z + 1)*(1 + pp.sol[0]/c) - 1)
        f1.write("%s,%s,%s,%s\n" % (wholeline00[i],pp.sol[1],pp.error[1],DER_SNR(lrflux)))
        print('Elapsed time in pPXF: %.2f s' % (clock() - t))
        print(DER_SNR(lrflux))
    # If the galaxy is at significant redshift z and the wavelength has been
    # de-redshifted with the three lines "z = 1.23..." near the beginning of
    # this procedure, the best-fitting redshift is now given by the following
    # commented line (equation 2 of Cappellari et al. 2009, ApJ, 704, L34;
    # http://adsabs.harvard.edu/abs/2009ApJ...704L..34C)
    #
    # print('Best-fitting redshift z:', (z + 1)*(1 + pp.sol[0]/c) - 1)

#------------------------------------------------------------------------------
x00=[];x01=[];x02=[];x03=[];x04=[];x05=[];x06=[];x07=[];x08=[];x09=[];x010=[];x011=[]
objid00=[];plate00=[];mjd00=[];fiberid00=[];veldisp00=[];veldispErr00=[];sigmaStars00=[];sigmaStarsErr00=[]
wholeline00=[]
f0 = open("/global/homes/k/ksaid/DESI_Work/Run/Good_Guys/fuji_FPT-fibermap-good-guys.csv","r")
f1 = open("/global/homes/k/ksaid/DESI_Work/Run/Good_Guys/fuji_FPT-fibermap-good-guys-results-reolution-matrix.txt","w")
f1.write('#fibermap_targetid,fibermap_i,ra,dec,targetid,id,healpix,survey,program,targetid.1,z,zerr,zwarn,spectype,subtype,deltachi2,healpix_id,targetid.2,target_ra,target_dec,obsconditions,release,brickid,brick_objid,fiberflux_ivar_g,fiberflux_ivar_r,fiberflux_ivar_z,morphtype,flux_g,flux_r,flux_z,flux_ivar_g,flux_ivar_r,flux_ivar_z,ebv,flux_w1,flux_w2,flux_ivar_w1,flux_ivar_w2,fiberflux_g,fiberflux_r,fiberflux_z,fibertotflux_g,fibertotflux_r,fibertotflux_z,sersic,coadd_numexp,coadd_exptime,coadd_numnight,coadd_numtile,healpix_id.1,objid,brickid.1,brickname,ra.1,dec.1,ppxf_sigma,ppxf_sigma_error,DER_SNR')
f1.write('\n')
for line in f0:
    if line[0]!='#':# and str(line.split(",")[4]) == '39628494279280105':
        wholeline00.append(str(line.split()[0]))
        x00.append(str(line.split(",")[4]))
        x02.append(float(line.split(",")[10]))
        x03.append(int(line.split(",")[1]))
        x04.append(str(line.split(",")[7]))
        x05.append(str(line.split(",")[8]))
        x06.append(int(line.split(",")[6]));x07.append(str(line.split(",")[2]));x08.append(str(line.split(",")[3]))
f0.close()
for i in range(10):
    #print ('spec-'+str(x01[i])+'-'+str(x02[i])+'-'+str(x03[i])+'.fits')
    name00 = x00[i]
    redshift00 = x02[i]
    index00 = x03[i]
    survey = x04[i]
    program = x05[i]
    healpix = x06[i]
    RA = x07[i]
    DEC = x08[i]
    print (survey,program,healpix)
    if __name__ == '__main__'  and os.path.isfile('/global/cfs/cdirs/desi/spectro/redux/fuji/healpix/'+survey+'/'+program+'/'+str(int(healpix/100))+'/'+str(healpix)+'/coadd-'+survey+'-'+program+'-'+str(healpix)+'.fits'):

        input_file = '/global/cfs/cdirs/desi/spectro/redux/fuji/healpix/'+survey+'/'+program+'/'+str(int(healpix/100))+'/'+str(healpix)+'/coadd-'+survey+'-'+program+'-'+str(healpix)+'.fits'
    #for i in range(68,500):
        index      = index00
        #print (i,i,i,i,i,i,i)
        ppxf_desi_example(input_file, index)

        import matplotlib.pyplot as plt
        plt.savefig('./plots/'+name00+'.pdf')
        plt.show()
        #plt.show(block=False)
        #plt.pause(1)
        #plt.close()
