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

    with fits.open(input_file) as hdu:
        flux = hdu['B_FLUX'].data
        nois = hdu['B_IVAR'].data
        wave = hdu['B_WAVELENGTH'].data
        resolution = hdu['B_RESOLUTION'].data
        #resolution_b = hdu[4].data
        #print ("resolution b=", resolution_b[10])
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

    indices = np.arange(len(nois), dtype=int)

    return indices[nois!=invalid_pixel_noise]



def ppxf_desi_example(input_file, index):

    ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))

    # Read a galaxy spectrum and define the wavelength range
    #
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
    FWHM_gal = full_res * 2.355

    z = 0. # Seems close to rest-frame.
    wave /= (1 + z)
    waverange /= (1 + z)
    FWHM_gal /= (1 + z)

    lrflux, lrwave, velscale = util.log_rebin(waverange, flux)
    median_lrflux = np.median(lrflux)
    lrflux /= median_lrflux  # Normalize spectrum to avoid numerical issues
    lrnois, _     , _        = util.log_rebin(waverange, nois**2.) 
    lrnois = np.sqrt(lrnois)
    lrnois /= median_lrflux

    # Read the list of filenames from the IndoUS Stellar Template library.
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
    c = 299792.458
    dv = (np.mean(ssp_lrwave[:velscale_ratio]) - lrwave[0])*c  # km/s

    z = redshift00 # Initial redshift estimate of the galaxy
    goodPixels = util.determine_goodpixels(lrwave, ssp_waverange, z)
    validPixels = valid_pixels(lrnois, invalid_pixel_noise=1.e8) # Notice lower value b/c of rebinning.
    goodPixels = np.intersect1d(goodPixels, validPixels)

    # Here the actual fit starts. The best fit is plotted on the screen.
    # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
    #
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
        signal = np.median(pp.galaxy)
        resid = pp.galaxy - pp.bestfit
        noise = np.std(resid)
        snr = signal / noise
        print("Formal errors:")
        print("     dV    dsigma")
        print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))
    #print('Best-fitting redshift:', (z + 1)*(1 + pp.sol[0]/c) - 1)
        f1.write("%s,%s,%s,%s,%s\n" % (wholeline00[i],pp.sol[1],pp.error[1],DER_SNR(lrflux),snr))
        print('Elapsed time in pPXF: %.2f s' % (clock() - t))
        print("SNR",DER_SNR(lrflux),snr)
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
f0 = open("/global/homes/k/ksaid/DESI_Work/Run/Good_Guys/fuji_FPT-fibermap_per_tile-good-guys.csv","r")
f1 = open("/global/homes/k/ksaid/DESI_Work/Run/Good_Guys/fuji_FPT-fibermap_per_tile-good-guys-results.txt","w")
f1.write('#fibermap_targetid,fibermap_i,ra,dec,targetid,id,tileid,petal,night,targetid.1,z,zerr,zwarn,spectype,subtype,deltachi2,cumultile_id,targetid.2,target_ra,target_dec,obsconditions,release,brickid,brick_objid,fiberflux_ivar_g,fiberflux_ivar_r,fiberflux_ivar_z,morphtype,flux_g,flux_r,flux_z,flux_ivar_g,flux_ivar_r,flux_ivar_z,ebv,flux_w1,flux_w2,flux_ivar_w1,flux_ivar_w2,fiberflux_g,fiberflux_r,fiberflux_z,fibertotflux_g,fibertotflux_r,fibertotflux_z,sersic,coadd_numexp,coadd_exptime,coadd_numnight,coadd_numtile,cumultile_id.1,objid,brickid.1,brickname,ra.1,dec.1,ppxf_sigma,ppxf_sigma_error,DER_SNR,snr')
f1.write('\n')
for line in f0:
    if line[0]!='#':# and 80604 < float(line.split()[4]) < 81068:
        wholeline00.append(str(line.split()[0]))
        x00.append(str(line.split(",")[4]))
        x02.append(float(line.split(",")[10]))
        x03.append(int(line.split(",")[1]))
        x04.append(str(line.split(",")[6]))
        x05.append(str(line.split(",")[8]))
        x06.append(str(line.split(",")[7]));x07.append(str(line.split(",")[2]));x08.append(str(line.split(",")[3]))
f0.close()
for i in range(len(x00)):
    #print ('spec-'+str(x01[i])+'-'+str(x02[i])+'-'+str(x03[i])+'.fits')
    name00 = x00[i]
    redshift00 = x02[i]
    index00 = x03[i]
    TILEID = x04[i]
    DATE = x05[i]
    PETAL = x06[i]
    RA = x07[i]
    DEC = x08[i]
    #print (survey,program,healpix)
    if __name__ == '__main__'  and os.path.isfile('/global/cfs/cdirs/desi/spectro/redux/fuji/tiles/cumulative/'+TILEID+'/'+DATE+'/coadd-'+PETAL+'-'+TILEID+'-thru'+DATE+'.fits'):

        input_file = '/global/cfs/cdirs/desi/spectro/redux/fuji/tiles/cumulative/'+TILEID+'/'+DATE+'/coadd-'+PETAL+'-'+TILEID+'-thru'+DATE+'.fits'
    #for i in range(68,500):
        index      = index00
        #print (i,i,i,i,i,i,i)
        ppxf_desi_example(input_file, index)

        #import matplotlib.pyplot as plt
        #plt.savefig('./plots/'+name00+'.pdf')
        #plt.show()
        #plt.show(block=False)
        #plt.pause(1)
        #plt.close()
