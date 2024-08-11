# Some code to do stuff associated with the DESI FP data.

import copy
import emcee
import numpy as np
import scipy as sp
import pandas as pd
from calc_kcor import *
from CosmoFunc import *
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import matplotlib.cm as cm
import matplotlib.ticker as tk
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from hyperfit.linfit import LinFit
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats, interpolate, optimize

omega_m = 0.31               # Matter density
deccel = 3.0*omega_m/2.0 - 1.0

# Simple function to truncate a colourmap to smaller ranges when plotting
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
	new_cmap = colors.LinearSegmentedColormap.from_list(
		'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
		cmap(np.linspace(minval, maxval, n)))
	return new_cmap

# Convenience function to return the weighted average, error on the weighted mean, and weighted variance for a set of values.
def weighted_avg_and_std(values, weights, axis=None):
    average = np.average(values, weights=weights, axis=axis)
    average_err = np.std(values)*np.sqrt(np.sum((weights/np.sum(weights))**2))
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return (average, average_err, np.sqrt(variance))

# Heliocentric to CMB frame conversion routine from Anthony Carr.
def perform_corr(z_inp, RA, Dec, corrtype="full", dipole="Planck"):

    """
    A function to perform heliocentric corrections on redshifts.
    Inputs:
        z: float, input heliocentric or CMB frame redshift
        RA: float, equatorial right ascension
        Dec: float, equatorial declination
        corrtype: string of the type of correction to be performed. Either 'full' (default), 'approx',
            '-full' or '-approx', respectively corresponding to the proper correction, low-z additive
            approximation, and the backwards corrections to go from z_CMB to z_helio.
        dipole: string specifying 'Planck', 'COBE' or 'Astropy' dipoles. 'Astropy' imports astropy (slow)
                and uses their co-ordinates and transforms and COBE dipole; different beyond first decimal 
                point.
    Outputs:
        z_CMB (if corrtype is 'full' or 'approx'): float
        z_helio (only if corrtype is '-full' or '-approx'): float

    Notes:
      Co-ords of North Galactic Pole (ICRS): RA = 192.729 ± 0.035 deg, Dec = 27.084 ± 0.023 deg (https://doi.org/10.1093/mnras/stw2772)
      Co-ords of Galactic Centre (ICRS): RA = 17h45m40.0409s, Dec = −29d00m28.118s (see above reference)
                                         RA = 266.41683708 deg, Dec = -29.00781056 deg
      Ascending node of the galactic plane = arccos(sin(Dec_GC)*cos(Dec_NGP)-cos(Dec_GC)*sin(Dec_NGP)*cos(RA_NGP-RA_GC))
                                           = 122.92828126730255 = l_0
      Transform CMB dipole from (l,b) to (RA,Dec):
          Dec = arcsin(sin(Dec_NGP)*sin(b)+cos(Dec_NGP)*cos(b)*cos(l_0-l))
              = -6.9895105228347 deg
          RA = RA_NGP + arctan((cos(b)*sin(l_0-l)) / (cos(Dec_NGP)*sin(b)-sin(Dec_NGP)*cos(b)*cos(l_0-l)))
             = 167.81671014708002 deg
      Astropy co-ordinates:
      RA_NGP_J2000 = 192.8594812065348, Dec_NGP_J2000 = 27.12825118085622, which are converted from B1950
      RA_NGP_B1950 = 192.25, Dec_NGP_B1950 = 27.4
      l_0_B1950 = 123
      l_0_J2000 = 122.9319185680026
    """

    import numpy as np

    v_Sun_Planck = 369.82  # +/- 0.11 km/s
    l_dipole_Planck = 264.021  # +/- 0.011 deg
    b_dipole_Planck = 48.253  # +/- 0.005 deg
    c = 299792.458  # km/s
    v_Sun_COBE = 371.0
    l_dipole_COBE = 264.14
    b_dipole_COBE = 48.26

    RA_Sun_Planck = 167.81671014708002  # deg
    Dec_Sun_Planck = -6.9895105228347
    RA_Sun_COBE = 167.88112630619747  # deg
    Dec_Sun_COBE = -7.024553155965497

    # RA_Sun_COBE = 168.01187366045565  # deg using Astropy values     # 168.0118667
    # Dec_Sun_COBE = -6.983037861854297  # # # -6.98303424

    if corrtype not in ["full", "approx", "-full", "-approx"]:
        print("Correction type unknown.")
        raise ValueError

    rad = np.pi / 180.0
    if dipole == "Planck":
        # Vincenty formula
        alpha = np.arctan2(
            np.hypot(
                np.cos(Dec_Sun_Planck * rad) * np.sin(np.fabs(RA - RA_Sun_Planck) * rad),
                np.cos(Dec * rad) * np.sin(Dec_Sun_Planck * rad)
                - np.sin(Dec * rad)
                * np.cos(Dec_Sun_Planck * rad)
                * np.cos(np.fabs(RA - RA_Sun_Planck) * rad),
            ),
            np.sin(Dec * rad) * np.sin(Dec_Sun_Planck * rad)
            + np.cos(Dec * rad)
            * np.cos(Dec_Sun_Planck * rad)
            * np.cos(np.fabs((RA - RA_Sun_Planck)) * rad),
        )
    elif dipole == "COBE":
        alpha = np.arctan2(
            np.hypot(
                np.cos(Dec_Sun_COBE * rad) * np.sin(np.fabs(RA - RA_Sun_COBE) * rad),
                np.cos(Dec * rad) * np.sin(Dec_Sun_COBE * rad)
                - np.sin(Dec * rad)
                * np.cos(Dec_Sun_COBE * rad)
                * np.cos(np.fabs(RA - RA_Sun_COBE) * rad),
            ),
            np.sin(Dec * rad) * np.sin(Dec_Sun_COBE * rad)
            + np.cos(Dec * rad)
            * np.cos(Dec_Sun_COBE * rad)
            * np.cos(np.fabs((RA - RA_Sun_COBE)) * rad),
        )
    elif dipole == "astropy":
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        coord_helio_COBE = SkyCoord(
            l_dipole_COBE * u.degree, b_dipole_COBE * u.degree, frame="galactic"
        ).galactic
        coord = SkyCoord(RA * u.degree, Dec * u.degree, frame="icrs").galactic
        alpha = coord_helio_COBE.separation(coord).radian
    if dipole == "Planck":
        v_Sun_proj = v_Sun_Planck * np.cos(alpha)
    elif dipole == "COBE" or dipole == "astropy":
        v_Sun_proj = v_Sun_COBE * np.cos(alpha)

    # z_Sun = -v_Sun_proj / c
    # Full special rel. correction since it is a peculiar vel
    #z_Sun = np.sqrt((1.0 + (-v_Sun_proj) / c) / (1.0 - (-v_Sun_proj) / c)) - 1.0
    z_Sun = -v_Sun_proj/c

    min_z = 0.0
    if corrtype == "full":
    	z_CMB = np.where(z_inp <= min_z, z_inp, (1 + z_inp) / (1 + z_Sun) - 1)
    elif (corrtype == "-full"):
   	 	# backwards correction where z_CMB is actually z_helio and vice versa
        z_helio = np.where(z_inp <= min_z, z_inp, (1 + z_inp) * (1 + z_Sun) - 1)
    elif corrtype == "approx":
    	z_CMB = np.where(z_inp <= min_z, z_inp, z_inp - z_Sun)
    elif corrtype == "-approx":
    	z_helio = np.where(z_inp <= min_z, z_inp,  z_inp + z_Sun)

    if corrtype[0] == "-":
        return z_helio
    else:
        return z_CMB

# Generate a set of redshift-distance relation splines
def rz_table(redmax = 1.0, nlookbins=400, om=0.31):

	print(om)

	# Generate a redshift-distance lookup table. Always handy!
	red = np.empty(nlookbins)
	ez = np.empty(nlookbins)
	dist = np.empty(nlookbins)
	for i in range(nlookbins):
	    red[i] = i*redmax/nlookbins
	    ez[i] = Ez(red[i], om, 1.0-om, 0.0, -1.0, 0.0, 0.0)
	    dist[i] = DistDc(red[i], om, 1.0-om, 0.0, 100.0, -1.0, 0.0, 0.0)
	red_spline = sp.interpolate.splrep(dist, red, s=0)
	lumred_spline = sp.interpolate.splrep((1.0+red)*dist, red, s=0)
	dist_spline = sp.interpolate.splrep(red, dist, s=0)
	lumdist_spline = sp.interpolate.splrep(red, (1.0+red)*dist, s=0)
	ez_spline = sp.interpolate.splrep(red, ez, s=0)

	return red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline

# Function to compute the log-likelihood of a set of FP parameters given some data. 
def FP_func(params, logdists, z_obs, r, s, i, err_r, err_s, err_i, Sn, smin, sumgals=True, chi_squared_only=False, freek=False):

	if freek:
		a, b, k, rmean, smean, imean, sigma1, sigma2, sigma3 = params
	else:
		a, b, rmean, smean, imean, sigma1, sigma2, sigma3 = params
		k = 0.0

	fac1, fac2, fac3, fac4 = k*a**2 + k*b**2 - a, k*a - 1.0 - b**2, b*(k+a), 1.0 - k*a
	norm1, norm2 = 1.0+a**2+b**2, 1.0+b**2+k**2*(a**2+b**2)-2.0*a*k
	dsigma31, dsigma23 = sigma3**2-sigma1**2, sigma2**2-sigma3**3
	sigmar2 =  1.0/norm1*sigma1**2 +      b**2/norm2*sigma2**2 + fac1**2/(norm1*norm2)*sigma3**2
	sigmas2 = a**2/norm1*sigma1**2 + k**2*b**2/norm2*sigma2**2 + fac2**2/(norm1*norm2)*sigma3**2
	sigmai2 = b**2/norm1*sigma1**2 +   fac4**2/norm2*sigma2**2 + fac3**2/(norm1*norm2)*sigma3**2
	sigmars =  -a/norm1*sigma1**2 -   k*b**2/norm2*sigma2**2 + fac1*fac2/(norm1*norm2)*sigma3**2
	sigmari =  -b/norm1*sigma1**2 +   b*fac4/norm2*sigma2**2 + fac1*fac3/(norm1*norm2)*sigma3**2
	sigmasi = a*b/norm1*sigma1**2 - k*b*fac4/norm2*sigma2**2 + fac2*fac3/(norm1*norm2)*sigma3**2

	sigma_cov = np.array([[sigmar2, sigmars, sigmari], [sigmars, sigmas2, sigmasi], [sigmari, sigmasi, sigmai2]])

	# Compute the chi-squared and determinant (quickly!)
	cov_r = err_r**2 + np.log10(1.0 + 300.0/(LightSpeed*z_obs))**2 + sigmar2
	cov_s = err_s**2 + sigmas2
	cov_i = err_i**2 + sigmai2
	cov_ri = -1.0*err_r*err_i + sigmari

	A = cov_s*cov_i - sigmasi**2
	B = sigmasi*cov_ri - sigmars*cov_i
	C = sigmars*sigmasi - cov_s*cov_ri
	E = cov_r*cov_i - cov_ri**2
	F = sigmars*cov_ri - cov_r*sigmasi
	I = cov_r*cov_s - sigmars**2	

	sdiff, idiff = s - smean, i - imean
	rnew = r - np.tile(logdists, (len(r), 1)).T
	rdiff = rnew - rmean

	det = cov_r*A + sigmars*B + cov_ri*C
	log_det = np.log(det)/Sn

	chi_squared = (A*rdiff**2 + E*sdiff**2 + I*idiff**2 + 2.0*rdiff*(B*sdiff + C*idiff) + 2.0*F*sdiff*idiff)/(det*Sn)

	# Compute the FN term for the Scut only
	delta = (A*F**2 + I*B**2 - 2.0*B*C*F)/det
	FN = np.log(0.5 * special.erfc(np.sqrt(E/(2.0*(det+delta)))*(smin-smean)))/Sn

	if chi_squared_only:
		return chi_squared
	elif sumgals:
		return 0.5 * np.sum(chi_squared + log_det + 2.0*FN)
	else:
		return 0.5 * (chi_squared + log_det)

# Function to compute the FN correction for a set of galaxies
def FN_func(FPparams, z_obs, err_r, err_s, err_i, lmin, lmax, smin, freek=False):

	if freek:
		a, b, k, rmean, smean, imean, sigma1, sigma2, sigma3 = FPparams
	else:
		a, b, rmean, smean, imean, sigma1, sigma2, sigma3 = FPparams
		k = 0.0

	fac1, fac2, fac3, fac4 = k*a**2 + k*b**2 - a, k*a - 1.0 - b**2, b*(k+a), 1.0 - k*a
	norm1, norm2 = 1.0+a**2+b**2, 1.0+b**2+k**2*(a**2+b**2)-2.0*a*k
	dsigma31, dsigma23 = sigma3**2-sigma1**2, sigma2**2-sigma3**3
	sigmar2 =  1.0/norm1*sigma1**2 +      b**2/norm2*sigma2**2 + fac1**2/(norm1*norm2)*sigma3**2
	sigmas2 = a**2/norm1*sigma1**2 + k**2*b**2/norm2*sigma2**2 + fac2**2/(norm1*norm2)*sigma3**2
	sigmai2 = b**2/norm1*sigma1**2 +   fac4**2/norm2*sigma2**2 + fac3**2/(norm1*norm2)*sigma3**2
	sigmars =  -a/norm1*sigma1**2 -   k*b**2/norm2*sigma2**2 + fac1*fac2/(norm1*norm2)*sigma3**2
	sigmari =  -b/norm1*sigma1**2 +   b*fac4/norm2*sigma2**2 + fac1*fac3/(norm1*norm2)*sigma3**2
	sigmasi = a*b/norm1*sigma1**2 - k*b*fac4/norm2*sigma2**2 + fac2*fac3/(norm1*norm2)*sigma3**2

	cov_r = err_r**2 + np.log10(1.0 + 300.0/(LightSpeed*z_obs))**2 + sigmar2
	cov_s = err_s**2 + sigmas2
	cov_i = err_i**2 + sigmai2
	cov_ri = -1.0*err_r*err_i + sigmari

	A = cov_s*cov_i - sigmasi**2
	B = sigmasi*cov_ri - sigmars*cov_i
	C = sigmars*sigmasi - cov_s*cov_ri
	E = cov_r*cov_i - cov_ri**2
	F = sigmars*cov_ri - cov_r*sigmasi
	I = cov_r*cov_s - sigmars**2	

	# Inverse of the determinant!!
	detinv = 1.0/(cov_r*A + sigmars*B + cov_ri*C)

	# Compute all the G, H and R terms
	G = np.sqrt(E)/(2*F-B)*(C*(2*F+B) - A*F - 2.0*B*I)
	delta = (I*B**2 + A*F**2 - 2.0*B*C*F)*detinv**2
	Edet = E*detinv
	Gdet = (G*detinv)**2
	Rmin = (lmin - rmean - imean/2.0)*np.sqrt(2.0*delta/detinv)/(2.0*F-B)
	Rmax = (lmax - rmean - imean/2.0)*np.sqrt(2.0*delta/detinv)/(2.0*F-B)

	G0 = -np.sqrt(2.0/(1.0+Gdet))*Rmax
	G2 = -np.sqrt(2.0/(1.0+Gdet))*Rmin
	G1 = -np.sqrt(Edet/(1.0+delta))*(smin - smean)

	H = np.sqrt(1.0+Gdet+delta)
	H0 = G*detinv*np.sqrt(delta) - np.sqrt(Edet/2.0)*(1.0+Gdet)*(smin - smean)/Rmax
	H2 = G*detinv*np.sqrt(delta) - np.sqrt(Edet/2.0)*(1.0+Gdet)*(smin - smean)/Rmin
	H1 = G*detinv*np.sqrt(delta) - np.sqrt(2.0/Edet)*(1.0+delta)*Rmax/(smin - smean)
	H3 = G*detinv*np.sqrt(delta) - np.sqrt(2.0/Edet)*(1.0+delta)*Rmin/(smin - smean)

	FN = special.owens_t(G0, H0/H)+special.owens_t(G1, H1/H)-special.owens_t(G2, H2/H)-special.owens_t(G1, H3/H)
	FN += 1.0/(2.0*np.pi)*(np.arctan2(H2,H)+np.arctan2(H3,H)-np.arctan2(H0,H)-np.arctan2(H1,H))
	FN += 1.0/4.0*(special.erf(G0/np.sqrt(2.0))-special.erf(G2/np.sqrt(2.0)))

	# This can go less than zero for very large distances if there are rounding errors, so set a floor
	# This shouldn't affect the measured logdistance ratios as these distances were already very low probability!
	index = np.where(FN < 1.0e-15)
	FN[index] = 1.0e-15

	return np.log(FN)

# Routine to read in the DESI FP data and fit the fundamental plane and recover log-distance ratios
def fit_logdist(outfile, zmin=0.0033, zmax=0.1, veldispmin=50.0, veldispmax=420.0, serrcut=1.0, mag_low=10.0, mag_high=18.0, evo_corr=0.85, spiralsinFP=False):

	# Get some redshift-distance lookup tables
	red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()

	# The magnitude limit and velocity dispersion limits (very important!)
	smin = np.log10(veldispmin)
	smax = np.log10(veldispmax)

	# Read in the DESI FP data
	data = pd.read_csv("./fuji_full_fp_sample_v03.csv")
	print(data.keys())
	data["zcmb"] = perform_corr(data["z_x"].to_numpy(), data["ra_1"].to_numpy(), data["dec_1"].to_numpy())
	data["zcmb_group"] = data["zcmb"]             # No group redshifts at the moment :( So just set the group redshift equal to the CMB redshift

	# Compute the FP variables from the raw data. Redshift uncertainties are negligible so we ignore them (for theta they only increase the error by, at most, 0.06%!)
	data["dz"] = sp.interpolate.splev(data["zcmb"].to_numpy(), dist_spline)
	data["dz_group"] = sp.interpolate.splev(data["zcmb_group"].to_numpy(), dist_spline)
	theta = data["uncor_radius"] * np.sqrt(data["BA_ratio"])
	theta_err = theta * np.sqrt((data["uncor_radius_err"]/data["uncor_radius"])**2 + 0.25*(data["BA_ratio_error"]/data["BA_ratio"])**2)
	data["r"] = np.log10(theta) + np.log10(data["dz_group"].to_numpy()) + np.log10(1000.0*np.pi/(180.0*3600.0)) - np.log10(1.0 + data["z_x"].to_numpy())
	data["er"] = theta_err/(np.log(10.0)*theta)
	data["kcor_r"] = calc_kcor('r', data["z_x"].to_numpy(), 'g - r', data["mag_g"].to_numpy() - data["mag_r_corrected"].to_numpy())
	data["kcor_g"] = calc_kcor('g', data["z_x"].to_numpy(), 'g - r', data["mag_g"].to_numpy() - data["mag_r_corrected"].to_numpy())
	data["i"] = 0.4*4.65 - 0.4*data["mag_r_corrected"] - 0.4*evo_corr*data["zcmb_group"] - np.log10(2.0*np.pi) - 2.0*np.log10(theta) + 4.0*np.log10(1.0 + data["z_x"]) + 0.4*data["kcor_r"] + 2.0*np.log10(180.0*3600.0/(10.0*np.pi))   # No extinction as this is already included in mag_r_corrected
	data["ei"] = np.sqrt(0.4**2*data["mag_r_err"]**2 + 4.0*(theta_err/(np.log(10.0)*theta))**2)
	theta_ap = 0.75               # DESI fiber radius in arcseconds
	sigma_corr = data["ppxf_sigma"] * (theta/8.0/theta_ap)**(-0.04)
	#sigma_corr = (0.94*data["ppxf_sigma"] + 9.53) * (theta/8.0/theta_ap)**(-0.04) #to correct to sdss ppxf sigma value
	sigma_corr_err = np.sqrt((sigma_corr * data["ppxf_sigma_error"]/data["ppxf_sigma"])**2 + (0.04*(theta/8.0/theta_ap)**(-1.04)*(theta_err/8.0/theta_ap))**2)
	data["s"] = np.log10(sigma_corr)
	data["es"] = sigma_corr_err/(np.log(10.0)*sigma_corr)  
	data["absmag_r"] = data["mag_r_corrected"] - 5.0*np.log10(data["dz_group"].to_numpy()) - 5.0*np.log10(1.0 + data["z_x"]) - 25.0 - data["kcor_r"] + evo_corr*data["zcmb_group"]

	# Trim the data to the magnitude limits (might not be necessary as the file already has 10.0 < r < 17.0)
	data = data.drop(data[(data["ppxf_sigma_error"]/data["ppxf_sigma"] > serrcut)].index)
	data = data.drop(data[(data["zcmb"] < zmin) | (data["zcmb"] > zmax)].index)
	data = data.drop(data[(data["mag_r_corrected"] < mag_low) | (data["mag_r_corrected"] > mag_high)].index)
	data = data.drop(data[(data["ppxf_sigma"] < 10.0**smin) | (data["ppxf_sigma"] > 10.0**smax)].index)

	# Compute the Sn weighting
	Vmin, Vmax = (1.0+zmin)**3*sp.interpolate.splev(zmin, dist_spline)**3, (1.0+zmax)**3*sp.interpolate.splev(zmax, dist_spline)**3
	Dlim = 10.0**((mag_high - data["mag_r_corrected"].to_numpy() + 5.0*np.log10(data["dz_group"].to_numpy()) + 5.0*np.log10(1.0 + data["z_x"]))/5.0)
	zlim = sp.interpolate.splev(Dlim, lumred_spline)
	data["Sn"] = np.where(zlim >= zmax, 1.0, np.where(zlim <= zmin, 0.0, (Dlim**3 - Vmin)/(Vmax - Vmin)))

	# Get data without any spirals (for fitting the Fundamental Plane parameters)
	if spiralsinFP:
		data_nospirals = data
	else:
		data_nospirals = data.drop(data[data["flag"] == 0].index)   # Remove VId spirals

	print(len(data), len(data_nospirals))

	# Fit the Fundamental Plane
	data_fit = data_nospirals
	badcount = len(data_nospirals)
	converged = False
	while not converged:
		avals, bvals, kvals = (1.0, 1.8), (-1.5, -0.5), (-0.2, 0.2)
		rvals, svals, ivals = (-0.5, 0.5), (2.0, 2.4), (2.4, 3.0)
		s1vals, s2vals, s3vals = (0.01, 0.12), (0.05, 0.5), (0.1, 0.3)
		FPparams = sp.optimize.differential_evolution(FP_func, bounds=(avals, bvals, rvals, svals, ivals, s1vals, s2vals, s3vals), 
			args=(0.0, data_fit["zcmb"].to_numpy(), data_fit["r"].to_numpy(), data_fit["s"].to_numpy(), data_fit["i"].to_numpy(), data_fit["er"].to_numpy(), data_fit["es"].to_numpy(), data_fit["ei"].to_numpy(), data_fit["Sn"].to_numpy(), smin), maxiter=10000, tol=1.0e-6)
		chi_squared = data_nospirals["Sn"].to_numpy()*FP_func(FPparams.x, 0.0, data_nospirals["zcmb"].to_numpy(), data_nospirals["r"].to_numpy(), data_nospirals["s"].to_numpy(), data_nospirals["i"].to_numpy(), data_nospirals["er"].to_numpy(), data_nospirals["es"].to_numpy(), data_nospirals["ei"].to_numpy(), data_nospirals["Sn"].to_numpy(), smin, sumgals=False, chi_squared_only=True)[0]
		pvals = sp.stats.chi2.sf(chi_squared, np.sum(chi_squared)/(len(data_nospirals) - 8.0))

		data_fit = data_nospirals.drop(data_nospirals[pvals < 0.01].index).reset_index(drop=True)
		badcountnew = len(np.where(pvals < 0.01)[0])
		converged = True if badcount == badcountnew else False
		print(FPparams.x, np.sum(chi_squared), len(data_fit), sp.stats.chi2.isf(0.01, np.sum(chi_squared)/(len(data_nospirals) - 8.0)), np.sum(chi_squared)/(len(data_nospirals) - 8.0), badcount, badcountnew, converged)
		badcount = badcountnew

	FPparams = FPparams.x

	# Fit the logdistance ratios
	dmin, dmax, nd = -1.5, 1.5, 1001
	dbins = np.linspace(dmin, dmax, nd, endpoint=True)

	d_H = np.outer(10.0**(-dbins), data["dz_group"].to_numpy())
	z_H = sp.interpolate.splev(d_H, red_spline, der=0)
	lmin = (4.65 + 5.0*np.log10(1.0+data["z_x"].to_numpy()) - evo_corr*data["zcmb_group"].to_numpy() + data["kcor_r"].to_numpy() + 10.0 - 2.5*np.log10(2.0*math.pi) + 5.0*np.log10(d_H) - mag_high)/5.0
	lmax = (4.65 + 5.0*np.log10(1.0+data["z_x"].to_numpy()) - evo_corr*data["zcmb_group"].to_numpy() + data["kcor_r"].to_numpy() + 10.0 - 2.5*np.log10(2.0*math.pi) + 5.0*np.log10(d_H) - mag_low)/5.0
	loglike = FP_func(FPparams, dbins, data["zcmb"].to_numpy(), data["r"].to_numpy(), data["s"].to_numpy(), data["i"].to_numpy(), data["er"].to_numpy(), data["es"].to_numpy(), data["ei"].to_numpy(), np.ones(len(data)), smin, sumgals=False)
	FNvals = FN_func(FPparams, data["zcmb"].to_numpy(), data["er"].to_numpy(), data["es"].to_numpy(), data["ei"].to_numpy(), lmin, lmax, smin)

	# Convert to the PDF for logdistance
	logP_dist = -1.5*np.log(2.0*math.pi) - loglike - FNvals

	# normalise logP_dist
	ddiff = np.log10(d_H[:-1])-np.log10(d_H[1:])
	valdiff = np.exp(logP_dist[1:])+np.exp(logP_dist[0:-1])
	norm = 0.5*np.sum(valdiff*ddiff, axis=0)

	logP_dist -= np.log(norm[:,None]).T
	
	# Calculate the mean and variance of the gaussian, then the skew
	mean = np.sum(dbins[0:-1,None]*np.exp(logP_dist[0:-1])+dbins[1:,None]*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0
	err = np.sqrt(np.sum(dbins[0:-1,None]**2*np.exp(logP_dist[0:-1])+dbins[1:,None]**2*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0 - mean**2)
	gamma1 = (np.sum(dbins[0:-1,None]**3*np.exp(logP_dist[0:-1])+dbins[1:,None]**3*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0 - 3.0*mean*err**2 - mean**3)/err**3
	gamma1 = np.where(gamma1 > 0.99, 0.99, gamma1)
	gamma1 = np.where(gamma1 < -0.99, -0.99, gamma1)
	delta = np.sign(gamma1)*np.sqrt(np.pi/2.0*1.0/(1.0 + ((4.0 - np.pi)/(2.0*np.abs(gamma1)))**(2.0/3.0)))
	scale = err*np.sqrt(1.0/(1.0 - 2.0*delta**2/np.pi))
	loc = mean - scale*delta*np.sqrt(2.0/np.pi)
	alpha = delta/(np.sqrt(1.0 - delta**2))

	data["logdist"] = mean
	data["logdist_err"] = err
	data["logdist_alpha"] = alpha

	if False:

		# Plot the PDFs of a few galaxies
		print(np.argmax(alpha), np.argmax(err), np.argmin(alpha), np.argmin(mean), np.argmax(mean))
		indices_good = np.array([np.argmax(alpha), np.argmax(err), np.argmin(alpha), np.argmin(mean), np.argmax(mean)])

		fig = plt.figure(figsize=(5,6))
		fig.patch.set_facecolor('None')
		gs = gridspec.GridSpec(6, 3, hspace=0.7, wspace=0.05, left=0.07, bottom=0.13, right=0.98, top=0.98) 

		ax1=fig.add_subplot(gs[2:,:])
		for index in indices_good:
			print(mean[index], err[index], alpha[index])
			skewindex = np.where(np.logical_and(np.logical_and(dbins > mean[index]-2.0*err[index], dbins < mean[index]+2.0*err[index]), np.abs(dbins-mean[index]) > 0.1*err[index]))
			gaussian = 1.0/(np.sqrt(2.0*math.pi)*err[index])*np.exp(-0.5*(dbins-mean[index])**2/err[index]**2)
			skew = np.mean((np.exp(logP_dist[skewindex,index])/gaussian[skewindex] - 1.0) * (((dbins[skewindex]-mean[index])/err[index])**3 - 3*(dbins[skewindex]-mean[index])/err[index]))
			skewpdf = 1.0/(np.sqrt(2.0*np.pi)*scale[index])*np.exp(-0.5*(dbins-loc[index])**2/scale[index]**2)*(1.0 + special.erf(alpha[index]*np.sqrt(0.5)*(dbins-loc[index])/scale[index]))
			ax1.errorbar(dbins, np.exp(logP_dist[:,index]), marker='o', color='k', markerfacecolor='k', markeredgecolor='k', linestyle="None", zorder=1)
			ax1.errorbar(dbins, skewpdf, marker='None', color='r', markerfacecolor='w', markeredgecolor='k', linestyle="-", zorder=1)
			#ax1.errorbar(dbins, gaussian, marker='None', color='g', markerfacecolor='w', markeredgecolor='k', linestyle="-", zorder=1)
		ax1.set_xlim(-0.75, 0.75)
		#ax1.set_xscale('log')
		ax1.set_xlabel(r'$\eta$',fontsize=14)
		#ax1.set_ylabel(r'$P(\Delta d_{n})$',fontsize=16)
		ax1.tick_params(width=1.3)
		ax1.tick_params('both',length=10, which='major')
		ax1.tick_params('both',length=5, which='minor')
		for axis in ['top','left','bottom','right']:
			ax1.spines[axis].set_linewidth(1.3)
		for tick in ax1.xaxis.get_ticklabels():
			tick.set_fontsize(12)
		for tick in ax1.yaxis.get_ticklabels():
			tick.set_fontsize(12)

		labels = [r"$\eta_{n}$", r"$\sigma_{\eta_{n}}$", r"$\alpha_{n}$"]
		extents = np.array([[-0.35, 0.35], [0.092, 0.115], [-0.8, -0.05]])
		xticks = np.array([[-0.2, 0.0, 0.2], [0.10, 0.11], [-0.7, -0.40, -0.10]])

		print(np.amin(alpha), np.amax(alpha))

		for i, plotdata in enumerate([mean, err, alpha]):
			mean, std = np.mean(plotdata), np.std(plotdata)
			xvals = np.linspace(mean-5.0*std, mean+5.0*std, 1000)
			pdf = 1.0/(np.sqrt(2.0*np.pi)*std)*np.exp(-0.5*(xvals-mean)**2/std**2)

			ax=fig.add_subplot(gs[:2,i])
			ax.hist(plotdata, 25, range=extents[i], color='b', histtype='stepfilled', alpha=0.2, density=True, zorder=2)
			ax.hist(plotdata, 25, range=extents[i], color='b', histtype='step', alpha=1.0, linewidth=1.3, density=True, zorder=3)
			ax.axvline(mean, color='k', linestyle='--', alpha=0.7)
			#ax.plot(xvals, pdf, color='k', linestyle='-')
			ax.set_xlim(extents[i][0], extents[i][1])
			ax.tick_params(width=1.3)
			ax.tick_params('both',length=8, which='major')
			ax.tick_params('both',length=4, which='minor')
			for axis in ['top','left','bottom','right']:
			    ax.spines[axis].set_linewidth(1.3)
			for tick in ax.xaxis.get_ticklabels():
			    tick.set_fontsize(12)
			for tick in ax.yaxis.get_ticklabels():
			    tick.set_fontsize(12)
			ax.set_yticks([])
			ax.set_xticks(xticks[i])
			if i == 0:
				ax.text(0.05, 0.85, labels[i], color='k', fontsize=14, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
			else:
				ax.text(0.75, 0.85, labels[i], color='k', fontsize=14, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

		plt.show()


	czmod = LightSpeed*data["zcmb"].to_numpy()*(1.0 + 0.5*(1.0 - deccel)*(data["zcmb"].to_numpy()) - (1.0/6.0)*(2.0 - deccel - 3.0*deccel*deccel)*(data["zcmb"].to_numpy())**2)
	data["pv"] = np.log(10.0)*czmod/(1.0 + czmod/LightSpeed)*data["logdist"].to_numpy()
	data["pverr"] = np.log(10.0)*czmod/(1.0 + czmod/LightSpeed)*data["logdist_err"].to_numpy()
	data.to_csv(outfile, index=False)

if __name__ == "__main__":

	fit_logdist("./DESI_FP_logdists_fiducial.csv")

