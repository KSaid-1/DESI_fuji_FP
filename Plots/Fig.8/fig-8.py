import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import getdist.plots as gdplt
import getdist.mcsamples
from getdist import MCSamples, plots


chain_sbf = pd.read_csv("mcmc_chain_only_H0.csv")
samples_sbf = getdist.mcsamples.MCSamples(
    samples=chain_sbf.to_numpy(),
    names=["H0"],
    labels=["H_0"],
)


chain_trgb = pd.read_csv("mcmc_chain_only_H0_trgb.csv")
samples_trgb = getdist.mcsamples.MCSamples(
    samples=chain_trgb.to_numpy(),
    names=["H0"],
    labels=["H_0"],
)

# chain_10percentsigma = pd.read_csv("mcmc_chain_only_H0_10percentsigma.csv")
# samples_10percentsigma = getdist.mcsamples.MCSamples(
#     samples=chain_10percentsigma.to_numpy(),
#     names=["H0"],
#     labels=["H_0"],
# )

chain_spiralsinFP = pd.read_csv("mcmc_chain_only_H0_spiralsinFP.csv")
samples_spiralsinFP = getdist.mcsamples.MCSamples(
    samples=chain_spiralsinFP.to_numpy(),
    names=["H0"],
    labels=["H_0"],
)

chain_corr_sigma_to_sdss = pd.read_csv("mcmc_chain_only_H0_corr_sigma_to_sdss.csv")
samples_corr_sigma_to_sdss = getdist.mcsamples.MCSamples(
    samples=chain_corr_sigma_to_sdss.to_numpy(),
    names=["H0"],
    labels=["H_0"],
)

chain_z_0034 = pd.read_csv("mcmc_chain_only_H0_z_0034.csv")
samples_z_0034 = getdist.mcsamples.MCSamples(
    samples=chain_z_0034.to_numpy(),
    names=["H0"],
    labels=["H_0"],
)

chain_z_001 = pd.read_csv("mcmc_chain_only_H0_z_001.csv")
samples_z_001 = getdist.mcsamples.MCSamples(
    samples=chain_z_001.to_numpy(),
    names=["H0"],
    labels=["H_0"],
)

chain_sbf_w_cal_err = pd.read_csv("mcmc_chain_only_H0_w_cal_err.csv")
samples_sbf_w_cal_err = getdist.mcsamples.MCSamples(
    samples=chain_sbf_w_cal_err.to_numpy(),
    names=["H0"],
    labels=["H_0"],
)

chain_sbf_minus_cal_err = pd.read_csv("mcmc_chain_only_H0_minus_cal_err.csv")
samples_sbf_minus_cal_err = getdist.mcsamples.MCSamples(
    samples=chain_sbf_minus_cal_err.to_numpy(),
    names=["H0"],
    labels=["H_0"],
)

chain_sbf_plus_cal_err = pd.read_csv("mcmc_chain_only_H0_plus_cal_err.csv")
samples_sbf_plus_cal_err = getdist.mcsamples.MCSamples(
    samples=chain_sbf_plus_cal_err.to_numpy(),
    names=["H0"],
    labels=["H_0"],
)

chain_sbf_zhd = pd.read_csv("mcmc_chain_only_H0_zhd.csv")
samples_sbf_zhd = getdist.mcsamples.MCSamples(
    samples=chain_sbf_zhd.to_numpy(),
    names=["H0"],
    labels=["H_0"],
)



chain_spiralsinFP_no = pd.read_csv("mcmc_chain_only_H0_no_spirals_anywhere.csv")
samples_spiralsinFP_no = getdist.mcsamples.MCSamples(
    samples=chain_spiralsinFP_no.to_numpy(),
    names=["H0"],
    labels=["H_0"],
)

g = gdplt.get_single_plotter(width_inch=9.3,ratio=1. / 3.)
g.settings.title_limit_fontsize = 13
g.settings.legend_frac_subplot_margin =0.01
g.settings.axes_fontsize = 17
g.settings.lab_fontsize = 17
g.settings.title_limit_fontsize = 15
g.plot_1d([samples_sbf, samples_spiralsinFP, samples_spiralsinFP_no, samples_corr_sigma_to_sdss, samples_z_0034, samples_z_001, samples_sbf_zhd, samples_trgb], 'H0', colors=['','','','','','','','grey'], ls=['-','-','-','-','-','-','-','--'], title_limit=None)
g.add_legend(['Fiducial: FP Analysis with SBF Calibration', 'Inclusion of Spirals', 'Exclusion of Spirals from FP, calibration, and HD', 'Correcting DESI $\sigma$ to SDSS', 'Higher Redshifts (z > 0.034)', 'No redshift cut applied', 'Applying the peculiar velocity correction', 'FP Analysis with TRGB Calibration'], colored_text=True)
plt.grid(visible=True, which='both', color='0.65',linestyle=':')
plt.xlabel(r'$H_0$ $[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$', fontsize=g.settings.axes_fontsize)
g.export('H0_sys_FP.pdf')