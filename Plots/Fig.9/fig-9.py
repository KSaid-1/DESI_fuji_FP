import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import getdist.plots as gdplt
import getdist.mcsamples
from getdist import loadMCSamples
from getdist import MCSamples, plots


def read_chains(file_prefix, num_chains):
    chains = []
    for i in range(num_chains):
        file_name = f"{file_prefix}_{i}.csv"
        chain_i = pd.read_csv(file_name)
        chains.append(chain_i)
    return chains

def calculate_statistical_error(chains):
    chain_arrays = [chain.values for chain in chains]
    
    samples = MCSamples(samples=chain_arrays, names=chains[0].columns, labels=['H_0'])
    g = plots.get_single_plotter(width_inch=9.3,ratio=1. / 3.)
    g.settings.title_limit_fontsize = 13
    g.settings.legend_frac_subplot_margin =0.01
    g.settings.axes_fontsize = 17
    g.settings.lab_fontsize = 17
    g.settings.title_limit_fontsize = 15
    plt.grid(visible=True, which='both', color='0.65', linestyle=':')
    plt.title('Systematics due to using SBF as Calibrator')
    for chain_array in chain_arrays:
        individual_samples = MCSamples(samples=[chain_array], names=chains[0].columns, labels=chains[0].columns)
        g.plot_1d(individual_samples, 'H0',line_args={'ls':'--', 'color':'grey','lw':0.5, 'alpha':0.2})
    g.plot_1d(samples, 'H0',lims=[60.,90.])#, title_limit=1)
    plt.xlabel(r'$H_0$ $[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$', fontsize=g.settings.axes_fontsize)
    g.export('H0_measured_sys_sbf.pdf')
    
    means = np.mean(chain_arrays, axis=0)
    std_devs = np.std(chain_arrays, axis=0)
    
    statistical_error = np.sqrt(np.mean(std_devs**2))
    
    return statistical_error

file_prefix = 'mcmc_chain_only_H0'
num_chains = 10 #change this number to 1000 if you have all chains here https://doi.org/10.5281/zenodo.13363598
chains = read_chains(file_prefix, num_chains)

statistical_error = calculate_statistical_error(chains)
print(f"Statistical Error: {statistical_error}")


def calculate_statistical_error(chains):
    chain_arrays = [chain.values for chain in chains]
    
    samples = MCSamples(samples=chain_arrays, names=chains[0].columns, labels=['H_0'])
    
    g = plots.get_single_plotter(width_inch=9.3,ratio=1. / 3.)
    g.settings.title_limit_fontsize = 13
    g.settings.legend_frac_subplot_margin =0.01
    g.settings.axes_fontsize = 17
    g.settings.lab_fontsize = 17
    g.settings.title_limit_fontsize = 15
    plt.grid(visible=True, which='both', color='0.65', linestyle=':')
    plt.title('Systematics due to using TRGB as Calibrator')
    for chain_array in chain_arrays:
        individual_samples = MCSamples(samples=[chain_array], names=chains[0].columns, labels=chains[0].columns)
        g.plot_1d(individual_samples, 'H0',line_args={'ls':'--', 'color':'grey','lw':0.5, 'alpha':0.2})
    g.plot_1d(samples, 'H0', lims=[60.,90.])#, title_limit=1)
    #g.add_legend(['Systematics due to the TRGB Calibration'])
    plt.xlabel(r'$H_0$ $[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$', fontsize=g.settings.axes_fontsize)
    g.export('H0_measured_sys_trgb.pdf')
    
    means = np.mean(chain_arrays, axis=0)
    std_devs = np.std(chain_arrays, axis=0)
    
    statistical_error = np.sqrt(np.mean(std_devs**2))
    
    return statistical_error


file_prefix = './Ho_sys_calibration_TRGB/mcmc_chain_only_H0'
num_chains = 10 #change this number to 1000 if you have the full data here https://doi.org/10.5281/zenodo.13363598
chains = read_chains(file_prefix, num_chains)

statistical_error = calculate_statistical_error(chains)
print(f"Statistical Error: {statistical_error}")