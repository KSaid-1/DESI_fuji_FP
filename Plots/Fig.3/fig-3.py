import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

scatter_data = pd.read_csv('fig_3_scatter_data.csv')
fit_data = pd.read_csv('fig_3_fit_data.csv')
fit_params = pd.read_csv('fig_3_fit_params.csv')
scatter_data_2 = pd.read_csv('fig_3_scatter_data_2.csv')
histogram_data = pd.read_csv('fig_3_histogram_data.csv')
gaussian_fit = pd.read_csv('fig_3_gaussian_fit.csv')
stats = pd.read_csv('fig_3_stats.csv')

f_1, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 4), sharex=False, sharey=False)

ax1.set_ylim(0.0, 350.0)
ax1.set_xlim(0.0, 350.0)
ax1.errorbar(scatter_data['sigma_1'], scatter_data['sigma_2'], 
             xerr=scatter_data['err_1'], yerr=scatter_data['err_2'], 
             fmt=".k", alpha=0.1)
ax1.scatter(scatter_data['sigma_1'], scatter_data['sigma_2'], color='red', edgecolor='none', alpha=0.01)
ax1.plot([0, 350], [0, 350], color="black")
ax1.plot(fit_data['x_fit'], fit_data['y_fit'], 'r--', label='Linear Fit')
ax1.text(20., 330., "Linear Fit", color='r', size=10)
ax1.text(20., 300., f"y = {fit_params['a_fit'].values[0]:.2f}*x + {fit_params['b_fit'].values[0]:.2f}", color='r', size=10)
ax1.grid(visible=True, which='both', color='0.65', linestyle=':')
ax1.set_aspect(1)
ax1.set_xlabel(r'$\sigma_{P}$', size=15)
ax1.set_ylabel(r'$\sigma_{S}$', size=15)

ax2.set_ylim(-200, 200)
ax2.set_xlim(0.0, 350.0)
ax2.scatter(scatter_data_2['x'], scatter_data_2['y'], color='black', edgecolor='none', s=3.5)
ax2.axhline(y=0.0, color="black")
ax2.grid(visible=True, which='both', color='0.65', linestyle=':')
ax2.set_xlabel(r'$\dfrac{\sigma_{p} + \sigma_{s}}{2}$ [km/s]', size=15)
ax2.set_ylabel(r'$\sigma_{P} - \sigma_{S}$', size=15)

ax3.set_ylim(0.0, 0.5)
ax3.set_xlim(-4.0, 4.0)
ax3.hist(histogram_data['pull_pipe'], bins=np.arange(-5.0, 5.0, 0.1), density=1, histtype='step', color='red', label='sigmastar_lamost vs. veldisp_sdss')
ax3.plot(gaussian_fit['edges_prop'], gaussian_fit['y'], 'k', linewidth=2, label='Gaussian with mean 0 and $\sigma$ 1')
ax3.grid(visible=True, which='both', color='0.65', linestyle=':')
ax3.set_xlabel(r'$\epsilon = \dfrac{\sigma_{p} - \sigma_{s}}{\sqrt{(\delta \sigma_{p})^2 + (\delta \sigma_{s})^2}}$', size=15)
ax3.set_ylabel(r'$N$', size=15)
ax3.text(1.5, 0.45, f"$\\bar{{\\epsilon}}\\/=\\/{stats['mean_pull'].values[0]:.3f}$", size=15, color="red")
ax3.text(1.5, 0.4, f"$\\sigma_{{\\epsilon}}\\/=\\/{stats['std_pull'].values[0]:.3f}$", size=15, color="red")

plt.tight_layout()
plt.savefig('internal_consistency.png')
plt.savefig('internal_consistency.pdf')
plt.show()