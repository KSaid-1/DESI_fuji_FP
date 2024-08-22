import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

scatter_data = pd.read_csv('fig_4_external_scatter_data.csv')
fit_data1 = pd.read_csv('fig_4_external_fit_data1.csv')
fit_params1 = pd.read_csv('fig_4_external_fit_params1.csv')
fit_data2 = pd.read_csv('fig_4_external_fit_data2.csv')
fit_params2 = pd.read_csv('fig_4_external_fit_params2.csv')
histogram_data = pd.read_csv('fig_4_external_histogram_data.csv')
gaussian_fit = pd.read_csv('fig_4_external_gaussian_fit.csv')

f_1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), sharex=False, sharey=False)

ax1.set_ylim(0.0, 350.0)
ax1.set_xlim(0.0, 350.0)
ax1.errorbar(scatter_data['ppxf_sigma'], scatter_data['veldisp'], 
             xerr=scatter_data['ppxf_sigma_error'], yerr=scatter_data['veldispErr'], 
             fmt=".k", alpha=0.1)
ax1.scatter(scatter_data['ppxf_sigma'], scatter_data['veldisp'], color='black', edgecolor='none', s=3.5)
ax1.plot([0, 350], [0, 350], color="black")
ax1.plot(fit_data1['x_fit1'], fit_data1['y_fit1'], 'r--', label='Linear Fit')
ax1.text(20., 330., "Linear Fit", color='r', size=10)
ax1.text(20., 300., f"y = {fit_params1['a_fit1'].values[0]:.2f}*x + {fit_params1['b_fit1'].values[0]:.2f}", color='r', size=10)
ax1.set_aspect(1)
ax1.set_xlabel(r'$\sigma_{\rm{DESI}}^{\rm{pPXF}}$ [km/s]', size=15)
ax1.set_ylabel(r'$\sigma_{\rm{SDSS}}^{\rm{pipeline}}$ [km/s]', size=15)
ax1.grid(visible=True, which='both', color='0.65', linestyle=':')

# Subplot 4
ax4.set_ylim(0.0, 350.0)
ax4.set_xlim(0.0, 350.0)
ax4.errorbar(scatter_data['ppxf_sigma'], scatter_data['sigmaStars'], 
             xerr=scatter_data['ppxf_sigma_error'], yerr=scatter_data['sigmaStarsErr'], 
             fmt=".k", alpha=0.1)
ax4.scatter(scatter_data['ppxf_sigma'], scatter_data['sigmaStars'], color='black', edgecolor='none', s=3.5)
ax4.plot([0, 350], [0, 350], color="black")
ax4.plot(fit_data2['x_fit2'], fit_data2['y_fit2'], 'r--', label='Linear Fit')
ax4.text(20., 330., "Linear Fit", color='r', size=10)
ax4.text(20., 300., f"y = {fit_params2['a_fit2'].values[0]:.2f}*x + {fit_params2['b_fit2'].values[0]:.2f}", color='r', size=10)
ax4.set_aspect(1)
ax4.set_xlabel(r'$\sigma_{\rm{DESI}}^{\rm{pPXF}}$ [km/s]', size=15)
ax4.set_ylabel(r'$\sigma_{\rm{SDSS}}^{\rm{pPXF}}$ [km/s]', size=15)
ax4.grid(visible=True, which='both', color='0.65', linestyle=':')

ax2.set_ylim(-200, 200)
ax2.set_xlim(0.0, 350.0)
ax2.scatter(((scatter_data['ppxf_sigma'] + scatter_data['veldisp']) / 2.), 
            scatter_data['ppxf_sigma'] - scatter_data['veldisp'], 
            color='black', edgecolor='none', s=3.5)
ax2.axhline(y=0.0, color="black")
ax2.grid(visible=True, which='both', color='0.65', linestyle=':')
ax2.set_xlabel(r'$\dfrac{\sigma_{\rm{DESI}}^{\rm{pPXF}} + \sigma_{\rm{SDSS}}^{\rm{pipeline}}}{2}$ [km/s]', size=15)
ax2.set_ylabel(r'$\sigma_{\rm{DESI}}^{\rm{pPXF}} - \sigma_{\rm{SDSS}}^{\rm{pipeline}}$ [km/s]', size=15)

ax5.set_ylim(-200, 200)
ax5.set_xlim(0.0, 350.0)
ax5.scatter(((scatter_data['ppxf_sigma'] + scatter_data['sigmaStars']) / 2.), 
            scatter_data['ppxf_sigma'] - scatter_data['sigmaStars'], 
            color='black', edgecolor='none', s=3.5)
ax5.axhline(y=0.0, color="black")
ax5.grid(visible=True, which='both', color='0.65', linestyle=':')
ax5.set_xlabel(r'$\dfrac{\sigma_{\rm{DESI}}^{\rm{pPXF}} + \sigma_{\rm{SDSS}}^{\rm{pPXF}}}{2}$ [km/s]', size=15)
ax5.set_ylabel(r'$\sigma_{\rm{DESI}}^{\rm{pPXF}} - \sigma_{\rm{SDSS}}^{\rm{pPXF}}$ [km/s]', size=15)

ax3.set_ylim(0.0, 0.5)
ax3.set_xlim(-4.0, 4.0)
ax3.hist(histogram_data['pull_pipe_1'], bins=np.arange(-5.0, 5.0, 0.1), density=1, histtype='step', color='red')
ax3.plot(gaussian_fit['edges_prop'], gaussian_fit['y'], 'k', linewidth=2)
ax3.grid(visible=True, which='both', color='0.65', linestyle=':')
ax3.set_xlabel(r'$\epsilon = \dfrac{\sigma_{\rm{DESI}}^{\rm{pPXF}} - \sigma_{\rm{SDSS}}^{\rm{pipeline}}}{\sqrt{(\delta \sigma_{\rm{DESI}}^{\rm{pPXF}})^2 + (\delta \sigma_{\rm{SDSS}}^{\rm{pipeline}})^2}}$', size=15)
ax3.set_ylabel(r'$N$', size=15)
ax3.text(2.0, 0.45, f"$N = {len(histogram_data['pull_pipe_1']):.0f}$", size=15, color='r')
ax3.text(2.0, 0.4, f"$\\bar{{\\epsilon}} = {np.mean(histogram_data['pull_pipe_1']):.2f}$", size=15, color='r')
ax3.text(2.0, 0.35, f"$\\sigma_{{\\epsilon}} = {np.std(histogram_data['pull_pipe_1']):.2f}$", size=15, color='r')

ax6.set_ylim(0.0, 0.5)
ax6.set_xlim(-4.0, 4.0)
ax6.hist(histogram_data['pull_pipe_2'], bins=np.arange(-5.0, 5.0, 0.1), density=1, histtype='step', color='red')
ax6.plot(gaussian_fit['edges_prop'], gaussian_fit['y'], 'k', linewidth=2)
ax6.grid(visible=True, which='both', color='0.65', linestyle=':')
ax6.set_xlabel(r'$\epsilon = \dfrac{\sigma_{\rm{DESI}}^{\rm{pPXF}} - \sigma_{\rm{SDSS}}^{\rm{pPXF}}}{\sqrt{(\delta \sigma_{\rm{DESI}}^{\rm{pPXF}})^2 + (\delta \sigma_{\rm{SDSS}}^{\rm{pPXF}})^2}}$', size=15)
ax6.set_ylabel(r'$N$', size=15)
ax6.text(2.0, 0.45, f"$N = {len(histogram_data['pull_pipe_2']):.0f}$", size=15, color='r')
ax6.text(2.0, 0.4, f"$\\bar{{\\epsilon}} = {np.mean(histogram_data['pull_pipe_2']):.2f}$", size=15, color='r')
ax6.text(2.0, 0.35, f"$\\sigma_{{\\epsilon}} = {np.std(histogram_data['pull_pipe_2']):.2f}$", size=15, color='r')

plt.tight_layout()
plt.savefig('external_consistency_v2.png')
plt.savefig('external_consistency_v2.pdf')
plt.show()