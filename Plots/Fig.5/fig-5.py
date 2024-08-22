import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroML.plotting import scatter_contour

xfit = np.loadtxt("fig_5_xfit.csv", delimiter=",")
sfit = np.loadtxt("fig_5_sfit.csv", delimiter=",")
ifit = np.loadtxt("fig_5_ifit.csv", delimiter=",")
desi_data = pd.read_csv("fig_5_desi_data.csv")
XFP_F_desi = desi_data["XFP_F_desi"].values
r_desi = desi_data["r_desi"].values
fit_params = pd.read_csv("fig_5_fit_params.csv")

a = fit_params["a"][0]
b = fit_params["b"][0]
c = fit_params["c"][0]
mean_r = fit_params["mean_r"][0]
mean_s = fit_params["mean_s"][0]
mean_i = fit_params["mean_i"][0]
sigma1 = fit_params["sigma1"][0]
sigma2 = fit_params["sigma2"][0]
sigma3 = fit_params["sigma3"][0]

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-0.4, 1.0)
ax.set_ylim(-0.4, 1.0)

scatter_contour(XFP_F_desi, r_desi, threshold=25, log_counts=True, ax=ax,
               histogram2d_args=dict(bins=25),
               plot_args=dict(marker=',', linestyle='none', color='black'),
               contour_args=dict(cmap=plt.cm.copper))


ax.plot(xfit, xfit, c='black')  # Plotting the xfit line
ax.text(-0.35, 1.125 - 0.2, f"3D Gaussian fit for Fuji r-band Fundamental Plane using {len(r_desi)} galaxies:")
ax.text(-0.35, 1.0 - 0.2, rf'$a$ = ${a:.3f}$, $b$ = ${b:.3f}$, $c$ = ${c:.3f}$')
ax.text(-0.35, 0.875 - 0.2, rf'$\bar r$ = ${mean_r:.3f}$, $\bar s$ = ${mean_s:.3f}$, $\bar i$ = ${mean_i:.3f}$')
ax.text(-0.35, 0.75 - 0.2, rf'$\sigma_1$ = ${sigma1:.3f}$, $\sigma_2$ = ${sigma2:.3f}$, $\sigma_3$ = ${sigma3:.3f}$')
ax.set_xlabel(r'$a \/\log_{10} \sigma + b\/ \log_{10} I_e + c$ [$h^{-1}$ kpc]', size=20)
ax.set_ylabel(r'$\log_{10} R_e$ [$h^{-1}$ kpc]', size=20)
plt.legend(framealpha=0.0)
ax.grid(visible=True, which='both', color='0.65', linestyle=':')
plt.tight_layout()

plt.savefig('FP_fuji_recreated.png', transparent=True, dpi=300)
plt.savefig('FP_fuji_recreated.pdf')
plt.show()