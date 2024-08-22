import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroML.plotting import scatter_contour

data = pd.read_csv('fig_6_fuji_sdss_data.csv')

xfit = np.linspace(-0.3, 0.3)
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-0.3, 0.3)
ax.set_ylim(-0.3, 0.3)

scatter_contour(data.logdist_x, data.logdist_y, threshold=10, log_counts=True, ax=ax,
                histogram2d_args=dict(bins=12),
                plot_args=dict(marker=',', linestyle='none', color='black'),
                contour_args=dict(cmap=plt.cm.copper))

ax.plot(xfit, xfit, c='black')

mean_x_err = data.logdist_err_x.mean()
mean_y_err = data.logdist_err_y.mean()

corner_x = -0.18
corner_y = 0.18
ax.errorbar(corner_x, corner_y, xerr=mean_x_err, yerr=mean_y_err, fmt='o', color='black', 
            capsize=5, capthick=1, elinewidth=1, markersize=0)

ax.text(corner_x + 0.02, corner_y + 0.04, f'Mean errors:\nu($\eta_x$): {mean_x_err:.3f}\nu($\eta_y$): {mean_y_err:.3f}', 
        verticalalignment='center', fontsize=8)

slope = 1.00877
intercept = -0.0122365
ax.plot([-0.4, 0.4], [slope * -0.4 + intercept, slope * 0.4 + intercept], 
        color="red", label=f'Hyper Fit (y={slope:.2f}x+{intercept:.2f})')

ax.set_xlabel(r'$\eta_{DESI}$')
ax.set_ylabel(r'$\eta_{SDSS}$')
plt.legend(framealpha=0.0)
ax.grid(visible=True, which='both', color='0.65', linestyle=':')
plt.tight_layout()
plt.savefig('fuji_vs_sdss_contour_with_errors.png', dpi=300)
plt.savefig('fuji_vs_sdss_contour_with_errors.pdf', dpi=300)
plt.show()