import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

scatter_data = pd.read_csv('fig_2_scatter_data.csv')
fit_data = pd.read_csv('fig_2_fit_data.csv')
fit_params = pd.read_csv('fig_2_fit_params.csv')
text_data = pd.read_csv('fig_2_text_data.csv')



d02 = plt.scatter(scatter_data['ppxf_sigma_error'] / scatter_data['ppxf_sigma'],
                  scatter_data['snr_ppxf'],
                  c=scatter_data['z'],
                  edgecolor='none',
                  s=2.5,
                  norm=colors.TwoSlopeNorm(vmin=0., vcenter=0.05, vmax=0.1),
                  cmap=plt.cm.copper)

cbar2 = plt.colorbar(d02)
cbar2.set_label(r'Redshift')

plt.plot(fit_data['x_fit'], fit_data['y_fit'], 
         label=f"Fitted Function: $e^{{a / (x + b)}}$, a = {fit_params['a_fit'].values[0]:.2f}, b = {fit_params['b_fit'].values[0]:.2f}", 
         color='blue')

plt.xlabel(r'$\delta\sigma/\sigma$')
plt.ylabel(r'$S/N$ [pixel$^{-1}$]')
plt.xscale('log')
plt.yscale('log')

plt.text(0.004, 1.0, f"{text_data['percentage'].values[0]:.0f}% of SV FP galaxies", rotation=0, color='green', size=12)
plt.text(0.6, 4, f"{text_data['complement_percentage'].values[0]:.0f}% of SV FP galaxies", rotation=90, color='red', size=12)

plt.axvline(0.1, 0, 100, c='black', linestyle=':')
plt.grid(visible=True, which='both', color='0.65', linestyle=':')
plt.legend(loc="lower left")

plt.tight_layout()
plt.savefig('SNR_vs_delta_sigma_sv.png')
plt.savefig('SNR_vs_delta_sigma_sv.pdf')
plt.show()