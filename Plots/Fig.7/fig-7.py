import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d

main_data = pd.read_csv('fig_7_main_data.csv')
fit_data = pd.read_csv('fig_7_fit_data.csv')
fit_params = pd.read_csv('fig_7_fit_params.csv')

def variable_window(z):
    if z > 0.04:
        return 50
    elif 0.02 <= z <= 0.04:
        return 50
    else:
        return 50

def moving_average_variable(x, y, z):
    result = np.zeros_like(y)
    for i in range(len(y)):
        window = variable_window(z[i])
        start = max(0, i - window // 2)
        end = min(len(y), i + window // 2)
        result[i] = np.average(y[start:end], weights=x[start:end])
    return result

def moving_std_variable(x, y, z):
    result = np.zeros_like(y)
    for i in range(len(y)):
        window = variable_window(z[i])
        start = max(0, i - window // 2)
        end = min(len(y), i + window // 2)
        weights = x[start:end]
        mean = np.average(y[start:end], weights=weights)
        result[i] = np.sqrt(np.average((y[start:end] - mean)**2, weights=weights))
    return result

plt.style.use('default')

sigma = 5  # Gaussian smoothing parameter

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

ax1 = plt.subplot(gs[0])
ax1.errorbar(main_data['zcmb'], main_data['mu_a_cal'], yerr=main_data['mu_a_err'], fmt=".k", alpha=0.05)
ax1.grid(visible=True, which='both', color='0.65', linestyle=':')
ax1.plot(fit_data['x_fit'], fit_data['fitted_k_modulus_only_H0'], label='Best fit $H_0$ for a $\Lambda$CDM with ($\Omega_m$, $\Omega_{\Lambda}$) = (0.3, 0.7)', color='k')
ax1.plot(fit_data['x_fit'], fit_data['fitted_k_modulus_Planck_values_q0'], ls="--", label='Best fit $H_0$ for a $\Lambda$CDM with ($\Omega_m$, $\Omega_{\Lambda}$) = (0.315, 0.685)', color='r')
ax1.plot(fit_data['x_fit'], fit_data['fitted_k_modulus_Planck'], label='Planck 2018 results with ($H_0$, $\Omega_m$, $\Omega_{\Lambda}$) = (67.4, 0.315, 0.685)', color='b')

ax1.plot([0.023,0.0999],[1.5,1.5],'-k',lw=1.5)
ax1.plot([0.023,0.023],[1.4,1.6],'-k',lw=1.5)
ax1.plot([0.0999,0.0999],[1.4,1.6],'-k',lw=1.5)
ax1.text(0.042,1.55,"Fiducial $H_0$ Fitting Range",color='k',size=15)
ax1.plot([0.016,0.034],[1.0,1.0],'-k',lw=1.5)
ax1.plot([0.016,0.016],[0.9,1.1],'-k',lw=1.5)
ax1.plot([0.034,0.034],[0.9,1.1],'-k',lw=1.5)
ax1.text(0.0165,1.1,"Zero-point\nCalibration Range",color='k',size=11)
ax1.set_ylabel(r'$\mu_{A}$')
ax1.legend()
ax1.set_xlim(0.001,np.max(main_data['zcmb']))
ax1.set_ylim(0.5,3.0)

ax2 = plt.subplot(gs[1])
ax2.errorbar(main_data['zcmb'], main_data['res_fit_H0_low'], yerr=main_data['mu_a_err'], fmt=".k", alpha=0.05)
ax2.grid(visible=True, which='both', color='0.65', linestyle=':')

sorted_indices = np.argsort(main_data['zcmb'])
z_sorted = main_data['zcmb'].iloc[sorted_indices]
res_fit_H0_low_sorted = main_data['res_fit_H0_low'].iloc[sorted_indices]
mask = ~np.isnan(res_fit_H0_low_sorted)
z_sorted = z_sorted[mask]
res_fit_H0_low_sorted = res_fit_H0_low_sorted[mask]

def plot_average_line(data, color, label):
    data_sorted = data.iloc[sorted_indices][mask]
    smoothed_data = moving_average_variable(np.ones_like(z_sorted), data_sorted, z_sorted)
    smoothed_data = gaussian_filter1d(smoothed_data, sigma=sigma)
    std_data = moving_std_variable(np.ones_like(z_sorted), data_sorted, z_sorted)
    std_data = gaussian_filter1d(std_data, sigma=sigma)
    sem_data = std_data / np.sqrt(np.array([variable_window(z) for z in z_sorted]))
    
    ax2.plot(z_sorted, smoothed_data, color=color, alpha=0.7, label=label)
    ax2.fill_between(z_sorted, 
                     smoothed_data - 1.96 * sem_data, 
                     smoothed_data + 1.96 * sem_data, 
                     color=color, alpha=0.2)

plot_average_line(main_data['res_fit_H0_low'], 'black', 'Residuals')
plot_average_line(main_data['residuals_pv_effect'], 'blue', '2M++ pv model (Said et al. 2020)')
plot_average_line(main_data['residuals_pv_effect_2'], 'green', '2MRS pv model (Lilow et al. 2021)')
plot_average_line(main_data['residuals_pv_effect_3'], 'red', '2M++ pv model (Carrick et al. 2015)')

ax2.set_xlabel(r'$z$')
ax2.set_ylabel(r'$\mu_{A} - \mu_{A}^{model}$')
ax2.set_xlim(0.0, np.max(main_data['zcmb']))
ax2.set_ylim(-0.5, 0.5)
ax2.legend(loc='best', fontsize='small')

plt.tight_layout()
plt.savefig('Hubble_diagram.pdf')
plt.savefig('Hubble_diagram.png', dpi=360)
plt.show()