import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)
    
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [-3.0, 0.0, 3.0]
        return np.ma.masked_array(np.interp(value, x, y))

#dust_data = pd.read_csv('dust_data_fig_1.csv')
cluster_data = pd.read_csv('cluster_data_fig_1.csv')
desi_data = pd.read_csv('desi_data_fig_1.csv')
fuji_data = pd.read_csv('fuji_data_fig_1.csv')

plt.rcParams.update({'font.size': 8})
plt.rcParams["font.family"] = "serif"
plt.rcParams['mathtext.fontset'] = 'cm'
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1, projection='mollweide')
plt.subplots_adjust(top=0.95, bottom=0.02, right=0.95, left=0.08)

norm = MidpointNormalize(midpoint=-1.0)
# ax.scatter(dust_data['ra_dust'], dust_data['dec_dust'], s=0.1, c=dust_data['ebv_dust'], 
#            edgecolors='none', cmap='Reds', norm=norm, alpha=0.1, rasterized=True)

for _, cluster in cluster_data.iterrows():
    ax.text(cluster['ra_rad'] + cluster['r_thing'], cluster['dec_rad'] + cluster['d_thing'], 
            r'{}'.format(cluster['Id']), fontsize=9, color='black')
    ax.plot(cluster['ra_rad'], cluster['dec_rad'], 'o', markersize=20, 
            color='black', markeredgecolor='black', mfc='none')

ax.scatter(desi_data['ra_desi'], desi_data['dec_desi'], s=5., alpha=0.05, facecolors='none', edgecolors='b')

ax.scatter(fuji_data['ra_fuji'], fuji_data['dec_fuji'], s=0.1, color='salmon')

ax.grid(visible=True, which='both', color='0.65', linestyle=':')
ax.set_ylabel('Declination', fontsize=10, labelpad=1)
ax.set_xlabel('Right Ascension', fontsize=10, labelpad=+4)
plt.tight_layout()

plt.savefig('fuji_New_Mollweide_RADEC.pdf')
plt.show()