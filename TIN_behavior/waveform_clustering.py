import nems.db as nd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import numpy as np

path = '/auto/users/hellerc/results/nat_pupil_ms/'
cellids_cache = path + 'celltypes.csv'

cellids = pd.DataFrame(pd.concat([nd.get_batch_cells(289), nd.get_batch_cells(307), nd.get_batch_cells(324), nd.get_batch_cells(325)]).cellid)
iso_query = f"SELECT cellid, isolation from gSingleRaw WHERE cellid in {tuple([x for x in cellids.cellid])}"
isolation = nd.pd_query(iso_query)
cellids = pd.DataFrame(data=isolation[isolation.isolation>=75].cellid.unique(), columns=['cellid'])

sw = [nd.get_gSingleCell_meta(cellid=c, fields='wft_spike_width') for c in cellids.cellid] 
cellids['spike_width'] = sw

# remove cellids that weren't sorted with KS (so don't have waveform stats)
cellids = cellids[cellids.spike_width!=-1]

# now save endslope and peak trough ratio
es = [nd.get_gSingleCell_meta(cellid=c, fields='wft_endslope') for c in cellids.cellid] 
pt = [nd.get_gSingleCell_meta(cellid=c, fields='wft_peak_trough_ratio') for c in cellids.cellid] 
cellids['end_slope'] = es
cellids['peak_trough'] = pt

g = sns.pairplot(cellids[['spike_width', 'end_slope', 'peak_trough']], diag_kind='kde')



# looks like endslope and spike width most effect for clustering... focus in on them
f, ax = plt.subplots(1, 2, figsize=(10, 5))
km = KMeans(n_clusters=2).fit(cellids[['spike_width', 'peak_trough', 'end_slope']])
cellids['type'] = km.labels_

g = sns.scatterplot(x='spike_width', y='peak_trough', hue='type', data=cellids, s=25, ax=ax[0])

ax[0].set_xlabel('Spike Width (s)')
ax[0].set_ylabel('Spike peak:trough ratio')
ax[0].set_title(r"$n_{type=1} = %s / %s$" % (cellids['type'].sum(), cellids.shape[0]))

km = KMeans(n_clusters=2).fit(cellids[['spike_width', 'end_slope']])
cellids['type'] = km.labels_

g = sns.scatterplot(x='spike_width', y='end_slope', hue='type', data=cellids, s=25, ax=ax[1])

ax[1].set_xlabel('Spike Width (s)')
ax[1].set_ylabel('Spike endslope (dV / dt)')
ax[1].set_title(r"$n_{type=1} = %s / %s$" % (cellids['type'].sum(), cellids.shape[0]))

f.tight_layout()

# one-D classification based on on spike width
f, ax = plt.subplots(1, 1, figsize=(6, 4))
km = KMeans(n_clusters=2).fit(cellids[['spike_width', 'peak_trough']])
cellids['type'] = km.labels_

bins = np.arange(0, 1, 0.02)
h = ax.hist(cellids['spike_width'], color='lightgrey', bins=bins)
ax.hist(cellids[cellids.type==0]['spike_width'], label='Type 1', histtype='step', lw=2, bins=bins)
ax.hist(cellids[cellids.type==1]['spike_width'], label='Type 2', histtype='step', lw=2, bins=bins)

# fit bimodal
def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)


expected=(0.3, .2, 50, 0.6, .2, 100)
params, cov = curve_fit(bimodal, h[1][:-1], h[0], expected)
sigma = np.sqrt(np.diag(cov))
ax.plot(h[1], bimodal(h[1], *params), color='red', lw=3, label='Bimodal model')

ax.legend(frameon=False)

plt.show()

