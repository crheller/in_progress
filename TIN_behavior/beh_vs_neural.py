"""
Attempt to relate behavioral performance changes to neural changes
(in d', and in noise space)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

df = pd.read_csv('/home/charlie/Desktop/lbhb/code/projects/in_progress/TIN_behavior/res.csv', index_col=0)
df.index = df.pair

val = 'dp_opt'  # centroid or optimal decoder
mask = (df.cat_tar) & (~df.tdr_overall) & (~df.pca)

# compare active/passive diff in dprime to behavior performance
delta_dp = (df.loc[mask & df.active][val] - df.loc[mask & ~df.active][val]) 
delta_dp_rel = delta_dp / \
                    (df.loc[mask & df.active][val] + df.loc[mask & ~df.active][val])
di = df.loc[mask & df.active][['DI', 'site', 'area', 'snr1', 'f1', 'f2']]
df_delt = pd.concat([delta_dp, di], axis=1)

mapping = {-5: 40, 0: 100, np.inf: 160}
f, ax = plt.subplots(1, 2, figsize=(10, 5))
for s in df_delt.site.unique():
    _df = df_delt.loc[df_delt.site==s].sort_values(by='snr1')
    size = [v for (k, v) in mapping.items() if k in _df.snr1.values]
    ec = ['white' if f1==f2 else 'k' for f1, f2 in zip(_df['f1'], _df['f2'])]
    #ec = 'k'
    area = _df.area.iloc[0]
    if area=='PEG':
        ax[1].scatter(_df['DI'], _df[val], edgecolor=ec, label=s, s=size, lw=1)
    else:
        ax[0].scatter(_df['DI'], _df[val], edgecolor=ec, label=s, s=size, lw=1)

ax[0].axvline(0.5, linestyle='--', color='grey')
ax[1].axvline(0.5, linestyle='--', color='grey')
ax[0].set_xlabel('Behavioral DI \n (catch vs. tar)')
ax[1].set_xlabel('Behavioral DI \n (catch vs. tar)')
ax[0].set_ylabel(r"$\Delta d'^2$")
ax[1].set_ylabel(r"$\Delta d'^2$")
ax[0].set_title('A1')
ax[1].set_title('PEG')
ax[0].legend(frameon=False)
ax[1].legend(frameon=False)

f.tight_layout()


plt.show()
