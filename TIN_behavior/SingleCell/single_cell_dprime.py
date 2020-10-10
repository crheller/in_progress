"""
Look at active / passive effects on single neuron discriminability of task stimuli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

df = pd.read_pickle('/auto/users/hellerc/code/projects/in_progress/TIN_behavior/singleCell_res.pickle')
df.index = df.pair
df['dprime'] = abs(df['dprime'])

# cat vs target decoding, grouped by cellid
f, ax = plt.subplots(2, 2, figsize=(8, 8))

a1_mask = (df.cat_tar) & (df.area=='A1') & (df.f1 == df.f2) 
peg_mask = (df.cat_tar) & (df.area=='PEG') & (df.f1 == df.f2)
ax[0, 0].scatter(df[a1_mask & ~df.active].groupby(by='cellid').mean()['dprime'], 
                df[a1_mask & df.active].groupby(by='cellid').mean()['dprime'], s=20, edgecolor='white', color='tab:blue')
ax[0, 0].scatter(df[a1_mask & ~df.active].groupby(by='cellid').mean()['dprime'].mean(), 
                df[a1_mask & df.active].groupby(by='cellid').mean()['dprime'].mean(), s=80, edgecolor='k', color='tab:blue')
m = np.min(ax[0, 0].get_xlim() + ax[0, 0].get_ylim())      
mi = np.max(ax[0, 0].get_xlim() + ax[0, 0].get_ylim())       
ax[0, 0].plot([m, mi], [m, mi], '--', color='grey')
ax[0, 0].set_xlabel("Passive")
ax[0, 0].set_ylabel("Active")
ax[0, 0].set_title(r"Catch vs. Tar $d'$, A1 Single Neurons")

ax[0, 1].scatter(df[peg_mask & ~df.active].groupby(by='cellid').mean()['dprime'], 
                df[peg_mask & df.active].groupby(by='cellid').mean()['dprime'], s=20, edgecolor='white', color='tab:orange')
ax[0, 1].scatter(df[peg_mask & ~df.active].groupby(by='cellid').mean()['dprime'].mean(), 
                df[peg_mask & df.active].groupby(by='cellid').mean()['dprime'].mean(), s=80, edgecolor='k', color='tab:orange')
m = np.min(ax[0, 1].get_xlim() + ax[0, 1].get_ylim())      
mi = np.max(ax[0, 1].get_xlim() + ax[0, 1].get_ylim())       
ax[0, 1].plot([m, mi], [m, mi], '--', color='grey')
ax[0, 1].set_xlabel("Passive")
ax[0, 1].set_ylabel("Active")
ax[0, 1].set_title(r"Catch vs. Tar $d'$, PEG Single Neurons")

a1_mask = (df.tar_tar) & (df.area=='A1') & (df.f1 == df.f2) 
peg_mask = (df.tar_tar) & (df.area=='PEG') & (df.f1 == df.f2)
ax[1, 0].scatter(df[a1_mask & ~df.active].groupby(by='cellid').mean()['dprime'], 
                df[a1_mask & df.active].groupby(by='cellid').mean()['dprime'], s=20, edgecolor='white', color='tab:blue')
ax[1, 0].scatter(df[a1_mask & ~df.active].groupby(by='cellid').mean()['dprime'].mean(), 
                df[a1_mask & df.active].groupby(by='cellid').mean()['dprime'].mean(), s=80, edgecolor='k', color='tab:blue')
m = np.min(ax[1, 0].get_xlim() + ax[1, 0].get_ylim())      
mi = np.max(ax[1, 0].get_xlim() + ax[1, 0].get_ylim())       
ax[1, 0].plot([m, mi], [m, mi], '--', color='grey')
ax[1, 0].set_xlabel("Passive")
ax[1, 0].set_ylabel("Active")
ax[1, 0].set_title(r"Tar vs. Tar $d'$, A1 Single Neurons")

ax[1, 1].scatter(df[peg_mask & ~df.active].groupby(by='cellid').mean()['dprime'], 
                df[peg_mask & df.active].groupby(by='cellid').mean()['dprime'], s=20, edgecolor='white', color='tab:orange')
ax[1, 1].scatter(df[peg_mask & ~df.active].groupby(by='cellid').mean()['dprime'].mean(), 
                df[peg_mask & df.active].groupby(by='cellid').mean()['dprime'].mean(), s=80, edgecolor='k', color='tab:orange')
m = np.min(ax[1, 1].get_xlim() + ax[1, 1].get_ylim())      
mi = np.max(ax[1, 1].get_xlim() + ax[1, 1].get_ylim())       
ax[1, 1].plot([m, mi], [m, mi], '--', color='grey')
ax[1, 1].set_xlabel("Passive")
ax[1, 1].set_ylabel("Active")
ax[1, 1].set_title(r"Tar vs. Tar $d'$, PEG Single Neurons")

f.tight_layout()

plt.show()