"""
Analyze state-dependent changes in the noise space / how this relates to decoding axes
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 14

# load decoding results
df = pd.read_pickle('/auto/users/hellerc/code/projects/in_progress/TIN_behavior/res_pr.pickle')
df.index = df.pair

# look at catch vs. tar discrimination, use pair-specific tdr, not pca
mask = (df.tar_tar) & (df.tdr_overall) & (~df.pca) #& (df.f1==df.f2)
a1_mask = mask & (df.area=='A1')
peg_mask = mask & (df.area=='PEG')

# add alignment column (noise (e1) vs. dU alignment)
evec1 = np.stack(df['evecs'].values)[:, :, 0]
evec2 = np.stack(df['evecs'].values)[:, :, 1]
dU = np.stack(df['dU'].values).squeeze() 
dU = (dU.T / np.linalg.norm(dU, axis=1)).T
cos_ev1_dU = abs(np.diag(evec1 @ dU.T))
cos_ev2_dU = abs(np.diag(evec2 @ dU.T))
df['cos_dU_evec1'] = cos_ev1_dU
df['cos_dU_evec2'] = cos_ev2_dU

# compare noise space in active / passive
f, ax = plt.subplots(2, 3, figsize=(12, 8))

for i, string in enumerate(['cos_dU_evec1', 'cos_dU_evec2']):

    ax[i, 0].scatter(np.stack(df[df.active & a1_mask]['evals'].values)[:, i], 
                    np.stack(df[~df.active & a1_mask]['evals'].values)[:, i], 
                    edgecolor='white', label='A1')
    ax[i, 0].scatter(np.stack(df[df.active & peg_mask]['evals'].values)[:, i], 
                    np.stack(df[~df.active & peg_mask]['evals'].values)[:, i], 
                    edgecolor='white', label='PEG')
    ax[i, 0].set_xlabel('Active')
    ax[i, 0].set_ylabel('Passive')
    ax[i, 0].set_title(r"Noise variance ($\lambda_%s$)" % int(i+1))
    mi = np.min([ax[i, 0].get_xlim()[0], ax[i, 0].get_ylim()[0]])
    ma = np.max([ax[i, 0].get_xlim()[1], ax[i, 0].get_ylim()[1]])
    ax[i, 0].plot([mi, ma], [mi, ma], 'k--')
    ax[i, 0].legend(frameon=False)

    ax[i, 1].scatter(df[df.active & a1_mask][string], 
                    df[~df.active & a1_mask][string], 
                    edgecolor='white')
    ax[i, 1].scatter(df[df.active & peg_mask][string], 
                    df[~df.active & peg_mask][string], 
                    edgecolor='white')
    ax[i, 1].set_xlabel('Active')
    ax[i, 1].set_ylabel('Passive')
    ax[i, 1].set_title(r"Noise interference with $\Delta \mu$")
    ax[i, 1].plot([0, 1], [0, 1], 'k--')

    ax[i, 2].scatter(np.stack(df[df.active & a1_mask]['evals'].values)[:,i] - np.stack(df[~df.active & a1_mask]['evals'].values)[:,i], 
                    df[df.active & a1_mask][string] - df[~df.active & a1_mask][string], 
                    edgecolor='white')
    ax[i, 2].scatter(np.stack(df[df.active & peg_mask]['evals'].values)[:,i] - np.stack(df[~df.active & peg_mask]['evals'].values)[:,i], 
                    df[df.active & peg_mask][string] - df[~df.active & peg_mask][string], 
                    edgecolor='white')
    ax[i, 2].set_ylabel(r"$\Delta$ Noise interference (a - p)")
    ax[i, 2].set_xlabel(r"$\Delta$ Noise variance (a - p)")
    ax[i, 2].axhline(0, linestyle='--', color='k')
    ax[i, 2].axvline(0, linestyle='--', color='k')

f.tight_layout()

# look at first order separation in active / passive and overall change in dprime
df['dU_mag'] = df['dU'].apply(lambda x: np.linalg.norm(x))

f, ax = plt.subplots(2, 1, figsize=(4, 8))

ax[0].scatter(df[df.active & a1_mask]['dU_mag'], 
            df[~df.active & a1_mask]['dU_mag'],
            edgecolor='white', label='A1')
ax[0].scatter(df[df.active & peg_mask]['dU_mag'], 
            df[~df.active & peg_mask]['dU_mag'],
            edgecolor='white', label='PEG')
ax[0].set_xlabel('Active')
ax[0].set_ylabel('Passive')
ax[0].set_title(r"Signal Magnitude ($|\Delta \mathbf{\mu}|$)")
mi = np.min([ax[0].get_xlim()[0], ax[0].get_ylim()[0]])
ma = np.max([ax[0].get_xlim()[1], ax[0].get_ylim()[1]])
ax[0].plot([mi, ma], [mi, ma], 'k--')
ax[0].legend(frameon=False)

val = 'dp_opt'
ax[1].scatter(df[df.active & a1_mask][val], 
            df[~df.active & a1_mask][val],
            edgecolor='white', label='A1')
ax[1].scatter(df[df.active & peg_mask][val], 
            df[~df.active & peg_mask][val],
            edgecolor='white', label='PEG')
ax[1].set_xlabel('Active')
ax[1].set_ylabel('Passive')
ax[1].set_title(r"$d'^2$")
mi = np.min([ax[1].get_xlim()[0], ax[1].get_ylim()[0]])
ma = np.max([ax[1].get_xlim()[1], ax[1].get_ylim()[1]])
ax[1].plot([mi, ma], [mi, ma], 'k--')
ax[1].legend(frameon=False)

f.tight_layout()

plt.show()
'''
# compare the change in noise stats to behavior
f, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(np.stack(df[df.active & a1_mask]['evals'].values)[:,0] - np.stack(df[~df.active & a1_mask]['evals'].values)[:,0], 
                df[df.active & a1_mask]['DI'], label='A1')
ax[0].scatter(np.stack(df[df.active & peg_mask]['evals'].values)[:,0] - np.stack(df[~df.active & peg_mask]['evals'].values)[:,0], 
                df[df.active & peg_mask]['DI'], label='PEG')
ax[0].axhline(0.5, linestyle='--', color='k')
ax[0].set_ylabel('Behavioral performacne (DI)')
ax[0].set_xlabel(r"$\Delta$ Noise variance (a - p)")

ax[1].scatter(df[df.active & a1_mask]['cos_dU_evec1'] - df[~df.active & a1_mask]['cos_dU_evec1'], 
                df[df.active & a1_mask]['DI'], label='A1')
ax[1].scatter(df[df.active & peg_mask]['cos_dU_evec1'] - df[~df.active & peg_mask]['cos_dU_evec1'], 
                df[df.active & peg_mask]['DI'], label='PEG')
ax[1].axhline(0.5, linestyle='--', color='k')
ax[1].set_ylabel('Behavioral performacne (DI)')
ax[1].set_xlabel(r"$\Delta$ Noise interference (a - p)")
ax[1].legend(frameon=False)

f.tight_layout()

# sanity check compare the change in noise stats to change in decoding
val = 'dp_opt'
f, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(np.stack(df[df.active & a1_mask]['evals'].values)[:,0] - np.stack(df[~df.active & a1_mask]['evals'].values)[:,0], 
                df[df.active & a1_mask][val] - df[~df.active & a1_mask][val], label='A1')
ax[0].scatter(np.stack(df[df.active & peg_mask]['evals'].values)[:,0] - np.stack(df[~df.active & peg_mask]['evals'].values)[:,0], 
                df[df.active & peg_mask][val] - df[~df.active & peg_mask][val], label='PEG')
#ax[0].axhline(0.5, linestyle='--', color='k')
ax[0].set_xlabel(r"$\Delta$ Noise variance (a - p)")
ax[0].set_ylabel(r"$\Delta d'^2$")

ax[1].scatter(df[df.active & a1_mask]['cos_dU_evec1'] - df[~df.active & a1_mask]['cos_dU_evec1'], 
                df[df.active & a1_mask][val] - df[~df.active & a1_mask][val], label='A1')
ax[1].scatter(df[df.active & peg_mask]['cos_dU_evec1'] - df[~df.active & peg_mask]['cos_dU_evec1'], 
                df[df.active & peg_mask][val] - df[~df.active & peg_mask][val], label='PEG')
ax[1].set_ylabel(r"$\Delta d'^2$")
#ax[1].axhline(0.5, linestyle='--', color='k')
ax[1].set_xlabel(r"$\Delta$ Noise interference (a - p)")
ax[1].legend(frameon=False)

f.tight_layout()

plt.show()
'''