"""
Unpack ref-ref/ref-tar decoding - for different octave differences. Does the frequency space difference
predict something about noise / signal overlap and/or their contribution to behavior dependent 
changes in coding?
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

# look at catch vs. tar discrimination
mask = (df.ref_ref) & (~df.tdr_overall) & (~df.pca)
a1_mask = mask & (df.area=='A1')
peg_mask = mask & (df.area=='PEG')
a1_only = True
peg_only = False

df['octave_sep'] = np.log2(df['f1']/df['f2'])

evec1 = np.stack(df['evecs'].values)[:, :, 0]
evec2 = np.stack(df['evecs'].values)[:, :, 1]
dU = np.stack(df['dU'].values).squeeze() 
dU = (dU.T / np.linalg.norm(dU, axis=1)).T
cos_ev1_dU = abs(np.diag(evec1 @ dU.T))
cos_ev2_dU = abs(np.diag(evec2 @ dU.T))
df['cos_dU_evec1'] = cos_ev1_dU
df['cos_dU_evec2'] = cos_ev2_dU
df['dU_mag'] = df['dU'].apply(lambda x: np.linalg.norm(x))


# compare noise space in active / passive
f, ax = plt.subplots(2, 3, figsize=(12, 8))

for i, string in enumerate(['cos_dU_evec1', 'cos_dU_evec2']):

    if a1_only:
        ax[i, 0].scatter(np.stack(df[df.active & a1_mask]['evals'].values)[:, i], 
                        np.stack(df[~df.active & a1_mask]['evals'].values)[:, i], 
                        c=abs(df[df.active & a1_mask]['octave_sep']), cmap='Blues', label='A1', s=10)
    if peg_only:
        ax[i, 0].scatter(np.stack(df[df.active & peg_mask]['evals'].values)[:, i], 
                        np.stack(df[~df.active & peg_mask]['evals'].values)[:, i], 
                        c=abs(df[df.active & peg_mask]['octave_sep']), cmap='Oranges', label='PEG', s=10)
    ax[i, 0].set_xlabel('Active')
    ax[i, 0].set_ylabel('Passive')
    ax[i, 0].set_title(r"Noise variance ($\lambda_%s$)" % int(i+1))
    mi = np.min([ax[i, 0].get_xlim()[0], ax[i, 0].get_ylim()[0]])
    ma = np.max([ax[i, 0].get_xlim()[1], ax[i, 0].get_ylim()[1]])
    ax[i, 0].plot([mi, ma], [mi, ma], 'k--')

    if a1_only:
        ax[i, 1].scatter(df[df.active & a1_mask][string], 
                        df[~df.active & a1_mask][string], 
                        c=abs(df[df.active & a1_mask]['octave_sep']), cmap='Blues', s=10)
    if peg_only:
        ax[i, 1].scatter(df[df.active & peg_mask][string], 
                        df[~df.active & peg_mask][string], 
                        c=abs(df[df.active & peg_mask]['octave_sep']), cmap='Oranges', s=10)
    ax[i, 1].set_xlabel('Active')
    ax[i, 1].set_ylabel('Passive')
    ax[i, 1].set_title(r"Noise interference with $\Delta \mu$")
    ax[i, 1].plot([0, 1], [0, 1], 'k--')

    if a1_only:
        ax[i, 2].scatter(np.stack(df[df.active & a1_mask]['evals'].values)[:,i] - np.stack(df[~df.active & a1_mask]['evals'].values)[:,i], 
                        df[df.active & a1_mask][string] - df[~df.active & a1_mask][string], 
                        c=abs(df[df.active & a1_mask]['octave_sep']), cmap='Blues', s=10)
    if peg_only:
        ax[i, 2].scatter(np.stack(df[df.active & peg_mask]['evals'].values)[:,i] - np.stack(df[~df.active & peg_mask]['evals'].values)[:,i], 
                        df[df.active & peg_mask][string] - df[~df.active & peg_mask][string], 
                        c=abs(df[df.active & peg_mask]['octave_sep']), cmap='Oranges', s=10)
    ax[i, 2].set_ylabel(r"$\Delta$ Noise interference (a - p)")
    ax[i, 2].set_xlabel(r"$\Delta$ Noise variance (a - p)")
    ax[i, 2].axhline(0, linestyle='--', color='k')
    ax[i, 2].axvline(0, linestyle='--', color='k')

f.tight_layout()

df['dU_mag'] = df['dU'].apply(lambda x: np.linalg.norm(x))

f, ax = plt.subplots(2, 1, figsize=(4, 8))

if a1_only:
    ax[0].scatter(df[df.active & a1_mask]['dU_mag'], 
                df[~df.active & a1_mask]['dU_mag'],
                c=abs(df[df.active & a1_mask]['octave_sep']), cmap='Blues', s=10)
if peg_only:
    ax[0].scatter(df[df.active & peg_mask]['dU_mag'], 
                df[~df.active & peg_mask]['dU_mag'],
                c=abs(df[df.active & peg_mask]['octave_sep']), cmap='Oranges', s=10)
ax[0].set_xlabel('Active')
ax[0].set_ylabel('Passive')
ax[0].set_title(r"Signal Magnitude ($|\Delta \mathbf{\mu}|$)")
mi = np.min([ax[0].get_xlim()[0], ax[0].get_ylim()[0]])
ma = np.max([ax[0].get_xlim()[1], ax[0].get_ylim()[1]])
ax[0].plot([mi, ma], [mi, ma], 'k--')

val = 'dp_opt'
if a1_only:
    ax[1].scatter(df[df.active & a1_mask][val], 
                df[~df.active & a1_mask][val],
                c=abs(df[df.active & a1_mask]['octave_sep']), cmap='Blues', s=10)
if peg_only:
    ax[1].scatter(df[df.active & peg_mask][val], 
                df[~df.active & peg_mask][val],
                c=abs(df[df.active & peg_mask]['octave_sep']), cmap='Oranges', s=10)
ax[1].set_xlabel('Active')
ax[1].set_ylabel('Passive')
ax[1].set_title(r"$d'^2$")
mi = np.min([ax[1].get_xlim()[0], ax[1].get_ylim()[0]])
ma = np.max([ax[1].get_xlim()[1], ax[1].get_ylim()[1]])
ax[1].plot([mi, ma], [mi, ma], 'k--')

f.tight_layout()


plt.show()