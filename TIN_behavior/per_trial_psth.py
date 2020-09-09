# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from nems_lbhb.baphy_experiment import BAPHYExperiment
import charlieTools.baphy_remote as br
import charlieTools.noise_correlations as nc
from charlieTools.plotting import compute_ellipse
from sklearn.decomposition import PCA
import nems.db as nd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# %% [markdown]
# ### Per-trial PSTHs
# * plot PSTH per trial for each behavior condition, aligned to catch, aligned to target, aligned to trial start. 
# %% [markdown]
# #### Options

# %%
site = 'CRD010b'
options = {'resp': True, 'pupil': False, 'rasterfs': 100}

trial_outcomes = ['HIT_TRIAL', 'MISS_TRIAL', 'CORRECT_REJECT_TRIAL', 'PASSIVE_EXPERIMENT']
# could also do: FALSE_ALARM_TRIAL, INCORRECT_HIT_TRIAL

# trial window (so can extract custom slices of data)
trial_window = options['rasterfs'] * 4  # 4 seconds of data

# define onset / offset window for extracting only complete sound data. Could load this from exptevents, just faster to hardcode
onset = int(0.1 * options['rasterfs'])
offset = int(0.4 * options['rasterfs'])

# %% [markdown]
# #### Load recording

# %%
# get parmfiles
sql = "SELECT sCellFile.cellid, sCellFile.respfile, gDataRaw.resppath from sCellFile INNER JOIN"             " gCellMaster ON (gCellMaster.id=sCellFile.masterid) INNER JOIN"             " gDataRaw ON (sCellFile.rawid=gDataRaw.id)"             " WHERE gCellMaster.siteid=%s"             " and gDataRaw.runclass='TBP' and gDataRaw.bad=0"
d = nd.pd_query(sql, (site,))
d['parmfile'] = [f.replace('.spk.mat', '.m') for f in d['respfile']]
parmfiles = np.unique(np.sort([os.path.join(d['resppath'].iloc[i], d['parmfile'].iloc[i]) for i in range(d.shape[0])])).tolist()
manager = BAPHYExperiment(parmfiles)
rec = manager.get_recording(**options)
rec['resp'] = rec['resp'].rasterize()

# %% [markdown]
# #### Plot PSTHs
# * For each trial outcome, plot:
#     * per-neuron trial psth as heatmap
#     * average across neuron psth (pop. psth)
#     * psth along PC1
#     * along PC2
#     * in state-space (PC1 vs. PC2)

# %%
rt = rec.copy()
rt = rt.and_mask('TRIAL')
for to in trial_outcomes:

    fig = plt.figure(figsize=(14, 12))
    pn = plt.subplot2grid((12, 1), (0, 0), rowspan=6)
    pop = plt.subplot2grid((12, 1), (6, 0), rowspan=2)
    pc1 = plt.subplot2grid((12, 1), (8, 0), rowspan=2)
    pc2 = plt.subplot2grid((12, 1), (10, 0), rowspan=2)

    r = rt.copy()
    r = r.and_mask(to)

    # single neuron psth
    psth = r['resp'].extract_epoch('TRIAL', mask=r['mask'])
    psth = np.nanmean(psth, axis=0)
    pn.imshow(psth, aspect='auto', cmap='Reds')
    pn.set_ylabel('Neuron')
    pn.set_xlabel(f"Trial time (binned at {options['rasterfs']} Hz)")

    # Mean population psth
    pop.plot(np.mean(psth, axis=0) * options['rasterfs'], lw=2, color='k', label='population PSTH')
    pop.set_ylabel('Mean FR (Hz)')
    pop.set_xlabel(f"Trial time (binned at {options['rasterfs']} Hz)")
    pop.legend(frameon=False, fontsize=8)

    # PCA on trial averaged responses
    targets = [f for f in rec['resp'].epochs.name.unique() if 'TAR_' in f]
    catch = [f for f in rec['resp'].epochs.name.unique() if 'CAT_' in f]

    sounds = targets + catch
    ref_stims = [x for x in rec['resp'].epochs.name.unique() if 'STIM_' in x]
    idx = np.argsort([int(s.split('_')[-1]) for s in ref_stims])
    ref_stims = np.array(ref_stims)[idx].tolist()
    all_stims = ref_stims + sounds

    rall = rt.copy()
    rall = rt.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'INCORRECT_HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'PASSIVE_EXPERIMENT'])
    # can't simply extract evoked for refs because can be longer/shorted if it came after target 
    # and / or if it was the last stim.So, masking prestim / postim doesn't work.Do it manually
    d = rall['resp'].extract_epochs(all_stims, mask=rall['mask'])
    d = {k: v[~np.isnan(v[:, :, onset:offset].sum(axis=(1, 2))), :, :] for (k, v) in d.items()}
    d = {k: v[:, :, onset:offset] for (k, v) in d.items()}

    Rall_u = np.vstack([d[k].sum(axis=2).mean(axis=0) for k in d.keys()])

    pca = PCA(n_components=2)
    pca.fit(Rall_u)
    pc_axes = pca.components_

    # project onto first PC and plot trial psth
    r['pc1'] = r['resp']._modified_copy(r['resp']._data.T.dot(pc_axes.T).T[[0], :])
    pc_psth = r['pc1'].extract_epoch('TRIAL', mask=r['mask'])
    pc_psth = np.nanmean(pc_psth, axis=0)
    pc_psth = np.mean(pc_psth, axis=0) * options['rasterfs']
    if pc_psth[np.argmax(np.abs(pc_psth))] < 0:
        pc_psth *= -1
    pc1.plot(pc_psth, lw=2, color='tab:blue', label=r"$PC_1 PSTH$")
    pc1.set_title(f"Variance Explained: {pca.explained_variance_ratio_[0]}")
    pc1.set_ylabel('Mean FR (Hz)')
    pc1.set_xlabel(f"Trial time (binned at {options['rasterfs']} Hz)")
    pc1.legend(frameon=False, fontsize=8)

    # project onto second PC and plot trial psth
    r['pc2'] = r['resp']._modified_copy(r['resp']._data.T.dot(pc_axes.T).T[[1], :])
    pc_psth = r['pc2'].extract_epoch('TRIAL', mask=r['mask'])
    pc_psth = np.nanmean(pc_psth, axis=0)
    pc_psth = np.mean(pc_psth, axis=0) * options['rasterfs']
    if pc_psth[np.argmax(np.abs(pc_psth))] < 0:
        pc_psth *= -1
    pc2.plot(pc_psth, lw=2, color='tab:orange', label=r"$PC_2 PSTH$")
    pc2.set_title(f"Variance Explained: {pca.explained_variance_ratio_[1]}")
    pc2.set_ylabel('Mean FR (Hz)')
    pc2.set_xlabel(f"Trial time (binned at {options['rasterfs']} Hz)")
    pc2.legend(frameon=False, fontsize=8)

    pn.set_title(to, fontsize=14)

    fig.tight_layout()


# %%
np.isnan(r['pc1']._data).sum()

