"""
Plot population PSTHs for catch stimuli / targets for each trial type (HIT / MISS / CORR. REJ. / INCORR. HIT)
    plot by:
        compute overall mean (collapse over cells)
        project on PC1 of evoked response?
        project on tar/catch discrimination axis
"""
from nems_lbhb.baphy_experiment import BAPHYExperiment
import charlieTools.baphy_remote as br
import charlieTools.noise_correlations as nc
from sklearn.decomposition import PCA
import nems.db as nd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# zscore for PCA?
zscore = False

# recording load options
options = {'resp': True, 'pupil': False, 'rasterfs': 20}

# epoch extraction window (to avoid poststim nans - bc sound was turned off 100ms post lick!!)
bins = int(options['rasterfs']*0.4)  # earliest lick is at stimonset + 200ms
prestim, poststim = int(options['rasterfs']*0.1), int(options['rasterfs']*0.3)

# define time window for plotting
t = np.linspace(0, 0.4 - 1/options['rasterfs'], int(options['rasterfs'] * 0.4))

# define trial outcomes for plotting (for each sound, plot it's psth for each outcome)
outcomes = ['HIT_TRIAL', 'MISS_TRIAL', 'INCORRECT_HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'PASSIVE_EXPERIMENT']

# siteids
sites = ['CRD009b', 'CRD010b', 'CRD011c', 'CRD012b', 'CRD013b', 'CRD016c', 'CRD017c', 'CRD018d', 'CRD019b']

for site in sites:
    # get parmfiles
    sql = "SELECT sCellFile.cellid, sCellFile.respfile, gDataRaw.resppath from sCellFile INNER JOIN" \
               " gCellMaster ON (gCellMaster.id=sCellFile.masterid) INNER JOIN" \
               " gDataRaw ON (sCellFile.rawid=gDataRaw.id)" \
               " WHERE gCellMaster.siteid=%s" \
               " and gDataRaw.runclass='TBP' and gDataRaw.bad=0"
    d = nd.pd_query(sql, (site,))
    d['parmfile'] = [f.replace('.spk.mat', '.m') for f in d['respfile']]
    parmfiles = np.unique(np.sort([os.path.join(d['resppath'].iloc[i], d['parmfile'].iloc[i]) for i in range(d.shape[0])])).tolist()
    manager = BAPHYExperiment(parmfiles)
    rec = manager.get_recording(**options)
    rec['resp'] = rec['resp'].rasterize()

    # find / sort epoch names
    files = [f for f in rec['resp'].epochs.name.unique() if 'FILE_' in f]
    targets = [f for f in rec['resp'].epochs.name.unique() if 'TAR_' in f]
    catch = [f for f in rec['resp'].epochs.name.unique() if 'CAT_' in f]

    sounds = targets + catch
    ref_stims = [x for x in rec['resp'].epochs.name.unique() if 'STIM_' in x]
    idx = np.argsort([int(s.split('_')[-1]) for s in ref_stims])
    ref_stims = np.array(ref_stims)[idx].tolist()
    all_stims = ref_stims + sounds


    # ======================================= POP PSTH1 =====================================
    # plot the mean population PSTH by just taking the mean psth across neurons
    f, ax = plt.subplots(1, len(sounds), figsize=(16, 3), sharey=True)
    i = 0
    colors = plt.cm.get_cmap('tab10', len(outcomes))
    for s in sounds:
        for oc, o in enumerate(outcomes):
            _r = rec.copy()
            _r = _r.create_mask(True)
            _r = _r.and_mask(o)
            try:
                resp = rec['resp'].extract_epoch(s, mask=_r['mask'])[:, :, :bins]
                psth = np.nanmean(resp, axis=(0, 1))
                ax[i].plot(t, psth, label=o, color=colors(oc))
                sem = np.nanstd(np.nanmean(resp, axis=1), axis=0) / np.sqrt(resp.shape[0])
                ax[i].fill_between(t, psth-sem, psth+sem, alpha=0.2, lw=0, color=colors(oc))
            except:
                pass

        ax[i].legend(frameon=False, fontsize=6)
        ax[i].axvline(0.1, linestyle='--', color='k')
        ax[i].set_title(s, fontsize=8)

        i+=1

    f.tight_layout()

    fig.canvas.set_window_title('Population PSTH (mean over neurons)')

    # ======================================= POP PSTH2 =====================================
    # project onto first PC of evoked target / catch responses, take mean across trials
    rall = rec.copy()
    rall = rall.create_mask(True)
    rall = rall.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'INCORRECT_HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'PASSIVE_EXPERIMENT'])
    # can't simply extract evoked for refs because can be longer/shorted if it came after target 
    # and / or if it was the last stim.So, masking prestim / postim doesn't work.Do it manually
    d = rall['resp'].extract_epochs(sounds, mask=rall['mask'])
    d = {k: v[~np.isnan(v[:, :, prestim:poststim].sum(axis=(1, 2))), :, :] for (k, v) in d.items()}
    d = {k: v[:, :, prestim:poststim] for (k, v) in d.items()}

    # zscore each neuron
    m = np.concatenate([d[e] for e in d.keys()], axis=0).mean(axis=-1).mean(axis=0)
    sd = np.concatenate([d[e] for e in d.keys()], axis=0).mean(axis=-1).std(axis=0)
    if zscore:
        d = {k: (v.transpose(0, -1, 1) - m).transpose(0, -1, 1)  for (k, v) in d.items()}
        d = {k: (v.transpose(0, -1, 1) / sd).transpose(0, -1, 1)  for (k, v) in d.items()}

    Rall_u = np.vstack([d[k].sum(axis=2).mean(axis=0) for k in d.keys()])

    pca = PCA(n_components=1)
    pca.fit(Rall_u)
    pc_axes = pca.components_

    rec['pca'] = rec['resp']._modified_copy(rec['resp']._data.T.dot(pc_axes.T).T)

    # plot the mean population PSTH of the PC1 projection
    fig, ax = plt.subplots(1, len(sounds), figsize=(16, 3), sharey=True)
    i = 0
    colors = plt.cm.get_cmap('tab10', len(outcomes))
    for s in sounds:
        for oc, o in enumerate(outcomes):
            _r = rec.copy()
            _r = _r.create_mask(True)
            _r = _r.and_mask(o)
            try:
                resp = rec['pca'].extract_epoch(s, mask=_r['mask'])[:, :, :bins]
                psth = np.nanmean(resp, axis=(0, 1))
                ax[i].plot(t, psth, label=o, color=colors(oc))
                sem = np.nanstd(np.nanmean(resp, axis=1), axis=0) / np.sqrt(resp.shape[0])
                ax[i].fill_between(t, psth-sem, psth+sem, alpha=0.2, lw=0, color=colors(oc))
            except:
                pass

        ax[i].legend(frameon=False, fontsize=6)
        ax[i].axvline(0.1, linestyle='--', color='k')
        ax[i].set_title(s, fontsize=8)

        i+=1

    fig.tight_layout()

    fig.canvas.set_window_title('Population PSTH (projection onto PC1)')


    # ======================================= POP PSTH3 =====================================
    # mean response along the target/catch discrimination axis
    # get mean catch response, and mean target response
    catch_u = np.vstack([d[k].sum(axis=2).mean(axis=0) for k in catch]).mean(axis=0)
    target_u = np.vstack([d[k].sum(axis=2).mean(axis=0) for k in targets]).mean(axis=0)

    # comptue dU for the target / catch discrimination
    dU = (target_u - catch_u)[np.newaxis, :]
    dU = dU / np.linalg.norm(dU)

    rec['discrim'] = rec['resp']._modified_copy(rec['resp']._data.T.dot(dU.T).T)

    # plot the mean population PSTH of the discrimination axis projection
    fig, ax = plt.subplots(1, len(sounds), figsize=(16, 3), sharey=True)
    i = 0
    colors = plt.cm.get_cmap('tab10', len(outcomes))
    for s in sounds:
        for oc, o in enumerate(outcomes):
            _r = rec.copy()
            _r = _r.create_mask(True)
            _r = _r.and_mask(o)
            try:
                resp = rec['discrim'].extract_epoch(s, mask=_r['mask'])[:, :, :bins]
                psth = np.nanmean(resp, axis=(0, 1))
                ax[i].plot(t, psth, label=o, color=colors(oc))
                sem = np.nanstd(np.nanmean(resp, axis=1), axis=0) / np.sqrt(resp.shape[0])
                ax[i].fill_between(t, psth-sem, psth+sem, alpha=0.2, lw=0, color=colors(oc))
            except:
                pass

        ax[i].legend(frameon=False, fontsize=6)
        ax[i].axvline(0.1, linestyle='--', color='k')
        ax[i].set_title(s, fontsize=8)

        i+=1

    fig.tight_layout()

    fig.canvas.set_window_title('Population PSTH (projection onto discrimination axis)')