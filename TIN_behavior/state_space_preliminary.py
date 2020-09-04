"""
For each recording site from CRD, plot target / catch responses in state-space.

Label each figure with:
    siteid
    ncells
    nfiles
    penArea
"""
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

# recording load options
options = {'resp': True, 'pupil': False, 'rasterfs': 10}

# state-space projection options
zscore = True

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

    # ================================================================================================
    # Plot "raw" data -- tuning curves / psth's .
    # PSTHs for REFs
    ref_dur = int(manager.get_baphy_exptparams()[0]['TrialObject'][1]['ReferenceHandle'][1]['Duration'] * options['rasterfs'])
    pre_post = int(manager.get_baphy_exptparams()[0]['TrialObject'][1]['ReferenceHandle'][1]['PostStimSilence'] * options['rasterfs'])
    d1 = int(np.sqrt(rec['resp'].shape[0]))
    f, ax = plt.subplots(d1+1, d1+1, figsize=(16, 12))
    for i in range(rec['resp'].shape[0]):
        if i == 0:
            br.psth(rec, chan=rec['resp'].chans[i], epochs=ref_stims, ep_dur=ref_dur, cmap='viridis', prestim=pre_post, ax=ax.flatten()[i])
        else:
            br.psth(rec, chan=rec['resp'].chans[i], epochs=ref_stims, ep_dur=ref_dur, cmap='viridis', prestim=pre_post, supp_legend=True, ax=ax.flatten()[i])
    f.tight_layout()

    # PSTHs for TARs (and CATCH)
    d1 = int(np.sqrt(rec['resp'].shape[0]))
    f, ax = plt.subplots(d1+1, d1+1, figsize=(16, 12))
    for i in range(rec['resp'].shape[0]):
        if i == 0:
            br.psth(rec, chan=rec['resp'].chans[i], epochs=sounds, ep_dur=ref_dur, cmap=None, prestim=pre_post, ax=ax.flatten()[i])
        else:
            br.psth(rec, chan=rec['resp'].chans[i], epochs=sounds, ep_dur=ref_dur, cmap=None, prestim=pre_post, supp_legend=True, ax=ax.flatten()[i])
    f.tight_layout()

    # TUNING CURVES
    cfs = [s.split('_')[-1] for s in ref_stims]
    prestim, poststim = int(options['rasterfs']*0.1), int(options['rasterfs']*0.3)
    ftc_all = []
    f, ax = plt.subplots(d1+1, d1+1, figsize=(16, 12))
    for i in range(rec['resp'].shape[0]):
        d = rec['resp'].extract_channels([rec['resp'].chans[i]]).extract_epochs(ref_stims)
        ftc = [d[e][~np.isnan(d[e][:,0,prestim:poststim].sum(axis=-1)), :, prestim:poststim].mean() for e in d.keys()]
        ftc_all.append(ftc)
        if i == 0:
            ax.flatten()[i].plot(ftc)
        else:
            ax.flatten()[i].plot(ftc)
        
        ax.flatten()[i].set_xticks(range(len(ftc)))
        ax.flatten()[i].set_xticklabels(cfs, fontsize=6, rotation=45)
        ax.flatten()[i].set_xlabel('CF', fontsize=6)
        ax.flatten()[i].set_ylabel('Mean response', fontsize=6)
        ax.flatten()[i].set_title(rec['resp'].chans[i], fontsize=6)

    f.tight_layout()


    # Mean tuning curve across population
    f, ax = plt.subplots(1, 1)

    tar_freq = int(np.array(targets)[np.argwhere([True if '+Inf' in t else False for t in targets])][0][0].split('_')[1].split('+')[0])

    ax.set_title('Population (mean) Tuning Curve')
    ax.plot(np.stack(ftc_all).mean(axis=0))
    ax.axvline(np.argwhere(np.array([int(cf) for cf in cfs])==tar_freq)[0][0], 
                        color='k', linestyle='--', label='Target Frequency')
    ax.set_xticks(range(np.stack(ftc_all).shape[-1]))
    ax.set_xticklabels(cfs, rotation=45)
    ax.set_xlabel('CF')
    ax.set_ylabel('Mean response')
    ax.legend(frameon=False)

    f.tight_layout()

    # =================================== trial-averaged PCA space ====================================
    rall = rec.copy()
    rall = rall.create_mask(True)
    rall = rall.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'INCORRECT_HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'PASSIVE_EXPERIMENT'])
    # can't simply extract evoked for refs because can be longer/shorted if it came after target 
    # and / or if it was the last stim.So, masking prestim / postim doesn't work.Do it manually
    d = rall['resp'].extract_epochs(all_stims, mask=rall['mask'])
    d = {k: v[~np.isnan(v[:, :, prestim:poststim].sum(axis=(1, 2))), :, :] for (k, v) in d.items()}
    d = {k: v[:, :, prestim:poststim] for (k, v) in d.items()}

    # zscore each neuron
    m = np.concatenate([d[e] for e in d.keys()], axis=0).mean(axis=-1).mean(axis=0)
    sd = np.concatenate([d[e] for e in d.keys()], axis=0).mean(axis=-1).std(axis=0)
    if zscore:
        d = {k: (v.transpose(0, -1, 1) - m).transpose(0, -1, 1)  for (k, v) in d.items()}
        d = {k: (v.transpose(0, -1, 1) / sd).transpose(0, -1, 1)  for (k, v) in d.items()}

    Rall_u = np.vstack([d[k].sum(axis=2).mean(axis=0) for k in d.keys()])

    pca = PCA(n_components=2)
    pca.fit(Rall_u)
    pc_axes = pca.components_

    ra = rec.copy().create_mask(True)
    ra = ra.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'CORRECT_REJECT_TRIAL'])
    da = ra['resp'].extract_epochs(all_stims, mask=ra['mask'])
    da = {k: v[:, :, prestim:poststim] for (k, v) in da.items()}
    da = {k: v[~np.isnan(v.sum(axis=(1, 2))), :, :] for (k, v) in da.items()}
    if zscore:
        da = {k: (v.transpose(0, -1, 1) - m).transpose(0, -1, 1)  for (k, v) in da.items()}
        da = {k: (v.transpose(0, -1, 1) / sd).transpose(0, -1, 1)  for (k, v) in da.items()}

    rp = rec.copy().create_mask(True)
    rp = rp.and_mask(['PASSIVE_EXPERIMENT'])
    dp = rp['resp'].extract_epochs(all_stims, mask=rp['mask'])
    dp = {k: v[:, :, prestim:poststim] for (k, v) in dp.items()}
    dp = {k: v[~np.isnan(v.sum(axis=(1, 2))), :, :] for (k, v) in dp.items()}
    if zscore:
        dp = {k: (v.transpose(0, -1, 1) - m).transpose(0, -1, 1)  for (k, v) in dp.items()}
        dp = {k: (v.transpose(0, -1, 1) / sd).transpose(0, -1, 1)  for (k, v) in dp.items()}

    # project active / passive responses onto PCA plane
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    for e in all_stims:
        passive = dp[e].mean(axis=-1).dot(pc_axes.T)
        active = da[e].mean(axis=-1).dot(pc_axes.T)

        if e in ref_stims:
            ax[0].plot(passive[:, 0], passive[:, 1], alpha=0.3, marker='.', color='lightgrey', lw=0)
            el = compute_ellipse(passive[:, 0], passive[:, 1])
            ax[0].plot(el[0], el[1], lw=1, color=ax[0].get_lines()[-1].get_color())
            
            ax[1].plot(active[:, 0], active[:, 1], alpha=0.3, marker='.', color='lightgrey', lw=0)
            el = compute_ellipse(active[:, 0], active[:, 1])
            ax[1].plot(el[0], el[1], lw=1, color=ax[1].get_lines()[-1].get_color())
        else:
            ax[0].plot(passive[:, 0], passive[:, 1], alpha=0.3, marker='.', lw=0, label=e)
            el = compute_ellipse(passive[:, 0], passive[:, 1])
            ax[0].plot(el[0], el[1], lw=1, color=ax[0].get_lines()[-1].get_color())
            
            ax[1].plot(active[:, 0], active[:, 1], alpha=0.3, marker='.', lw=0, label=e)
            el = compute_ellipse(active[:, 0], active[:, 1])
            ax[1].plot(el[0], el[1], lw=1, color=ax[1].get_lines()[-1].get_color())

    ax[0].axhline(0, linestyle='--', lw=2, color='grey')
    ax[0].axvline(0, linestyle='--', lw=2, color='grey')
    ax[1].axhline(0, linestyle='--', lw=2, color='grey')
    ax[1].axvline(0, linestyle='--', lw=2, color='grey')

    ax[0].set_title('Passive')
    ax[1].set_title('Active')
    ax[0].set_xlabel(r'$PC_1$ (var. explained: {})'.format(round(pca.explained_variance_ratio_[0], 3)))
    ax[1].set_xlabel(r'$PC_1$ (var. explained: {})'.format(round(pca.explained_variance_ratio_[0], 3)))
    ax[0].set_ylabel(r'$PC_2$ (var. explained: {})'.format(round(pca.explained_variance_ratio_[1], 3)))
    ax[1].set_ylabel(r'$PC_2$ (var. explained: {})'.format(round(pca.explained_variance_ratio_[1], 3)))

    ax[0].legend(frameon=False, fontsize=6)

    fig.canvas.set_window_title("PCA decompostion")

    fig.tight_layout()

    # =================================== TDR space - tar/cat only ====================================
    # project onto tar/cat discrim axis / principle noise axis for these stims
    d = rall['resp'].extract_epochs(sounds, mask=rall['mask'])
    d = {k: v[~np.isnan(v[:, :, prestim:poststim].sum(axis=(1, 2))), :, :] for (k, v) in d.items()}
    d = {k: v[:, :, prestim:poststim] for (k, v) in d.items()}

    # zscore each neuron
    m = np.concatenate([d[e] for e in d.keys()], axis=0).mean(axis=-1).mean(axis=0)
    sd = np.concatenate([d[e] for e in d.keys()], axis=0).mean(axis=-1).std(axis=0)
    if zscore:
        d = {k: (v.transpose(0, -1, 1) - m).transpose(0, -1, 1)  for (k, v) in d.items()}
        d = {k: (v.transpose(0, -1, 1) / sd).transpose(0, -1, 1)  for (k, v) in d.items()}

    # get mean catch response, and mean target response
    catch_u = np.vstack([d[k].sum(axis=2).mean(axis=0) for k in catch]).mean(axis=0)
    target_u = np.vstack([d[k].sum(axis=2).mean(axis=0) for k in targets]).mean(axis=0)

    # comptue dU for the target / catch discrimination
    dU = (target_u - catch_u)[np.newaxis, :] 
    dU /= np.linalg.norm(dU)

    # get pooled noise data over catch / target stimuli
    noise_data = np.vstack([d[k].sum(axis=2) - d[k].sum(axis=2).mean(axis=0) for k in d.keys()])
    pca = PCA()
    pca.fit(noise_data)
    noise_axis = pca.components_[[0], :]

    # define second TDR axis
    noise_on_dec = (np.dot(noise_axis, dU.T)) * dU
    orth_ax = noise_axis - noise_on_dec
    orth_ax /= np.linalg.norm(orth_ax)

    tdr_weights = np.concatenate((dU, orth_ax), axis=0)

    ra = rec.copy().create_mask(True)
    ra = ra.and_mask(['HIT_TRIAL', 'MISS_TRIAL', 'CORRECT_REJECT_TRIAL'])
    da = ra['resp'].extract_epochs(sounds, mask=ra['mask'])
    da = {k: v[:, :, prestim:poststim] for (k, v) in da.items()}
    da = {k: v[~np.isnan(v.sum(axis=(1, 2))), :, :] for (k, v) in da.items()}
    if zscore:
        da = {k: (v.transpose(0, -1, 1) - m).transpose(0, -1, 1)  for (k, v) in da.items()}
        da = {k: (v.transpose(0, -1, 1) / sd).transpose(0, -1, 1)  for (k, v) in da.items()}

    rp = rec.copy().create_mask(True)
    rp = rp.and_mask(['PASSIVE_EXPERIMENT'])
    dp = rp['resp'].extract_epochs(sounds, mask=rp['mask'])
    dp = {k: v[:, :, prestim:poststim] for (k, v) in dp.items()}
    dp = {k: v[~np.isnan(v.sum(axis=(1, 2))), :, :] for (k, v) in dp.items()}
    if zscore:
        dp = {k: (v.transpose(0, -1, 1) - m).transpose(0, -1, 1)  for (k, v) in dp.items()}
        dp = {k: (v.transpose(0, -1, 1) / sd).transpose(0, -1, 1)  for (k, v) in dp.items()}

    # get mean active / passive noise correlations
    rsc_active = nc.compute_rsc(da, chans=rec['resp'].chans)['rsc'].mean()
    rsc_passive = nc.compute_rsc(dp, chans=rec['resp'].chans)['rsc'].mean()

    # project active / passive responses onto PCA plane
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    for e in sounds:
        passive = dp[e].mean(axis=-1).dot(tdr_weights.T)
        active = da[e].mean(axis=-1).dot(tdr_weights.T)

        ax[0].plot(passive[:, 0], passive[:, 1], alpha=0.3, marker='.', lw=0, label=e)
        el = compute_ellipse(passive[:, 0], passive[:, 1])
        ax[0].plot(el[0], el[1], lw=1, color=ax[0].get_lines()[-1].get_color())
        
        ax[1].plot(active[:, 0], active[:, 1], alpha=0.3, marker='.', lw=0, label=e)
        el = compute_ellipse(active[:, 0], active[:, 1])
        ax[1].plot(el[0], el[1], lw=1, color=ax[1].get_lines()[-1].get_color())

    ax[0].axhline(0, linestyle='--', lw=2, color='grey')
    ax[0].axvline(0, linestyle='--', lw=2, color='grey')
    ax[1].axhline(0, linestyle='--', lw=2, color='grey')
    ax[1].axvline(0, linestyle='--', lw=2, color='grey')

    ax[0].set_title(r'Passive, $r_{sc}$: %s' %round(rsc_passive, 3))
    ax[1].set_title(r'Active, $r_{sc}$: %s' %round(rsc_active, 3))
    ax[0].set_xlabel(r'$TDR_1$ ($\Delta \mu$)')
    ax[1].set_xlabel(r'$TDR_1$ ($\Delta \mu$)')
    ax[0].set_ylabel(r'$TDR_2$')
    ax[1].set_ylabel(r'$TDR_2$')

    ax[0].legend(frameon=False, fontsize=6)

    fig.canvas.set_window_title("TDR decompostion")

    fig.tight_layout()



    # ================================= Explicitly compute pop. coding angles ==================================