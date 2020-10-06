"""
Quick and dirty quantify discriminability of pairwise combos of tars / catches.
In active / passive.
    deal with pupil?

define TDR over all stims? Or on pairwise basis? Both? Use PC-space too?
"""

from nems_lbhb.baphy_experiment import BAPHYExperiment
import charlieTools.baphy_remote as br
import charlieTools.noise_correlations as nc
from charlieTools.plotting import compute_ellipse
from charlieTools.decoding import compute_dprime
from charlieTools.dim_reduction import TDR
import nems_lbhb.tin_helpers as thelp
from sklearn.decomposition import PCA
import nems.db as nd
from itertools import combinations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 14

# fig path
fpath = '/auto/users/hellerc/results/tmp_figures/tbp_decoding/'

# recording load options
options = {'resp': True, 'pupil': False, 'rasterfs': 10}
batches = [324, 325]
recache = False

# state-space projection options
zscore = False

# plot ref
plot_ref = False
if plot_ref:
    fext = '_withREF'
else:
    fext = ''

# extract evoked periods
start = int(0.1 * options['rasterfs'])
end = int(0.4 * options['rasterfs'])

# siteids
dfs = []
for batch in batches:
    sites = np.unique([c[:7] for c in nd.get_batch_cells(batch).cellid])
    sites = [s for s in sites if s!='CRD013b']
    for site in sites:
        skip_site = False
        # set up subplots for PCA / TDR projections
        f, ax = plt.subplots(2, 2, figsize=(12, 10))
        f.canvas.set_window_title(site)

        print("Analyzing site: {}".format(site))
        manager = BAPHYExperiment(batch=batch, siteid=site)
        rec = manager.get_recording(recache=recache, **options)
        rec['resp'] = rec['resp'].rasterize()

        # mask appropriate trials
        rec = rec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])

        ra = rec.copy()
        ra = ra.create_mask(True)
        ra = ra.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])

        rp = rec.copy()
        rp = rp.create_mask(True)
        rp = rp.and_mask(['PASSIVE_EXPERIMENT'])

        _rp = rp.apply_mask(reset_epochs=True)
        _ra = ra.apply_mask(reset_epochs=True)

        # find / sort epoch names
        targets = thelp.sort_targets([f for f in _ra['resp'].epochs.name.unique() if 'TAR_' in f])
        # only keep target presented at least 5 times
        targets = [t for t in targets if (_ra['resp'].epochs.name==t).sum()>=5]
        # remove "off-center targets"
        on_center = thelp.get_tar_freqs([f.strip('REM_') for f in _ra['resp'].epochs.name.unique() if 'REM_' in f])[0]
        targets = [t for t in targets if str(on_center) in t]
        if len(targets)==0:
            # NOT ENOUGH REPS AT THIS SITE
            skip_site = True
        catch = [f for f in _ra['resp'].epochs.name.unique() if 'CAT_' in f]
        # remove off-center catches
        catch = [c for c in catch if str(on_center) in c]
        ref_stim = thelp.sort_refs([f for f in _ra['resp'].epochs.name.unique() if 'STIM_' in f])
        rem = [f for f in rec['resp'].epochs.name.unique() if 'REM_' in f]
        sounds = targets + catch

        if not skip_site:
            # define colormaps for each sound
            tar_colors = plt.get_cmap('Reds', len(targets)+2)
            cat_colors = plt.get_cmap('Greys', len(catch)+2)
            ref_colors = plt.get_cmap('viridis', len(ref_stim))
            BwG, gR = thelp.make_tbp_colormaps(ref_stim, catch+targets)
            # get all pairwise combos of targets / catches
            pairs = list(combinations(sounds, 2))
            index = [p[0]+'_'+p[1] for p in pairs]
            df = pd.DataFrame(columns=['dp_opt', 'wopt', 'evecs', 'evec_sim', 'dU', 'dp_diag', 'tdr_overall', 'pca', 'active', 'pair' \
                                    'snr1', 'snr2', 'cat_cat', 'tar_tar', 'cat_tar', 'f1', 'f2', 'DI'])

            # get overall TDR axes (grouping target / catch)
            tar = np.vstack([v[:, :, start:end].mean(axis=-1) for (k, v) in rec['resp'].extract_epochs(targets, mask=rec['mask']).items()])
            cat = np.vstack([v[:, :, start:end].mean(axis=-1) for (k, v) in rec['resp'].extract_epochs(catch, mask=rec['mask']).items()])
            m = np.concatenate((tar, cat), axis=0).mean(axis=0)
            sd = np.concatenate((tar, cat), axis=0).std(axis=0)
            sd[sd==0] = 1
            if not zscore:
                m = 0
                sd = 1
            tar = (tar - m) / sd
            cat = (cat - m) / sd
            tdr = TDR()
            tdr.fit(tar, cat)
            all_tdr_weights = tdr.weights

            # get first two PCs of REF space, and try decoding there
            dref = rec['resp'].extract_epochs(ref_stim, mask=rec['mask'])
            mpca = np.concatenate([dref[e][:, :, start:end] for e in dref.keys()], axis=0).mean(axis=-1).mean(axis=0)
            sdpc = np.concatenate([dref[e][:, :, start:end] for e in dref.keys()], axis=0).mean(axis=-1).std(axis=0)
            sdpc[sdpc==0] = 1
            if not zscore:
                mpca = 0
                sdpc = 1

            dref = {k: (v.transpose(0, -1, 1) - mpca).transpose(0, -1, 1)  for (k, v) in dref.items()}
            dref = {k: (v.transpose(0, -1, 1) / sdpc).transpose(0, -1, 1)  for (k, v) in dref.items()}
            Rall_u = np.vstack([dref[k].sum(axis=2).mean(axis=0) for k in dref.keys()])
            pca = PCA(n_components=2)
            pca.fit(Rall_u)
            pc_axes = pca.components_

            # plot projections for data into the PCA space and "all" TDR space

            # REF
            if plot_ref:
                for i, t in enumerate(ref_stim):
                    # ================================ TDR ==========================================
                    r1 = rec['resp'].extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1)
                    r1 = (r1 - m) / sd
                    r1 = r1.dot(all_tdr_weights.T).T
                    ax[0, 0].set_title('Active')
                    ax[0, 0].scatter(r1[0], r1[1], alpha=0.2, s=10, lw=0, color=BwG(i))
                    el = thelp.compute_ellipse(r1[0], r1[1])
                    ax[0, 0].plot(el[0], el[1], color=BwG(i), alpha=0.2, label=t.split('STIM_')[1])

                    r1 = rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, start:end].mean(axis=-1)
                    r1 = (r1 - m) / sd
                    r1 = r1.dot(all_tdr_weights.T).T
                    ax[0, 1].set_title('Passive')
                    ax[0, 1].scatter(r1[0], r1[1], alpha=0.2, s=10, lw=0, color=BwG(i))
                    el = thelp.compute_ellipse(r1[0], r1[1])
                    ax[0, 1].plot(el[0], el[1], color=BwG(i), alpha=0.2)

                    # =============================== PCA ========================================
                    r1 = rec['resp'].extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1)
                    r1 = (r1 - m) / sd
                    r1 = r1.dot(pc_axes.T).T
                    ax[1, 0].set_title('Active')
                    ax[1, 0].scatter(r1[0], r1[1], alpha=0.2, s=10, lw=0, color=BwG(i))
                    el = thelp.compute_ellipse(r1[0], r1[1])
                    ax[1, 0].plot(el[0], el[1], color=BwG(i), alpha=0.2)

                    r1 = rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, start:end].mean(axis=-1)
                    r1 = (r1 - m) / sd
                    r1 = r1.dot(pc_axes.T).T
                    ax[1, 1].set_title('Passive')
                    ax[1, 1].scatter(r1[0], r1[1], alpha=0.2, s=10, lw=0, color=BwG(i))
                    el = thelp.compute_ellipse(r1[0], r1[1])
                    ax[1, 1].plot(el[0], el[1], color=BwG(i), alpha=0.2)
                

            # TARGETS / CATCHES
            for i, t in enumerate(catch + targets):
                # ================================ TDR ==========================================
                r1 = rec['resp'].extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r1 = r1.dot(all_tdr_weights.T).T
                ax[0, 0].set_title('Active')
                ax[0, 0].scatter(r1[0], r1[1], alpha=1, s=10, lw=0, color=gR(i))
                el = thelp.compute_ellipse(r1[0], r1[1])
                ax[0, 0].plot(el[0], el[1], color=gR(i), label=t, lw=2)

                r1 = rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r1 = r1.dot(all_tdr_weights.T).T
                ax[0, 1].set_title('Passive')
                ax[0, 1].scatter(r1[0], r1[1], alpha=1, s=10, lw=0, color=gR(i))
                el = thelp.compute_ellipse(r1[0], r1[1])
                ax[0, 1].plot(el[0], el[1], color=gR(i), lw=2)

                # =============================== PCA ========================================
                r1 = rec['resp'].extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r1 = r1.dot(pc_axes.T).T
                ax[1, 0].set_title('Active')
                ax[1, 0].scatter(r1[0], r1[1], alpha=1, s=10, lw=0, color=gR(i))
                el = thelp.compute_ellipse(r1[0], r1[1])
                ax[1, 0].plot(el[0], el[1], color=gR(i), label=t, lw=2)

                r1 = rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r1 = r1.dot(pc_axes.T).T
                ax[1, 1].set_title('Passive')
                ax[1, 1].scatter(r1[0], r1[1], alpha=1, s=10, lw=0, color=gR(i))
                el = thelp.compute_ellipse(r1[0], r1[1])
                ax[1, 1].plot(el[0], el[1], color=gR(i), lw=2)


            ylims = (np.min([ax[0, 0].get_ylim()[0], ax[0, 1].get_ylim()[0]]), np.max([ax[0, 0].get_ylim()[1], ax[0, 1].get_ylim()[1]]))
            xlims = (np.min([ax[0, 0].get_xlim()[0], ax[0, 1].get_xlim()[0]]), np.max([ax[0, 0].get_xlim()[1], ax[0, 1].get_xlim()[1]]))
            ax[0, 0].set_xlim(xlims)
            ax[0, 0].set_ylim(ylims)
            ax[0, 1].set_xlim(xlims)
            ax[0, 1].set_ylim(ylims)

            ylims = (np.min([ax[1, 0].get_ylim()[0], ax[1, 1].get_ylim()[0]]), np.max([ax[1, 0].get_ylim()[1], ax[1, 1].get_ylim()[1]]))
            xlims = (np.min([ax[1, 0].get_xlim()[0], ax[1, 1].get_xlim()[0]]), np.max([ax[1, 0].get_xlim()[1], ax[1, 1].get_xlim()[1]]))
            ax[1, 0].set_xlim(xlims)
            ax[1, 0].set_ylim(ylims)
            ax[1, 1].set_xlim(xlims)
            ax[1, 1].set_ylim(ylims)


            # CATCH
            '''
            for i, t in enumerate(catch):
                # ================================ TDR ==========================================
                r1 = rec['resp'].extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r1 = r1.dot(all_tdr_weights.T).T
                ax[0, 0].set_title('Active')
                ax[0, 0].scatter(r1[0], r1[1], alpha=1, s=10, lw=0, color=cat_colors(i))
                el = thelp.compute_ellipse(r1[0], r1[1])
                ax[0, 0].plot(el[0], el[1], color=cat_colors(i), label=t, lw=2)

                r1 = rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r1 = r1.dot(all_tdr_weights.T).T
                ax[0, 1].set_title('Passive')
                ax[0, 1].scatter(r1[0], r1[1], alpha=1, s=10, lw=0, color=cat_colors(i))
                el = thelp.compute_ellipse(r1[0], r1[1])
                ax[0, 1].plot(el[0], el[1], color=cat_colors(i), lw=2)

                # =============================== PCA ========================================
                r1 = rec['resp'].extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r1 = r1.dot(pc_axes.T).T
                ax[1, 0].set_title('Active')
                ax[1, 0].scatter(r1[0], r1[1], alpha=1, s=10, lw=0, color=cat_colors(i))
                el = thelp.compute_ellipse(r1[0], r1[1])
                ax[1, 0].plot(el[0], el[1], color=cat_colors(i), label=t, lw=2)

                r1 = rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r1 = r1.dot(pc_axes.T).T
                ax[1, 1].set_title('Passive')
                ax[1, 1].scatter(r1[0], r1[1], alpha=1, s=10, lw=0, color=cat_colors(i))
                el = thelp.compute_ellipse(r1[0], r1[1])
                ax[1, 1].plot(el[0], el[1], color=cat_colors(i), lw=2)
            '''

            leg = ax[0, 0].legend(frameon=False, handlelength=0, bbox_to_anchor=(-0.05, 1.0), loc='upper right')
            for line, text in zip(leg.get_lines(), leg.get_texts()):
                text.set_color(line.get_color())
            ax[0, 0].set_xlabel(r"$TDR_1$ ($\Delta \mu$)")
            ax[0, 1].set_xlabel(r"$TDR_1$ ($\Delta \mu$)")
            ax[0, 0].set_ylabel(r"$TDR_2$")
            ax[0, 1].set_ylabel(r"$TDR_2$")

            ax[1, 0].set_xlabel(r"$PC_1$")
            ax[1, 1].set_xlabel(r"$PC_1$")
            ax[1, 0].set_ylabel(r"$PC_2$")
            ax[1, 1].set_ylabel(r"$PC_2$")

            f.tight_layout()

            if zscore:
                f.savefig(fpath + f'{site}{fext}_zscore.pdf')
            else:
                f.savefig(fpath + f'{site}{fext}.pdf')

            # get behavior performance for this site
            behavior_performance = manager.get_behavior_performance(**options)

            # for each pair, project into TDR (overall and pair-specific) and compute dprime
            for pair in pairs:
                idx = pair[0] + '_' + pair[1]
                snr1 = thelp.get_snrs([pair[0]])[0]
                snr2 = thelp.get_snrs([pair[1]])[0]
                f1 = thelp.get_tar_freqs([pair[0]])[0]
                f2 = thelp.get_tar_freqs([pair[1]])[0]
                cat_cat = ('CAT_' in pair[0]) & ('CAT_' in pair[1])
                tar_tar = ('TAR_' in pair[0]) & ('TAR_' in pair[1])
                cat_tar = (('CAT_' in pair[0]) & ('TAR_' in pair[1])) | (('CAT_' in pair[1]) & ('TAR_' in pair[0]))

                # get behavioral DI
                if 'TAR_' in pair[0]:
                    di = behavior_performance['LI'][pair[0].strip('TAR_').strip('CAT_')+'_'+pair[1].strip('TAR_').strip('CAT_')]
                else:
                    di = behavior_performance['LI'][pair[1].strip('TAR_').strip('CAT_')+'_'+pair[0].strip('TAR_').strip('CAT_')]

                # extract data over all trials for TDR
                r1 = rec['resp'].extract_epoch(pair[0], mask=rec['mask'])[:, :, start:end].mean(axis=-1)
                r2 = rec['resp'].extract_epoch(pair[1], mask=rec['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r2 = (r2 - m) / sd

                tdr = TDR()
                tdr.fit(r1, r2)
                pair_tdr_weights = tdr.weights

                # ================================= active data ======================================
                r1 = rec['resp'].extract_epoch(pair[0], mask=ra['mask'])[:, :, start:end].mean(axis=-1)
                r2 = rec['resp'].extract_epoch(pair[1], mask=ra['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r2 = (r2 - m) / sd

                # using overall tdr
                dp, wopt, evals, evecs, evec_sim, dU = compute_dprime(r1.dot(all_tdr_weights.T).T, r2.dot(all_tdr_weights.T).T)
                dp_diag, _, _, _, _, _ = compute_dprime(r1.dot(all_tdr_weights.T).T, r2.dot(all_tdr_weights.T).T, diag=True)
                df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evec_sim, dU, dp_diag, True, False, True, idx, snr1, snr2, cat_cat, tar_tar, cat_tar, f1, f2, di], \
                            index=['dp_opt', 'wopt', 'evecs', 'evec_sim', 'dU', 'dp_diag', 'tdr_overall', 'pca', 'active', 'pair', \
                                'snr1', 'snr2', 'cat_cat', 'tar_tar', 'cat_tar', 'f1', 'f2', 'DI']).T)

                
                # using pair-specific tdr
                dp, wopt, evals, evecs, evec_sim, dU = compute_dprime(r1.dot(pair_tdr_weights.T).T, r2.dot(pair_tdr_weights.T).T)
                dp_diag, _, _, _, _, _ = compute_dprime(r1.dot(pair_tdr_weights.T).T, r2.dot(pair_tdr_weights.T).T, diag=True)
                df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evec_sim, dU, dp_diag, False, False, True, idx, snr1, snr2, cat_cat, tar_tar, cat_tar, f1, f2, di], \
                            index=['dp_opt', 'wopt', 'evecs', 'evec_sim', 'dU', 'dp_diag', 'tdr_overall', 'pca', 'active', 'pair', \
                                'snr1', 'snr2', 'cat_cat', 'tar_tar', 'cat_tar', 'f1', 'f2', 'DI']).T)

                # using PCA
                dp, wopt, evals, evecs, evec_sim, dU = compute_dprime(r1.dot(pc_axes.T).T, r2.dot(pc_axes.T).T)
                dp_diag, _, _, _, _, _ = compute_dprime(r1.dot(pc_axes.T).T, r2.dot(pc_axes.T).T, diag=True)
                df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evec_sim, dU, dp_diag, False, True, True, idx, snr1, snr2, cat_cat, tar_tar, cat_tar, f1, f2, di], \
                            index=['dp_opt', 'wopt', 'evecs', 'evec_sim', 'dU', 'dp_diag', 'tdr_overall', 'pca', 'active', 'pair', \
                                'snr1', 'snr2', 'cat_cat', 'tar_tar', 'cat_tar', 'f1', 'f2', 'DI']).T)
            
            
                # ================================= passive data ======================================
                r1 = rec['resp'].extract_epoch(pair[0], mask=rp['mask'])[:, :, start:end].mean(axis=-1)
                r2 = rec['resp'].extract_epoch(pair[1], mask=rp['mask'])[:, :, start:end].mean(axis=-1)

                # using overall tdr
                dp, wopt, evals, evecs, evec_sim, dU = compute_dprime(r1.dot(all_tdr_weights.T).T, r2.dot(all_tdr_weights.T).T)
                dp_diag, _, _, _, _, _ = compute_dprime(r1.dot(all_tdr_weights.T).T, r2.dot(all_tdr_weights.T).T, diag=True)
                df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evec_sim, dU, dp_diag, True, False, False, idx, snr1, snr2, cat_cat, tar_tar, cat_tar, f1, f2, di], \
                            index=['dp_opt', 'wopt', 'evecs', 'evec_sim', 'dU', 'dp_diag', 'tdr_overall', 'pca', 'active', 'pair', \
                                'snr1', 'snr2', 'cat_cat', 'tar_tar', 'cat_tar', 'f1', 'f2', 'DI']).T)
                
                # using pair-specific tdr
                dp, wopt, evals, evecs, evec_sim, dU = compute_dprime(r1.dot(pair_tdr_weights.T).T, r2.dot(pair_tdr_weights.T).T)
                dp_diag, _, _, _, _, _ = compute_dprime(r1.dot(pair_tdr_weights.T).T, r2.dot(pair_tdr_weights.T).T, diag=True)
                df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evec_sim, dU, dp_diag, False, False, False, idx, snr1, snr2, cat_cat, tar_tar, cat_tar, f1, f2, di], \
                            index=['dp_opt', 'wopt', 'evecs', 'evec_sim', 'dU', 'dp_diag', 'tdr_overall', 'pca', 'active', 'pair', \
                                'snr1', 'snr2', 'cat_cat', 'tar_tar', 'cat_tar', 'f1', 'f2', 'DI']).T)

                # using PCA
                dp, wopt, evals, evecs, evec_sim, dU = compute_dprime(r1.dot(pc_axes.T).T, r2.dot(pc_axes.T).T)
                dp_diag, _, _, _, _, _ = compute_dprime(r1.dot(pc_axes.T).T, r2.dot(pc_axes.T).T, diag=True)
                df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evec_sim, dU, dp_diag, False, True, False, idx, snr1, snr2, cat_cat, tar_tar, cat_tar, f1, f2, di], \
                            index=['dp_opt', 'wopt', 'evecs', 'evec_sim', 'dU', 'dp_diag', 'tdr_overall', 'pca', 'active', 'pair', \
                                'snr1', 'snr2', 'cat_cat', 'tar_tar', 'cat_tar', 'f1', 'f2', 'DI']).T)

            df['site'] = site
            if batch==324: area='A1'
            else: area='PEG'
            df['area'] = area

            dfs.append(df)

df = pd.concat(dfs)
dtypes = {
    'dp_opt': 'float32',
    'wopt': 'object',
    'evecs': 'object',
    'evec_sim': 'float',
    'dU': 'object',
    'dp_diag': 'float32',
    'tdr_overall': 'bool',
    'pca': 'bool',
    'active': 'bool',
    'pair': 'object',
    'snr1': 'float',
    'snr2': 'float',
    'cat_cat': 'bool',
    'tar_tar': 'bool',
    'cat_tar': 'bool',
    'f1': 'int32',
    'f2': 'int32',
    'DI': 'float32'
    }
dtypes_new = {k: v for k, v in dtypes.items() if k in df.columns}
df = df.astype(dtypes_new)

df.to_csv('/home/charlie/Desktop/lbhb/code/projects/in_progress/TIN_behavior/res.csv')

plt.close('all')