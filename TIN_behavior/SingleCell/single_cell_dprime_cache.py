"""
Compute dprimes for single neurons in active / passive. Before / after removing first order pupil.
"""


from nems_lbhb.baphy_experiment import BAPHYExperiment
import charlieTools.baphy_remote as br
import charlieTools.noise_correlations as nc
import charlieTools.preprocessing as preproc
from charlieTools.plotting import compute_ellipse
from nems_lbhb.decoding import compute_dprime
from charlieTools.dim_reduction import TDR
import nems_lbhb.tin_helpers as thelp
from sklearn.decomposition import PCA
import nems.db as nd
from itertools import combinations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 14

# fig path
fpath = '/auto/users/hellerc/results/tmp_figures/tbp_decoding/SingleCells/'

# recording load options
options = {'resp': True, 'pupil': True, 'rasterfs': 10}
batches = [324, 325]
recache = False

# regress out first order pupil?
regress_pupil = False

# extract evoked periods
start = int(0.1 * options['rasterfs'])
end = int(0.4 * options['rasterfs'])

# siteids
df = pd.DataFrame()
for batch in batches:
    sites = np.unique([c[:7] for c in nd.get_batch_cells(batch).cellid])
    sites = [s for s in sites if s!='CRD013b']
    for site in sites:
        manager = BAPHYExperiment(batch=batch, siteid=site)
        rec = manager.get_recording(recache=recache, **options)
        rec['resp'] = rec['resp'].rasterize()

        # mask appropriate trials
        rec = rec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])
        rec = rec.apply_mask(reset_epochs=True)

        if regress_pupil:
            rec = preproc.regress_state(rec, state_sigs=['pupil'])

        ra = rec.copy()
        ra = ra.create_mask(True)
        ra = ra.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])

        rp = rec.copy()
        rp = rp.create_mask(True)
        rp = rp.and_mask(['PASSIVE_EXPERIMENT'])

        # stim pairs
        targets = thelp.sort_targets([f for f in ra['resp'].epochs.name.unique() if 'TAR_' in f])
        targets = [t for t in targets if (ra.apply_mask(reset_epochs=True)['resp'].epochs.name==t).sum()>=5]
        on_center = thelp.get_tar_freqs([f.strip('REM_') for f in ra['resp'].epochs.name.unique() if 'REM_' in f])[0]
        targets = [t for t in targets if str(on_center) in t]
        catch = [f for f in ra['resp'].epochs.name.unique() if 'CAT_' in f]
        catch = [c for c in catch if str(on_center) in c]
        ref_stim = thelp.sort_refs([f for f in ra['resp'].epochs.name.unique() if 'STIM_' in f])
        rem = [f for f in rec['resp'].epochs.name.unique() if 'REM_' in f]
        sounds = targets + catch
        pairs = list(combinations(['REFERENCE'] + ref_stim + sounds, 2))

        behavior_performance = manager.get_behavior_performance(**options)

        # Perform analysis for each cellid
        for cellid in ra['resp'].chans:
            print(f"Analyzing {cellid}")
            _ra = ra['resp'].extract_channels([cellid])
            _rp = rp['resp'].extract_channels([cellid])
            
            # plot target / catch distributions
            dat = pd.DataFrame()
            for ep in targets + catch:
                r1 = _ra.extract_epoch(ep, mask=ra['mask']).squeeze()[:, start:end].mean(axis=-1, keepdims=True)
                r2 = _ra.extract_epoch(ep, mask=rp['mask']).squeeze()[:, start:end].mean(axis=-1, keepdims=True)
                dat = pd.concat([dat, pd.DataFrame(np.stack([np.concatenate([r1, r2]).squeeze(), 
                                r1.shape[0]*['Active'] + r2.shape[0]*['Passive'], [ep]*(r1.shape[0] + r2.shape[0])]).T, columns=['Spike Count', 'state', 'Epoch'])])
                dat = dat.astype({'Spike Count': 'float32'})
            g = sns.FacetGrid(dat, col='state', hue='Epoch')
            g.map(sns.kdeplot, "Spike Count", fill=True)
            g.add_legend(frameon=False)
            g.fig.suptitle(cellid)
            g.fig.set_size_inches(8,4)
            g.fig.set_tight_layout(True)
            g.fig.savefig(fpath + f'{cellid}_tarDist.pdf')
            plt.close('all')

            _df = pd.DataFrame()
            for pair in pairs:
                if ('STIM_' in pair[0]) | ('REFERENCE' in pair[0]): snr1 = np.inf
                else: snr1 = thelp.get_snrs([pair[0]])[0]
                if ('STIM_' in pair[1]) | ('REFERENCE' in pair[1]): snr2 = np.inf
                else: snr2 = thelp.get_snrs([pair[1]])[0]
                
                if ('REFERENCE' in pair[0]): f1 = 0
                else: f1 = thelp.get_tar_freqs([pair[0].strip('STIM_')])[0]
                if 'REFERENCE' in pair[1]: f2 = 0
                else: f2 = thelp.get_tar_freqs([pair[1].strip('STIM_')])[0]

                # get behavioral DI
                if ('REFERENCE' in pair[0]) & (('TAR_' in pair[1]) | ('CAT_' in pair[1])):
                    di = behavior_performance['DI'][pair[1].strip('TAR_').strip('CAT_')]
                elif ('REFERENCE' in pair[1]) & (('TAR_' in pair[0]) | ('CAT_' in pair[0])):
                    di = behavior_performance['DI'][pair[0].strip('TAR_').strip('CAT_')]
                elif ('STIM_' in pair[0]) | ('STIM_' in pair[1]):
                    di = np.inf
                elif 'TAR_' in pair[0]:
                    di = behavior_performance['LI'][pair[0].strip('TAR_').strip('CAT_')+'_'+pair[1].strip('TAR_').strip('CAT_')]
                elif 'TAR_' in pair[1]:
                    di = behavior_performance['LI'][pair[1].strip('TAR_').strip('CAT_')+'_'+pair[0].strip('TAR_').strip('CAT_')]
                else:
                    di = np.inf 

                r1 = _ra.extract_epoch(pair[0], mask=ra['mask']).squeeze()[:, start:end].mean(axis=-1, keepdims=True)
                r2 = _ra.extract_epoch(pair[1], mask=ra['mask']).squeeze()[:, start:end].mean(axis=-1, keepdims=True)
                dpa = compute_dprime(r1.T, r2.T)
                _d = pd.DataFrame(data=[dpa], columns=['dprime'])
                _d['cat_tar'] = (('CAT_' in pair[0]) & ('TAR_' in pair[1])) | (('CAT_' in pair[1]) | ('TAR_' in pair[0]))
                _d['tar_tar'] = (('TAR_' in pair[0]) & ('TAR_' in pair[1]))
                _d['ref_tar'] = (('STIM_' in pair[0]) & ('TAR_' in pair[1])) | (('STIM_' in pair[1]) | ('TAR_' in pair[0]))
                _d['ref_cat'] = (('STIM_' in pair[0]) & ('CAT_' in pair[1])) | (('STIM_' in pair[1]) | ('CAT_' in pair[0]))
                _d['ref_ref'] = (('STIM_' in pair[0]) & ('STIM_' in pair[1]))
                _d['aref_ref'] = (('REFERENCE' in pair[0]) & ('STIM_' in pair[1])) | (('REFERENCE_' in pair[1]) | ('STIM_' in pair[0]))
                _d['aref_tar'] = (('REFERENCE' in pair[0]) & ('TAR_' in pair[1])) | (('REFERENCE_' in pair[1]) | ('TAR_' in pair[0]))
                _d['aref_cat'] = (('REFERENCE' in pair[0]) & ('CAT_' in pair[1])) | (('REFERENCE_' in pair[1]) | ('CAT_' in pair[0]))

                _d['active'] = True
                _d['pair'] = '_'.join(pair)
                _d['snr1'] = snr1
                _d['snr2'] = snr2
                _d['f1'] = f1
                _d['f2'] = f2
                _d['DI'] = di
                _d['cellid'] = cellid
                _d['site'] = site
                if batch==324: area='A1'
                else: area='PEG'
                _d['area'] = area
                
                _df = pd.concat([_df, _d])

                r1 = _rp.extract_epoch(pair[0], mask=rp['mask']).squeeze()[:, start:end].mean(axis=-1, keepdims=True)
                r2 = _rp.extract_epoch(pair[1], mask=rp['mask']).squeeze()[:, start:end].mean(axis=-1, keepdims=True)
                dpp = compute_dprime(r1.T, r2.T)

                _d = pd.DataFrame(data=[dpp], columns=['dprime'])
                _d['cat_tar'] = (('CAT_' in pair[0]) & ('TAR_' in pair[1])) | (('CAT_' in pair[1]) | ('TAR_' in pair[0]))
                _d['tar_tar'] = (('TAR_' in pair[0]) & ('TAR_' in pair[1]))
                _d['ref_tar'] = (('STIM_' in pair[0]) & ('TAR_' in pair[1])) | (('STIM_' in pair[1]) | ('TAR_' in pair[0]))
                _d['ref_cat'] = (('STIM_' in pair[0]) & ('CAT_' in pair[1])) | (('STIM_' in pair[1]) | ('CAT_' in pair[0]))
                _d['ref_ref'] = (('STIM_' in pair[0]) & ('STIM_' in pair[1]))
                _d['aref_ref'] = (('REFERENCE' in pair[0]) & ('STIM_' in pair[1])) | (('REFERENCE_' in pair[1]) | ('STIM_' in pair[0]))
                _d['aref_tar'] = (('REFERENCE' in pair[0]) & ('TAR_' in pair[1])) | (('REFERENCE_' in pair[1]) | ('TAR_' in pair[0]))
                _d['aref_cat'] = (('REFERENCE' in pair[0]) & ('CAT_' in pair[1])) | (('REFERENCE_' in pair[1]) | ('CAT_' in pair[0]))

                _d['active'] = False
                _d['pair'] = '_'.join(pair)
                _d['snr1'] = snr1
                _d['snr2'] = snr2
                _d['f1'] = f1
                _d['f2'] = f2
                _d['DI'] = di
                _d['cellid'] = cellid
                _d['site'] = site
                if batch==324: area='A1'
                else: area='PEG'
                _d['area'] = area

                _df = pd.concat([_df, _d])

            df = pd.concat([df, _df])

dtypes = {
    'dp': 'float32',
    'active': 'bool',
    'area': 'object',
    'pair': 'object',
    'snr1': 'float32',
    'snr2': 'float32',
    'cat_cat': 'bool',
    'tar_tar': 'bool',
    'cat_tar': 'bool',
    'ref_ref': 'bool',
    'ref_tar': 'bool',
    'ref_cat': 'bool',
    'aref_tar': 'bool',
    'aref_cat': 'bool',
    'aref_ref': 'bool',
    'f1': 'int32',
    'f2': 'int32',
    'DI': 'float32',
    'site': 'object',
    'cellid': 'object'
    }
dtypes_new = {k: v for k, v in dtypes.items() if k in df.columns}
df = df.astype(dtypes_new)
df.to_pickle('/auto/users/hellerc/code/projects/in_progress/TIN_behavior/singleCell_res.pickle')


