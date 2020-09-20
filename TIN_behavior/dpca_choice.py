"""
Use dPCA to see if there's choice information
"""
from dPCA.dPCA import dPCA
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

# fig path
fpath = '/home/charlie/Desktop/lbhb/tmp_figures/tbp_decoding/'

# recording load options
options = {'resp': True, 'pupil': False, 'rasterfs': 10}
batches = [324, 325]
recache = False

# state-space projection options
zscore = False

# extract evoked periods for PCA
start = int(0.1 * options['rasterfs'])
end = int(0.4 * options['rasterfs'])
tend = int(0.5 * options['rasterfs'])

for batch in batches:
    sites = np.unique([c[:7] for c in nd.get_batch_cells(batch).cellid])
    sites = [s for s in sites if s!='CRD013b']
    for site in sites:
        print("Analyzing site: {}".format(site))
        manager = BAPHYExperiment(batch=batch, siteid=site)
        rec = manager.get_recording(recache=recache, **options)
        rec['resp'] = rec['resp'].rasterize()

        # mask appropriate trials
        rec = rec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'MISS_TRIAL'])
        
        ra = rec.copy()
        ra = ra.create_mask(True)
        ra = ra.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'MISS_TRIAL'])

        rp = rec.copy()
        rp = rp.create_mask(True)
        rp = rp.and_mask(['PASSIVE_EXPERIMENT'])

        _rp = rp.apply_mask(reset_epochs=True)
        _ra = ra.apply_mask(reset_epochs=True)

        # find / sort epoch names
        targets = thelp.sort_targets([f for f in _ra['resp'].epochs.name.unique() if 'TAR_' in f])
        catch = [f for f in _ra['resp'].epochs.name.unique() if 'CAT_' in f]
        ref_stim = thelp.sort_refs([f for f in _ra['resp'].epochs.name.unique() if 'STIM_' in f])
        rem = [f for f in rec['resp'].epochs.name.unique() if 'REM_' in f]
        sounds = targets + catch


        # perform dPCA -- need to get mean response to each target/catch in each condition
        
        # actually, this is confusing... what to do about targets that were never missed?

        # HIT
        hit = rec['resp'].extract_epochs(targets, mask=rec.and_mask(['HIT_TRIAL'])['mask'])
        hit = np.stack([v[:, :, 0:tend].mean(axis=0) for (k, v) in hit.items()])
        # MISS
        try:
            miss = rec['resp'].extract_epoch('MISS_TRIAL')
        except:
            miss = None
        # CORR. REJECT
        cmiss = rec['resp'].extract_epoch('CORRECT_REJECT_TRIAL')