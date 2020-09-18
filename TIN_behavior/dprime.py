"""
Quick and dirty quantify discriminability of pairwise combos of tars / catches.
In active / passive.
    deal with pupil?

define TDR over all stims? Or on pairwise basis?
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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# recording load options
options = {'resp': True, 'pupil': False, 'rasterfs': 10}
recache = True
# state-space projection options
zscore = False

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
    rec = manager.get_recording(recache=recache, **options)
    rec['resp'] = rec['resp'].rasterize()

    # mask appropriate trials
    rec = rec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'MISS_TRIAL'])

    # find / sort epoch names
    targets = [f for f in rec['resp'].epochs.name.unique() if 'TAR_' in f]
    catch = [f for f in rec['resp'].epochs.name.unique() if 'CAT_' in f]

    sounds = targets + catch

    # get all pairwise combos of targets / catches
    pairs = list(combinations(sounds, 2))

    # get overall TDR axes (grouping target / catch)
    tar = np.stack([v for (k, v) in rec['resp'].extract_epochs(targets, mask=rec['mask']).items()])
    cat = np.stack([v for (k, v) in rec['resp'].extract_epochs(catch, mask=rec['mask']).items()])

    # for each pair, project to TDR space (overall, or pair-specific) and compute dprime
    for pair in pairs: