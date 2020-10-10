"""
For each site, find the delta noise correlation axis. Compute in 3 different ways:
    over all stimuli
    over targets only
    over targets + catches
For each, plot:
    active / passive / difference covariance matrices
    eigendecomposition of difference, along with noise floor
    similary matrix of the first eigenvector for each method
"""
from nems_lbhb.baphy_experiment import BAPHYExperiment
import charlieTools.baphy_remote as br
import charlieTools.noise_correlations as nc
import charlieTools.preprocessing as preproc
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

options = {'resp': True, 'pupil': True, 'rasterfs': 10}
batches = [324, 325]
recache = False

site = 'CRD010b'
batch = 325
cmap = 'bwr'
nshuff = 10

# extract evoked period decision window only (and collapse over this for nc measurements)
# TODO: might want to also try different time windows (like with the PTD data)
start = int(0.1 * options['rasterfs'])
end = int(0.3 * options['rasterfs'])

manager = BAPHYExperiment(batch=batch, siteid=site)
rec = manager.get_recording(recache=recache, **options)
rec['resp'] = rec['resp'].rasterize()
rec = rec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])

# mask appropriate trials
ra = rec.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])
rp = rec.and_mask(['PASSIVE_EXPERIMENT'])
targets = thelp.sort_targets([f for f in ra['resp'].epochs.name.unique() if 'TAR_' in f])
catch = thelp.sort_targets([f for f in ra['resp'].epochs.name.unique() if 'CAT_' in f])
# keep only on-center tar / cat
oncenter = thelp.get_tar_freqs([f.strip('REM_') for f in ra['resp'].epochs.name.unique() if 'REM_' in f])[0]
targets = [t for t in targets if thelp.get_tar_freqs([t])[0]==oncenter]
catch = [t for t in catch if thelp.get_tar_freqs([t])[0]==oncenter]
all_stim = [s for s in ra['resp'].epochs.name.unique() if 'STIM_' in s] + targets + catch


# set up figure
f, ax = plt.subplots(3, 4, figsize=(12, 9))
acov1, pcov1, dcov1, e1, acov2, pcov2, dcov2, e2, acov3, pcov3, dcov3, e3 = ax.flatten()

EVECS = []
# ================================= TARGET ONLY ======================================
respa = []
respp = []
for t in targets:
    _r = rec['resp'].extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1, keepdims=True)
    m = _r.mean(axis=0)
    sd = _r.std(axis=0)
    sd[sd==0] = 1
    _r = (_r - m) / sd
    respa.append(_r)

    _r = rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, start:end].mean(axis=-1, keepdims=True)
    m = _r.mean(axis=0)
    sd = _r.std(axis=0)
    sd[sd==0] = 1
    _r = (_r - m) / sd
    respp.append(_r)
respa = np.concatenate(respa, axis=0).squeeze()
respp = np.concatenate(respp, axis=0).squeeze()

AMAT = np.cov(respa.T); np.fill_diagonal(AMAT, 0)
PMAT = np.cov(respp.T); np.fill_diagonal(PMAT, 0)
DIFF = PMAT - AMAT

evals, evecs = np.linalg.eig(DIFF)
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

EVECS.append(evecs[:, 0])

acov1.imshow(AMAT, aspect='auto', cmap=cmap, vmin=-1, vmax=1)
pcov1.imshow(PMAT, aspect='auto', cmap=cmap, vmin=-1, vmax=1)
dcov1.imshow(DIFF, aspect='auto', cmap=cmap, vmin=-1, vmax=1)
e1.plot(evals, '.-', label='Raw data')
e1.axhline(0, linestyle='--', color='grey', lw=1)
acov1.set_title(r"$\Sigma$ Active")
pcov1.set_title(r"$\Sigma$ Passive")
dcov1.set_title("Difference (Passive - Active)")
e1.set_title("Eigendecomposition \n of difference")
e1.set_ylabel(r"$\lambda$")
e1.set_xlabel(r"$\alpha$")

# get noise floor
shuff_evals = []
for i in range(nshuff):
    respa = []
    respp = []
    for t in targets:
        areps = rec['resp'].extract_epoch(t, mask=ra['mask']).shape[0]
        preps = rec['resp'].extract_epoch(t, mask=rp['mask']).shape[0]
        idx1 = np.random.choice(np.arange(0, areps+preps), areps)
        idx2 = np.array(list(set(np.arange(0, areps+preps)).difference(set(idx1))))
        _r = rec['resp'].extract_epoch(t, mask=rec['mask'])[idx1, :, start:end].mean(axis=-1, keepdims=True)
        m = _r.mean(axis=0)
        sd = _r.std(axis=0)
        sd[sd==0] = 1
        _r = (_r - m) / sd
        respa.append(_r)

        _r = rec['resp'].extract_epoch(t, mask=rec['mask'])[idx2, :, start:end].mean(axis=-1, keepdims=True)
        m = _r.mean(axis=0)
        sd = _r.std(axis=0)
        sd[sd==0] = 1
        _r = (_r - m) / sd
        respp.append(_r)
    respa = np.concatenate(respa, axis=0).squeeze()
    respp = np.concatenate(respp, axis=0).squeeze()

    AMAT = np.cov(respa.T); np.fill_diagonal(AMAT, 0)
    PMAT = np.cov(respp.T); np.fill_diagonal(PMAT, 0)
    DIFF = PMAT - AMAT

    evals, evecs = np.linalg.eig(DIFF)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    shuff_evals.append(evals)

seval_mean = np.stack(shuff_evals).mean(axis=0)
seval_sem = np.stack(shuff_evals).std(axis=0) #/ np.sqrt(nshuff)
e1.fill_between(np.arange(0, evals.shape[0]), seval_mean - seval_sem, seval_mean+seval_sem, color='tab:orange', lw=0, label='Noise Floor')
e1.legend(frameon=False)
# ============================= TARGET + CATCH ONLY ==================================
respa = []
respp = []
for t in targets+catch:
    _r = rec['resp'].extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1, keepdims=True)
    m = _r.mean(axis=0)
    sd = _r.std(axis=0)
    sd[sd==0] = 1
    _r = (_r - m) / sd
    respa.append(_r)

    _r = rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, start:end].mean(axis=-1, keepdims=True)
    m = _r.mean(axis=0)
    sd = _r.std(axis=0)
    sd[sd==0] = 1
    _r = (_r - m) / sd
    respp.append(_r)
respa = np.concatenate(respa, axis=0).squeeze()
respp = np.concatenate(respp, axis=0).squeeze()

AMAT = np.cov(respa.T); np.fill_diagonal(AMAT, 0)
PMAT = np.cov(respp.T); np.fill_diagonal(PMAT, 0)
DIFF = PMAT - AMAT

evals, evecs = np.linalg.eig(DIFF)
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

EVECS.append(evecs[:, 0])

acov2.imshow(AMAT, aspect='auto', cmap=cmap, vmin=-1, vmax=1)
pcov2.imshow(PMAT, aspect='auto', cmap=cmap, vmin=-1, vmax=1)
dcov2.imshow(DIFF, aspect='auto', cmap=cmap, vmin=-1, vmax=1)
e2.plot(evals, '.-', label='Raw data')
e2.axhline(0, linestyle='--', color='grey', lw=1)
acov2.set_title(r"$\Sigma$ Active")
pcov2.set_title(r"$\Sigma$ Passive")
dcov2.set_title("Difference (Passive - Active)")
e2.set_title("Eigendecomposition \n of difference")
e2.set_ylabel(r"$\lambda$")
e2.set_xlabel(r"$\alpha$")

# get noise floor
shuff_evals = []
for i in range(nshuff):
    respa = []
    respp = []
    for t in targets+catch:
        areps = rec['resp'].extract_epoch(t, mask=ra['mask']).shape[0]
        preps = rec['resp'].extract_epoch(t, mask=rp['mask']).shape[0]
        idx1 = np.random.choice(np.arange(0, areps+preps), areps)
        idx2 = np.array(list(set(np.arange(0, areps+preps)).difference(set(idx1))))
        _r = rec['resp'].extract_epoch(t, mask=rec['mask'])[idx1, :, start:end].mean(axis=-1, keepdims=True)
        m = _r.mean(axis=0)
        sd = _r.std(axis=0)
        sd[sd==0] = 1
        _r = (_r - m) / sd
        respa.append(_r)

        _r = rec['resp'].extract_epoch(t, mask=rec['mask'])[idx2, :, start:end].mean(axis=-1, keepdims=True)
        m = _r.mean(axis=0)
        sd = _r.std(axis=0)
        sd[sd==0] = 1
        _r = (_r - m) / sd
        respp.append(_r)
    respa = np.concatenate(respa, axis=0).squeeze()
    respp = np.concatenate(respp, axis=0).squeeze()

    AMAT = np.cov(respa.T); np.fill_diagonal(AMAT, 0)
    PMAT = np.cov(respp.T); np.fill_diagonal(PMAT, 0)
    DIFF = PMAT - AMAT

    evals, evecs = np.linalg.eig(DIFF)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    shuff_evals.append(evals)

seval_mean = np.stack(shuff_evals).mean(axis=0)
seval_sem = np.stack(shuff_evals).std(axis=0) #/ np.sqrt(nshuff)
e2.fill_between(np.arange(0, evals.shape[0]), seval_mean - seval_sem, seval_mean+seval_sem, color='tab:orange', lw=0, label='Noise Floor')
e2.legend(frameon=False)

# ================================== ALL STIM ========================================
respa = []
respp = []
for t in all_stim:
    _r = rec['resp'].extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1, keepdims=True)
    m = _r.mean(axis=0)
    sd = _r.std(axis=0)
    sd[sd==0] = 1
    _r = (_r - m) / sd
    respa.append(_r)

    _r = rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, start:end].mean(axis=-1, keepdims=True)
    m = _r.mean(axis=0)
    sd = _r.std(axis=0)
    sd[sd==0] = 1
    _r = (_r - m) / sd
    respp.append(_r)
respa = np.concatenate(respa, axis=0).squeeze()
respp = np.concatenate(respp, axis=0).squeeze()

AMAT = np.cov(respa.T); np.fill_diagonal(AMAT, 0)
PMAT = np.cov(respp.T); np.fill_diagonal(PMAT, 0)
DIFF = PMAT - AMAT

evals, evecs = np.linalg.eig(DIFF)
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

EVECS.append(evecs[:, 0])

acov3.imshow(AMAT, aspect='auto', cmap=cmap, vmin=-1, vmax=1)
pcov3.imshow(PMAT, aspect='auto', cmap=cmap, vmin=-1, vmax=1)
dcov3.imshow(DIFF, aspect='auto', cmap=cmap, vmin=-1, vmax=1)
e3.plot(evals, '.-', label='Raw data')
e3.axhline(0, linestyle='--', color='grey', lw=1)
acov3.set_title(r"$\Sigma$ Active")
pcov3.set_title(r"$\Sigma$ Passive")
dcov3.set_title("Difference (Passive - Active)")
e3.set_title("Eigendecomposition \n of difference")
e3.set_ylabel(r"$\lambda$")
e3.set_xlabel(r"$\alpha$")

# get noise floor
shuff_evals = []
for i in range(nshuff):
    respa = []
    respp = []
    for t in all_stim:
        areps = rec['resp'].extract_epoch(t, mask=ra['mask']).shape[0]
        preps = rec['resp'].extract_epoch(t, mask=rp['mask']).shape[0]
        idx1 = np.random.choice(np.arange(0, areps+preps), areps)
        idx2 = np.array(list(set(np.arange(0, areps+preps)).difference(set(idx1))))
        _r = rec['resp'].extract_epoch(t, mask=rec['mask'])[idx1, :, start:end].mean(axis=-1, keepdims=True)
        m = _r.mean(axis=0)
        sd = _r.std(axis=0)
        sd[sd==0] = 1
        _r = (_r - m) / sd
        respa.append(_r)

        _r = rec['resp'].extract_epoch(t, mask=rec['mask'])[idx2, :, start:end].mean(axis=-1, keepdims=True)
        m = _r.mean(axis=0)
        sd = _r.std(axis=0)
        sd[sd==0] = 1
        _r = (_r - m) / sd
        respp.append(_r)
    respa = np.concatenate(respa, axis=0).squeeze()
    respp = np.concatenate(respp, axis=0).squeeze()

    AMAT = np.cov(respa.T); np.fill_diagonal(AMAT, 0)
    PMAT = np.cov(respp.T); np.fill_diagonal(PMAT, 0)
    DIFF = PMAT - AMAT

    evals, evecs = np.linalg.eig(DIFF)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    shuff_evals.append(evals)

seval_mean = np.stack(shuff_evals).mean(axis=0)
seval_sem = np.stack(shuff_evals).std(axis=0) #/ np.sqrt(nshuff)
e3.fill_between(np.arange(0, evals.shape[0]), seval_mean - seval_sem, seval_mean+seval_sem, color='tab:orange', lw=0, label='Noise Floor')
e3.legend(frameon=False)

f.tight_layout()

f.canvas.set_window_title(site)

# ============================== SIMILARITY MATRIX ================================
SIM = abs(np.stack(EVECS).dot(np.stack(EVECS).T))

f, ax = plt.subplots(1, 1, figsize=(5, 4))

pcm = ax.imshow(SIM, aspect='auto', cmap='Reds', vmax=1)
ax.set_title("First eigenvector\nsimilarity for each method")
f.colorbar(pcm, ax=ax)

f.tight_layout()
f.canvas.set_window_title(site)

plt.show()