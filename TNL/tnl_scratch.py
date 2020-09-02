"""
Load / plot sample TNL data set.
"""

from nems_lbhb.baphy_experiment import BAPHYExperiment
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import charlieTools.plotting as cplt

zscore = True
pca_axis = True
freq_snr_axis = False  # either this or PCA should be true
if pca_axis & freq_snr_axis:
    raise ValueError
parmfile = '/auto/data/daq/Cordyceps/CRD002/CRD002a10_a_TNL.m'
rasterfs = 10
options = {'resp': True, 'pupil': True, 'rasterfs': rasterfs}

manager = BAPHYExperiment(parmfile)
rec = manager.get_recording(**options)
rec['resp'] = rec['resp'].rasterize()
rec = rec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)

# z-score spikes
if zscore:
    m = rec.apply_mask()['resp']._data.mean(axis=-1)
    sd = rec.apply_mask()['resp']._data.std(axis=-1)
    z = rec['resp']._data.T - m
    z /= sd
    rec['resp'] = rec['resp']._modified_copy(z.T)

# parse stimulus epochs to sort by tone SNR and frequency
stim_epochs = [s for s in rec.epochs.name.unique() if 'STIM_' in s]
snrs = np.unique([e.split('_')[-1].split('+')[1][:-2] for e in stim_epochs]).tolist()
non_inf = np.sort([s for s in snrs if 'Inf' not in s]).tolist()
minus_inf = [s for s in snrs if '-Inf' in s]
inf = [s for s in snrs if ('Inf' in s) & ('-Inf' not in s)]
snrs = minus_inf + non_inf + inf
cfs =  [str(cf) for cf in np.sort(np.unique([int(e.split('_')[-1].split('+')[0]) for e in stim_epochs]))]
noise_only = [s for s in stim_epochs if '-Inf' in s]
sidx = np.argsort([int(s.split('_')[1].split('+')[0]) for s in noise_only])
noise_only = np.array(noise_only)[sidx].tolist()

# plot tuning curves for each cell using the noise alone stimuli
ftc_data = []
for i, n in enumerate(rec['resp'].chans):
    # get mean response to each noise burst
    r = rec['resp'].extract_channels([n])
    d = r.extract_epochs(noise_only, mask=rec['mask'], allow_incomplete=True)
    ftc = []
    sem = []
    for e in d.keys():
        ftc.append(d[e].mean() * rasterfs) 
        sem.append(d[e].std() / np.sqrt(d[e].shape[0]))
    ftc_data.append(ftc)

ftc = np.stack(ftc_data)

# plot tuning curves by depth with heatmap
f, ax = plt.subplots(1, 1, figsize=(4, 8))

im = ax.imshow(ftc, cmap='Reds', aspect='auto')
f.colorbar(im, ax=ax)

ax.set_ylabel('Neuron sorted by depth')
ax.set_xlabel('CF (Hz)')
ax.set_xticks(range(0, ftc.shape[-1]))
ax.set_xticklabels(cfs, fontsize=8, rotation=45)
ax.set_yticks(range(0, ftc.shape[0]))
ax.set_yticklabels([r[8:] for r in rec['resp'].chans])
ax.set_title("Per-neuron FTC (cmap in units of Spk / sec)")

f.tight_layout()


# get 4D matrix: reps x neuron x freq x SNR
resp = np.zeros((rec['resp'].extract_epoch(noise_only[0]).shape[0],
                len(rec['resp'].chans),
                len(cfs),
                len(snrs)))
resp_bp = np.zeros((int(rec['resp'].extract_epoch(noise_only[0]).shape[0] / 2),
                len(rec['resp'].chans),
                len(cfs),
                len(snrs)))
resp_sp = np.zeros((int(rec['resp'].extract_epoch(noise_only[0]).shape[0] / 2),
                len(rec['resp'].chans),
                len(cfs),
                len(snrs)))
for i, f in enumerate(cfs):
    for j, snr in enumerate(snrs):
        epoch = [s for s in stim_epochs if (f in s) & (s.split('+')[1]==snr+'dB')][0]
        r = rec['resp'].extract_epoch(epoch, mask=rec['mask'], allow_incomplete=True)
        # mean over time bins for each neuron
        rm = r.mean(axis=(2))
        resp[:, :, i, j] = rm
        
        # split on median pupil for each stimulus
        p = rec['pupil'].extract_epoch(epoch, mask=rec['mask'], allow_incomplete=True)
        pm = p.mean(axis=-1).squeeze()
        median = np.median(pm)
        bidx = np.argwhere(pm > median).squeeze()
        sidx = np.argwhere(pm <= median).squeeze()

        resp_bp[:, :, i, j] = rm[bidx]
        resp_sp[:, :, i, j] = rm[sidx]

if freq_snr_axis:
    # marginalize over tone levels to get the "frequency axis"
    l_marg = np.mean(resp, axis=3).transpose([1, 0, -1]).reshape(len(rec['resp'].chans), -1)
    pca = PCA(n_components=1)
    pca.fit(l_marg.T)
    freq_axis = pca.components_

    # marginalize over frequencies to get the "SNR axis"
    f_marg = np.mean(resp, axis=2).transpose([1, 0, -1]).reshape(len(rec['resp'].chans), -1)
    pca = PCA(n_components=1)
    pca.fit(f_marg.T)
    snr_axis = pca.components_

    # define plane, make frequency the x-axis
    level_on_freq = (np.dot(snr_axis, freq_axis.T)) * freq_axis
    orth_ax = snr_axis - level_on_freq
    orth_ax /= np.linalg.norm(orth_ax)

    weights = np.concatenate((freq_axis, orth_ax), axis=0)
elif pca_axis:
    pca = PCA(n_components=2)
    pca.fit(resp.mean(axis=0).reshape(len(rec['resp'].chans), -1).T)
    weights = pca.components_

# project snr axis into this plane (for visualization)
snr_proj_axis = snr_axis.dot(weights.T)
snr_proj_axis *= 5

# project single trials onto plane
rec['proj_freq'] = rec['resp']._modified_copy(np.matmul(rec['resp']._data.T, weights[[0], :].T).T)
rec['proj2'] = rec['resp']._modified_copy(np.matmul(rec['resp']._data.T, weights[[1], :].T).T)

# plot by epoch (dot size = SNR, color = CF)
cmap = 'viridis'
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
colors = plt.cm.get_cmap(cmap, len(cfs))
dot_sizes = np.linspace(20, 100, len(snrs))
for i, f in enumerate(cfs):
    for j, snr in enumerate(snrs):
        # get single trial data for x and y axes
        epoch = [s for s in stim_epochs if (f in s) & (s.split('+')[1]==snr+'dB')][0]
        x = rec['proj_freq'].extract_epoch(epoch, mask=rec['mask'], allow_incomplete=True).mean(axis=(1, 2)) 
        y = rec['proj2'].extract_epoch(epoch, mask=rec['mask'], allow_incomplete=True).mean(axis=(1, 2)) 

        # plot this stimulus
        ax.scatter(x, y, s=10, color=colors(i), lw=0, alpha=0.2, zorder=0)
        el = cplt.compute_ellipse(x, y)
        ax.plot(el[0], el[1], color=colors(i), zorder=-2)
        if '-Inf' in snr:
            ax.scatter(x.mean(), y.mean(), s=dot_sizes[j], color=colors(i), edgecolor='k', zorder=-1)
        else:
            ax.scatter(x.mean(), y.mean(), s=dot_sizes[j], color=colors(i), edgecolor='white', zorder=-1)

# plot level axis
ax.plot([0, snr_proj_axis[0, 0]], [0, snr_proj_axis[0, 1]], lw=2, color='k', label='SNR axis')
ax.plot([0, -snr_proj_axis[0, 0]], [0, -snr_proj_axis[0, 1]], lw=2, color='k')

# plot 0s
ax.axhline(0, linestyle='--', zorder=0, color='grey')
ax.axvline(0, linestyle='--', zorder=0, color='grey')

if freq_snr_axis:
    ax.set_xlabel('Signal plane, Frequency Axis')
    ax.set_ylabel('Signal plane, axis orthogonal to Frequency Axis')
elif pca_axis:
    ax.set_xlabel(r"$PC_1$")
    ax.set_ylabel(r"$PC_2$")

norm = mpl.colors.Normalize(vmin=0, vmax=15)
divider = make_axes_locatable(ax)
cbarax = divider.append_axes('right', size='2%', pad=0.05)
cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cb1.set_label('CF (Hz)')
cb1.set_ticks(range(0, len(cfs)))
cb1.set_ticklabels(cfs)

ax.legend(frameon=False)

if zscore:
    ax.set_title('z-scored population responses')
else:
    ax.set_title('raw population responses')

fig.tight_layout()

# ======================= REDO BUT SPLIT BY PUPIL STATE =================================

cmap = 'viridis'
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
colors = plt.cm.get_cmap(cmap, len(cfs))
dot_sizes = np.linspace(20, 100, len(snrs))
# BIG PUPIL
if freq_snr_axis:
    # marginalize over tone levels to get the "frequency axis"
    l_marg = np.mean(resp_bp, axis=3).transpose([1, 0, -1]).reshape(len(rec['resp'].chans), -1)
    pca = PCA(n_components=1)
    pca.fit(l_marg.T)
    freq_axis = pca.components_

    # marginalize over frequencies to get the "SNR axis"
    f_marg = np.mean(resp_bp, axis=2).transpose([1, 0, -1]).reshape(len(rec['resp'].chans), -1)
    pca = PCA(n_components=1)
    pca.fit(f_marg.T)
    snr_axis = pca.components_

    # define plane, make frequency the x-axis
    level_on_freq = (np.dot(snr_axis, freq_axis.T)) * freq_axis
    orth_ax = snr_axis - level_on_freq
    orth_ax /= np.linalg.norm(orth_ax)

    weights = np.concatenate((freq_axis, orth_ax), axis=0)
elif pca_axis:
    pca = PCA(n_components=2)
    pca.fit(resp_bp.mean(axis=0).reshape(len(rec['resp'].chans), -1).T)
    weights = pca.components_

# project snr axis into this plane (for visualization)
snr_proj_axis = snr_axis.dot(weights.T)
snr_proj_axis *= 5

# project single trials onto plane
rec['proj_freq'] = rec['resp']._modified_copy(np.matmul(rec['resp']._data.T, weights[[0], :].T).T)
rec['proj2'] = rec['resp']._modified_copy(np.matmul(rec['resp']._data.T, weights[[1], :].T).T)

# plot by epoch (dot size = SNR, color = CF)
for i, f in enumerate(cfs):
    for j, snr in enumerate(snrs):
        # get single trial data for x and y axes
        epoch = [s for s in stim_epochs if (f in s) & (s.split('+')[1]==snr+'dB')][0]
        x = rec['proj_freq'].extract_epoch(epoch, mask=rec['mask'], allow_incomplete=True).mean(axis=(1, 2)) 
        y = rec['proj2'].extract_epoch(epoch, mask=rec['mask'], allow_incomplete=True).mean(axis=(1, 2)) 

        p = rec['pupil'].extract_epoch(epoch, mask=rec['mask'], allow_incomplete=True)
        pm = p.mean(axis=-1).squeeze()
        median = np.median(pm)
        bidx = np.argwhere(pm > median).squeeze()
        sidx = np.argwhere(pm <= median).squeeze()

        x = x[bidx]
        y = y[bidx]

        # plot this stimulus
        ax[0].scatter(x, y, s=10, color=colors(i), lw=0, alpha=0.2, zorder=0)
        el = cplt.compute_ellipse(x, y)
        ax[0].plot(el[0], el[1], color=colors(i), zorder=-2)
        if '-Inf' in snr:
            ax[0].scatter(x.mean(), y.mean(), s=dot_sizes[j], color=colors(i), edgecolor='k', zorder=-1)
        else:
            ax[0].scatter(x.mean(), y.mean(), s=dot_sizes[j], color=colors(i), edgecolor='white', zorder=-1)

# plot level axis
ax[0].plot([0, snr_proj_axis[0, 0]], [0, snr_proj_axis[0, 1]], lw=2, color='k', label='SNR axis')
ax[0].plot([0, -snr_proj_axis[0, 0]], [0, -snr_proj_axis[0, 1]], lw=2, color='k')

# plot 0s
ax[0].axhline(0, linestyle='--', zorder=0, color='grey')
ax[0].axvline(0, linestyle='--', zorder=0, color='grey')

if freq_snr_axis:
    ax[0].set_xlabel('Signal plane, Frequency Axis')
    ax[0].set_ylabel('Signal plane, axis orthogonal to Frequency Axis')
elif pca_axis:
    ax[0].set_xlabel(r"$PC_1$")
    ax[0].set_ylabel(r"$PC_2$")

norm = mpl.colors.Normalize(vmin=0, vmax=15)
divider = make_axes_locatable(ax[0])
cbarax = divider.append_axes('right', size='2%', pad=0.05)
cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cb1.set_label('CF (Hz)')
cb1.set_ticks(range(0, len(cfs)))
cb1.set_ticklabels(cfs)

ax[0].legend(frameon=False)

if zscore:
    ax[0].set_title('Big pupil \n z-scored population responses')
else:
    ax[0].set_title('Small pupil \n raw population responses')



# SMALL PUPIL
if freq_snr_axis:
    # marginalize over tone levels to get the "frequency axis"
    l_marg = np.mean(resp_bp, axis=3).transpose([1, 0, -1]).reshape(len(rec['resp'].chans), -1)
    pca = PCA(n_components=1)
    pca.fit(l_marg.T)
    freq_axis = pca.components_

    # marginalize over frequencies to get the "SNR axis"
    f_marg = np.mean(resp_sp, axis=2).transpose([1, 0, -1]).reshape(len(rec['resp'].chans), -1)
    pca = PCA(n_components=1)
    pca.fit(f_marg.T)
    snr_axis = pca.components_

    # define plane, make frequency the x-axis
    level_on_freq = (np.dot(snr_axis, freq_axis.T)) * freq_axis
    orth_ax = snr_axis - level_on_freq
    orth_ax /= np.linalg.norm(orth_ax)

    weights = np.concatenate((freq_axis, orth_ax), axis=0)
elif pca_axis:
    pca = PCA(n_components=2)
    pca.fit(resp_sp.mean(axis=0).reshape(len(rec['resp'].chans), -1).T)
    weights = pca.components_

# project snr axis into this plane (for visualization)
snr_proj_axis = snr_axis.dot(weights.T)
snr_proj_axis *= 5

# project single trials onto plane
rec['proj_freq'] = rec['resp']._modified_copy(np.matmul(rec['resp']._data.T, weights[[0], :].T).T)
rec['proj2'] = rec['resp']._modified_copy(np.matmul(rec['resp']._data.T, weights[[1], :].T).T)

# plot by epoch (dot size = SNR, color = CF)
for i, f in enumerate(cfs):
    for j, snr in enumerate(snrs):
        # get single trial data for x and y axes
        epoch = [s for s in stim_epochs if (f in s) & (s.split('+')[1]==snr+'dB')][0]
        x = rec['proj_freq'].extract_epoch(epoch, mask=rec['mask'], allow_incomplete=True).mean(axis=(1, 2)) 
        y = rec['proj2'].extract_epoch(epoch, mask=rec['mask'], allow_incomplete=True).mean(axis=(1, 2)) 

        p = rec['pupil'].extract_epoch(epoch, mask=rec['mask'], allow_incomplete=True)
        pm = p.mean(axis=-1).squeeze()
        median = np.median(pm)
        bidx = np.argwhere(pm > median).squeeze()
        sidx = np.argwhere(pm <= median).squeeze()

        x = x[sidx]
        y = y[sidx]

        # plot this stimulus
        ax[1].scatter(x, y, s=10, color=colors(i), lw=0, alpha=0.2, zorder=0)
        el = cplt.compute_ellipse(x, y)
        ax[1].plot(el[0], el[1], color=colors(i), zorder=-2)
        if '-Inf' in snr:
            ax[1].scatter(x.mean(), y.mean(), s=dot_sizes[j], color=colors(i), edgecolor='k', zorder=-1)
        else:
            ax[1].scatter(x.mean(), y.mean(), s=dot_sizes[j], color=colors(i), edgecolor='white', zorder=-1)

# plot level axis
ax[1].plot([0, snr_proj_axis[0, 0]], [0, snr_proj_axis[0, 1]], lw=2, color='k', label='SNR axis')
ax[1].plot([0, -snr_proj_axis[0, 0]], [0, -snr_proj_axis[0, 1]], lw=2, color='k')

# plot 0s
ax[1].axhline(0, linestyle='--', zorder=0, color='grey')
ax[1].axvline(0, linestyle='--', zorder=0, color='grey')

if freq_snr_axis:
    ax[1].set_xlabel('Signal plane, Frequency Axis')
    ax[1].set_ylabel('Signal plane, axis orthogonal to Frequency Axis')
elif pca_axis:
    ax[1].set_xlabel(r"$PC_1$")
    ax[1].set_ylabel(r"$PC_2$")

norm = mpl.colors.Normalize(vmin=0, vmax=15)
divider = make_axes_locatable(ax[1])
cbarax = divider.append_axes('right', size='2%', pad=0.05)
cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cb1.set_label('CF (Hz)')
cb1.set_ticks(range(0, len(cfs)))
cb1.set_ticklabels(cfs)

ax[1].legend(frameon=False)

if zscore:
    ax[1].set_title('Small pupil \n z-scored population responses')
else:
    ax[1].set_title('Small pupil \n raw population responses')


fig.tight_layout()


plt.show()