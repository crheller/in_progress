"""
Goal is to see if we can show evidence for learning during the TIN behavior (TBP with variable SNR, single target)

Date that we switched back (from two tone reward learning to single tone) was 8.5.2020. Behavior was crummy on 8.5, though.
Start from 08.06.2020
"""

import charlieTools.TIN_behavior.tin_helpers as thelp

import statistics
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems.db as nd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
import datetime as dt

plot_rt_histograms = False

# recording load options
options = {"resp": False, "pupil": False, "rasterfs": 20}

runclass = 'TBP'
ed = '2020-08-05'
ld = '2020-09-10'
ed = dt.datetime.strptime(ed, '%Y-%m-%d')
ld = dt.datetime.strptime(ld, '%Y-%m-%d')

# get list of parmfiles and sort by date (all files from one day go into one analysis??)
sql = "SELECT gDataRaw.resppath, gDataRaw.parmfile, pendate FROM gPenetration INNER JOIN gCellMaster ON (gCellMaster.penid = gPenetration.id)"\
                " INNER JOIN gDataRaw ON (gCellMaster.id = gDataRaw.masterid) WHERE" \
                " gDataRaw.runclass='{0}' and gDataRaw.bad=0 and gDataRaw.trials>50 and"\
                " gPenetration.animal = 'Cordyceps' and gDataRaw.behavior='active'".format(runclass)
d = nd.pd_query(sql)
d['date'] = d['pendate'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))

# screen for date
d = d[(d['date'] >= ed) & (d['date'] <= ld)]
d = d.sort_values(by='date')

# join path
d['parmfile_path'] = [os.path.join(d['resppath'].iloc[i], d['parmfile'].iloc[i]) for i in range(d.shape[0])]

# define set of unique dates
uDate = d['date'].unique()

# make some empty lists / arrays to hold results

# LI (i.e. DI between targets and catch)
# for now, only save results for the onCF masker (not off-center masker)
# assume levels are [-10, -5, 0, Inf]
all_snrs = np.array([-10, -5, 0, np.inf])
LI_EARLY = np.full((4, len(uDate)), np.nan)
LI_LATE = np.full((4, len(uDate)), np.nan)
LI_ALL = np.full((4, len(uDate)), np.nan)

for idx, ud in enumerate(uDate):
    parmfiles = d[d.date==ud].parmfile_path.values.tolist()
    # add catch to make sure "siteid" the same for all files
    sid = [p.split(os.path.sep)[-1][:7] for p in parmfiles]
    if np.any(np.array(sid) != sid[0]):
        bad_idx = (np.array(sid)!=sid[0])
        parmfiles = np.array(parmfiles)[~bad_idx].tolist()
    manager = BAPHYExperiment(parmfiles)

    # make sure only loaded actives
    pf_mask = [True if k['BehaveObjectClass']=='RewardTargetLBHB' else False for k in manager.get_baphy_exptparams()]
    if sum(pf_mask) == len(manager.parmfile):
        pass
    else:
        parmfiles = np.array(manager.parmfile)[pf_mask].tolist()
        manager = BAPHYExperiment(parmfiles)

    rec = manager.get_recording(**options)
    rec = rec.and_mask(['ACTIVE_EXPERIMENT'])

    # define overlapping trial windows for behavior metrics
    # divide trials into quantiles (25/50/75 percentile)
    nTrials = rec['fileidx'].extract_epoch('TRIAL', mask=rec['mask']).shape[0]
    edges = np.quantile(range(nTrials), [.25, .5, .75]).astype(int)
    # then define three overlapping ranges. Early, middle, late
    t1 = range(1, edges[1])
    t2 = range(edges[0], edges[2])
    t3 = range(edges[1], nTrials)


    # =================================== plot RT histogram =========================================
    # plot an overall RT histogram
    if plot_rt_histograms:
        bins = np.arange(0, 3, 0.1)
        # leave behavior options as defaults, if want to include
        # invalid trials, update options dict with appropriate vals
        f, ax = plt.subplots(1, 4, figsize=(16, 4))
        for i, trials in enumerate([t1, t2, t3, range(nTrials)]):
            manager.plot_RT_histogram(trials=trials, bins=bins, ax=ax[i], **options)
            ax[i].set_title(f"Trials {min(trials)}:{max(trials)}", fontsize=8)

        f.tight_layout()

        f.canvas.set_window_title(str(ud)[:10])


    # ============================== plot DI between targets / catch ===============================
    # for each target, get its DI relative to Catch
    targets = thelp.sort_targets([t.split('_')[1] for t in rec.epochs.name.unique() if 'TAR_' in t])
    snrs = thelp.get_snrs(targets)
    freqs = thelp.get_tar_freqs(targets)
    catch = thelp.sort_targets([t.split('_')[1] for t in rec.epochs.name.unique() if 'CAT_' in t])
    # if multiple +Inf targets (reminder trials), drop the reminder
    if sum(np.array(snrs)==np.inf) > 1:
        drop_idx = np.argwhere(np.array(snrs)==np.inf)[-1]
        keep_idx = np.array(list(set(range(len(snrs))).difference(set(drop_idx))))
        targets = np.array(targets)[keep_idx].tolist()
        snrs = np.array(snrs)[keep_idx].tolist()
        freqs = np.array(freqs)[keep_idx].tolist()

    # get performance in the three windows (and overall)
    perf_all = manager.get_behavior_performance(**options)
    perf_early = manager.get_behavior_performance(trials=t1, **options)
    perf_late = manager.get_behavior_performance(trials=t3, **options)
    li = []
    li_early = []
    li_late = []
    if len(catch)==1:
        for t in targets:
            li.append(perf_all['LI'][t+'_'+catch[0]])
            li_early.append(perf_early['LI'][t+'_'+catch[0]])
            li_late.append(perf_late['LI'][t+'_'+catch[0]])
    elif len(catch)==2:
        # only compute LI for the "on-BF" targets vs. on-BF catch -- discard off-center masker for this analysis
        f = statistics.mode(freqs)
        c = np.array(catch)[f==np.array(thelp.get_tar_freqs(catch))].tolist()
        targets = np.array(targets)[f==np.array(freqs)].tolist()
        snrs = np.array(snrs)[f==np.array(freqs)].tolist()
        for t in targets:
            li.append(perf_all['LI'][t+'_'+catch[0]])
            li_early.append(perf_early['LI'][t+'_'+catch[0]])
            li_late.append(perf_late['LI'][t+'_'+catch[0]])


    f, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.plot(range(len(targets)), li, 'o-', label='All trials')
    ax.plot(range(len(targets)), li_early, 'o-', label='Early')
    ax.plot(range(len(targets)), li_late, 'o-', label='Late')
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, fontsize=6)
    ax.set_xlabel('Target')
    ax.set_ylabel('Tar vs. Catch DI')
    ax.legend(frameon=False, fontsize=8)

    ax.set_title(str(ud)[:10], fontsize=10)

    f.tight_layout()


    snr_idx = [np.argwhere(all_snrs==s)[0][0] for s in snrs]

    nValidTrials_early = np.sum([v for k,v in perf_early['nTrials'].items() if k!='Reference'])
    nValidTrials_late = np.sum([v for k,v in perf_late['nTrials'].items() if k!='Reference'])
    if (nValidTrials_early >= 20) & (nValidTrials_late >= 20):
        # only save days with sufficient data
        LI_ALL[snr_idx, idx] = li
        LI_EARLY[snr_idx, idx] = li_early
        LI_LATE[snr_idx, idx] = li_late


# summary plot of LI
snr_strings = ['-10', '-5', '0', 'Inf']
f, ax = plt.subplots(1, 1, figsize=(5, 4))

m = np.nanmean(LI_ALL, axis=-1)
mse = np.nanstd(LI_ALL, axis=-1) / np.sqrt(LI_ALL.shape[-1])
m_early = np.nanmean(LI_EARLY, axis=-1)
mse_early = np.nanstd(LI_EARLY, axis=-1) / np.sqrt(LI_EARLY.shape[-1])
m_late = np.nanmean(LI_LATE, axis=-1)
mse_late = np.nanstd(LI_LATE, axis=-1) / np.sqrt(LI_LATE.shape[-1])

ax.plot(range(len(snr_strings)), m, 'o-', label='All Trials')
ax.fill_between(range(len(snr_strings)), m-mse, m+mse, alpha=0.3)

ax.plot(range(len(snr_strings)), m_early, 'o-', label='Early')
ax.fill_between(range(len(snr_strings)), m_early-mse_early, m_early+mse_early, alpha=0.3)

ax.plot(range(len(snr_strings)), m_late, 'o-', label='Late')
ax.fill_between(range(len(snr_strings)), m_late-mse_late, m_late+mse_late, alpha=0.3)

ax.set_xticks(range(len(snr_strings)))
ax.set_xticklabels(snr_strings)
ax.set_xlabel('Target SNR')

ax.set_ylabel('Target vs. Catch DI')

ax.legend(frameon=False, fontsize=10)

f.tight_layout()

plt.show()