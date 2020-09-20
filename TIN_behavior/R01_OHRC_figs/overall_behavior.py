"""
Overall behavior (shown with cum. RT histogram)

Also generate RTs for early/middle/late epochs. See if any evidence of learning.
"""

"""
Goal is to see if we can show evidence for learning during the TIN behavior (TBP with variable SNR, single target)

Date that we switched back (from two tone reward learning to single tone) was 8.5.2020. Behavior was crummy on 8.5, though.
Start from 08.06.2020
"""

import charlieTools.TIN_behavior.tin_helpers as thelp

import statistics
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.runclass import TBP
import nems.db as nd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
import datetime as dt

# one figure for overall behavior, one for learning
figsave1 = '/home/charlie/Desktop/lbhb/code/projects/in_progress/TIN_behavior/R01_OHRC_figs/behavior1.pdf'
figsave2 = '/home/charlie/Desktop/lbhb/code/projects/in_progress/TIN_behavior/R01_OHRC_figs/behavior2.pdf'

import charlieTools.TIN_behavior.tin_helpers as thelp

import statistics
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.behavior import get_reaction_times
from nems_lbhb.behavior_plots import plot_RT_histogram
import nems.db as nd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
import datetime as dt

rel = True       # define early / middle / late trials with relative trial number (e.g. first 50% of trials during that session)
absolute = True # define early / middle / late trials with absolute trial number

# recording load options
options = {"resp": False, "pupil": False, "rasterfs": 20}

runclass = 'TBP'
ed = '2020-08-05'
ld = '2020-09-20'
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

# store RTs according to SNR
snrs = ['-inf', '-10', '-5', '0', 'inf']
all_snrs = [-10, -5, 0, np.inf]
rts = {k: [] for k in snrs}
rts_early = {k: [] for k in snrs} 
rts_middle = {k: [] for k in snrs}
rts_late = {k: [] for k in snrs}
LI_EARLY = np.full((4, len(uDate)), np.nan)
LI_MID = np.full((4, len(uDate)), np.nan)
LI_LATE = np.full((4, len(uDate)), np.nan)
LI_ALL = np.full((4, len(uDate)), np.nan)
VALID_TRIALS = np.full(len(uDate), np.nan)
VALID_TRIALS_EAR = np.full(len(uDate), np.nan)
VALID_TRIALS_LAT = np.full(len(uDate), np.nan)
for idx, ud in enumerate(uDate):
    print(f"Loading data from {ud}")
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

    # get reaction times of targets, only for "correct" trials
    bev = manager.get_behavior_events(**options)
    bev = manager._stack_events(bev)
    bev = bev[bev.invalidTrial==False]
    bev, params = TBP(bev, manager.get_baphy_exptparams()[0])
    _rts = get_reaction_times(params, bev, **options)

    # get early / middle / late rts
    nTrials = len(bev.Trial.unique())
    edges = np.quantile(range(nTrials), [.25, .5, .75]).astype(int)
    edges = bev.Trial.unique()[edges]
    # then define three overlapping ranges. Early, middle, late
    if rel:
        t1 = range(1, edges[1])
        t2 = range(edges[0], edges[2])
        t3 = range(edges[1], max(bev.Trial))
    if absolute:
        t1 = range(1, 100)
        t2 = range(50, 100)
        t3 = range(100, 1000)
    _rts_early = get_reaction_times(params, bev[bev.Trial.isin(t1)], **options)
    _rts_middle = get_reaction_times(params, bev[bev.Trial.isin(t2)], **options)
    _rts_late = get_reaction_times(params, bev[bev.Trial.isin(t3)], **options)

    targets = [k for k in _rts['Target'].keys() if ('-Inf' not in k) & ('reminder' not in k)]
    catch = [k for k in _rts['Target'].keys() if '-Inf' in k]
    snrs = thelp.get_snrs(targets)
    catch_snrs = thelp.get_snrs(catch)
    for s, t in zip(snrs+catch_snrs, targets+catch):
        rts[str(s)].extend(_rts['Target'][t])
        try: rts_early[str(s)].extend(_rts_early['Target'][t])
        except: pass
        try: rts_middle[str(s)].extend(_rts_middle['Target'][t])
        except: pass
        try: rts_late[str(s)].extend(_rts_late['Target'][t])
        except: pass

    # GET TAR vs. CAT DI
    # get performance in the three windows (and overall)
    perf_all = manager.get_behavior_performance(**options)
    perf_early = manager.get_behavior_performance(trials=t1, **options)
    perf_mid = manager.get_behavior_performance(trials=t2, **options)
    perf_late = manager.get_behavior_performance(trials=t3, **options)
    li = []
    li_early = []
    li_mid = []
    li_late = []
    if len(catch)==1:
        for t in targets:
            li.append(perf_all['LI'][t+'_'+catch[0].strip('CAT_')])
            li_early.append(perf_early['LI'][t+'_'+catch[0].strip('CAT_')])
            li_mid.append(perf_mid['LI'][t+'_'+catch[0].strip('CAT_')])
            li_late.append(perf_late['LI'][t+'_'+catch[0].strip('CAT_')])
    elif len(catch)==2:
        # only compute LI for the "on-BF" targets vs. on-BF catch -- discard off-center masker for this analysis
        f = int(params['TrialObject'][1]['TargetHandle'][1]['Names'][params['TrialObject'][1]['CueTarIdx']].split('+')[0]) 
        c = np.array(catch)[f==np.array(thelp.get_tar_freqs(catch))].tolist()
        snrs = np.array(snrs)[f==np.array(thelp.get_tar_freqs(targets))].tolist()
        targets = np.array(targets)[f==np.array(thelp.get_tar_freqs(targets))].tolist()
        for t in targets:
            li.append(perf_all['LI'][t+'_'+catch[0].strip('CAT_')])
            li_early.append(perf_early['LI'][t+'_'+catch[0].strip('CAT_')])
            li_mid.append(perf_early['LI'][t+'_'+catch[0].strip('CAT_')])
            li_late.append(perf_late['LI'][t+'_'+catch[0].strip('CAT_')])
    
    snr_idx = [np.argwhere(np.array(all_snrs)==s)[0][0] for s in snrs]

    nValidTrials_early = np.sum([v for k,v in perf_early['nTrials'].items() if k!='Reference'])
    nValidTrials_late = np.sum([v for k,v in perf_late['nTrials'].items() if k!='Reference'])
    if 1: #(nValidTrials_early >= 20) & (nValidTrials_late >= 20):
        # only save days with sufficient data
        LI_ALL[snr_idx, idx] = li
        LI_EARLY[snr_idx, idx] = li_early
        LI_MID[snr_idx, idx] = li_mid
        LI_LATE[snr_idx, idx] = li_late

    nValidTrials = np.sum([v for k,v in perf_all['nTrials'].items() if k!='Reference'])
    VALID_TRIALS[idx] = nValidTrials
    VALID_TRIALS_EAR[idx] = nValidTrials_early
    VALID_TRIALS_LAT[idx] = nValidTrials_late

# tweak target names for plot legend
name_map = {'-10': "-10 dB", '-5': "-5 dB", '0': "0 dB", "inf": "Inf dB", '-inf': "Catch"}
rts = {name_map[name]: val for name, val in rts.items()}
rts_early = {name_map[name]: val for name, val in rts_early.items()}
rts_middle = {name_map[name]: val for name, val in rts_middle.items()}
rts_late = {name_map[name]: val for name, val in rts_late.items()}
rts.pop('-10 dB')
rts_early.pop('-10 dB')
rts_middle.pop('-10 dB')
rts_late.pop('-10 dB')
# ========================== OVERALL BEHAVIOR PLOT ===========================
f, ax = plt.subplots(1, 2, figsize=(10, 4))

# RT
bins = np.arange(0, 1.2, 0.001)
plot_RT_histogram(rts, bins=bins, ax=ax[0])
ax[0].set_ylim((0, 1))

# TAR vs. CAT DI
keep_idx = np.nanmean(LI_ALL[1:,:], axis=0)>0.5
sem = np.nanstd(LI_ALL[1:, keep_idx], axis=-1) / np.sqrt(LI_ALL[1:, keep_idx].shape[-1])
ax[1].plot(LI_ALL[1:, keep_idx], color='lightgrey', marker='.', lw=0)
ax[1].errorbar([0, 1, 2], np.nanmean(LI_ALL[1:, keep_idx], axis=-1), yerr=sem, marker='o', capsize=2, zorder=10000)
ax[1].set_ylabel('Target vs. Catch Discriminability (DI)')
ax[1].set_xlabel('Target SNR (dB)')
ax[1].set_xticks([0, 1, 2])
ax[1].axhline(0.5, linestyle='--', color='grey')
ax[1].set_xticklabels([str(s) for s in all_snrs[1:]])

f.tight_layout()

f.savefig(figsave1)

# plot early, late, middle
f, ax = plt.subplots(1, 3, figsize=(16, 4))

bins = np.arange(0, 1.2, 0.001)
plot_RT_histogram(rts_early, bins=bins, ax=ax[0])
ax[0].set_title('Early trials')
ax[0].grid(True)
ax[0].set_ylim((0, 1))

bins = np.arange(0, 1.2, 0.001)
plot_RT_histogram(rts_late, bins=bins, ax=ax[1])
ax[1].set_title('Late trials')
ax[1].grid(True)
ax[1].set_ylim((0, 1))

# DI
ear = [
    np.average(LI_EARLY[1,~np.isnan(LI_EARLY[1,:])], weights=VALID_TRIALS_EAR[~np.isnan(LI_EARLY[1,:])]),
    np.average(LI_EARLY[2,~np.isnan(LI_EARLY[2,:])], weights=VALID_TRIALS_EAR[~np.isnan(LI_EARLY[2,:])]),
    np.average(LI_EARLY[3,~np.isnan(LI_EARLY[3,:])], weights=VALID_TRIALS_EAR[~np.isnan(LI_EARLY[3,:])])
]
lat = [
    np.average(LI_LATE[1,~np.isnan(LI_LATE[1,:])], weights=VALID_TRIALS_LAT[~np.isnan(LI_LATE[1,:])]),
    np.average(LI_LATE[2,~np.isnan(LI_LATE[2,:])], weights=VALID_TRIALS_LAT[~np.isnan(LI_LATE[2,:])]),
    np.average(LI_LATE[3,~np.isnan(LI_LATE[3,:])], weights=VALID_TRIALS_LAT[~np.isnan(LI_LATE[3,:])])
]
sem = np.nanstd(LI_EARLY, axis=-1) / np.sqrt(LI_EARLY.shape[-1])
ax[2].errorbar([0, 1, 2], ear, yerr=sem[1:], marker='o', capsize=2, label='Early')
sem = np.nanstd(LI_LATE, axis=-1) / np.sqrt(LI_LATE.shape[-1])
ax[2].errorbar([0, 1, 2], lat, yerr=sem[1:], marker='o', capsize=2, label='Late')

ax[2].legend(frameon=False)
ax[2].set_ylabel('Target vs. Catch Discriminability (DI)')
ax[2].set_xlabel('Target SNR (dB)')
ax[2].set_xticks([0, 1, 2])
ax[2].axhline(0.5, linestyle='--', color='grey')
ax[2].set_xticklabels([str(s) for s in all_snrs[1:]])


f.tight_layout()
f.savefig(figsave2)


plt.show()