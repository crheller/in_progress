"""
Goal is to see if we can show evidence for learning during the TIN behavior (TBP with variable SNR, single target)

Date that we switched back (from two tone reward learning to single tone) was 8.5.2020. Behavior was crummy on 8.5, though.
Start from 08.06.2020
"""

from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems.db as nd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
import datetime as dt

# recording load options
options = {"resp": False, "pupil": False, "rasterfs": 20}

runclass = 'TBP'
ed = '2020-08-05'
ld = '2020-09-02'
ed = dt.datetime.strptime(ed, '%Y-%m-%d')
ld = dt.datetime.strptime(ld, '%Y-%m-%d')

# get list of parmfiles and sort by date (all files from one day go into one analysis??)
sql = "SELECT gDataRaw.resppath, gDataRaw.parmfile, pendate FROM gPenetration INNER JOIN gCellMaster ON (gCellMaster.penid = gPenetration.id)"\
                " INNER JOIN gDataRaw ON (gCellMaster.id = gDataRaw.masterid) WHERE" \
                " gDataRaw.runclass='{0}' and gDataRaw.bad=0 and gDataRaw.trials>50 and"\
                " gPenetration.animal = 'Cordyceps'".format(runclass)
d = nd.pd_query(sql)
d['date'] = d['pendate'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))

# screen for date
d = d[(d['date'] >= ed) & (d['date'] <= ld)]
d = d.sort_values(by='date')

# join path
d['parmfile_path'] = [os.path.join(d['resppath'].iloc[i], d['parmfile'].iloc[i]) for i in range(d.shape[0])]

uDate = d['date'].unique()
for ud in uDate:
    parmfiles = d[d.date==ud].parmfile_path.values.tolist()
    # add catch to make sure "siteid" the same for all files
    sid = [p.split(os.path.sep)[-1][:7] for p in parmfiles]
    if np.any(np.array(sid) != sid[0]):
        bad_idx = (np.array(sid)!=sid[0])
        parmfiles = np.array(parmfiles)[~bad_idx].tolist()
    manager = BAPHYExperiment(parmfiles)
    rec = manager.get_recording(**options)
