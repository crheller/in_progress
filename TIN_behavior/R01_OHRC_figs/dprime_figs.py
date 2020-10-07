import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

df = pd.read_csv('/auto/users/hellerc/code/projects/in_progress/TIN_behavior/res_pr.csv', index_col=0)
df.index = df.pair

# ================================  neurometric function overall ===========================================
a1_mask = (df.cat_tar) & (df.area=='A1') & (df.tdr_overall==False) & (df.f1 == df.f2) & (~df.pca)
peg_mask = (df.cat_tar) & (df.area=='PEG') & (df.tdr_overall==False) & (df.f1 == df.f2) & (~df.pca)
val = 'dp_opt'
f, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

# ==================== A1 ====================
ax[0].set_title('A1')
ax[0].set_xlabel('SNR')
ax[0].set_ylabel(r"$d'^2$")

# active
dpa = df[a1_mask & df.active][[val, 'snr1']].groupby(by='snr1').mean().sort_values(by='snr1')
dpa_sem = df[a1_mask & df.active][[val, 'snr1']].groupby(by='snr1').sem().sort_values(by='snr1')
snrs = dpa.index.values
x = np.arange(0, len(snrs))
ax[0].errorbar(x, dpa[val].values, yerr=dpa_sem[val].values, marker='o', capsize=5, label='Active')

# passive
dpp = df[a1_mask & (df.active==False)][[val, 'snr1']].groupby(by='snr1').mean().sort_values(by='snr1')
dpp_sem = df[a1_mask & (df.active==False)][[val, 'snr1']].groupby(by='snr1').sem().sort_values(by='snr1')
snrs = dpp.index.values
x = np.arange(0, len(snrs))
ax[0].errorbar(x, dpp[val].values, yerr=dpp_sem[val].values, marker='o', capsize=5, label='Passive')

ax[0].legend(frameon=False)
ax[0].set_xticks(x)
ax[0].set_xticklabels(snrs)

# =================== PEG ====================
ax[1].set_title('PEG')
ax[1].set_xlabel('SNR')
ax[1].set_ylabel(r"$d'^2$")

# active
dpa = df[peg_mask & df.active][[val, 'snr1']].groupby(by='snr1').mean().sort_values(by='snr1')
dpa_sem = df[peg_mask & df.active][[val, 'snr1']].groupby(by='snr1').sem().sort_values(by='snr1')
snrs = dpa.index.values
x = np.arange(0, len(snrs))
ax[1].errorbar(x, dpa[val].values, yerr=dpa_sem[val].values, marker='o', capsize=5, label='Active')

# passive
dpp = df[peg_mask & (df.active==False)][[val, 'snr1']].groupby(by='snr1').mean().sort_values(by='snr1')
dpp_sem = df[peg_mask & (df.active==False)][[val, 'snr1']].groupby(by='snr1').sem().sort_values(by='snr1')
snrs = dpp.index.values
x = np.arange(0, len(snrs))
ax[1].errorbar(x, dpp[val].values, yerr=dpp_sem[val].values, marker='o', capsize=5, label='Passive')

ax[1].legend(frameon=False)
ax[1].set_xticks(x)
ax[1].set_xticklabels(snrs)

f.suptitle('Neurometric functions')

f.tight_layout()

# ===================================== Neurometric showing individual sites ==========================================
a1_mask = (df.cat_tar) & (df.area=='A1') & (df.tdr_overall==False) & (df.f1 == df.f2) & (~df.pca)
peg_mask = (df.cat_tar) & (df.area=='PEG') & (df.tdr_overall==False) & (df.f1 == df.f2) & (~df.pca)
val = 'dp_opt'
all_snrs = np.sort(df[a1_mask].snr1.unique())
f, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

# ==================== A1 ====================
ax[0].set_title('A1')
ax[0].set_xlabel('SNR')
ax[0].set_ylabel(r"$d'^2$")

# active
dpa = df[a1_mask & df.active][[val, 'snr1', 'site']].groupby(by=['snr1', 'site']).mean().sort_values(by='snr1')
colors = plt.get_cmap('jet', len(dpa.index.get_level_values(1).unique()))
for i, site in enumerate(dpa.index.get_level_values(1).unique()):
    snrs = dpa.loc[pd.IndexSlice[:, site], :].index.get_level_values(0)
    idx = [True if s in snrs else False for s in all_snrs ]
    x = np.arange(0, len(all_snrs))[idx]
    ax[0].plot(x, dpa.loc[pd.IndexSlice[:, site], val].values, label=site, color=colors(i))

# passive
dpp = df[a1_mask & ~df.active][[val, 'snr1', 'site']].groupby(by=['snr1', 'site']).mean().sort_values(by='snr1')
for i, site in enumerate(dpp.index.get_level_values(1).unique()):
    snrs = dpp.loc[pd.IndexSlice[:, site], :].index.get_level_values(0)
    idx = [True if s in snrs else False for s in all_snrs ]
    x = np.arange(0, len(all_snrs))[idx]
    ax[0].plot(x, dpp.loc[pd.IndexSlice[:, site], val].values, color=colors(i), linestyle='--')


ax[0].legend(frameon=False)
ax[0].set_xticks(np.arange(0, len(all_snrs)))
ax[0].set_xticklabels(all_snrs)

# =================== PEG ====================
ax[1].set_title('PEG')
ax[1].set_xlabel('SNR')
ax[1].set_ylabel(r"$d'^2$")

# active
dpa = df[peg_mask & df.active][[val, 'snr1', 'site']].groupby(by=['snr1', 'site']).mean().sort_values(by='snr1')
colors = plt.get_cmap('jet', len(dpa.index.get_level_values(1).unique()))
for i, site in enumerate(dpa.index.get_level_values(1).unique()):
    snrs = dpa.loc[pd.IndexSlice[:, site], :].index.get_level_values(0)
    idx = [True if s in snrs else False for s in all_snrs ]
    x = np.arange(0, len(all_snrs))[idx]
    ax[1].plot(x, dpa.loc[pd.IndexSlice[:, site], val].values, label=site, color=colors(i))

# passive
dpp = df[peg_mask & ~df.active][[val, 'snr1', 'site']].groupby(by=['snr1', 'site']).mean().sort_values(by='snr1')
for i, site in enumerate(dpp.index.get_level_values(1).unique()):
    snrs = dpp.loc[pd.IndexSlice[:, site], :].index.get_level_values(0)
    idx = [True if s in snrs else False for s in all_snrs ]
    x = np.arange(0, len(all_snrs))[idx]
    ax[1].plot(x, dpp.loc[pd.IndexSlice[:, site], val].values, color=colors(i), linestyle='--')


ax[1].legend(frameon=False)
ax[1].set_xticks(np.arange(0, len(all_snrs)))
ax[1].set_xticklabels(all_snrs)

f.suptitle('Neurometric functions')

f.tight_layout()

# ================================ TARGET INVARIANCE =================================
# show active vs. passive scatter plot of tar vs. tar and cat vs. tar (mean for both - one point for each per site)
val = 'dp_opt'
f, ax = plt.subplots(1, 2, figsize=(10, 5))
m = 70
# ==== A1 ===== 
tar_mask = (df.tar_tar) & (df.area=='A1') & (df.tdr_overall==False) & (df.f1 == df.f2) & (~df.pca)
cat_mask = (df.cat_tar) & (df.area=='A1') & (df.tdr_overall==False) & (df.f1 == df.f2) & (~df.pca)
ax[0].scatter(df[tar_mask & df.active].groupby(by='site').mean()[val],
            df[tar_mask & ~df.active].groupby(by='site').mean()[val], edgecolor='white', s=50, label='Tar vs. Tar')
ax[0].scatter(df[cat_mask & df.active].groupby(by='site').mean()[val],
            df[cat_mask & ~df.active].groupby(by='site').mean()[val], edgecolor='white', s=50, label='Tar vs. Cat')
ax[0].set_xlabel('Active')
ax[0].set_ylabel('Passive')
ax[0].plot([0, m], [0, m], '--', color='grey', lw=2)
ax[0].set_title('A1')
ax[0].legend(frameon=False)

# ==== PEG ===== 
tar_mask = (df.tar_tar) & (df.area=='PEG') & (df.tdr_overall==False) & (df.f1 == df.f2) & (~df.pca)
cat_mask = (df.cat_tar) & (df.area=='PEG') & (df.tdr_overall==False) & (df.f1 == df.f2) & (~df.pca)
ax[1].scatter(df[tar_mask & df.active].groupby(by='site').mean()[val],
            df[tar_mask & ~df.active].groupby(by='site').mean()[val], edgecolor='white', s=50)
ax[1].scatter(df[cat_mask & df.active].groupby(by='site').mean()[val],
            df[cat_mask & ~df.active].groupby(by='site').mean()[val], edgecolor='white', s=50, label='Tar vs. Cat')
ax[1].set_xlabel('Active')
ax[1].set_ylabel('Passive')
ax[1].plot([0, m], [0, m], '--', color='grey', lw=2)
ax[1].set_title('PEG')

f.tight_layout()

# ============================ TARGET INVARIANCE DELTA DPRIME ==============================
# normalize changes and look within site
val = 'dp_opt'
absolute = False

f, ax = plt.subplots(1, 2, figsize=(10, 5))

tar_mask = (df.tar_tar) & (df.area=='A1') & (df.tdr_overall==False) & (df.f1 == df.f2) & (~df.pca)
cat_mask = (df.cat_tar) & (df.area=='A1') & (df.tdr_overall==False) & (df.f1 == df.f2) & (~df.pca)
tt_act = df[tar_mask & df.active][[val, 'site']].set_index('site')
tt_pass = df[tar_mask & ~df.active][[val, 'site']].set_index('site')
if absolute:
    tt_delta = ((tt_act - tt_pass)).groupby(level=0).mean()
else:
    tt_delta = ((tt_act - tt_pass) / (tt_act + tt_pass)).groupby(level=0).mean()
ct_act = df[cat_mask & df.active][[val, 'site']].set_index('site')
ct_pass = df[cat_mask & ~df.active][[val, 'site']].set_index('site')
if absolute:
    ct_delta = ((ct_act - ct_pass)).groupby(level=0).mean()
else:
    ct_delta = ((ct_act - ct_pass) / (ct_act + ct_pass)).groupby(level=0).mean()


colors = plt.get_cmap('jet', len(ct_delta.index))
cells = nd.get_batch_cells(324).cellid
for i, s in enumerate(list(set(ct_delta.index).intersection(set(tt_delta.index)))):
    ncells = len([c for c in cells if s in c])
    lab = s + f' ({ncells} cells)'
    ax[0].plot([0, 1], [tt_delta.loc[s], ct_delta.loc[s]], color=colors(i), label=lab)

ax[0].legend(frameon=False)
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(['Tar. vs. Tar', 'Cat vs. Tar'])
ax[0].set_ylabel(r"$\Delta d'^2$")
ax[0].set_title('A1')

tar_mask = (df.tar_tar) & (df.area=='PEG') & (df.tdr_overall==False) & (df.f1 == df.f2) & (~df.pca)
cat_mask = (df.cat_tar) & (df.area=='PEG') & (df.tdr_overall==False) & (df.f1 == df.f2) & (~df.pca)
tt_act = df[tar_mask & df.active][[val, 'site']].set_index('site')
tt_pass = df[tar_mask & ~df.active][[val, 'site']].set_index('site')
if absolute:
    tt_delta = ((tt_act - tt_pass)).groupby(level=0).mean()
else:
    tt_delta = ((tt_act - tt_pass) / (tt_act + tt_pass)).groupby(level=0).mean()
ct_act = df[cat_mask & df.active][[val, 'site']].set_index('site')
ct_pass = df[cat_mask & ~df.active][[val, 'site']].set_index('site')
if absolute:
    ct_delta = ((ct_act - ct_pass)).groupby(level=0).mean()
else:
    ct_delta = ((ct_act - ct_pass) / (ct_act + ct_pass)).groupby(level=0).mean()

colors = plt.get_cmap('jet', len(ct_delta.index))
cells = nd.get_batch_cells(325).cellid
for i, s in enumerate(list(set(ct_delta.index).intersection(set(tt_delta.index)))):
    ncells = len([c for c in cells if s in c])
    lab = s + f' ({ncells} cells)'
    ax[1].plot([0, 1], [tt_delta.loc[s], ct_delta.loc[s]], color=colors(i), label=lab)

ax[1].legend(frameon=False)
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(['Tar. vs. Tar', 'Cat vs. Tar'])
ax[1].set_ylabel(r"$\Delta d'^2$")
ax[1].set_title('PEG')


plt.show()