"""
tar vs. tar / cat vs. tar discriminability in active vs passive
"""
import nems.db as nd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 14

figsave = '/auto/users/hellerc/code/projects/in_progress/TIN_behavior/R01_OHRC_figs/invariance.pdf'

df = pd.read_csv('/auto/users/hellerc/code/projects/in_progress/TIN_behavior/res_pr.csv', index_col=0)
df.index = df.pair

val = 'dp_opt'
df[val] = np.sqrt(df[val])
m = 8

tar_mask = (df.tar_tar) & (df.tdr_overall==True) & (~df.pca) & (df.f1 == df.f2) 
cat_mask = (df.cat_tar) & (df.tdr_overall==True) & (~df.pca) & (df.f1 == df.f2) 
ref_mask = (df.ref_ref) & (df.tdr_overall==False) & (~df.pca)

f, ax = plt.subplots(2, 2, figsize=(8, 8))

# ==== A1 ===== 
ax[0, 0].scatter(df[tar_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val],
            df[tar_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val], s=80, edgecolor='k', label='Tar vs. Tar')
ax[0, 0].scatter(df[cat_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val],
            df[cat_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val], s=80, edgecolor='k', label='Tar vs. Cat')
ax[0, 0].scatter(df[ref_mask & df.active & (df.area=='A1')].groupby(by='site').mean()[val],
            df[ref_mask & ~df.active & (df.area=='A1')].groupby(by='site').mean()[val], s=80, edgecolor='k', label='Ref vs. Ref')
ax[0, 0].set_xlabel(r"$d'$ Active")
ax[0, 0].set_ylabel(r"$d'$ Passive")
ax[0, 0].plot([0, m], [0, m], '--', color='grey', lw=2)
ax[0, 0].set_title('A1')
ax[0, 0].legend(frameon=False)

# ==== PEG ===== 
ax[0, 1].scatter(df[tar_mask & df.active & (df.area=='PEG')].groupby(by='site').mean()[val],
            df[tar_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val], s=80, edgecolor='k')
ax[0, 1].scatter(df[cat_mask & df.active & (df.area=='PEG')].groupby(by='site').mean()[val],
            df[cat_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val], s=80, edgecolor='k')
ax[0, 1].scatter(df[ref_mask & df.active & (df.area=='PEG')].groupby(by='site').mean()[val],
            df[ref_mask & ~df.active & (df.area=='PEG')].groupby(by='site').mean()[val], s=80, edgecolor='k')
ax[0, 1].set_xlabel(r"$d'$ Active")
ax[0, 1].set_ylabel(r"$d'$ Passive")
ax[0, 1].plot([0, m], [0, m], '--', color='grey', lw=2)
ax[0, 1].set_title('PEG')

# normalize changes and look within site
tt_act = df[tar_mask & df.active & (df.area=='A1')][[val, 'site']].set_index('site')
tt_pass = df[tar_mask & ~df.active & (df.area=='A1')][[val, 'site']].set_index('site')
tt_delta = ((tt_act - tt_pass) / (tt_act + tt_pass)).groupby(level=0).mean()
tt_sem = ((tt_act - tt_pass) / (tt_act + tt_pass)).groupby(level=0).sem()
ct_act = df[cat_mask & df.active & (df.area=='A1')][[val, 'site']].set_index('site')
ct_pass = df[cat_mask & ~df.active & (df.area=='A1')][[val, 'site']].set_index('site')
ct_delta = ((ct_act - ct_pass) / (ct_act + ct_pass)).groupby(level=0).mean()
ct_sem = ((ct_act - ct_pass) / (ct_act + ct_pass)).groupby(level=0).sem()
rr_act = df[ref_mask & df.active & (df.area=='A1')][[val, 'site']].set_index('site')
rr_pass = df[ref_mask & ~df.active & (df.area=='A1')][[val, 'site']].set_index('site')
rr_delta = ((rr_act - rr_pass) / (rr_act + rr_pass)).groupby(level=0).mean()
rr_sem = ((rr_act - rr_pass) / (rr_act + rr_pass)).groupby(level=0).sem()

colors = plt.get_cmap('jet', len(ct_delta.index))
cells = nd.get_batch_cells(324).cellid
for i, s in enumerate(list(set(ct_delta.index).intersection(set(tt_delta.index)))):
    ncells = len([c for c in cells if s in c])
    lab = s + f' ({ncells} cells)'
    ax[1, 0].errorbar([0, 1, 2], [rr_delta.loc[s].values[0], tt_delta.loc[s].values[0], ct_delta.loc[s].values[0]], \
        yerr=[rr_sem.loc[s].values[0], tt_sem.loc[s].values[0], ct_sem.loc[s].values[0]], capsize=3, color=colors(i), label=lab, marker='o')

leg = ax[1, 0].legend(frameon=False, handlelength=0)
for line, text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
ax[1, 0].set_xticks([0, 1, 2])
ax[1, 0].axhline(0, linestyle='--', color='grey', lw=2)
ax[1, 0].set_xticklabels(['Ref vs. Ref,', 'Tar. vs. Tar', 'Cat vs. Tar'], rotation=45)
ax[1, 0].set_ylabel(r"$\Delta d'$")

tt_act = df[tar_mask & df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
tt_pass = df[tar_mask & ~df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
tt_delta = ((tt_act - tt_pass) / (tt_act + tt_pass)).groupby(level=0).mean()
tt_sem = ((tt_act - tt_pass) / (tt_act + tt_pass)).groupby(level=0).sem()
ct_act = df[cat_mask & df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
ct_pass = df[cat_mask & ~df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
ct_delta = ((ct_act - ct_pass) / (ct_act + ct_pass)).groupby(level=0).mean()
ct_sem = ((ct_act - ct_pass) / (ct_act + ct_pass)).groupby(level=0).sem()
rr_act = df[ref_mask & df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
rr_pass = df[ref_mask & ~df.active & (df.area=='PEG')][[val, 'site']].set_index('site')
rr_delta = ((rr_act - rr_pass) / (rr_act + rr_pass)).groupby(level=0).mean()
rr_sem = ((rr_act - rr_pass) / (rr_act + rr_pass)).groupby(level=0).sem()

colors = plt.get_cmap('jet', len(ct_delta.index))
cells = nd.get_batch_cells(325).cellid
for i, s in enumerate(list(set(ct_delta.index).intersection(set(tt_delta.index)))):
    ncells = len([c for c in cells if s in c])
    lab = s + f' ({ncells} cells)'
    ax[1, 1].errorbar([0, 1, 2], [rr_delta.loc[s].values[0], tt_delta.loc[s].values[0], ct_delta.loc[s].values[0]], \
        yerr=[rr_sem.loc[s].values[0], tt_sem.loc[s].values[0], ct_sem.loc[s].values[0]], capsize=3, color=colors(i), label=lab, marker='o')

leg = ax[1, 1].legend(frameon=False, handlelength=0)
for line, text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
ax[1, 1].set_xticks([0, 1, 2])
ax[1, 1].axhline(0, linestyle='--', color='grey', lw=2)
ax[1, 1].set_xticklabels(['Ref vs. Ref', 'Tar. vs. Tar', 'Cat vs. Tar'], rotation=45)
ax[1, 1].set_ylabel(r"$\Delta d'$")

f.tight_layout()

f.savefig(figsave)

plt.show()