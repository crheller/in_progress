""" 
Compare first order / second order changes / how they relate to dprime
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 14

figsave = '/auto/users/hellerc/code/projects/in_progress/TIN_behavior/R01_OHRC_figs/invariance.pdf'

df = pd.read_csv('/auto/users/hellerc/code/projects/in_progress/TIN_behavior/res.csv', index_col=0)
df.index = df.pair

a1_mask = (df.cat_tar) & (df.area=='A1') & (df.tdr_overall==False) & (df.f1 == df.f2) & (~df.pca)
peg_mask = (df.cat_tar) & (df.area=='PEG') & (df.tdr_overall==False) & (df.f1 == df.f2) & (~df.pca)

# add column for dU mag

