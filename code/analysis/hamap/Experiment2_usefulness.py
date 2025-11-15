# %%
import pandas as pd
import os.path as osp
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

TNSEP = 'TNsep'
SEP = 'Standalone'
HALF = 'Half-half'
methods = {SEP:'Normal_tiles_from_normal_slides',
           HALF:'Half_normal_tiles_from_tumor_slides'}
VGG, VGGs = 'vgg19', 'VGG19'
RESNET, RESNETs = 'resnet50', 'ResNet50'
INCEPTION, INCEPTIONs = 'inceptionv3', 'Inceptionv3'
VIT, VITs = 'vit', 'ViT'

# read in results 
df_res = pd.read_csv('results/exp1_results_summary.csv')
df_res



# %%
### Main Text Figure. compare fixation-reuction to viewport and weighted-viewport   
#
metric = 'specificity'
df_select = df_res[(df_res['metric']==metric) & (df_res['method']==SEP) &
                   (df_res['case_type']=='Fixation-reduction (large entire test)') &
                   (df_res['model']=='VGG19')]

fig, ax = plt.subplots(figsize=(7,4))
g = sns.barplot(data=df_select, x='case', y='value',
                capsize=0.05, palette='Greys_r', 
                order=['Fixation-reduction', 'Optics', 'Weighted-optics',
                       'Viewport', 'Weighted-viewport'],
                width=0.3, zorder=2, ax=ax)

plt.ylim(0.0,1.0) 
plt.xlabel('Gaze map method')
plt.yticks(np.arange(0, 1.01, step=0.1))
plt.grid(axis='y')
plt.ylabel(metric.capitalize())
# plt.legend(title='Gaze map', bbox_to_anchor=(1.0,1.02))
plt.title('Fixation-reduction vs. other methods')

plt.savefig('results_plots/exp2_compare_gaze_maps_%s.png' % metric, dpi=200, bbox_inches='tight')




# %%
### calculate values for paper
df_select = df_res[(df_res['method']==SEP) &
                   (df_res['case_type']=='Fixation-reduction (large entire test)') &
                   (df_res['model']=='VGG19')]
res = df_select.groupby(['model','case','metric'], as_index=False).agg({'value':['mean','std']})
res.columns = ['model','case','metric','mean','std']
res.to_csv('./results/exp_v2_gaze_method_performance.csv', index=False)
res


# %%
