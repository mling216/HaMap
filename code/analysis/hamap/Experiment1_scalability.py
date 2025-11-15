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


# %%
### get all model results
result_vgg = '/users/PAS1575/chen8028/Pathology/individual_mask/vgg19_testing'
result_resnet = '/users/PAS1575/chen8028/Pathology/individual_mask/resnet50_testing'
result_inception = '/users/PAS1575/chen8028/Pathology/individual_mask/inceptionv3_testing'
result_vit = '/users/PAS1575/chen8028/Pathology/individual_mask/vit_testing'

result_all_paths = []
other_cnt = 0
for res_folder in [result_vgg, result_resnet, result_inception, result_vit]:
    result_files = glob.glob(osp.join(res_folder, '*_report.csv'))
    for result_f in result_files:
        result_all_paths.append(result_f)
print(other_cnt)

### extract and save prediction results
res = []
for result_f in result_all_paths:
    if VGG in result_f:
        model = VGGs
    elif RESNET in result_f:
        model = RESNETs
    elif INCEPTION in result_f:
        model = INCEPTIONs
    elif VIT in result_f:
        model = VITs
    else:
        print(result_f, 'not in model list')
        continue
    
    if TNSEP in result_f:
        method = SEP
        method_str = methods[SEP]
    else:
        method = HALF
        method_str = methods[HALF]

    segs = osp.basename(result_f).split('_s')
    case = segs[0].replace('TNsep_','')
    
    if 'small' in result_f:
        case_type = 'Fixation-reduction (large)'
    elif '20test' in result_f:
        case_type = 'Fixation-reduction (small)'
    else:
        if 'w_viewport' in result_f:
            case = 'Viewport'
        elif 'viewport' in result_f:
            case = 'Weighted-viewport'
        elif 'w_optics' in result_f:
            case = 'Weighted-optics'
        elif 'optics' in result_f:
            case = 'Optics'                        
        else:
            case = 'Fixation-reduction'
        case_type = 'Fixation-reduction (large entire test)'
    thresh = segs[0].split('_')[-1]

    seed = segs[1].split('_')[0]
    df = pd.read_csv(result_f)
    accu = df['precision'].values[2]    # this is accuracy
    f1score = df['f1-score'].values[4]  # 3 macro avg, 4 weighted avg
    sensitivity = df['recall'].values[1]
    specificity = df['recall'].values[0]

    res.append([model, method, method_str, case_type, thresh, case, seed, 'accuracy', accu])
    res.append([model, method, method_str, case_type, thresh, case, seed, 'f1-score', f1score])
    res.append([model, method, method_str, case_type, thresh, case, seed, 'sensitivity', sensitivity])
    res.append([model, method, method_str, case_type, thresh, case, seed, 'specificity', specificity])

df_res = pd.DataFrame(res, columns=['model','method','method_str','case_type','threshold','case','seed','metric','value'])

# add model orders
df_res.loc[df_res['model']==VGGs, 'model_order'] = 1
df_res.loc[df_res['model']==RESNETs, 'model_order'] = 2
df_res.loc[df_res['model']==INCEPTIONs, 'model_order'] = 3
df_res.loc[df_res['model']==VITs, 'model_order'] = 4

df_res.sort_values(['model_order','method','case_type','threshold','case','metric','seed'], inplace=True)
df_res.to_csv('results/exp1_results_summary.csv', index=False)



# %%
### Get results from previous study and merge them
df_pre = pd.read_csv('/users/PAS1575/chen8028/Pathology/patchCodeNew/results/exp1_results_summary.csv').fillna('')
df_pre.loc[df_pre['model']=='Resnet50', 'model'] = RESNETs

# add model orders
df_pre.loc[df_pre['model']==VGGs, 'model_order'] = 1
df_pre.loc[df_pre['model']==RESNETs, 'model_order'] = 2
df_pre.loc[df_pre['model']==INCEPTIONs, 'model_order'] = 3
df_pre.loc[df_pre['model']==VITs, 'model_order'] = 4

# merge together

# the following line is for old small test using merged masks
# df_pre.loc[df_pre['case_type']=='Fixation-reduction (ours)','case_type'] = 'Fixation-reduction (small)'

# remove the old case and merge
df_pre = df_pre[df_pre['case_type']!='Fixation-reduction (ours)']
df_res = pd.concat([df_res, df_pre])
df_res



# %%
### Main Text Figure. compare fixation-reuction to baseline and random   
#
metric = 'specificity'
df_select = df_res[(df_res['metric']==metric) & (df_res['method']==SEP) &
                   ((df_res['threshold']=='') | (df_res['case_type']=='Fixation-reduction (small)') |
                                        (df_res['case_type']=='Fixation-reduction (large)'))]

g = sns.barplot(data=df_select, x='model', y='value', hue='case_type',
                hue_order=['Baseline-large', 'Baseline-small',
                           'Fixation-reduction (large)',
                           'Fixation-reduction (small)',
                           'Random-small','Random-large'],                
                capsize=0.05, palette='Greys_r', zorder=2)

"""
# ['/', '//', '+', '-', 'x', '\\', '*', 'o', 'O', '.']
hatches = ["", "", "", "", "", ""]
# Loop over the bars
for bars, hatch in zip(g.containers, hatches):
    # Set a different hatch for each group of bars
    for bar in bars:
        bar.set_hatch(hatch)
"""

plt.ylim(0.0,1.0) 
plt.xlabel('DNN model')
plt.yticks(np.arange(0, 1.01, step=0.1))
plt.grid(axis='y')
plt.ylabel(metric.capitalize())
plt.legend(title='Fixation map', bbox_to_anchor=(1.0,1.02))
plt.title('Eye-tracking experiment fixation-reduction vs. others')

plt.savefig('results_plots/exp1_compare_wt_baseline_%s.png' % metric, dpi=300, bbox_inches='tight')




# %%
### calculate values for paper
df_select = df_res[(df_res['method']==SEP) &
                   ((df_res['threshold']=='') | (df_res['case_type']=='Fixation-reduction (small)') |
                                        (df_res['case_type']=='Fixation-reduction (large)'))]
df_select.to_csv('./results/exp_v1_model_performance_metrics_details.csv', index=False)

res = df_select.groupby(['model','case_type','metric'], as_index=False).agg({'value':['mean','std']})
res.columns = ['model','case_type','metric','mean','std']
res.to_csv('./results/exp_v1_model_performance_metrics.csv', index=False)
res




# %%
### plot results by levels  -- Deprecated
df_res = pd.read_csv('/users/PAS1575/chen8028/Pathology/individual_mask/results/exp1_results_summary.csv').fillna('')

df_select = df_res[(df_res['metric']=='f1-score') & (df_res['method']==SEP) &
                   (df_res['case_type']=='Fixation-reduction (large)')]

pat = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=False, n_colors=6)
g = sns.barplot(data=df_select, x='model', y='value', hue='case',               
                capsize=0.05, palette=pat, 
                width=0.6, zorder=2)

plt.ylim(0.0,1.0) 
plt.xlabel('DNN model')
plt.yticks(np.arange(0, 1.01, step=0.1))
plt.grid(axis='y')
plt.ylabel('F1-score')
plt.legend(title='Threshold', bbox_to_anchor=(1.0,1.02))
plt.title('Testing results for different models\nat different thresholds')
plt.gcf().set_size_inches(5,5)

plt.savefig('results_plots/exp1_compare_levels_f1.png', dpi=300, bbox_inches='tight')






# %%
