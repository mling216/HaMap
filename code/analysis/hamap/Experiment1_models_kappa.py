# %%
import pandas as pd
import os.path as osp
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import krippendorff as kd
from statsmodels.stats import inter_rater as irr
from sklearn.metrics import cohen_kappa_score

TNSEP = 'TNsep'
SEP = 'Standalone'
HALF = 'Half-half'
methods = {SEP:'Normal_tiles_from_normal_slides',
           HALF:'Half_normal_tiles_from_tumor_slides'}
VGG, VGGs = 'vgg19', 'VGG19'
RESNET, RESNETs = 'resnet50', 'Resnet50'
INCEPTION, INCEPTIONs = 'inceptionv3', 'Inceptionv3'
VIT, VITs = 'vit', 'ViT'


# %%
### get all model results
result_vgg = '/users/PAS1575/chen8028/Pathology/individual_mask/vgg19_testing'
result_resnet = '/users/PAS1575/chen8028/Pathology/individual_mask/resnet50_testing'
result_inception = '/users/PAS1575/chen8028/Pathology/individual_mask/inceptionv3_testing'
result_vit = '/users/PAS1575/chen8028/Pathology/individual_mask/vit_testing'

def binary_pred(x):
    if x>0.5:
        return 1.0
    else:
        return 0.0

def binary_pred_int(x):
    if x>0.5:
        return 1
    else:
        return 0

res = []    # to save cohen's kappa
inter_rater = []    # to save inter-rater kappa on predictions
inter_rater_correct = []    # to save inter-rater kappa on true/false
for res_folder in [result_vgg, result_resnet, result_inception, result_vit]:
    if VGG in res_folder:
        model = VGGs
    elif RESNET in res_folder:
        model = RESNETs
    elif INCEPTION in res_folder:
        model = INCEPTIONs
    elif VIT in res_folder:
        model = VITs
    else:
        print(result_f, 'not in model list')
        continue

    result_type = '*_small_test.csv'
    result_files = glob.glob(osp.join(res_folder, result_type))
    if 'small' in result_type:
        case_type = 'Fixation-reduction (large)'
    else:
        case_type = 'Fixation-reduction (large test)'
    
    for idx, result_f in enumerate(result_files): 
        # skip the 0.7_s26 tumor dataset has only 77k tiles
        # if '0.7_s26' in result_f:
        #     continue
        segs = osp.basename(result_f).split('_s')
        thresh = segs[0].split('_')[-1]
        seed = segs[1].split('_')[0]

        df = pd.read_csv(result_f)
        df['tile'] = df.index
        df['thresh'] = thresh
        df['seed'] = seed
        df['pred'] = df['yhat'].apply(binary_pred_int)
        df['correct'] = df['pred'] == df['ytruth']

        # calculate cohen's kapps between groundtruth and prediction
        kappa = cohen_kappa_score(df['ytruth'], df['pred'])
        res.append([model, thresh, seed, case_type,kappa])
        
        # concat prediction results into columns for inter-rater evaluation
        df_col = df[['pred']]
        df_col.columns = [thresh+'_'+seed]
        if idx==0:
            df_inter_rater = df_col
        else:
            df_inter_rater = pd.concat([df_inter_rater, df_col], axis=1)

        # concat 'correct' results into columns for inter-rater evaluation
        df_col_correct = df[['correct']]
        df_col_correct.columns = [thresh+'_'+seed]
        if idx==0:
            df_inter_rater_correct = df_col_correct
        else:
            df_inter_rater_correct = pd.concat([df_inter_rater_correct, df_col_correct], axis=1)

    # calculate inter-rater fleiss_kappa
    dats, cats = irr.aggregate_raters(df_inter_rater)
    fleiss_kappa = irr.fleiss_kappa(dats, method='fleiss')
    inter_rater.append([model, fleiss_kappa])

    # calculate inter-rater 'correct' fleiss_kappa
    dats, cats = irr.aggregate_raters(df_inter_rater_correct)
    fleiss_kappa_correct = irr.fleiss_kappa(dats, method='fleiss')
    inter_rater_correct.append([model, fleiss_kappa_correct])    

    print(fleiss_kappa, fleiss_kappa_correct)
 

# buid the dataframe for cohen's kappa between groundtruth and prediction
df_kappa = pd.DataFrame(res, columns=['model','threshold','seed','case_type','kappa'])
df_kappa.to_csv('results/exp1_cohen_kappa.csv', index=False)
# calculate model-wise stat for cohen's kappa
df_kappa_agg = df_kappa.groupby('model', as_index=False).agg({'kappa':['mean','std']})
df_kappa_agg.columns = ['model','kappa-mean','kappa-std']
df_kappa_agg.to_csv('results/exp1_cohen_kappa_by_model.csv', index=False)

# build the datafram for inter-kappa on predictions and save results
df_fleiss_kappa = pd.DataFrame(inter_rater, columns=['model','inter_kappa'])
df_fleiss_kappa.to_csv('results/exp1_inter_rater_fleiss_kappa.csv', index=False)

# build the datafram for inter-kappa on true/false predictions and save results
df_fleiss_kappa_correct = pd.DataFrame(inter_rater_correct, columns=['model','inter_kappa'])
df_fleiss_kappa_correct.to_csv('results/exp1_inter_rater_fleiss_kappa_correct_or_not.csv', index=False)



# %%
### plot kappa results of the models  
#
df_select = df_kappa
df_select.sort_values('model', inplace=True)

fig, ax = plt.subplots(figsize=(4.5,3.5))
g = sns.barplot(data=df_select, x='model', y='kappa',
                capsize=0.05, zorder=2, width=0.4, color='grey', ax=ax)

plt.ylim(0.0,1.0) 
plt.xlabel('DNN model')
plt.yticks(np.arange(0, 1.01, step=0.1))
plt.grid(axis='y')
plt.ylabel("Cohen's Kappa")
plt.title("Eye-tracking experiment model prediction\nConsistency by Cohen's Kappa")

plt.savefig('results_plots/exp1_cohen_kappa.png', dpi=300, bbox_inches='tight')





# %%
