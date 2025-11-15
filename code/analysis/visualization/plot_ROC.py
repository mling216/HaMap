# %%
### load library
import os.path as osp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import RocCurveDisplay
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html#sklearn.metrics.RocCurveDisplay.from_predictions
from sklearn.metrics import roc_auc_score

result_path = '/users/PAS1575/chen8028/Pathology/individual_mask/MODEL_testing/'


# %%
### calculate ROC's AUC scores
models = ['vgg19', 'resnet50', 'inceptionv3', 'vit']
model_names = {'vgg19':'VGG19', 'resnet50':'ResNet50', 'inceptionv3':'Inceptionv3', 'vit':'VIT'}
thresholds = ['0.3', '0.5', '0.7']
seeds = [6, 16, 26]

res = []
for model in models:
    for thresh in thresholds:
        for seed in seeds:
            test_res_path = osp.join(result_path.replace('MODEL', model), 
                                     'TNsep_%s_s%d_small_test.csv' % 
                                     (thresh, seed))
            # read in result
            df_test = pd.read_csv(test_res_path)
            # ground truth labels
            y_test = np.array(df_test['ytruth'].to_list())
            # prediction scores or probability
            y_pred = np.array(df_test['yhat'].to_list())
            
            # calculate AUC
            auc = roc_auc_score(y_test, y_pred)
            res.append([model_names[model], thresh, seed, auc])

df_res = pd.DataFrame(data=res, 
                      columns=['Model','Threshold','Seed','AUC'])
df_res.to_csv('./results/roc_auc_results.csv', index=False)



# %%
### plot ROC curve
# RocCurveDisplay.from_predictions(y_test, y_pred)
# plt.show()


# %%
### plot the results
import seaborn as sns

df_res = pd.read_csv('./results/roc_auc_results.csv')

pat = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=False, n_colors=6)
g = sns.barplot(data=df_res, x='Model', y='AUC', hue='Threshold',               
                capsize=0.05, palette=pat, 
                width=0.6, zorder=2)

plt.ylim(0.0,1.0) 
plt.xlabel('')
plt.yticks(np.arange(0, 1.01, step=0.1))
plt.grid(axis='y')
plt.ylabel('AUC')
plt.legend(title='Threshold', bbox_to_anchor=(1.0,1.02))
plt.title('ROC Area under Curve (AUC) for different models\nat different thresholds')
plt.gcf().set_size_inches(5,5)

plt.savefig('results_plots/AUC_compare_levels.png', dpi=300, bbox_inches='tight')



# %%
### calculate stats
res = df_res.groupby(['Model', 'Threshold'], as_index=False).agg({'AUC':['mean','std']})
res.columns = ['Model','Threshold','Mean','Std']
res.to_csv('./results/AUC_stats.csv', index=False)

res = df_res.groupby(['Model'], as_index=False).agg({'AUC':['mean','std']})
res.columns = ['Model','Mean','Std']
res.to_csv('./results/AUC_stats_model.csv', index=False)


# %%
