# %%
### load library
import os.path as osp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import RocCurveDisplay
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html#sklearn.metrics.RocCurveDisplay.from_predictions
from sklearn.metrics import roc_auc_score

result_path = '/users/PAS1575/chen8028/Pathology/individual_mask/results/voting_scores_entire_test.csv'


# %%
### calculate ROC's AUC scores
df_scores = pd.read_csv(result_path)

models = ['VGG19', 'ResNet50', 'Inceptionv3', 'ViT']

res = []
for model in models:
    # read in result
    df_test = df_scores[df_scores['model']==model]
    # ground truth labels
    y_test = np.array(df_test['ytruth'].to_list())
    # prediction scores or probability
    y_pred = np.array(df_test['avg'].to_list())
    
    # calculate AUC
    auc = roc_auc_score(y_test, y_pred)
    res.append([model, auc])

df_res = pd.DataFrame(data=res, 
                      columns=['Model','AUC'])
df_res.to_csv('./results/roc_auc_by_voting.csv', index=False)



# %%
### plot ROC curve
# RocCurveDisplay.from_predictions(y_test, y_pred)
# plt.show()


# %%
### plot the results
import seaborn as sns

df_res = pd.read_csv('./results/roc_auc_by_voting.csv')

pat = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=False, n_colors=6)
g = sns.barplot(data=df_res, x='Model', y='AUC', 
                color='grey', width=0.4, zorder=2)

plt.ylim(0.0,1.0) 
plt.xlabel('')
plt.yticks(np.arange(0, 1.01, step=0.1))
plt.grid(axis='y')
plt.ylabel('AUC')
# plt.legend(title='Threshold', bbox_to_anchor=(1.0,1.02))
plt.title('ROC area under curve (AUC) for different models\nmajority voting results')
plt.gcf().set_size_inches(5,5)

plt.savefig('results_plots/AUC_compare_voting.png', dpi=300, bbox_inches='tight')



# %%
