# %%
import pandas as pd
import os.path as osp
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

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

res = []
df_vote_score = pd.DataFrame([])
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

    result_type = '*_test.csv'
    result_files = glob.glob(osp.join(res_folder, result_type))
    result_files = [x for x in result_files if not ('small' in x)]
    if 'small' in result_type:
        case_type = 'Voting result (single)'
    else:
        case_type = 'Voting result (single entire test set)'
    
    num_models = 0
    for idx, result_f in enumerate(result_files):    
        segs = osp.basename(result_f).split('_s')
        thresh = segs[0].split('_')[-1]
        seed = segs[1].split('_')[0]

        df = pd.read_csv(result_f)
        df['tile'] = df.index
        df['thresh'] = thresh
        df['seed'] = seed
        
        if idx==0:
            df_results = df
        else:
            df_results = pd.concat([df_results, df])
        num_models += 1

    df_results['pred'] = df_results['yhat'].apply(binary_pred)
    df_vote = df_results.groupby(['tile', 'ytruth'])['pred'].sum().reset_index()
    del df_vote['tile']
    df_vote['avg'] = df_vote['pred']/num_models
    df_vote['vote'] = df_vote['avg'].apply(binary_pred_int)
    # save prediciton score for AUC
    df_vote['model'] = model
    df_vote_score = pd.concat([df_vote_score, df_vote])
    # df_vote['decision'] = df_vote['vote']==df_vote['ytruth']
    # accu = df_vote['decision'].sum() / len(df_vote)
    

    target_names = ['normal', 'tumor']
    report = classification_report(df_vote['ytruth'], df_vote['vote'], target_names=target_names)
    print(report)

    # get specific results
    report = classification_report(df_vote['ytruth'], df_vote['vote'], target_names=target_names,
                                   output_dict=True)
    df = pd.DataFrame(report).transpose()
    accu = df['precision'].values[2]
    f1score = df['f1-score'].values[4]  # 3 macro avg, 4 weighted avg
    sensitivity = df['recall'].values[1]
    specificity = df['recall'].values[0]

    res.append([model, case_type, 'accuracy', accu])
    res.append([model, case_type, 'f1-score', f1score])
    res.append([model, case_type, 'sensitivity', sensitivity])
    res.append([model, case_type, 'specificity', specificity])


df_result = pd.DataFrame(res, columns=['model','case_type','metric','value'])
df_result.to_csv('results/exp1_voting_results_entire_test.csv', index=False)

df_vote_score = df_vote_score[['model','ytruth','avg']]
df_vote_score.to_csv('results/voting_scores_entire_test.csv', index=False)




# %%
