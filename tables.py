# -*- coding: utf-8 -*-
"""
Functions for printing tables or other text-based results


Authors: Veronika Cheplygina, Adria Perez-Rovira
URL: https://github.com/adriapr/crowdairway

"""

import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt
import seaborn as sns


import statsmodels.api as sm

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

import scipy.stats


import analysis as crowdanalysis



#Print statistics about results
def print_result(df_res_valid, df_res_invalid):
    
    
    num_valid = df_res_valid['result_id'].count()
    num_invalid = df_res_invalid['result_id'].count()
    num_total = num_valid + num_invalid
    
    print('Total ' + str(num_total))
    print('Valid ' + str(num_valid))
    print('Invalid ' + str(num_invalid))
    
    print('Invalid + excluded % {0:.{1}f} '.format(num_invalid / num_total * 100, 1))
    print('Invalid % preliminary {0:.{1}f} '.format(610/900*100, 1))
    
    
    # Why are results invalid? 
    cond_inside = df_res_invalid['inside'] == True
    cond_resized = df_res_invalid['resized'] == True
    
    cond_one = df_res_invalid['num_annot'] == 1
    cond_two = df_res_invalid['num_annot'] == 2
    cond_many = df_res_invalid['num_annot'] >= 3
    
    cond_see = df_res_invalid['cantsee'] == True
    
    # Not inside, not resized, 1 annotation - did not see airway or spam  
    val2 = df_res_invalid['result_id'].loc[~cond_inside & ~cond_resized & cond_one].count()
    
    val22 = df_res_invalid['result_id'].loc[cond_see].count()
    
    
    
    # Not inside and not resized, 2+ annotations - probably tried to annotate
    val3 = df_res_invalid['result_id'].loc[~cond_inside & ~cond_resized & (cond_two | cond_many)].count()
    
    # Inside but not resized, or vice versa - probably tried to annotate
    val4 = df_res_invalid['result_id'].loc[cond_inside & ~cond_resized].count()
    val5 = df_res_invalid['result_id'].loc[~cond_inside & cond_resized].count()
    
    # Resized and inside, but not one pair - technically valid, but now exluded for simpler analysis 
    val6 = df_res_invalid['result_id'].loc[cond_inside & cond_resized & ~cond_two].count()
    
    print('Only one annotation (did not see airway or did not read): ' + str(val2))
    print('Did not see airway: ' + str(val22))
    print('Two ore more annotations, but did not read instructions: ' + str(val3+val4+val5))
    print('Excluded for simpler analysis: ' + str(val6 / num_total * 100))
    print('Truly invalid: ' + str( (num_invalid - val6) / num_total * 100))
    


#Print statistics about workers
def print_worker(df_res):
    res_per_worker=df_res.groupby(['result_creator']).count()[['result_id']]
    
    res = df_res['result_creator'].value_counts(ascending=True)
        
    
    print('Total workers: ' + str(res_per_worker.count()))
    print('Minimum results per worker: ' + str(res.min()))
    print('Maximum results per worker: ' + str(res.max()))
    
    
    
    
#Print subject table
def print_subject(df_subject, df_task_combined, df_truth, combine_type):
          
    df_corr = crowdanalysis.get_subject_correlation(df_subject, df_task_combined, df_truth, combine_type)
        
    df_corr = pd.merge(df_subject, df_corr, on='subject_id', how='outer')    
    
    df_select = df_corr[['subject_id', 'has_cf', 'FVC1_ppred', 'FEV1_ppred', 'n', 'inner1_inner', 'outer1_outer', 'wap1_wap', 'wtr1_wtr']]
    

    print(df_select.to_latex(float_format="%.2f"))
    
    
   
#Print correlations between subject characteristics
def print_subject_correlation(df_subject, df_task_combined, df_truth, combine_type):
  
        
    df_corr = crowdanalysis.get_subject_correlation(df_subject, df_task_combined, df_truth, combine_type)
    df_corr = pd.merge(df_corr, df_subject, on='subject_id', how='outer')
    
    #Get average airway geenration from subjects
    df_truth_subject = df_truth.groupby('subject_id').mean()
    
    df_corr = pd.merge(df_corr, df_truth_subject[['generation']], on='subject_id', how='outer')
    
    
    #scipy.stats.pearsonr(x, y)    
    #scipy.stats.spearmanr(x, y)
    
    [corr_cf, p_cf] = scipy.stats.spearmanr(df_corr['inner1_inner'], df_corr['has_cf'])
    [corr_generation, p_generation] = scipy.stats.spearmanr(df_corr['inner1_inner'], df_corr['generation'])
    [corr_fvc, p_fvc] = scipy.stats.spearmanr(df_corr['inner1_inner'], df_corr['FVC1_ppred'])
    [corr_fev, p_fev] = scipy.stats.spearmanr(df_corr['inner1_inner'], df_corr['FEV1_ppred'])
    [corr_n, p_n] = scipy.stats.spearmanr(df_corr['inner1_inner'], df_corr['n'])
    
    
    
    X = df_corr[['inner1_inner', 'has_cf', 'FVC1_ppred', 'FEV1_ppred', 'n', 'generation']]
  
    print(X.corr())
      
    print(corr_cf)
    print(corr_generation)
    
    print('Has CF: {:01.3f}, p={:01.3f}'.format(corr_cf, p_cf))
    print('Generation: {:01.3f}, p={:01.3f}'.format(corr_generation, p_generation))
    print('FVC CF: {:01.3f}, p={:01.3f}'.format(corr_fvc, p_fvc))
    print('FEV: {:01.3f}, p={:01.3f}'.format(corr_fev, p_fev))
    print('Number airways: {:01.3f}, p={:01.3f}'.format(corr_n, p_n))
    