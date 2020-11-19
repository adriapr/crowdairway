# -*- coding: utf-8 -*-
"""
analysis.py

Decision-making functions on how to filter, combine and analyse crowd results. 

Authors: Veronika Cheplygina, Adria Perez-Rovira
URL: https://github.com/adriapr/crowdairway


"""

import pandas as pd
import numpy as np
import data as crowddata


def get_valid_results(df_res):
    """Splits up results into valid and invalid results.
    Based on properties computed in data.py."""
    
    # Select valid results (one pair of resized ellipses)
    cond_valid = df_res['inside'] & df_res['resized'] & (df_res['num_annot']==2)
    df_res_valid = df_res.loc[cond_valid]
    
    # Everything else is excluded/invalid 
    df_res_invalid =  df_res.loc[~cond_valid]
           
    return df_res_valid, df_res_invalid


def get_task_random(df_task, df_res):
    """Selects a random result for a task."""
   
       
    task_list = []

    for task_id in df_task['task_id']:
         
        
        task_results = df_res.loc[df_res['task_id'] == task_id]
                
        if len(task_results)>0:
            res_random = task_results.sample(random_state=task_id) #For reproducibility 
            num_combined = len(task_results)
            outer_random = res_random['outer'].to_numpy()[0]
            inner_random = res_random['inner'].to_numpy()[0]
            wap_random = crowddata.compute_wap(inner_random,outer_random) 
            wtr_random = crowddata.compute_wtr(inner_random,outer_random) 
        else:
            num_combined = 0
            outer_random = np.nan
            inner_random = np.nan
            wap_random = np.nan
            wtr_random = np.nan
           
            
        task_dict = {
                    'task_id':      task_id,
                    'num_combined': num_combined,
                    'outer_random': outer_random,
                    'inner_random': inner_random,
                    'wap_random': wap_random, 
                    'wtr_random': wtr_random,
                    }
        task_list.append(task_dict)
       

    df_task_combined = pd.DataFrame(task_list)
    df_task_combined = pd.merge(df_task_combined, df_task, on='task_id', how='outer')    
           
    return df_task_combined
    

def get_task_median(df_task, df_res):
    """Combines all the results for a task using median combining."""

    task_list = []

    for task_id in df_task['task_id']:
       
      
        task_results = df_res.loc[df_res['task_id'] == task_id]
       
        outer_median = task_results['outer'].median()
        inner_median = task_results['inner'].median()
      
        task_dict = {
                    'task_id':      task_id,
                    'num_combined': len(task_results),
                    'outer_median': outer_median,
                    'inner_median': inner_median,
                    'wap_median': crowddata.compute_wap(inner_median,outer_median),
                    'wtr_median': crowddata.compute_wtr(inner_median,outer_median),
                    }
        task_list.append(task_dict)
       

    df_task_combined = pd.DataFrame(task_list)
    df_task_combined = pd.merge(df_task_combined, df_task, on='task_id', how='outer')    
           
    return df_task_combined


def get_task_best(df_task, df_res_valid, df_truth):
    """Selects the best possible result for a task, based on the ground truth.
    Optimistically biased, only for determining upper bound! 
    """
    
    task_list = []


    for task_id in df_task['task_id']:
       
       
        task_results = df_res_valid.loc[df_res_valid['task_id'] == task_id]
       
        
        task_truth = df_truth.loc[df_truth['task_id'] == task_id]
        
        # Measuring best by sum of absolute differences of inner and outer airways 
        outer_diff = np.abs(task_results['outer'].to_numpy() - task_truth['outer1'].to_numpy()) 
        inner_diff = np.abs(task_results['inner'].to_numpy() - task_truth['inner1'].to_numpy()) 

        truth_diff = outer_diff + inner_diff
        
        
        if all(np.isnan(truth_diff)):
            outer_best = np.nan
            inner_best = np.nan
            wap_best = np.nan
            wtr_best = np.nan
            num_combined = 0
        else:
            ix_res_best = np.argmin(truth_diff) 
            res_best = task_results.iloc[ix_res_best]
            outer_best = res_best['outer']
            inner_best = res_best['inner']   
            wap_best = crowddata.compute_wap(inner_best,outer_best)
            wtr_best = crowddata.compute_wtr(inner_best,outer_best)
            num_combined = np.sum(~np.isnan(truth_diff))
        
    
        task_dict = {
                    'task_id':      task_id,
                    'num_combined': num_combined,
                    'outer_best':   outer_best,
                    'inner_best':   inner_best,
                    'wap_best':     wap_best,
                    'wtr_best':     wtr_best,
                    }
        task_list.append(task_dict)
      
    df_task = pd.DataFrame(task_list)
     
    return df_task



def get_subject_correlation(df_subject, df_task_combined, df_truth, combine_type=''):
    """Returns a dataframe of correlations bewteen the combined crowd and the expert"""
    
    cols_to_use = ['task_id', 'inner1', 'outer1', 'wap1', 'wtr1']
    
    df_task_combined = pd.merge(df_task_combined, df_truth[cols_to_use], on='task_id', how='outer')
 
         
    key = '_' + combine_type
   
    subject_ids = df_subject['subject_id'].unique()
    corr_list = []
    
    for idx, subject_id in enumerate(subject_ids):
    
        subject_tasks = df_task_combined.loc[df_task_combined['subject_id'] == subject_id]
           
        n = len(subject_tasks['subject_id'])
        inner1_inner = subject_tasks['inner1'].corr(subject_tasks['inner' + key])
        outer1_outer = subject_tasks['outer1'].corr(subject_tasks['outer' + key])
        wap1_wap = subject_tasks['wap1'].corr(subject_tasks['wap' + key])
        wtr1_wtr = subject_tasks['wtr1'].corr(subject_tasks['wtr' + key])
        
    
        corr_dict = {
            'n': n,
            'subject_id': subject_id,
            'inner1_inner': inner1_inner,
            'outer1_outer': outer1_outer,
            'wap1_wap': wap1_wap,
            'wtr1_wtr': wtr1_wtr,
            }
        corr_list.append(corr_dict)  
     
    df_corr = pd.DataFrame(corr_list)
    
    return df_corr
