# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:10:55 2020

@author: vcheplyg
"""

import pandas as pd
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import random

from skimage import draw
from parse import *

import load_data as crowdload

# Define how to combine multiple results per task 

#Although not combining, there is a decision made on what counts as a valid result
def get_valid_results(df_res):
    
    # Select "easy" valid results (one pair of resized ellipses)
    cond_valid = df_res['inside'] & df_res['resized'] & (df_res['num_annot']==2)
    df_res_valid = df_res.loc[cond_valid]
    
    # Everything else is invalid 
    df_res_invalid =  df_res.loc[~cond_valid]
    
    
    # Add ground truth to valid results - Should not be here to avoid leakage
    #df_res_valid = pd.merge(df_res_valid, df_truth, on='task_id', how='outer')    
    
    return df_res_valid, df_res_invalid



#Select a random result for a task
def get_task_random(df_task, df_res):
   
       
    task_list = []

    for task_id in df_task['task_id']:
      
        
        
        task_results = df_res.loc[df_res['task_id'] == task_id]
                
        if len(task_results)>0:
            res_random = task_results.sample(random_state=task_id) #For reproducibility 
            num_combined = len(task_results)
            outer_random = res_random['outer'].to_numpy()[0]
            inner_random = res_random['inner'].to_numpy()[0]
            wap_random = crowdload.compute_wap(inner_random,outer_random) 
            wtr_random = crowdload.compute_wap(inner_random,outer_random) 
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
    

#Combine all the results for a task with median combining
def get_task_median(df_task, df_res):

    task_list = []

    for task_id in df_task['task_id']:
        
        subject_id = df_task['subject_id'].to_numpy()[0]
        airway_id = df_task['airway_id'].to_numpy()[0]
        
      
        task_results = df_res.loc[df_res['task_id'] == task_id]
       
        outer_median = task_results['outer'].median()
        inner_median = task_results['inner'].median()
      
        task_dict = {
                    'task_id':      task_id,
                    'num_combined': len(task_results),
                    'outer_median': outer_median,
                    'inner_median': inner_median,
                    'wap_median': crowdload.compute_wap(inner_median,outer_median),
                    'wtr_median': crowdload.compute_wtr(inner_median,outer_median),
                    }
        task_list.append(task_dict)
       

    df_task_combined = pd.DataFrame(task_list)
    df_task_combined = pd.merge(df_task_combined, df_task, on='task_id', how='outer')    
           
    return df_task_combined



#Select the best possible result for a task (optimistically biased!) 
def get_task_best(df_task, df_res_valid, df_truth):

    task_list = []


    for task_id in df_task['task_id']:
       
       
        task_results = df_res_valid.loc[df_res_valid['task_id'] == task_id]
       
        
        task_truth = df_truth.loc[df_truth['task_id'] == task_id]
        
               
        truth_diff = np.abs(task_results['outer'].to_numpy() - task_truth['outer1'].to_numpy()) #Assumption - selecting based on outer, but could be one of the others

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
            wap_best = crowdload.compute_wap(inner_best,outer_best)
            wtr_best = crowdload.compute_wtr(inner_best,outer_best)
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
        
    #df_task = pd.merge(df_task, df_truth, on='task_id', how='outer')    
    
    return df_task

