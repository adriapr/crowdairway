# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:10:55 2020

@author: vcheplyg
"""

import pandas as pd
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import math

from skimage import draw
from parse import *

import load_data as crowdload

# Define how to combine multiple results per task 

#Although not combining, there is a decision made on what counts as a valid result
def get_valid_results(df_res, df_truth):
    
    # Select "easy" valid results (one pair of resized ellipses)
    cond_valid = df_res['inside'] & df_res['resized'] & (df_res['num_annot']==2)
    df_res_valid = df_res.loc[cond_valid]
    
    # Everything else is invalid 
    df_res_invalid =  df_res.loc[~cond_valid]
    
    
    # Add ground truth to valid results
    df_res_valid = pd.merge(df_res_valid, df_truth, on='task_id', how='outer')    
    
    return df_res_valid, df_res_invalid




def get_task_median(df_task, df_res, df_truth):

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
                    'wap_median': (outer_median - inner_median) / outer_median * 100,
                     # TODO: We should not be calling a function in the middle of nowhere (area_to_diam), which at the moment is defined multiple times
                    'wtr_median': ((crowdload.area_to_diam(outer_median) - crowdload.area_to_diam(inner_median)) / 2) / crowdload.area_to_diam(outer_median),
                    }
        task_list.append(task_dict)
       

    df_task = pd.DataFrame(task_list)
    
    
    df_task = pd.merge(df_task, df_truth, on='task_id', how='outer')    
        
   
    return df_task



def get_task_best(df_task, df_res_valid, df_truth):

    task_list = []

    for task_id in df_task['task_id']:
      
        task_results = df_res_valid.loc[df_res_valid['task_id'] == task_id]
       
        truth_diff = np.abs(task_results['outer'] - task_results['outer1']) 

        ix_res_best = np.argmin(truth_diff) 
        res_best = task_results.iloc[ix_res_best]
     
        task_dict = {
                    'task_id':      task_id,
                    'outer_best':   res_best['outer'],
                    'inner_best':   res_best['inner'],
                    'wap_best': (res_best['outer'] - res_best['inner']) / res_best['outer'] * 100,
                    'wtr_best': ((crowdload.area_to_diam(res_best['outer']) - crowdload.area_to_diam(res_best['inner'])) / 2) / crowdload.area_to_diam(res_best['outer']),
                    }
        task_list.append(task_dict)
      
    df_task = pd.DataFrame(task_list)
        
    df_task = pd.merge(df_task, df_truth, on='task_id', how='outer')    
    
    return df_task

