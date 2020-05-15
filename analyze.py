# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:16:34 2020

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

import show_results as crowdshow


#Here a decision is made - maybe should be elsewhere?
def get_valid_results(df_res, df_truth):
    
    # Select "easy" valid results (one pair of resized ellipses)
    cond_valid = df_res['inside'] & df_res['resized'] & (df_res['num_annot']==2)
    df_res_valid = df_res.loc[cond_valid]
    
    # Everything else is invalid 
    df_res_invalid =  df_res.loc[~cond_valid]
    
    
    # Add ground truth to valid results
    df_res_valid = pd.merge(df_res_valid, df_truth, on='task_id', how='outer')    
    
    return df_res_valid, df_res_invalid



def print_result_stats(df_res_valid, df_res_invalid):
    
    
    num_valid = df_res_valid['result_id'].count()
    num_invalid = df_res_invalid['result_id'].count()
    num_total = num_valid + num_invalid
    
    print('Total ' + str(num_total))
    print('Valid ' + str(num_valid))
    print('Invalid ' + str(num_invalid))
    
    print('Invalid % ' + str(num_invalid / num_total))
    print('Invalid % preliminary ' + str(610/900))
    
    
    # Why are results invalid? 
    cond_inside = df_res_invalid['inside'] == True
    cond_resized = df_res_invalid['resized'] == True
    
    cond_one = df_res_invalid['num_annot'] == 1
    cond_two = df_res_invalid['num_annot'] == 2
    cond_many = df_res_invalid['num_annot'] >= 3
    
    # Not inside, not resized, 1 annotation - did not see airway or spam  
    val2 = df_res_invalid['result_id'].loc[~cond_inside & ~cond_resized & cond_one].count()
    
    # Not inside and not resized, 2+ annotations - probably tried to annotate
    val3 = df_res_invalid['result_id'].loc[~cond_inside & ~cond_resized & (cond_two | cond_many)].count()
    
    # Inside but not resized, or vice versa - probably tried to annotate
    val4 = df_res_invalid['result_id'].loc[cond_inside & ~cond_resized].count()
    val5 = df_res_invalid['result_id'].loc[~cond_inside & cond_resized].count()
    
    # Resized and inside, but not one pair - technically valid, but now exluded for simpler analysis 
    val6 = df_res_invalid['result_id'].loc[cond_inside & cond_resized & ~cond_two].count()
    
    print('Did not see airway or spam: ' + str(val2))
    print('Tried to annotate but did not read instructions: ' + str(val3+val4+val5))
    print('Excluded for simpler analysis: ' + str(val6))



def plot_results_per_worker(df_res):
    
    res = df_res['result_creator'].value_counts(ascending=True)
    
    res_per_worker=df_res.groupby(['result_creator']).count()[['result_id']]
    
    print('Total workers: ' + str(res_per_worker.count()))
    
    print('Minimum results per worker: ' + str(res.min()))
    print('Maximum results per worker: ' + str(res.max()))
    
    x = np.arange(1,len(res))
    y =  np.cumsum(res)
    plt.plot(x, y[:-1])
    
    plt.xlabel('Number of annotators')
    plt.ylabel('Cumulative results made') 
    
    plt.show()
    
    
    
def plot_valid_per_worker(df_res_valid, df_res_invalid):
    
        
    valid_per_worker=df_res_valid.groupby(['result_creator'], as_index=False).count().reset_index()
    num_worker = len(valid_per_worker)
    
    worker_num_valid = np.zeros(num_worker)
    worker_num_invalid = np.zeros(num_worker)
    
    for index, worker in valid_per_worker.iterrows():
        
        worker_res_valid = df_res_valid.loc[df_res_valid['result_creator'] == worker['result_creator']] 
        worker_num_valid[index] = worker_res_valid['task_id'].count()
    
        worker_res_invalid = df_res_invalid.loc[df_res_invalid['result_creator'] == worker['result_creator']] 
        worker_num_invalid[index] = worker_res_invalid['task_id'].count()
    
    
    #Plot stuff
    x = worker_num_valid
    y = worker_num_invalid
    
    
    max_value = np.max((np.max(x), np.max(y)))
    
    plt.scatter(x,y)
    
    m, b = np.polyfit(x, y, 1) 
    plt.plot(x, m*x + b) 
    
    plt.plot(np.arange(0,max_value), np.arange(0,max_value))
    
    plt.xlabel('Valid results created')
    plt.ylabel('Invalid results created')
    plt.legend(('fit to data', 'equal ratio', 'worker'))
    plt.show()
