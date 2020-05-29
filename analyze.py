# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:16:34 2020

@author: vcheplyg
"""

import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt

import combine as crowdcombine


#Path where figures will be stored (TODO put these all in 1 place)
fig_path ='figures'


#Print statistics about results
def print_result(df_res_valid, df_res_invalid):
    
    
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



#Print statistics about workers
def print_worker(df_res):
    res_per_worker=df_res.groupby(['result_creator']).count()[['result_id']]
    
    res = df_res['result_creator'].value_counts(ascending=True)
        
    
    print('Total workers: ' + str(res_per_worker.count()))
    print('Minimum results per worker: ' + str(res.min()))
    print('Maximum results per worker: ' + str(res.max()))
    
    

#Plot number of results, created by number of workers
def plot_result_worker(df_res):
    
    res = df_res['result_creator'].value_counts(ascending=True)
        
    x = np.arange(1,len(res))
    y =  np.cumsum(res)
    plt.plot(x, y[:-1])
    
    plt.xlabel('Number of annotators')
    plt.ylabel('Cumulative results made') 
    
    plt.show()
    
    
    
#Scatter workers, represented by number of their valid/invalid results   
def scatter_worker_valid(df_res_valid, df_res_invalid):
    
        
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
    
    
    
#Scatter task correlations between expert and crowd 
def scatter_correlation_expert_crowd(df_task_combined, df_truth, combine_type):
    
    
    df_task_combined = pd.merge(df_task_combined, df_truth, on='task_id', how='outer')    
    
    key = '_' + combine_type
        
    #Get the correlations
    corr_inner = df_task_combined['inner1'].corr(df_task_combined['inner' + key])
    corr_outer = df_task_combined['outer1'].corr(df_task_combined['outer' + key])
    corr_wap = df_task_combined['wap1'].corr(df_task_combined['wap' + key])
    corr_wtr = df_task_combined['wtr1'].corr(df_task_combined['wtr' + key])
    
        
    # Plot the areas
    fig, axes = plt.subplots(nrows=2, ncols=2)
        
    ax0 = df_task_combined.plot.scatter(ax=axes[0][0], x='inner1', y='inner'+key)
    ax0.set_xlabel('Expert 1')
    ax0.set_ylabel('Crowd, ' + combine_type)  
    ax0.set_title('Inner airway, corr={:01.3f}'.format(corr_inner))
    ax0.axis('equal')
    ax0.set_aspect('equal', adjustable="datalim")
    
      
    
    ax1 = df_task_combined.plot.scatter(ax=axes[0][1], x='outer1', y='outer'+key)
    ax1.set_xlabel('Expert 1')
    ax1.set_ylabel('Crowd, ' + combine_type)  
    ax1.set_title('Outer airway, corr={:01.3f}'.format(corr_outer))
    ax1.axis('equal')
    ax1.set_aspect('equal', adjustable="datalim")
    
      
    ax2 = df_task_combined.plot.scatter(ax=axes[1][0], x='wap1',  y='wap'+key)
    ax2.set_xlabel('Expert 1')
    ax2.set_ylabel('Crowd, ' + combine_type)  
    ax2.set_title('WAP, corr={:01.3f}'.format(corr_wap)) #TODO OMG WHY does this print out an extra line
    ax2.axis('equal')
    ax2.set_aspect('equal', adjustable="datalim")
    
    ax3 = df_task_combined.plot.scatter(ax=axes[1][1], x='wtr1',  y='wtr'+key)
    ax3.set_xlabel('Expert 1')
    ax3.set_ylabel('Crowd, ' + combine_type)  
    ax3.set_title('WTR, corr={:01.3f}'.format(corr_wtr)) #TODO OMG WHY does this print out an extra line
    ax3.axis('equal')
    ax3.set_aspect('equal', adjustable="datalim")
    
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path, 'scatter_correlation' + key + '.png'))
 


#Plot the correlation against mininum number of valid results for each task
def plot_correlation_valid(df_task_combined, df_truth, combine_type):
       
    
    df_task_combined = pd.merge(df_task_combined, df_truth, on='task_id', how='outer')
    
    key = '_' + combine_type
        
    minimum_results = np.arange(1,11)
    n_min = len(minimum_results)
    corr_inner = np.zeros(n_min)
    corr_outer = np.zeros(n_min)
    num_tasks = np.zeros(n_min)
    
    
    for idx, m in enumerate(minimum_results):
    
        df_task_subset = df_task_combined.loc[df_task_combined['num_combined']>=m]
    
        corr_inner[idx] = df_task_subset['inner1'].corr(df_task_subset['inner' + key])
        corr_outer[idx] = df_task_subset['outer1'].corr(df_task_subset['outer' + key])
        num_tasks[idx] = df_task_subset['task_id'].count()
            
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Minimum number of valid results')
    ax1.set_ylabel('Correlation crowd with expert 1', color=color)
    ax1.plot(minimum_results, corr_inner, label='Inner',  color=color, linestyle='-')
    ax1.plot(minimum_results, corr_outer, label='Outer',  color=color, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend()
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('Number of tasks analyzed', color=color)  # we already handled the x-label with ax1
    ax2.plot(minimum_results, num_tasks, color=color, linestyle=':', label='Tasks')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend() #TODO combine legend
    
    fig.tight_layout() 
    fig.savefig(os.path.join(fig_path, 'plot_correlation_valid' + key + '.png'))
    


def get_subject_correlation(df_subject, df_task_combined, combine_type=''):

    key = '_' + combine_type
   
    subject_ids = df_task_combined['subject_id'].unique()
    corr_list = []
    
    for idx, subject_id in enumerate(subject_ids):
    
        subject_tasks = df_task_combined.loc[df_task_combined['subject_id'] == subject_id]
        
        print(subject_tasks.head())
    
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
    


    
#Scatter correlations per subject, while displaying subject characteristics
def scatter_subject_correlation(df_subject, df_task_combined, combine_type):
  
    df_corr = get_subject_correlation(df_subject, df_task_combined, combine_type)
    df_corr = pd.merge(df_corr, df_subject, on='subject_id', how='outer')
 
    
    groups = df_corr.groupby('has_cf')
    
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
       ax.plot(group['inner1_inner'], group['outer1_outer'], marker='o', linestyle='', label=name)
       #TODO: needs ms=group['FEV1_ppred'].astype(float) (doesn't work)
       
    L = ax.legend(numpoints=1)
    L.get_texts()[0].set_text('no CF')
    L.get_texts()[1].set_text('has CF')

    
    plt.xlabel('Correlation with expert, inner')
    plt.ylabel('Correlation with expert, outer')
    plt.show()
    
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path, 'scatter_subject_correlation.png'))
    
    

