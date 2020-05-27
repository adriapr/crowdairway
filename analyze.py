# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:16:34 2020

@author: vcheplyg
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import os.path


import combine as crowdcombine


fig_path ='figures'



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




def print_worker(df_res):
    res_per_worker=df_res.groupby(['result_creator']).count()[['result_id']]
    
    res = df_res['result_creator'].value_counts(ascending=True)
        
    
    print('Total workers: ' + str(res_per_worker.count()))
    print('Minimum results per worker: ' + str(res.min()))
    print('Maximum results per worker: ' + str(res.max()))
    
    


def plot_result_worker(df_res):
    
    res = df_res['result_creator'].value_counts(ascending=True)
        
    x = np.arange(1,len(res))
    y =  np.cumsum(res)
    plt.plot(x, y[:-1])
    
    plt.xlabel('Number of annotators')
    plt.ylabel('Cumulative results made') 
    
    plt.show()
    
    
    
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
    
    
    
    
def scatter_correlation_expert_crowd(df_task, combine_type=''):
    
    #combine_type = 'median' ## '' for no combining, 'median', 'best'
       
    if combine_type != '':
        key = '_' + combine_type
    else:
        key = combine_type
        
    #Get the correlations
    corr_inner = df_task['inner1'].corr(df_task['inner' + key])
    corr_outer = df_task['outer1'].corr(df_task['outer' + key])
    corr_wap = df_task['wap1'].corr(df_task['wap' + key])
    corr_wtr = df_task['wtr1'].corr(df_task['wtr' + key])
    
        
    # Plot the areas
    fig, axes = plt.subplots(nrows=2, ncols=2)
    
    print(axes.shape)
    
    ax0 = df_task.plot.scatter(ax=axes[0][0], x='inner1', y='inner'+key)
    ax0.set_xlabel('Expert 1')
    ax0.set_ylabel('Crowd, ' + combine_type)  
    ax0.set_title('Inner airway, corr={:01.3f}'.format(corr_inner))
    ax0.axis('equal')
    ax0.set_aspect('equal', adjustable="datalim")
    
      
    
    ax1 = df_task.plot.scatter(ax=axes[0][1], x='outer1', y='outer'+key)
    ax1.set_xlabel('Expert 1')
    ax1.set_ylabel('Crowd, ' + combine_type)  
    ax1.set_title('Outer airway, corr={:01.3f}'.format(corr_outer))
    ax1.axis('equal')
    ax1.set_aspect('equal', adjustable="datalim")
    
      
    ax2 = df_task.plot.scatter(ax=axes[1][0], x='wap1',  y='wap'+key)
    ax2.set_xlabel('Expert 1')
    ax2.set_ylabel('Crowd, ' + combine_type)  
    ax2.set_title('WAP, corr={:01.3f}'.format(corr_wap)) #TODO OMG WHY does this print out an extra line
    ax2.axis('equal')
    ax2.set_aspect('equal', adjustable="datalim")
    
    ax3 = df_task.plot.scatter(ax=axes[1][1], x='wtr1',  y='wtr'+key)
    ax3.set_xlabel('Expert 1')
    ax3.set_ylabel('Crowd, ' + combine_type)  
    ax3.set_title('WTR, corr={:01.3f}'.format(corr_wtr)) #TODO OMG WHY does this print out an extra line
    ax3.axis('equal')
    ax3.set_aspect('equal', adjustable="datalim")
    
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path, 'scatter_correlation' + key + '.png'))
 


#TODO generalize this to any combining
def plot_correlation_valid(df_task, df_res_valid, df_truth, combine_type=''):
    
    
    if combine_type == 'median':
        df_task = crowdcombine.get_task_median(df_task, df_res_valid, df_truth)
    
    elif combine_type == 'best':
        df_task = crowdcombine.get_task_best(df_task, df_res_valid, df_truth)
    else:
        print('No such combine type')
            
    
    key = '_' + combine_type
        
    minimum_results = np.arange(1,11)
    n_min = len(minimum_results)
    corr_inner = np.zeros(n_min)
    corr_outer = np.zeros(n_min)
    num_tasks = np.zeros(n_min)
    
    
    for idx, m in enumerate(minimum_results):
    
        df_task_subset = df_task.loc[df_task['num_combined']>=m]
    
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
    fig.savefig(os.path.join(fig_path, 'plot_correlation_valid.png'))
    


def get_subject_correlation(df_subject, df_task, df_res_valid):

    subject_ids = df_task['subject_id'].unique()
    corr_list = []
    
    for idx, subject_id in enumerate(subject_ids):
    
        subject_results = df_res_valid.loc[df_res_valid['subject_id'] == subject_id]
    
        n = len(subject_results['subject_id'])
        inner1_inner = subject_results['inner1'].corr(subject_results['inner'])
        outer1_outer = subject_results['outer1'].corr(subject_results['outer'])
        wap1_wap = subject_results['wap1'].corr(subject_results['wap'])
        wtr1_wtr = subject_results['wtr1'].corr(subject_results['wtr'])
        
    
        corr_dict = {
            'n': n,
            'subject_id': subject_id,
            'inner1_inner': inner1_inner,
            'outer1_outer': outer1_outer,
            'wap1_wap': outer1_outer,
            'wtr1_wtr': outer1_outer,
            }
        corr_list.append(corr_dict)  
     
    df_corr = pd.DataFrame(corr_list)
    
    return df_corr
    
    
#TODO generalize this
def scatter_subject_correlation(df_subject, df_task, df_res_valid):
       
    # TODO - correlation per subject - are some subjects more difficult than others? 
    # Are there differences between CF and non CF? 
    
    
    df_corr = get_subject_correlation(df_subject, df_task, df_res_valid)
    corr_inner = df_corr['inner1_inner'].to_numpy()
    corr_outer = df_corr['outer1_outer'].to_numpy()
    
    
    
    #Different variables related to the subjects, that we can display in the plot
   
    id = df_subject['subject_id'].to_numpy()
    has_cf = df_subject['has_cf'].to_numpy()
    fev = df_subject['FEV1_ppred'].to_numpy()
    fvc= df_subject['FVC1_ppred'].to_numpy()
    
    #Plot the correlations, visualize subject status and FEV as color/size
    fig = plt.figure()
    plt.scatter(corr_inner[has_cf==1],corr_outer[has_cf==1], edgecolor='red', facecolor='none', s=fev[has_cf==1])
    plt.scatter(corr_inner[has_cf==0],corr_outer[has_cf==0], edgecolor='blue', facecolor='none', s=fev[has_cf==0])
    
 
    
     #ax = df_corr.plot.scatter(x='inner1_inner',  y='outer1_outer')
    
#
    #TODO this is out of order now 
    #for idx, subject_id in enumerate(id):
    #    plt.annotate(subject_id, (corr_inner[idx]+0.01, corr_outer[idx]-0.02))
    
    plt.xlabel('Correlation with expert, inner')
    plt.ylabel('Correlation with expert, outer')
    plt.legend('CF', 'no CF')
    plt.show()
    
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path, 'scatter_subject_correlation.png'))
    
    

