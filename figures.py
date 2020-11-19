# -*- coding: utf-8 -*-
"""
Functions for printing figures in the paper. 

Authors: Veronika Cheplygina, Adria Perez-Rovira
URL: https://github.com/adriapr/crowdairway
"""

import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import seaborn as sns
from zipfile import ZipFile

import data as crowddata

#Path where the task images are stored
zip_path = os.path.join('data','tasks.zip')


#Path where figures will be stored 
fig_path ='figures'



def set_style():
    """Set general style for plots"""

    sns.set_style("ticks") # style
    sns.set_context("talk") # make things bigger
    plt.rcParams['figure.figsize'] = [12, 10] # make figures larger
    

def show_task(df_task, df_res, df_annot, *args, **kwargs):
    """ Show a task and optionally a result drawn on top of it
    Required: either task_id, or both subject_id and airway_id
    Optional: result_index (between 0 and 19) to show the result 
    """
    
    task_id = kwargs.get('task_id', None)
    subject_id = kwargs.get('subject_id', None)
    airway_id = kwargs.get('airway_id', None)
    result_index = kwargs.get('result_index', None)
    
    # TODO need potential error handling, like not existing IDs
   
    if task_id != None:    
        
        #Select corresponding task from data frame
        df = df_task.loc[df_task['task_id'] == task_id]
           
        subject_id = df['subject_id'].values[0]  #TODO also need to handle what happens if task_id isn't in the data
        airway_id =  df['airway_id'].values[0]
        
    else:
        df = df_task.loc[df_task['subject_id'] == subject_id]
        df = df.loc[df['airway_id'] == airway_id]
        task_id =  df['task_id'].values[0]
  
        
    
    task_file = 'data({}).airways({}).viewpoints(1).png'.format(subject_id, airway_id)
        
    #Unzip images if necessary
    with ZipFile(zip_path, 'r') as zip: 
        with zip.open(task_file) as myfile:
            im = mpimg.imread(myfile)
            plt.imshow(im, cmap="gray")
            plt.title('task {}, subject {}, airway {}'.format(task_id, subject_id, airway_id))
      
    if result_index != None:
        res = df_res.loc[df_res['task_id'] == task_id].reset_index()
        result_id = res['result_id'][result_index]
        
        annot = df_annot.loc[df_annot['result_id'] == result_id]
        
        #Display everything
        ax = plt.gca()  
           
        for (index,a) in annot.iterrows():
        
            ell_patch, vertices = crowddata.get_ellipse_patch_vertices(a)
            ell_patch.set_edgecolor('#6699FF')
            ell_patch.set_linewidth(3)
            ax.add_patch(ell_patch)
            
        plt.xlabel('result {}'.format(result_id))
        plt.show()    
    



def plot_result_worker(df_res):
    """Plot number of results, created by number of workers"""
        
    res = df_res['result_creator'].value_counts(ascending=True)
        
    x = np.arange(1,len(res))
    y =  np.cumsum(res)
    
    fig = plt.figure()
    plt.plot(x, y[:-1])
    
    plt.xlabel('Number of annotators')
    plt.ylabel('Cumulative results made') 
    
    sns.despine()
    
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path, 'plot_result_worker.png'), format="png")
    #doesn't work
    
    
        
    

def scatter_worker_valid(df_res_valid, df_res_invalid):
    """Scatter workers, represented by number of their valid/invalid results"""
        
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
    
    
    end_line = np.min((np.max(x), np.max(y)))
    
    
    fig = plt.figure()
    plt.scatter(x,y, alpha=0.3) # transaprency makes easier to see dense areas
    
    m, b = np.polyfit(x, y, 1) 
    plt.plot(x, m*x + b) 
    
    plt.plot([0, end_line], [0, end_line]) 
    plt.xlabel('Valid results created')
    plt.ylabel('Invalid results created')
    plt.legend(('fit to data', 'equal ratio', 'worker'))
    
    sns.despine()
    
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path, 'scatter_worker_valid.png'), format="png")
    #doesn't work
    
    
   

def scatter_correlation_by_part(df_random, df_median, df_best, df_truth, part):
    """Scatter task correlations between expert and crowd, for a specific measurement"""
    
    df_random = pd.merge(df_random, df_truth, on='task_id', how='outer')    
    df_median = pd.merge(df_median, df_truth, on='task_id', how='outer')
    df_best = pd.merge(df_best, df_truth, on='task_id', how='outer')
    
        
    #Get the correlations - TODO ideally we should have a method that does all combining at once
    
    corr1 = df_random[part+ '1'].corr(df_random[part+'_random'])
    corr2 = df_median[part+ '1'].corr(df_median[part+'_median'])
    corr3 = df_best[part+ '1'].corr(df_best[part+'_best'])
    corr4 = df_best[part+ '1'].corr(df_best[part+'2'])
  
    # Plot the areas
    fig, axes = plt.subplots(nrows=2, ncols=2)
        
    ax0 = df_random.plot.scatter(ax=axes[0][0], x=part+'1', y=part+'_random', alpha = 0.3)
    ax0.set_xlabel('Expert 1')
    ax0.set_ylabel('Crowd random')  
    
    t = part + ', corr={:01.3f}'.format(corr1)
    print(t)
    ax0.set_title(t)
  
    
    ax1 = df_median.plot.scatter(ax=axes[0][1], x=part+'1', y=part+'_median', alpha = 0.3)
    ax1.set_xlabel('Expert 1')
    ax1.set_ylabel('Crowd median')  
    t = part + ', corr={:01.3f}'.format(corr2)
    ax1.set_title(t)
    print(t)
    max_data = max(ax1.get_xlim()[1], ax1.get_ylim()[1])
    ax1.set_xlim(-5, max_data)
    ax1.set_ylim(-5, max_data)
    
      
    ax2 = df_best.plot.scatter(ax=axes[1][0], x=part+'1',  y=part+'_best', alpha = 0.3)
    ax2.set_xlabel('Expert 1')
    ax2.set_ylabel('Crowd best')  
    t = part + ', corr={:01.3f}'.format(corr3)
    ax2.set_title(t) 
    print(t)
    
    ax3 = df_best.plot.scatter(ax=axes[1][1], x=part+'1',  y=part+'2', alpha = 0.3)
    ax3.set_xlabel('Expert 1')
    ax3.set_ylabel('Expert 2')  
    t = part + ', corr={:01.3f}'.format(corr4)
    ax3.set_title(t)
    print(t)
    
    max_x = max(ax0.get_xlim()[1], ax1.get_xlim()[1], ax2.get_xlim()[1], ax3.get_xlim()[1])
    max_y = max(ax0.get_ylim()[1], ax1.get_ylim()[1], ax2.get_ylim()[1], ax3.get_ylim()[1])
    
    offset = -5
    if part=='wtr':
        offset = -0.05
    
    ax0.set_xlim(offset, max_x)
    ax0.set_ylim(offset, max_y)
    ax1.set_xlim(offset, max_x)
    ax1.set_ylim(offset, max_y)
    ax2.set_xlim(offset, max_x)
    ax2.set_ylim(offset, max_y)
    ax3.set_xlim(offset, max_x)
    ax3.set_ylim(offset, max_y)
    
    
    
    sns.despine()
    
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path, 'scatter_correlation_' + part + '.png'), format="png")



def plot_correlation_valid(df_task_combined, df_truth, combine_type):
    """Plot the correlation against mininum number of valid results for each task"""   
    
    df_task_combined = pd.merge(df_task_combined, df_truth, on='task_id', how='outer')
    
    key = '_' + combine_type
        
    minimum_results = np.arange(1,19)
    n_min = len(minimum_results)
    corr_inner = np.zeros(n_min)
    corr_outer = np.zeros(n_min)
    corr_WTR = np.zeros(n_min)
    corr_WAP = np.zeros(n_min)
    num_tasks = np.zeros(n_min)
    
    
    for idx, m in enumerate(minimum_results):
    
        df_task_subset = df_task_combined.loc[df_task_combined['num_combined']>=m]
        corr_inner[idx] = df_task_subset['inner1'].corr(df_task_subset['inner' + key])
        corr_outer[idx] = df_task_subset['outer1'].corr(df_task_subset['outer' + key])
        corr_WTR[idx] = df_task_subset['wtr1'].corr(df_task_subset['wtr' + key])
        corr_WAP[idx] = df_task_subset['wap1'].corr(df_task_subset['wap' + key])
        num_tasks[idx] = df_task_subset['task_id'].count()
           
    print(num_tasks)
    print(corr_inner)    
        
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Minimum number of valid results')
    ax1.set_ylabel('Correlation crowd with expert 1', color=color)
    ax1.plot(minimum_results, corr_outer, label='Outer',  color=color, linestyle='--')
    ax1.plot(minimum_results, corr_inner, label='Inner',  color=color, linestyle='-')
    ax1.plot(minimum_results, corr_WAP, label='WAP',  color='k', linestyle=':')
    ax1.plot(minimum_results, corr_WTR, label='WTR',  color='k', linestyle='-.')
    ax1.set_xticks(minimum_results)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='lower left')
    ax1.set_ylim(0,1)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('Number of tasks analyzed', color=color)  # we already handled the x-label with ax1
    ax2.plot(minimum_results, num_tasks, color=color, linestyle=':', label='Tasks')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='lower right')
    ax2.set_ylim(0,np.max(num_tasks)+50)
    
    fig.tight_layout() 
    fig.savefig(os.path.join(fig_path, 'plot_correlation_valid' + key + '.png'), format="png")



    


