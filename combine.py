# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:10:55 2020

@author: vcheplyg
"""

# Define how to combine multiple results per task 

def get_task_median(df_task, df_res):

    task_list = []

    for task_id in df_task['task_id']:
      
        task_results = df_res.loc[df_res['task_id'] == task_id]
       
        combined_outer = task_results['outer'].median()
        combined_inner = task_results['inner'].median()
      
        task_dict = {
                    'task_id':      task_id,
                    'num_combined': len(task_results),
                    'outer_median': combined_outer,
                    'inner_median': combined_inner,
                    }
        task_list.append(task_dict)
       

    df_task = pd.DataFrame(task_list)
   
    return df_task


df_task_median = get_task_median(df_task, df_res_valid)
df_task_median.head(5)