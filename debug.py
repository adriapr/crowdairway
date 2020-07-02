# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:19:28 2020

@author: vcheplyg
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:22:02 2020

@author: vcheplyg
"""

import pandas as pd
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage import draw
from zipfile import ZipFile

import os

import load_data as crowdload

zip_path = os.path.join('data','tasks.zip')



def sort_tasks_by_valid_results(df_task, df_res_valid):
    
    #How many valid results does each task have
    res = df_res_valid['task_id'].value_counts(ascending=True).to_frame()
    res.reset_index(level=0, inplace=True)
   
    res.columns = ['task_id', 'valid_count']
  

    for index, task in df_task.iterrows():
        
        #print(task)
        task_id = task['task_id']
        subject_id = task['subject_id']
        airway_id =  task['airway_id']
        
        valid_count = res.loc[res['task_id'] == task_id]['valid_count']
        if valid_count.size>0:
            valid_count = valid_count.values[0]
            
            task_file = 'data({}).airways({}).viewpoints(1).png'.format(subject_id, airway_id)
            
            with ZipFile(zip_path, 'r') as zip: 
                with zip.open(task_file) as myfile:
                    im = mpimg.imread(myfile)
                    
                    fig = plt.figure()
                    plt.imshow(im, cmap="gray")
                    plt.title('valid {}, task {}, subject {}, airway {}'.format(valid_count, task_id, subject_id, airway_id))
                    
                    save_file = 'valid({}).task({}).png'.format(valid_count, task_id)
                    fig.savefig(os.path.join('tasks', save_file), format="png")
