# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:47:40 2020

@author: vcheplyg
"""

import pandas as pd
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from skimage import draw
from parse import *

import load_data as crowdload

import analyze as crowdanalyze
import combine as crowdcombine


#Whether to import/process all data again
process_data = False


if process_data:
    crowdload.process_data()  
    
# Load all the processed files 
df_task, df_res, df_annot, df_truth, df_subject = crowdload.get_df_processed() 
    
# Select valid results 
df_res_valid, df_res_invalid = crowdcombine.get_valid_results(df_res)
    
# Analyze result statistics
#crowdanalyze.print_result(df_res_valid, df_res_invalid)
#crowdanalyze.print_worker(df_res)
#crowdanalyze.plot_result_worker(df_res_valid)
#crowdanalyze.scatter_worker_valid(df_res_valid, df_res_invalid)
    
    
#Combine results per task in different ways
df_task_random = crowdcombine.get_task_random(df_task, df_res_valid)
df_task_median = crowdcombine.get_task_median(df_task, df_res_valid)
df_task_best = crowdcombine.get_task_best(df_task, df_res_valid, df_truth) #optimistically biased!

#crowdanalyze.scatter_correlation_expert_crowd(df_task_random, df_truth, 'random')
#crowdanalyze.scatter_correlation_expert_crowd(df_task_median, df_truth, 'median')
#crowdanalyze.scatter_correlation_expert_crowd(df_task_best, df_truth, 'best')  


#Correlation vs minimum number of available valid results 
#crowdanalyze.plot_correlation_valid(df_task_random, df_truth, 'random')
#crowdanalyze.plot_correlation_valid(df_task_median, df_truth, 'median')
#crowdanalyze.plot_correlation_valid(df_task_best, df_truth, 'best')
    
#crowdanalyze.plot_subject_correlation(df_subject, df_task_median, df_truth, 'median')

#Print subject characteristics    
#crowdanalyze.print_subject(df_subject, df_task_median, df_truth, 'median')
    
crowdanalyze.scatter_correlation_by_part(df_task_random, df_task_median, df_task_best, df_truth, 'inner')
crowdanalyze.scatter_correlation_by_part(df_task_random, df_task_median, df_task_best, df_truth, 'outer')
crowdanalyze.scatter_correlation_by_part(df_task_random, df_task_median, df_task_best, df_truth, 'wap')
crowdanalyze.scatter_correlation_by_part(df_task_random, df_task_median, df_task_best, df_truth, 'wtr')


#crowdanalyze.scatter_correlation_experts(df_task_median, df_truth, 'median')

#crowdanalyze.print_subject_correlation(df_subject, df_task_median, df_truth, 'median')

     

    
    