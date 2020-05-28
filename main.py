# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:47:40 2020

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



import analyze as crowdanalyze
import combine as crowdcombine

#Things that are working
def main():
    # Load all the processed files 
    df_task, df_res, df_annot, df_truth, df_subject = crowdload.get_df_processed() 
    
    # Select valid results and analyze their statistics
    df_res_valid, df_res_invalid = crowdcombine.get_valid_results(df_res)
    
    # How many results are there? How many workers are there? ? 
    crowdanalyze.print_result(df_res_valid, df_res_invalid)
    crowdanalyze.plot_result_worker(df_res_valid)
    crowdanalyze.scatter_worker_valid(df_res_valid, df_res_invalid)
    
    
    #Combine results per task in different ways, first, pick a random result
    df_task_random = crowdcombine.get_task_random(df_task, df_res_valid)
    crowdanalyze.scatter_correlation_expert_crowd(df_task_random, df_truth, 'random')
    
    
    #Combine all results per task with median combining
    df_task_median = crowdcombine.get_task_median(df_task, df_res_valid)
    crowdanalyze.scatter_correlation_expert_crowd(df_task_median, df_truth, 'median')

    #Select best result per task (optimistically biased, uses ground truth!)
    df_task_best = crowdcombine.get_task_best(df_task, df_res_valid, df_truth) 
    crowdanalyze.scatter_correlation_expert_crowd(df_task_best, df_truth, 'best')



#Development
def temp():
        
    # Load all the processed files 
    df_task, df_res, df_annot, df_truth, df_subject = crowdload.get_df_processed() 
    
    # Select valid results and analyze their statistics
    df_res_valid, df_res_invalid = crowdcombine.get_valid_results(df_res)
    
    #df_task_best = crowdcombine.get_task_best(df_task, df_res_valid, df_truth) 
    #crowdanalyze.scatter_correlation_expert_crowd(df_task_best, df_truth, 'best')

    
    
    
    
    
    