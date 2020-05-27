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





def main():
        
    # Process data - Only do this if there are no processed files, or a flag is set? (TODO)
    # crowdload.process_data() 
    
    # Load all the processed files 
    df_task, df_res, df_annot, df_truth, df_subject = crowdload.get_df_processed() 
    
    # Select valid results and analyze their statistics
    df_res_valid, df_res_invalid = crowdcombine.get_valid_results(df_res,df_truth)
    
    # How many results are there? How many are valid? 
    crowdanalyze.print_result(df_res_valid, df_res_invalid)
    
    crowdanalyze.plot_worker_result(df_res_valid)
    
    crowdanalyze.scatter_worker_valid(df_res_valid, df_res_invalid)


    #Combine results in different ways and compare to expert
    
    df_task_median = crowdcombine.get_task_median(df_task, df_res_valid, df_truth)
    #df_task_best = crowdcombine.get_task_best(df_task, df_res_valid, df_truth) #Bug in get_task_best
        
    # Scatter individual results without combining
    crowdanalyze.scatter_correlation(df_res_valid)
    
    # Scatter combined results
    crowdanalyze.scatter_corr(df_task_median, 'median')
    #crowdanalyze.scatter_corr(df_task_best, 'best')     
    
    # Scatter subjects, do the correlations depend on something?
    crowdanalyze.scatter_subjects(df_subject, df_task, df_res_valid)
    
    
    
    