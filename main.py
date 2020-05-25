# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:47:40 2020

@author: vcheplyg
"""

#import gdown
import json, csv
import pandas as pd
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import math

from skimage import draw
from parse import *

import load_data as crowdload

import show_results as crowdshow


import analyze as crowdanalyze





def main():
        
    # Process data - Only do this if there are no processed files, or a flag is set? (TODO)
    # crowdload.process_data() 
    
    # Load all the processed files 
    df_task, df_res, df_annot, df_truth, df_subjects = crowdload.get_df_processed() 
    
    
    df_res_valid, df_res_invalid = crowdanalyze.get_valid_results(df_res,df_truth)
    
    # How many results are there? How many are valid? 
    crowdanalyze.print_result_stats(df_res_valid, df_res_invalid)
    
    
    crowdanalyze.plot_workers_vs_results(df_res_valid)
    crowdanalyze.scatter_valid_vs_invalid(df_res_valid, df_res_invalid)

    
    
    
    