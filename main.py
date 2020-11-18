# -*- coding: utf-8 -*-
"""
Steps for reproducible results, similar to the Google Colab notebook

Created on Wed May 13 11:47:40 2020

@author: vcheplyg
"""

import data as crowddata
import analysis as crowdanalysis
import figures as crowdfigures
import tables as crowdtables


#####################
# Data
#####################

# Process data and save the processed data frames. This only needs to be done if the preprocessing code changes
use_processed_data = True

# Process data and save the proce
if use_processed_data == False:
    crowddata.process_data()  
    
# Load all the processed files 
df_task, df_res, df_annot, df_truth, df_subject = crowddata.get_df_processed() 


#####################
# Analysis
#####################
    
# Select valid results 
df_res_valid, df_res_invalid = crowdanalysis.get_valid_results(df_res)
    
#Combine results per task in different ways
df_task_random = crowdanalysis.get_task_random(df_task, df_res_valid)
df_task_median = crowdanalysis.get_task_median(df_task, df_res_valid)
df_task_best = crowdanalysis.get_task_best(df_task, df_res_valid, df_truth) #optimistically biased!


# From here on, medium combining is selected where only combining method is used. 
# TODO this should be handled by a single variable
df_task_combined = df_task_median
combine_type = 'median'

#Get correlations between crowd and expert
df_corr = crowdanalysis.get_subject_correlation(df_subject, df_task_combined, df_truth, combine_type)



#####################
# Table
#####################

# Statistics about workers and results
crowdtables.print_result(df_res_valid, df_res_invalid)
crowdtables.print_worker(df_res)

# Table 2- Correlations of different combining methods vs the expert 
#TODO

# Table 3 - Characteristics per subjects
crowdtables.print_subject(df_subject, df_task_median, df_truth, combine_type)

# Table 4 - Correlations between crowd quality and subject characteristics
crowdtables.print_subject_correlation(df_subject, df_task_median, df_truth, combine_type)


#####################
# Figures
#####################

# Figures 1 to 3 are illustrating the method and are not produced from the data

# Figure 4, statistics about workers and results
crowdfigures.plot_result_worker(df_res_valid)    #
crowdfigures.scatter_worker_valid(df_res_valid, df_res_invalid)
 
# Figure 5, Inner airway
crowdfigures.scatter_correlation_by_part(df_task_random, df_task_median, df_task_best, df_truth, 'inner')

# Figure 6, Outer airway
crowdfigures.scatter_correlation_by_part(df_task_random, df_task_median, df_task_best, df_truth, 'outer')

# Figure 7, WAP
crowdfigures.scatter_correlation_by_part(df_task_random, df_task_median, df_task_best, df_truth, 'wap')

# Figure 8, WTR 
crowdfigures.scatter_correlation_by_part(df_task_random, df_task_median, df_task_best, df_truth, 'wtr')

#Figure 9, Correlation vs minimum number of available valid results 
crowdfigures.plot_correlation_valid(df_task_combined, df_truth, combine_type)
    

    
    