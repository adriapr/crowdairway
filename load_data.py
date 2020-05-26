# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:59:52 2020

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
from parse import search

import os.path

# Define some constants
image_width = 500
image_height = 500

path_raw = os.path.join('crowdairway', 'data')
path_processed = os.path.join('crowdairway', 'data_processed')

file_subject = os.path.join(path_processed, 'subjects.csv')
file_truth = os.path.join(path_processed, 'airways_ground_truth.csv')
file_task = os.path.join(path_processed, 'tasks.csv')
file_res = os.path.join(path_processed, 'results.csv')
file_annot = os.path.join(path_processed, 'annotations.csv')



def get_df_processed():
    
    print(file_subject)
    
    df_subject = pd.read_csv(file_subject)
    
    df_truth=pd.read_csv(file_truth)
    df_task = pd.read_csv(file_task)
    df_res = pd.read_csv(file_res)
    df_annot = pd.read_csv(file_annot)
        
    return df_task, df_res, df_annot, df_truth, df_subject
    
    

# This only needs to be run once to get from raw data to processed files (a bit slow)
def process_data():
    #Load files
   
    results_file = os.path.join(path_raw, 'crowd_results.json')
    df_task, df_res, df_annot = get_df_crowd(results_file)
            
    print("Starting annotations...")
    df_ellipse = df_annot.apply(lambda annotation: get_annotation_ellipse(annotation), axis='columns', result_type='expand')
    df_annot_ellipse = pd.concat([df_annot, df_ellipse], axis='columns')

    # Process results  
    print("Starting results...")
    
    df_props = df_res.apply(lambda res: get_result_properties(res, df_annot_ellipse), axis='columns', result_type='expand')
    df_res_props = pd.concat([df_res,df_props],axis=1)
    
    
    df_truth=pd.read_csv(os.path.join(path_raw, 'airways_ground_truth.csv'))
    df_truth['wap1'] = df_truth.apply(lambda row: compute_wap(row), axis=1)
    df_truth['wtr1'] = df_truth.apply(lambda row: compute_wtr(row), axis=1)
    
    
    #Write processed files to CSV
    df_annot_ellipse.to_csv(file_annot, index=False, quoting=csv.QUOTE_NONNUMERIC)
    df_res_props.to_csv(file_res, index=False, quoting=csv.QUOTE_NONNUMERIC)
    df_task.to_csv(file_task, index=False, quoting=csv.QUOTE_NONNUMERIC)   
    df_truth.to_csv(file_truth, index=False, quoting=csv.QUOTE_NONNUMERIC)

    


# Compute wap and wtr for GT. This should probably be done somewhere else? Ans ussing apply()
def area_to_diam(a): # This is approxiamte, assuming a circle
  return np.sqrt(a * math.pi / 4)

def compute_wap(row):
  return (row['outer1'] - row['inner1']) / row['outer1'] * 100 # wall area percentage

def compute_wtr(row):
  return ((area_to_diam(row['outer1']) - area_to_diam(row['inner1'])) / 2) / area_to_diam(row['outer1']) # wall area percentage
    

def get_df_crowd(results_file):
    
    # Process annotations file, creating data frames for tasks, results, and annotations
    with open(results_file) as json_file:
        data = json.load(json_file)
    
    tasks = data['project']['tasks']
    
    # Initialize variables
    task_list = []
    results_list = []
    annotations_list = []
    result_id = 0
    
    for task in tasks:
    
        # Save information about the task
        task_id = task['frame']['frameIndex']
        filename = task['frame']['original']
        parts=search('data({:d}).airways({:d})',filename)
    
        task_dict = {
            'task_id': task_id,
            'subject_id': parts[0],
            'airway_id': parts[1]
            }
        task_list.append(task_dict)  
            
        for result in task['results']:
            
            for annotation in result['annotations']:
                
                # Coordinates of the ellipse
                x = annotation['points'][0][0::2]
                y = annotation['points'][0][1::2]
                
                points = list(zip(x, y))
              
                # Save information about the annotation 
                ann_dict = {
                    'result_id': result_id,
                    'annotation_id': annotation['id'],
                    'points': points,
                }
                annotations_list.append(ann_dict)   
    
            #Save information about the result
            res_dict = {
                    'task_id': task_id,
                    'result_id': result_id,
                    'result_creator': annotation['meta']['creator'],
                     }
            results_list.append(res_dict)    
            result_id = result_id+1 
    
    df_task = pd.DataFrame(task_list)
    df_res = pd.DataFrame(results_list)
    df_annot = pd.DataFrame(annotations_list)
        
    return df_task, df_res, df_annot




# Extract ellipse measurements (center point, two radii, orientation, area) from one annotation
def get_annotation_ellipse(annotation):

    points = annotation.points

    x = [p[0] for p in points]
    y = [p[1] for p in points]

    # The center of ellipse is always stored first 
    c = np.array([x[0], y[0]])

    # Calculate angle of ellipse relative to vector [1, 0]
    v1 = [x[0]-x[1], y[0]-y[1]]
    v2 = [0, 1]
    dot = v1[0]*v2[0] + v1[1]*v2[1] 
    det = v1[0]*v2[1] - v1[1]*v2[0] 
    degrees = -1*math.atan2(det, dot)*180/np.pi

    # Get vectors of orientation of ellipse
    a = np.array([x[1], y[1]])
    b = np.array([x[2], y[2]])

    # Measure width of ellipse in both directions
    radius1 = np.linalg.norm(c-a)
    radius2 = np.linalg.norm(c-b)

    # Sort radii by size
    major = np.maximum(radius1,radius2)
    minor = np.minimum(radius1,radius2)

    voxel_size = 0.55  #Actual voxel size is 0.5508 x 0.5508 x 0.6000, assuming equal size for simplicity 
    image_scaling = 0.1 #Has to do with scaling between exported airway images, and the annotation interface
    scaling = voxel_size*image_scaling
    
    area = (major*scaling)*(minor*scaling)*np.pi
    
    return pd.Series({'centre_x': x[0],'centre_y': y[0],'major_ax': major, 'minor_ax': minor, 'rotation': degrees, 'area': area})
    


# Convert an annotation (that has been measured) to a masked image
def get_annotation_mask(annotation):
    
    ell_patch = Ellipse((annotation['centre_x'], annotation['centre_x']), 2*annotation['major_ax'], 2*annotation['minor_ax'], annotation['rotation'], edgecolor='red', facecolor='none')
    coords = ell_patch.get_path()
    vertices = ell_patch.get_patch_transform().transform(coords.vertices)
    fill_row_coords, fill_col_coords = draw.polygon(vertices[:,0], vertices[:,1],[image_width,image_height]) #width and height are global variables
    
    mask = np.zeros([image_width,image_height], dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    
    return mask



# Get the drawing coordinates (patch and vertices) from an annotation
def get_ellipse_patch_vertices(annotation):

    ell_patch = Ellipse((annotation['centre_x'], annotation['centre_y']), 2*annotation['major_ax'], 2*annotation['minor_ax'], annotation['rotation'], edgecolor='red', facecolor='none')
    coords = ell_patch.get_path()
    vertices = ell_patch.get_patch_transform().transform(coords.vertices)

    return ell_patch, vertices


# Calculate several properties of the result's annotations, to be able to later decide if the result is valid or not 

def get_result_properties(res, df_annot):

    annotations = df_annot.loc[df_annot['result_id'] == res['result_id']]
    
    # Are ellipses inside each other?
    num_annot = len(annotations)
    annot_inside = np.zeros(num_annot)
    
    #Check if ellipses are inside each other - TODO maybe neater to already do this when loading data?
    
    if np.mod(num_annot,2) == 0 :
        for i in range(0, num_annot):
          mask1 = get_annotation_mask(annotations.iloc[i])
          for j in range(i+1,num_annot):
              mask2 = get_annotation_mask(annotations.iloc[j])
                      
              intersection = np.logical_and(mask1, mask2)

              diff1 = np.logical_xor(mask1,intersection)
              diff2 = np.logical_xor(mask2,intersection)
              
              if (np.sum(diff1) == 0)  | (np.sum(diff2) == 0):
                  annot_inside[i] = annotations.iloc[j]['annotation_id']
                  annot_inside[j] = annotations.iloc[i]['annotation_id']

    inside = annot_inside.all()

   
    # Have ellipses been resized? 
    resized_annot = (annotations['major_ax'] - annotations['minor_ax'] > 0.0001) 
    resized = resized_annot.all()
    
    # Assume only results with a single pair are used (otherwise select "best" pair?)
    if num_annot == 2:
        outer = annotations['area'].max()
        inner = annotations['area'].min()
        wap = (outer - inner) / outer * 100 # wall area percentage
        wtr = ((area_to_diam(outer) - area_to_diam(inner)) / 2) / area_to_diam(outer) # wall thickness ratio
    else:
        outer = np.nan
        inner = np.nan
        wap = np.nan
        wtr = np.nan
    
    return pd.Series({'num_annot': num_annot,'inside': inside, 'resized': resized, 'outer': outer, 'inner': inner,  'wap': wap, 'wtr': wtr})




