# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:22:02 2020

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


# Display an annotation - TODO fix duplication with show_result 
def show_annotation(annotation):
    
    if annotation['centre_x'] == False:
        annotation = add_ellipse(annotation)

    ell_patch = Ellipse((annotation['centre_x'], annotation['centre_y']), 2*annotation['major_ax'], 2*annotation['minor_ax'], annotation['rotation'], edgecolor='red', facecolor='none')


    plt.figure()
    ax = plt.gca()
    ax.add_patch(ell_patch)

    plt.axis('equal')
    plt.show()


# Show all annotations of the same result
def show_result(annotations):

    plt.figure()
    ax = plt.gca()

    for (index,a) in annotations.iterrows():
    
        ell_patch, vertices = get_ellipse(a)
        
        ax.add_patch(ell_patch)
        plt.scatter(vertices[:,0], vertices[:,1])

    plt.axis('equal')
    plt.xlim(0,image_width)
    plt.ylim(0,image_height)
    plt.show()