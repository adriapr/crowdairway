# Crowdsourcing Airway Segmentation - open data & code

This notebook contains the analysis for the paper "Crowdsourcing Airway Segmentation", submitted to X. 

The preprint is available here: 

The study investigates whether annotators from Amazon Mechanical Turk are able to outline airways in slices of chest CT images. The available data contains:

* Meta-data of subjects, whose airways are measured
* Images of airway slices 
* Ground truth area measurements of airways made by experts
* Airway coordinates made by crowd annotators 

# Repository structure

* /data/ - raw data - airway slides, subject characteristics, ground truth, crowd output
* /data_processed/ - processed data frames
* /figures/ - figures created by figures.py
* main.py - main script that reproduces the results (similar to the notebook)
* data.py - functions for loading and preprocessing the data
* analysis.py - functions for filtering/combining the annotations 
* figures.py - plot the figures
* tables.py - print the tables


# Notebook (Google Colaboratory) 

https://colab.research.google.com/drive/1Bnwg3tg5JNJCpUcOhN6h0FJfZ_Cdb9Uc#scrollTo=UGvdEkkQHNI9

This is a notebook which reproduces the analysis and results in the paper


# Dataframes

This code works with three types of objects, **task**, **result** and **annotation**, and their properties. 

**task** - an airway image that needs to be annotated

* subject_id - which subject is this image from
* airway_id - which airway of the subject is this image from
* task_id - combination of subject_id and airway_id uniquely identifies a task_id

* **result** - a set of annotations made by one annotator. A task will typically have 20 results. 
  * result_id
  * result_creator 
  * **annotation** - a single annotation (ellipse) drawn in the image. A result has 1 or more annotations. 
    * annotation_id 
    * points - coordinates of the ellipse

