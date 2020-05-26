# Crowdsourcing Airway Segmentation - open data & code

This notebook contains the analysis for the paper "Crowdsourcing Airway Segmentation", submitted to X. 

The study investigates whether annotators from Amazon Mechanical Turk are able to outline airways in slices of chest CT images. The available data contains:

* Images of airway slices TODO
* Ground truth area measurements of airways made by experts
* Airway coordinates made by crowd annotators 

# Notebook (Google Colaboratory) 

https://colab.research.google.com/drive/1Bnwg3tg5JNJCpUcOhN6h0FJfZ_Cdb9Uc#scrollTo=UGvdEkkQHNI9

Use this to load the processed data and plot different types of results (including results not shown in the paper)


# Code (this repository)

* main.py - main script that loads the data and creates the results
* load_data.py - functions for loading and preprocessing the data
* analyze.py - functions for creating plots 
* TODO


# Data structure

The first part of the script loads all the required data. This section describes three types of objects, **task**, **result** and **annotation**, and their properties. 

**task** - an airway image that needs to be annotated

* subject_id - which subject is this image from
* airway_id - which airway of the subject is this image from
* task_id - combination of subject_id and airway_id uniquely identifies a task_id

* **result** - a set of annotations made by one annotator. A task will typically have 10 results. 
  * result_id
  * result_creator 
  * **annotation** - a single annotation (ellipse) drawn in the image. A result has 1 or more annotations. 
    * annotation_id 
    * points - coordinates of the ellipse

