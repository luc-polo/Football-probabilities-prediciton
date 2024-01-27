"""This module contains functions needed to plot learning curves of our pipeline and calibrator"""

import pandas as pd
from sklearn.model_selection import  learning_curve
import numpy as np
from sklearn.base import clone
import sys
import matplotlib.pyplot as plt

# Modify the sys.path list to include the path to the data directory that contains the constant_variables module that we need to import
sys.path.append('C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/src')

#Import my modules
from configuration import constant_variables


# --------------------------------------------------------------
# Learning curve for Pipeline
# --------------------------------------------------------------

def pipeline_learning_curve(X_train_0, X_valid_0, Y_train_0, Y_valid_0, grid_search_0, scoring_0):
    """
        This function plots learning curves of our pipeline, using the combination of train and valid sets as its base data.

    Args:
        X_train_0 (DataFrame): Features dataframe of our trainset
        
        X_valid_0 (DataFrame): Features dataframe of our validationset
        
        Y_train_0 (DataFrame): Target dataframe of our trainset
        
        Y_valid_0 (DataFrame): Target dataframe of our validationset
        
        grid_search_0 (): The GridSearchCV() object that contains the results of the research of best parameters for our pipeline.
        
        scoring_0 (str): The scoring function we want to use to measure model performances and plot its learning curves

    Returns:
        None
    """
    #On fussionne le train set et le valid set pour entrer l'ensemble des données (sauf les test_sets) dans la funct learning_curve()
    lc_X = pd.concat([X_train_0, X_valid_0], ignore_index = True)
    lc_Y = pd.concat([Y_train_0, Y_valid_0], ignore_index = True)


    #On calcule les données nécessaires au tracage des learning curves
    train_sizes, train_scores, test_scores = learning_curve(clone(grid_search_0.best_estimator_), lc_X, lc_Y, train_sizes = np.linspace(0.04, 1, 55), cv = constant_variables.CV, scoring = scoring_0, random_state = 765, n_jobs= -1, shuffle = True)

    plt.plot(train_sizes, train_scores.mean(axis=1), label='Train set')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='Test set')

    #increasing nb of graduation a grid lines
    plt.minorticks_on()
    plt.grid( which='major', linewidth=2)
    plt.grid( which = 'minor', linewidth=1)
    plt.xlabel('training_set size')
    plt.ylabel('neg log loss')
    plt.title('Learning curve of the pipeline')
    plt.legend() 