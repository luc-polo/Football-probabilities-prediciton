"""This module contains functions needed to plot learning curves of our pipeline and calibrator"""

import pandas as pd
from sklearn.model_selection import  learning_curve
import numpy as np
from sklearn.base import clone
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve


# Modify the sys.path list to include the path to the data directory that contains the constant_variables module that we need to import
sys.path.append('C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/src')

#Import my modules
from configuration import constant_variables


# --------------------------------------------------------------
# Learning curve for Pipeline
# --------------------------------------------------------------

def pipeline_learning_curve(X_train_0, X_valid_0, Y_train_0, Y_valid_0, pipeline_0, scoring_0):
    """
        This function plots learning curves of our pipeline, using the combination of train and valid sets as its base data.

    Args:
        X_train_0 (DataFrame): Features dataframe of our trainset
        
        X_valid_0 (DataFrame): Features dataframe of our validationset
        
        Y_train_0 (DataFrame): Target dataframe of our trainset
        
        Y_valid_0 (DataFrame): Target dataframe of our validationset
        
        pipeline_0 (): The pipeline chosen (scaler + model + features selector (not always))
        
        scoring_0 (str): The scoring function we want to use to measure model performances and plot its learning curves

    Returns:
        None
    """
    #On fussionne le train set et le valid set pour entrer l'ensemble des données (sauf les test_sets) dans la funct learning_curve()
    lc_X = pd.concat([X_train_0, X_valid_0], ignore_index = True)
    lc_Y = pd.concat([Y_train_0, Y_valid_0], ignore_index = True)


    #On calcule les données nécessaires au tracage des learning curves
    train_sizes, train_scores, test_scores = learning_curve(clone(pipeline_0), lc_X, lc_Y, train_sizes = np.linspace(0.04, 1, 55), cv = constant_variables.CV, scoring = scoring_0, random_state = 765, n_jobs= -1, shuffle = True)

    plt.plot(train_sizes, train_scores.mean(axis=1), label='Train set')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='Test set')

    #Legend
    plt.xlabel('training_set size')
    plt.ylabel('neg log loss')
    plt.title('Learning curve of the pipeline')
    plt.legend() 
    # setting graduations
    plt.ylim(-0.65, -0.61)  # Adjust the y axis limits
    #increasing nb of graduation a grid lines
    plt.minorticks_on()
    plt.grid( which='major', linewidth=2)
    plt.grid( which = 'minor', linewidth=1)
    plt.show()

# --------------------------------------------------------------
# Learning curve for calibrator
# --------------------------------------------------------------

def learning_curve_calibrator(nb_test_sets_sizes, X_test_0, X_valid_0, Y_test_0, Y_valid_0, test_size_0, pipe_0, n_bins_0, cross_val_nb):
    """
        This function plots learning curves of our calibrator, using the combination of test and valid sets as its base data. Train set mustn't be added to the base data as the pipeline we will train a calibrator on, has been trained on train set. It would skew the test. 
        The calibrator, training, computation of perf, train test size definition... has been handmade. I found no function as learning_curve() to plot learning curve for a calibrator.

    Args:
        nb_test_sets_sizes (int): Number of differents train set sizes we want to test our calibrator on.
        
        X_test_0 (DataFrame): Features dataframe of our test set
        
        X_valid_0 (DataFrame): Features dataframe of our validation set
        
        Y_test_0 (DataFrame): Target dataframe of our test set
        
        Y_valid_0 (DataFrame): Target dataframe of our validation set
        
        test_size_0 (float): The proportion, of the combination of valid and test sets we want to use, to test the calibrator performances on.
        
        pipe_0 (Pipeline): The optimal pipeline found with GridSearchCV
        
        n_bins_0 (int): The number of bins we want to use to calculate the calibrator perf (by plotting its calibration curves and getting its points coordinates) for each test on a given train set size.
        
        cross_val_nb (int): How many cross validations we want to do for each test of calibrator perf on a given train set size.

    Returns:
        None
    """
    
    #On fussionne le valid et test set pour entrer l'ensemble des données (sauf le train) dans la funct 
    lc_calib_X = pd.concat([X_valid_0, X_test_0], ignore_index = True)
    lc_calib_Y = pd.concat([Y_valid_0, Y_test_0], ignore_index = True) 
    
    #On définit X_test, Y_test qui permettront de tester les perf du calibrateur
    X_prem, X_test, Y_prem, Y_test = train_test_split(lc_calib_X, lc_calib_Y, test_size = test_size_0, stratify = lc_calib_Y, shuffle = True, random_state = 78)
    
    #Ordonnées, c-a-d: Liste des perf du calibrator en fonction te la taille du train set
    deviation_list=[]
    
    #Abscisses, c-a-d: liste de la proportion du train set sur lequel on entraine le calibrator
    train_size_list = np.linspace(1/nb_test_sets_sizes,1-(1/nb_test_sets_sizes),nb_test_sets_sizes)
    
    
    #On fait les tests de perf du calibrateur avec differentes proportions de X_prem
    for i in range (nb_test_sets_sizes):
        deviation_sum = 0

        #On va calibrer et tester cross_val_nb models sur X_prem
        for j in range(cross_val_nb):
            #On définit la pipeline qu'on va calibrer:
            pipe = pipe_0
            
            #On sélectionne i/n du dataset
            X_calib, X_useless, Y_calib, Y_useless2 = train_test_split(X_prem, Y_prem, train_size = train_size_list[i], stratify = Y_prem, shuffle = True,  random_state=j+(10*i))

            #On calibre la pipeline sur le j ème X_calib
            calibrated_pipeline = CalibratedClassifierCV(pipe, cv = 'prefit' , method = 'isotonic', ensemble = True)
            calibrated_pipeline.fit(X_calib, Y_calib)

            #On calcule les perf du j ème calibrateur
            prob_true, prob_pred = calibration_curve(Y_test, calibrated_pipeline.predict_proba(X_test)[:, 1], n_bins=n_bins_0, strategy='quantile')
            #On calcule le ratio deviation (moyenne des différences entre prob_pred and prob_true)
            # Calcul des différences terme à terme des proba predicted and true
            differences = np.abs(prob_pred - prob_true)
            # Calcul de la moyenne des valeurs absolues des différences qui constituent la deviation
            deviation = np.mean(differences)
            deviation_sum += deviation
        
        #On calcule la deviation moyenne pour une proportion de X_prem de i/n et on l'ajoute à la liste
        deviation_list.append(deviation_sum/cross_val_nb)
    
    
    #on convertit train_size en nombre de samples
    train_size_list = train_size_list * X_prem.shape[0]
    #On convertit deviation_list en pourcentage
    deviation_list = [x*100 for x in deviation_list]
    #plotting learning curve
    plt.clf()
    plt.plot(train_size_list, deviation_list)
    plt.title('Learning curve of the calibrator')
    plt.xlabel('Training Size')
    plt.ylabel('Avg deviation')
    # Afficher le nb_bins utilisé pour évaluer la déviation moyenne de chaque calibrator
    plt.text(0, -2.1, f"nb_bins = {n_bins_0}", fontsize=10)
    # Afficher le cross_val_nb utilisé pour évaluer la déviation moyenne de chaque calibrator
    plt.text(0, -2.8, f"cross validation number for each point = {cross_val_nb}", fontsize=10)
    # Afficher la taille du test set utilisé pour évaluer le calibrator
    plt.text(0, -3.5, f"Test set size = {round(test_size_0*lc_calib_X.shape[0])}", fontsize=10)
    # Plot an horizontal line at the last point avg variation value
    plt.axhline(y=deviation_list[-1], color='red', linestyle='--')
    
    #Défini les valeurs affichées sur l'échelle de l'axe y
    y_axis_scale_values = np.linspace(0,10,8)
    y_axis_scale_values = y_axis_scale_values.tolist()
    #On ajoute aux valeurs affichées sur l'échelle de l'axe y celle de l'ordonnée du dernier point
    y_axis_scale_values.append(deviation_list[-1])
    #On arrondit au dixème les valeurs:
    y_axis_scale_values = np.round(y_axis_scale_values, 1)
    
    plt.yticks(y_axis_scale_values)
    plt.grid()
    plt.show()
    
    



