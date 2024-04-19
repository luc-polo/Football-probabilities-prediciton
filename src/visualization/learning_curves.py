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
from tabulate import tabulate


# Modify the sys.path list to include the path to the data directory that contains the constant_variables module that we need to import
sys.path.append('C:/Users/polol/OneDrive/Documents/ML/Projet Mbappe (11.23- )/Projet Mbappe Cookiestructure/src')

#Import my modules
from configuration import constant_variables
from data import preprocessing
from pipeline import results

# --------------------------------------------------------------
# Learning curve for Pipeline performances (logloss)
# --------------------------------------------------------------

def pipeline_learning_curve(X_train_0, Y_train_0, pipeline_0, scoring_0):
    """
        This function plots learning curves of our pipeline, using the combination of train and valid sets as its base data.

    Args:
        X_train_0 (DataFrame): Features dataframe of our trainset
        
        Y_train_0 (DataFrame): Target dataframe of our trainset
        
        pipeline_0 (): The pipeline chosen (scaler + model + features selector (not always))
        
        scoring_0 (str): The scoring function we want to use to measure model performances and plot its learning curves

    Returns:
        None
    """

    #On calcule les données nécessaires au tracage des learning curves
    train_sizes, train_scores, test_scores = learning_curve(clone(pipeline_0), X_train_0, Y_train_0, train_sizes = np.linspace(0.04, 1, 55), cv = constant_variables.CV, scoring = scoring_0, random_state = 765, n_jobs= -1, shuffle = True)

    plt.plot(train_sizes, train_scores.mean(axis=1), label='Train set')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='Test set')

    #Legend
    plt.xlabel('training_set size')
    plt.ylabel('neg log loss')
    plt.title('Learning curve of the pipeline')
    plt.legend() 
    # setting graduations
    plt.ylim(-0.61, -0.54)  # Adjust the y axis limits
    #increasing nb of graduation a grid lines
    plt.minorticks_on()
    plt.grid( which='major', linewidth=2)
    plt.grid( which = 'minor', linewidth=1)
    plt.show()

# --------------------------------------------------------------
# Learning curve for Pipeline calibration (deviation) 
# --------------------------------------------------------------
#That's not real learning curves. Just comparaison of calibration curves for different sizes of train-set

def data_formatting_partitionning_seasonally(names_col_to_concat_0, names_col_concatenated_0, col_to_remove_0, contextual_col_0, dataset_0, test_seasons_0, train_seasons_0, chosen_features_0):
    """From the last_dataset_xx, returns subdatasets of different size that we will use to compare the pipeline calibration depending on the train-set size. The function apply the formatting process of the function preprocessing.formatting_splitting_seasons

    Args:
        names_col_to_concat_0 (_type_): _description_
        names_col_concatenated_0 (_type_): _description_
        col_to_remove_0 (_type_): _description_
        contextual_col_0 (_type_): _description_
        dataset_0 (_type_): _description_
        test_seasons_0 (_type_): _description_
        train_seasons_0 (_type_): _description_
        chosen_features_0 (_type_): _description_

    Returns:
        _type_: _description_
    """
    X_train = []  # List that will contain the different X train set we will return
    Y_train = []  # List that will contain the different Y train set we will return
    for train_seasons_x in train_seasons_0:
        # Formatting and splitting (following seasons) dataset to get: train and test sets ( V)1) )
        X_train_info, X_train_00, Y_train_00, X_test_info,  X_test_00, Y_test_00 = preprocessing.formatting_splitting_seasons(names_col_to_concat_0, names_col_concatenated_0, col_to_remove_0, contextual_col_0, dataset_0, test_seasons_0, train_seasons_x)
        
        #On choisit une pipeline enregistrée dans pipeline.model et la selection de features qui va avec ( VI)3) )
        X_train_01 = X_train_00.copy()[chosen_features_0]
        X_test_01 = X_test_00.copy()[chosen_features_0]

        X_train.append(X_train_01)
        Y_train.append(Y_train_00)
    
    return X_train, Y_train, X_test_01, Y_test_00

def pipeline_calibration_learning_curve(X_train_0, X_test_0, Y_train_0, Y_test_0, chosen_pipeline_0, nb_bins_0, strategy_0):
    """Plots the learning curves of the pipeline for different size of train_set. It allows to evaluate the impact of trainset size on the performance of the model

    Args:
        X_train_0 (_type_): _description_
        X_test_0 (_type_): _description_
        Y_train_0 (_type_): _description_
        Y_test_0 (_type_): _description_
        chosen_pipeline_0 (_type_): _description_
        nb_bins_0 (_type_): _description_
        strategy_0 (_type_): _description_
    """
    learning_curves_col = ['b', 'r', 'k', 'g', 'c', 'm', 'y', 'k', 'w']   #A list of colours used below to give a different colour to each curve.
    plt.figure(figsize=(8, 6))
    deviations = []  #List that will contain the avg deviation for each curve
    curves_training_set_sizes = []   # List that will contain the number of samples in the train-set corresponding to each learning curve
    curves_labels = []   # List that will contain the labels of the learning curves
    
    for i, (X_train_x, Y_train_x) in enumerate(zip(X_train_0, Y_train_0)):
        # Train the pipeline chosen ( VI)3) )
        chosen_pipeline_trained_0 = chosen_pipeline_0.fit(X_train_x.copy(), Y_train_x.copy().values.ravel())
 
        #Make 'normal' proba predictions on the test_seasons ( VII)1) )
        normal_proba_pred_x = chosen_pipeline_trained_0.predict_proba(X_test_0)[:,1]
        
        #Plot Calibration curve of the non calibrated pipeline and info about its bins ( VII)1) )
        prob_true_x, prob_pred_x, samples_nb_per_bin_x, bins_x = results.calibration_curve_bis(Y_test_0, normal_proba_pred_x, n_bins= nb_bins_0, strategy=strategy_0)
        
        label_of_the_curve_x = 'Train_set samples =' + str(X_train_x.shape[0])
        plt.scatter(prob_pred_x, prob_true_x, s = 1.2, marker='o', linestyle='-', color= learning_curves_col[i], label=label_of_the_curve_x)
        plt.plot(prob_pred_x, prob_true_x, color= learning_curves_col[i])
        
        # Deviation computation
        # Calcul des différences terme à terme des proba predicted and true
        differences_x = np.abs(prob_pred_x - prob_true_x)
        # Calcul de la moyenne des valeurs absolues des différences qui constituent la deviation
        deviations.append(np.mean(differences_x))
        # We add to the training_set size for the curve x to the corresponding list
        curves_training_set_sizes.append(X_train_x.shape[0])
        # We add the label of the curve to the corresponding list
        curves_labels.append(label_of_the_curve_x)
        
    #We add legend, grid, title etc... to the graph
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probabilities')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curves: Impact of training-set size on calibration')
    plt.grid(True)
    plt.minorticks_on()
    plt.grid( which='major', linewidth=2)
    plt.grid( which = 'minor', linewidth=1)
    plt.legend()
    plt.show()
    
    #We display a table that display the avg deviation for each curve
    avg_deviation_df = pd.DataFrame({'Curve label': curves_labels, 'Curves avg deviation': deviations})  
    # Improve table design
    fancy_avg_deviation_df = tabulate(avg_deviation_df, headers='keys', tablefmt='fancy_grid')
    print(fancy_avg_deviation_df)

    
    


# --------------------------------------------------------------
# Learning curve for calibrator (nt used anymore)
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
    
    
