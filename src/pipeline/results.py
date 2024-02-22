"""This module contains functions used to display the results of our pipeline"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.calibration import calibration_curve
import warnings
from sklearn.utils import check_matplotlib_support, column_or_1d,  check_consistent_length
from sklearn.calibration import _check_pos_label_consistency
from tabulate import tabulate
from dateutil.relativedelta import relativedelta

from configuration import constant_variables
from data import preprocessing

# --------------------------------------------------------------
# GridSearchCV() Results
# --------------------------------------------------------------


# Displaying the results of the GridSearchCV() function that optimised pipeline parameters
def GridSearchCV_results(grid_search_0, X_train_0):
    """  
        Present the results of GridSearchCV() on our pipeline. Display the optimal parameters, score, and features selected by GridSearchCV() on our pipeline.
    
    Args:
        grid_search_0 (object): Name of the GridSearchCV() object ran before and that we want to display the results of.
        
        X_train_0 (DataFrame): The trainset on which we ran grid_search_0 before.
    
    Returns:
        None
    """
    # Display the best parameters and score
    best_params = grid_search_0.best_params_
    best_score = grid_search_0.best_score_
    print("Best Parameters:", best_params, "\n\nBest score with these hyper parameters:", best_score)

    #Display features selected
    # Obtenir le sélecteur de caractéristiques à partir du meilleur estimateur
    best_selector = grid_search_0.best_estimator_['features_selector']
    # Obtenir les indices des features sél"ectionnées
    selected_feature_indices = best_selector.get_support(indices=True)
    # Obtenir les noms des caractéristiques à partir des indices
    selected_feature_names = X_train_0.columns[selected_feature_indices]

    print("\n\nFeatures selected:",selected_feature_names)


# --------------------------------------------------------------
# Calibration Results
# --------------------------------------------------------------

#We modify the calibration_curve funct° of scikit to make it returns the width of the bins we need to display stats
def calibration_curve_bis(
    y_true,
    y_prob,
    *,
    pos_label=None,
    normalize="deprecated",
    n_bins=5,
    strategy="uniform"):
    """Compute true and predicted probabilities for a calibration curve.

    The method assumes the inputs come from a binary classifier, and
    discretize the [0, 1] interval into bins.

    Calibration curves may also be referred to as reliability diagrams.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.

    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.

    pos_label : int or str, default=None
        The label of the positive class.

        .. versionadded:: 1.1

    normalize : bool, default="deprecated"
        Whether y_prob needs to be normalized into the [0, 1] interval, i.e.
        is not a proper probability. If True, the smallest value in y_prob
        is linearly mapped onto 0 and the largest one onto 1.

        .. deprecated:: 1.1
            The normalize argument is deprecated in v1.1 and will be removed in v1.3.
            Explicitly normalizing `y_prob` will reproduce this behavior, but it is
            recommended that a proper probability is used (i.e. a classifier's
            `predict_proba` positive class).

    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `y_prob`) will not be returned, thus the
        returned arrays may have less than `n_bins` values.

    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.

        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.

    Returns
    -------
    prob_true : ndarray of shape (n_bins,) or smaller
        The proportion of samples whose class is the positive class, in each
        bin (fraction of positives).

    prob_pred : ndarray of shape (n_bins,) or smaller
        The mean predicted probability in each bin.

    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.calibration import calibration_curve
    >>> y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9,  1.])
    >>> prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=3)
    >>> prob_true
    array([0. , 0.5, 1. ])
    >>> prob_pred
    array([0.2  , 0.525, 0.85 ])
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)
    pos_label = _check_pos_label_consistency(pos_label, y_true)

    # TODO(1.3): Remove normalize conditional block.
    if normalize != "deprecated":
        warnings.warn(
            "The normalize argument is deprecated in v1.1 and will be removed in v1.3."
            " Explicitly normalizing y_prob will reproduce this behavior, but it is"
            " recommended that a proper probability is used (i.e. a classifier's"
            " `predict_proba` positive class or `decision_function` output calibrated"
            " with `CalibratedClassifierCV`).",
            FutureWarning,
        )
        if normalize:  # Normalize predicted values into interval [0, 1]
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}."
        )
    y_true = y_true == pos_label

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    
    bin_sums = np.bincount(binids, weights = y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))


    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    return prob_true, prob_pred, bin_total, bins


# Plot calibration curves of calibrated and not calibrated pipeline/model
def plot_calibration_curve_2(Y_test_0, proba_pred_0, n_bins_0, strategy_0, color_0, calibrated_model_or_not):
    """  
        Display the annotated calibration curves either for the non calibrated pipeline or for the calibrated one.
     
     Args:
        pipeline_0 (pipeline): The calibrated or non calibrated pipeline we want to plot the calibration curve of.
        
        X_test_0 (DataFrame): The features Dataframe used to plot the calibration curve.
        
        Y_test_0 (DataFrame): The labels/targets Dataframe used to plot the calibration curve.
        
        proba_pred_0 (list): the one d array containing the probabilities predicted by our pipeline on X_test_0
        
        n_bins_0 (int): Number of bins to discretize the predicted probabilities.
        
        strategy_0 (str): strategy to discretize the probabilities interval to define the bins intervals. Etither 'uniform' or 'quantile'
        
        color_0 (str): Color for plotting the calibration curve. Blue for the calibrated model and Red for non calibrated one.
        
        calibrated_model_or_not (Boolean): Wether the pipeline inputed is the calibrated one or not (used to define graph anotations)
    
     Returns:
        sklearn.calibration.CalibrationDisplay : The figure of calibration curve of pipeline_0
     """
    
    prob_true, prob_pred, samples_nb_per_bin, bins = calibration_curve_bis(Y_test_0, proba_pred_0, n_bins= n_bins_0, strategy=strategy_0)
    
    
    # Plot the calibration curve
    plt.figure(figsize=(8, 6))
    plt.scatter(prob_pred, prob_true, s = samples_nb_per_bin[samples_nb_per_bin != 0]/3, marker='o', linestyle='-', color= color_0, label='Calibrat° curve')
    plt.plot(prob_pred, prob_true, color= color_0)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probabilities')
    plt.ylabel('Fraction of positives')
    if calibrated_model_or_not == True:
        plt.title('Calibration curve for calibrated pipeline')
    else:
        plt.title('Calibration curve for non calibrated pipeline')
    plt.grid(True)
    
    
    #increasing nb of graduation a grid lines
    plt.minorticks_on()
    plt.grid( which='major', linewidth=2)
    plt.grid( which = 'minor', linewidth=1)
    
    #plot the vertical
    for i in range(len(prob_pred)):
        plt.plot([prob_pred[i],prob_true[i]], [prob_true[i], prob_true[i]], linestyle='-', color='black')
        annotation_text = round(abs(prob_pred[i]-prob_true[i]), 2)
        plt.annotate(annotation_text, ((prob_pred[i]+prob_true[i]) / 2, (prob_true[i])), textcoords="offset points", xytext=(0, 10), ha='center')

    # Display the parameter n_bins_0 value
    plt.text(0.00, 0.9, f'n_bins: {n_bins_0}', ha='left', va='top', fontsize=12)
    # Display the number of samples/predictions used to plot the curve
    plt.text(0.00, 0.84, f'test_set size: {Y_test_0.shape[0]}', ha='left', va='top', fontsize=12)
    
    plt.legend()
    plt.show()
    
    #Display stats on bins
    print('Above learning curve statistics on bins:\n')
    learning_curve_bins_stats = pd.DataFrame({
        'Bin interval':[[round(bins[i], 2), round(bins[i+1], 2)]  for i in range(len(bins)-1)],
        'Predictions nb in the bin': [samples_nb_per_bin[i] for i in range(len(bins)-1)]})
    
    # Improve table design
    fancy_learning_curve_bins_stats = tabulate(learning_curve_bins_stats, headers='keys', tablefmt='fancy_grid')
    print(fancy_learning_curve_bins_stats)
    
    return prob_true, prob_pred

# Print the statistics about pipeline calibration
def print_calibration_stats(prob_pred_0, prob_true_0, calibrated_or_not, *X_valid_0):
    """
        Present the stats related to the calibrated pipeline calibration:
        - The size of datasets used to train the calibrator and test the pipeline (if it has been calibrated)
        - A table with the numerical data of the pipeline calibration curve (The coordinates of the points on the graph) 
        - The average deviation of probabilities for our pipeline.
    
    Args:
        CalibrationDisplay_0 (sklearn.calibration.CalibrationDisplay): The figure of the calibration curve on X_test of the calibrated pipeline.
        
        X_valid_0 (DataFrame): The features Dataframe we used to train the calibrator. We use it to display its size (nb of rows).
        
        X_test_0 (DataFrame): The features Dataframe we used to test the calibrator. We use it to display its size (nb of rows)
    
    Returns:
        None
    """
    #On calcule et affiche, pour le model calibré, le ratio deviation (moyenne des différence entre prob_pred and prob_true)
    # Calcul des différences terme à terme des proba predicted and true
    differences = np.abs(prob_pred_0 - prob_true_0)
 
    # Calcul de la moyenne des valeurs absolues des différences qui constituent la deviation
    deviation = np.mean(differences)
 
    # Créer le tableau qui contiendra les ratiaux de deviation ainsi que prob_true
    table_data = {
         'Proba True': np.round(prob_true_0, decimals=3),
         f'Diff with {calibrated_or_not} pred proba': np.round(differences, decimals=3),
     }
    
    #Convert table_data into a DataFrame
    calibration_df = pd.DataFrame(table_data)
    
    print('\n\nAbove learning curve statistics on probabilities deviation:\n')
    if calibrated_or_not == 'calibrated':
        #On affiche la taille du train_set du calibrateur et du test_set sur lequel on a testé notre model calibré
        print('Train_set size of the calibrator : ', X_valid_0.shape[0])
    
    # Improve table design
    fancy_table = tabulate(calibration_df, headers='keys', tablefmt='fancy_grid')

    # Afficher le tableau 
    print(fancy_table)

    print('\nLa deviation moyenne pour ce paramétrage est de ', round(deviation*100, 2), "%")
    
# Plot histogram of predicted probabilities 
def plot_histo_predicted_proba( proba_pred_0, bins_0, color_0, calibrated_or_not ):
    """Plot the histogram of the proba predicted by the pipeline inputted on the features dataset inputted.

    Args:
        proba_pred_0 (np.ndarray): The proba predicted by the model we want to plot the histogram of
        bins_0 (int): The number of bins we want for our histogram
        color (str): The colour we want for our histogram bars
        calibrated_or_not (str): Only two possible values: 'calibrated' or 'non calibrated'. Used to print the title and axes names.
    """

    plt.hist(
        proba_pred_0,
        range=(0, 1),
        bins= bins_0,
        color= color_0)
    plt.title(f'Histogram of predicted probabilities of the {calibrated_or_not} pipeline')
    plt.xlabel('Probability value')
    plt.ylabel('Number of predicted proba')
    plt.grid()
    plt.show

# Ratio probabilities pred/sum of true target
def ratio_proba__sum_true_target(X_train_0, Y_train_0, X_test_0, Y_test_0, pipeline_0):
    """Prints the ratio of the sum of predicted probabilities by the pipeline on the train set to the sum of true labels on the train set, as well as the same ratios for the test set. This function serves as a diagnostic tool to assess the coherence of the model training. In a well-trained model, the ratio for the train set must be equal to 1t (according to info on internet). The function also provides valuable insights into the precision of predicted probabilities by calculating the ratio on the test set.

    Args:
        X_train_0 (_type_): Feature data for the train set
        Y_train_0 (_type_): True labels for the train set.
        X_test_0 (_type_): Feature data for the test set.
        Y_test_0 (_type_): True labels for the test set.
        pipeline_0 (_type_): The trained machine learning pipeline.
    """
    proba_pred_train = pipeline_0.predict_proba(X_train_0)[:,1]
    proba_pred_test = pipeline_0.predict_proba(X_test_0)[:,1]
    print(proba_pred_test.sum())
    
    proba_train_sum=round(proba_pred_train.sum(), 2)
    sum_true_values_train = Y_train_0.sum()
    ratio_train = proba_train_sum/sum_true_values_train
    
    print(f"\n\nLa somme des proba prédites sur le TRAIN SET est {proba_train_sum}, la somme des true event est {sum_true_values_train.item()}")
    print(f'Le rapport de la somme des proba prédites sur la somme de true target est {ratio_train.item()}')
    
    proba_test_sum=round(proba_pred_test.sum(), 2)
    sum_true_values_test = Y_test_0.sum()
    ratio_test = proba_test_sum/sum_true_values_test
    
    print(f"\n\nLa somme des proba prédites sur le TEST SET est {proba_test_sum}, la somme des true event est {sum_true_values_test.item()}")
    print(f'Le rapport de la somme des proba prédites sur la somme de true target est {ratio_test.item()}')


#Testtt
def proba_prediction_model_retrained_each_GW(H_A_col_to_concat_0, col_concatenated_names_0, col_to_delete_list_0, contextual_col_0, pipeline_0, seasons_0, dataset_0):
    """Make proba predictions on the seasons we selelected for testing, retraining the pipeline before each GW predictions. It's different from the classic predictions process because the model has the experience of the season running past matches.

    Args:
        H_A_col_to_concat_0 (list): List of column names we want to include in the final dataset for our pipeline. It contains the Home and Away teams col names that we will concatenate.
        
        col_concatenated_names_0 (list): List of names we will assign to the concatenated col (H_A_col_to_concat).
        
        col_to_delete_list_0 (list): List of column names to be deleted
        
        contextual_col_0 (list): List with the names of the concatenated columns containing a contextual information. That's the col we do not want to give to our model
        
        pipeline_0 (pipeline): The non fitted pipeline selected by GridSearchCV we will use to make proba predictions
        
        seasons_0 (list): list of the seasons years we want to make the tests on. We usually input 'test_seasons' defined in V)1)
        
        dataset_0 (_type_): The original dataset containing ALL data.

    Returns:
        Tuple: (proba_predicted, Y, X_info) The np.array containing the proba predicted on the test seasons, the Results corresponding to these predictions, the contextual columns corresponding to these predictions. 
    """
    
    proba_predicted = []
    Y = []
    X_info = pd.DataFrame(columns=contextual_col_0)
    
    #We define the seasons end dates of the the seaons we want to make predictions on
    seasons = []
    for seas in seasons_0:
        for date in constant_variables.seasons:
            if date.year == seas:
                seasons.append(date)
    
    for season in seasons:
        nb_of_GW_for_this_season = dataset_0['Game Week'].max()
        for game_week in range(constant_variables.min_played_matchs_nb +2, nb_of_GW_for_this_season + 1):
            
            #we start fining the date of the first match of this game week
            combined_conditions = (dataset_0['date_GMT']<season) & ((season - relativedelta(years=1)) <dataset_0['date_GMT'])& (dataset_0['Game Week'] == game_week)
            first_match_date = dataset_0[combined_conditions]['date_GMT'].min()
            
    
            # We test if at least one of the Game Week matches has been completed
            if not (dataset_0[combined_conditions]['status'] == 'incomplete').all():
                #We build up the dataset of all the data beofore this game week:
                train_dataset_for_this_gw = dataset_0[dataset_0['date_GMT']<first_match_date]
                
                #We apply the formatting and train_test_split on this dataset
                X_train_for_this_gw, X_train_info_for_this_gw, Y_train_for_this_gw = preprocessing.formatting_cleaning(H_A_col_to_concat_0, col_concatenated_names_0, col_to_delete_list_0, contextual_col_0, train_dataset_for_this_gw )
                
                #We train the pipeline on this formatted dataset
                pipeline_0_trained = pipeline_0.fit(X_train_for_this_gw, np.ravel(Y_train_for_this_gw))
                
                
                #We build up the test_datset for this GW
                test_dataset_for_this_gw = dataset_0[combined_conditions]
                
                #We apply the formatting and train_test_split on this dataset
                X_test_for_this_gw, X_test_info_for_this_gw, Y_test_for_this_gw = preprocessing.formatting_cleaning(H_A_col_to_concat_0, col_concatenated_names_0, col_to_delete_list_0, contextual_col_0, test_dataset_for_this_gw )
                
                #We predict proba on this gw matches:
                porba_pred_for_this_gw = pipeline_0_trained.predict_proba(X_test_for_this_gw)[:,1]
                
                #We add to the general datasets the proba pred, X_test_info, Y_test for this GW
                #Proba pred
                for proba in porba_pred_for_this_gw:
                    proba_predicted.append(proba)
                    
                #Y_test
                for R in Y_test_for_this_gw['Result']:
                    Y.append(R)

                
                #X_test_info
                if not X_info.empty:
                    X_info = pd.concat([X_info, X_test_info_for_this_gw], ignore_index=True, axis = 0)
                else:
                    X_info = X_test_info_for_this_gw
                

    return (np.array(proba_predicted), pd.DataFrame(Y, columns =['Result']), X_info)

