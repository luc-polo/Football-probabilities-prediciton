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
from dateutil.relativedelta import relativedelta
from tabulate import tabulate
import time

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
def plot_calibration_curve_2(Y_test_0, X_train_0, proba_pred_0, n_bins_0, strategy_0, color_0, GW_training_or_not):
    """  
        Display the annotated calibration curves either for the non calibrated pipeline or for the calibrated one.
     
     Args:
        Y_test_0 (DataFrame): The labels/targets Dataframe used to plot the calibration curve.
        
        X_train_0(DataFrame): The X_train used to train the pipeline that predicted proba_pred_0. We use it to display its nb of elements on our graph.
        
        proba_pred_0 (list): the one d array containing the probabilities predicted by our pipeline on X_test_0
        
        n_bins_0 (int): Number of bins to discretize the predicted probabilities.
        
        strategy_0 (str): strategy to discretize the probabilities interval to define the bins intervals. Etither 'uniform' or 'quantile'
        
        color_0 (str): Color for plotting the calibration curve. Blue for the calibrated model and Red for non calibrated one.
        
        GW_training_or_not (Boolean): Wether the pipeline inputed has been retrained every GW or Season before making prediction (used to display the number of sample in trainset)
    
     Returns:
        sklearn.calibration.CalibrationDisplay : The figure of calibration curve of pipeline_0
     """
    
    prob_true, prob_pred, samples_nb_per_bin, bins = calibration_curve_bis(Y_test_0, proba_pred_0, n_bins= n_bins_0, strategy=strategy_0, pos_label = 1)
    
    # Plot the calibration curve
    plt.figure(figsize=(8, 6))
    plt.scatter(prob_pred, prob_true, s = samples_nb_per_bin[samples_nb_per_bin != 0]/3, marker='o', linestyle='-', color= color_0, label='Calibrat° curve')
    plt.plot(prob_pred, prob_true, color= color_0)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probabilities')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration curve of the pipeline')
    plt.grid(True)
    
    
    #Increasing nb of graduation a grid lines
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
    # Display the train-set size used to train the pipeline used to plot the curve
    if GW_training_or_not == False:
        plt.text(0.00, 0.78, f'train_set size: {X_train_0.shape[0]}', ha='left', va='top', fontsize=12)
    
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
        prob_pred_0 (): The proba predicted by our pipeline that were returned by results.plot_calibration_curve_2
        
        prob_true_0 (): The matches results corresponding to the above proba returned by results.plot_calibration_curve_2
        
        calibrated_or_not (str): Wether the probabilities inputed are calibrated or not. 2 values: 'non calibrated' or 'calibrated'
        
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
def plot_histo_predicted_proba( proba_pred_0, bins_0, color_0, title ):
    """Plot the histogram of the proba predicted by the pipeline inputted on the features dataset inputted.

    Args:
        proba_pred_0 (np.ndarray): The proba predicted by the model we want to plot the histogram of
        bins_0 (int): The number of bins we want for our histogram
        color (str): The colour we want for our histogram bars
        calibrated_or_not (str): Only two possible values: 'calibrated' or 'non calibrated'. Used to print the title and axes names.
    """
    plt.figure()  # Create a new figure for each histogram
    
    plt.hist(
        proba_pred_0,
        range=(0, 1),
        bins= bins_0,
        color= color_0)
    plt.title(f'Histogram of the {title}')
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

#Proba prediction retraining model before each seas
def proba_prediction_retrained_each_seas(X_0, Y_0, X_info_0, pipeline_0, week_or_seas, test_seasons, chosen_features_0 = None):
    """Make proba predictions on the seasons we selelected for testing, retraining the pipeline before each GW or season predictions. It's different from the classic predictions process because the model has the experience of the season running past matches

    Args:                                                                         
        X_0 (DataFrame): Features dataframe
        
        Y_0 (DataFrame): Target dataframe
        
        X_info_0 (DataFrame): Contextual information dataframe
        
        pipeline_0 (Pipeline): The machine learning pipeline to use for training and predictions
        
        week_or_seas (Boolean): Do we want to train the model before each GW ('week') or each season ('season').
        
        test_seasons (list): List of seasons we want to make predictions on.
        
        chosen_features_0(list): The features selection associated to the model_chosen (VI)3)) Not needed if the pipeline comes from GridSearchCV
        

    Returns:
        Tuple: (proba_predicted, Y_pred, X_info) The np.array containing the proba predicted on the test seasons, the results corresponding to these predictions, the contextual columns corresponding to these predictions. 
    """
    
    proba_predicted = []
    Y_pred = pd.DataFrame(columns=["Result"])    # Initialize an empty DataFrame to contain the results of the matches we will make predicitons on
    Y_0['Result'] = Y_0['Result'].astype(int)
    X_info_pred = pd.DataFrame(columns = X_info_0.columns)
    
    #We define the seasons end dates of the the seaons we want to make predictions on
    seasons = []
    for seas in test_seasons:
        for date in constant_variables.seasons:
            if date.year == seas:
                seasons.append(date)
                
    #Loop over each season in this dataset
    for season in seasons:
        
        if week_or_seas == 'week':
            
            # Find the max number of Game Week in this dataset
            nb_of_GW = X_info_0['Game Week'].max()
            
            # Loop over each GW of this season
            for game_week in range(constant_variables.min_played_matchs_nb +2, nb_of_GW + 1):
                
                #we start finding the date of the first match of this game week
                test_condition_for_this_gw = (X_info_0['Date'] < season) & \
                                      (X_info_0['Date'] > (season - relativedelta(years=1))) &(X_info_0['Game Week'] == game_week)
                
                # Verify that we have the data for this GW matches (if the season is incomplete, the data is missing for the last GW)
                if test_condition_for_this_gw.sum()>0: 
                    first_match_date_this_gw = X_info_0[test_condition_for_this_gw]['Date'].min()
                    train_condition_for_this_gw = X_info_0['Date']<first_match_date_this_gw

                    #We build up the TRAIN datasets for this GW (targets and features datasets of matches before this GW):
                    X_train_for_this_gw = X_0[train_condition_for_this_gw]
                    Y_train_for_this_gw = Y_0[train_condition_for_this_gw]
                    
                    if chosen_features_0 is not None: #We check wether this parameter is defined was imputed in the function
                        X_train_for_this_gw = X_train_for_this_gw[chosen_features_0]  # Features selection linked to the model chosen ( VI)3 )
                    
                    #We train the pipeline on this formatted dataset
                    pipeline_0_trained = pipeline_0.fit(X_train_for_this_gw, np.ravel(Y_train_for_this_gw))
                    
                    
                    #We build up the TEST datasets for this GW (targets and features datasets of this GW matches):
                    X_test_for_this_gw  = X_0[test_condition_for_this_gw]
                    X_test_info_for_this_gw  = X_info_0[test_condition_for_this_gw]
                    Y_test_for_this_gw = Y_0[test_condition_for_this_gw]
                    
                    if chosen_features_0 is not None: #We check wether this parameter is defined was imputed in the function
                        X_test_for_this_gw = X_test_for_this_gw[chosen_features_0]  # Features selection linked to the model chosen ( VI)3 )
                    
                    #We PREDICT proba on this GW matches:
                    proba_pred_for_this_gw = pipeline_0_trained.predict_proba(X_test_for_this_gw)[:,1]
                    
                    #We add to the general datasets the proba pred, X_test_info, Y_test for this GW
                    #Proba pred
                    for proba in proba_pred_for_this_gw:
                        proba_predicted.append(proba)
                        
                    #Y_pred
                    Y_pred = pd.concat([Y_pred,Y_test_for_this_gw], axis=0, ignore_index = False)

                    #X_test_info
                    if not X_info_pred.empty:
                        X_info_pred = pd.concat([X_info_pred, X_test_info_for_this_gw], ignore_index=True, axis = 0)
                    else:
                        X_info_pred = X_test_info_for_this_gw
        else:
            
            #We build up the train_dataset for this season
            train_condition = (X_info_0['Date'] < season)
            X_train_for_this_seas = X_0[train_condition]
            Y_train_for_this_seas = Y_0[train_condition]
            
            if chosen_features_0 is not None: #We check wether this parameter is defined was imputed in the function
                X_train_for_this_seas = X_train_for_this_seas[chosen_features_0]  # Features selection linked to the model chosen ( VI)3 )
            
            #We train the pipeline on this formatted dataset
            pipeline_0_trained = pipeline_0.fit(X_train_for_this_seas, np.ravel(Y_train_for_this_seas))
            
            
            #We build up the TEST datasets for this season
            test_conditions = (X_info_0['Date'] < season) & \
                              (X_info_0['Date'] > (season - relativedelta(years=1)))
            X_test_for_this_seas = X_0[test_conditions]
            X_test_info_for_this_seas = X_info_0[test_conditions]
            Y_test_for_this_seas = Y_0[test_conditions]
            
            if chosen_features_0 is not None: #We check wether this parameter is defined was imputed in the function
                X_test_for_this_seas = X_test_for_this_seas[chosen_features_0] # Features selection linked to the model chosen ( VI)3 )
            
            #We PREDICT proba on this season matches:
            proba_pred_for_this_seas = pipeline_0_trained.predict_proba(X_test_for_this_seas)[:, 1]

    
            #We add to the general datasets the proba_pred, X_test_info, Y_test for this season
            #Proba pred
            proba_predicted.extend(proba_pred_for_this_seas)
            #Y_test
            Y_pred = pd.concat([Y_pred, Y_test_for_this_seas], axis=0, ignore_index=False)
            #X_test_info
            X_info_pred = pd.concat([X_info_pred, X_test_info_for_this_seas], ignore_index=True, axis=0)
            
    
    # We reset the indexes of the Y dataframe because certain rows within the dataframe have the same index, which causes issues when using the dataframe later in the code.
    Y_pred.reset_index(drop=True, inplace=True)           
              
    return (np.array(proba_predicted), Y_pred, X_info_pred)

#Model features coefficients study

def features_coeff_report(chosen_pipeline_trained_0, X_train_0):
    """This function displays the coefficients computed by the model for every feature.

    Args:
        chosen_pipeline_trained_0 (_type_): The pipeline trained we want to display the features coefficients.
        X_train_0 (_type_): The train features set
    """

    #Only for the "normal" proba prediction process

    # Access the coefficients from the Logistic Regression step
    coefficients = chosen_pipeline_trained_0.named_steps['model'].coef_
    # Get the indices of the selected features
    selected_indices = chosen_pipeline_trained_0.named_steps['features_selector'].get_support(indices=True)
    features = X_train_0.columns[selected_indices]
    print(coefficients)
    # Create lists to store feature names and coefficients
    feature_names = []
    coeff_values = []

    # Display feature names and coefficients
    for feature, coef in zip(features, coefficients[0]):
        feature_names.append(feature)
        coeff_values.append(coef)

    # Create a dictionary for the model_coeff_table
    model_coeff_table = {
        'Feature': feature_names,
        'Coefficient': coeff_values,
    }

    fancy_model_coeff_table = tabulate(model_coeff_table, headers='keys', tablefmt='fancy_grid')

    print(fancy_model_coeff_table)


#Calibration performances displaying depending on the nb of matches statistics are computed on (the nb of GW already played)
def calibration_over_season_advancement(season_divisions_nb, X_info_0, proba_pred_0, Y_0):
    """Returns a subset of datasets that are divisions of the original test_set obtained grouping it by played_matches_nb values. For instance the first dataset returned is all the matches of the test_set where played_matches_nb is in [6,21], the second one all the matches where played_matches_nb is in [22, 37]. These dataset will be used to compare calibration of pred_proba through the advencement of the season.

    Args:
        season_divisions_nb (_type_): The number of subdatasets we want to return. It's 2, 3  or maximum 4. If more there wont't be anough data to plot representative calibration curves.
        X_info_0 (_type_): The dataset containing contextual features of test_dataset
        proba_pred_0 (_type_): The predicted probabilities for the test_dataset
        Y_0 (_type_): The results of the matches of the test_dataset

    Returns:
        _type_: _description_
    """
    # Compute the limits of played matchs nb for each dataframe division
    max_nb_of_gw = X_info_0['Played_matchs_nb'].max() 
    split_points = np.linspace(constant_variables.min_played_matchs_nb, max_nb_of_gw, num=season_divisions_nb+1, dtype=int)[:]

    # Concat probabilities predicted, contextual features and matchs result
    proba_pred_01 = pd.DataFrame(proba_pred_0, columns = ['Proba pred'])
    col_names = []
    col_names.extend(X_info_0.columns.tolist())
    col_names.extend(proba_pred_01.columns.tolist())
    col_names.extend(Y_0.columns.tolist())
    
    dataset_concatenated = pd.concat([X_info_0, proba_pred_01, Y_0] , axis=1, ignore_index=True)
    dataset_concatenated.columns = col_names
    
    #Divide the dataframe
    subsets = []  #List that will contain the divisions of dataframe
    for i in range(season_divisions_nb):
        subsets.append(dataset_concatenated[
            (dataset_concatenated['Played_matchs_nb']>split_points[i])
            &
            (dataset_concatenated['Played_matchs_nb']<=split_points[i+1])])
    
    return subsets
        
#Plot learning curves for several subdataframes
def calibration_curves_subdataframes(subdatasets_0, nb_bins_01, histo_bars_nb_0, GW_training_or_not_0):
    """Plot calibration curves and histograms for the subset of datasets (returned by calibration_over_season_advancement()) to compare calibration over season advencement.

    Args:
        subdatasets_0 (_type_): The subset of dataset returned by calibration_over_season_advancement()
        nb_bins_01 (_type_): The number of bins we want for plotting calibration curves
        histo_bars_nb_0 (_type_): The number of bars we want in histograms plot
        GW_training_or_not_0 (Boolean): Wether the pipeline used to predict the probabilities we will plot the calib curves has been retrained evey GW or season.
    """
    #Plot Calibration curves for each subdataframe
    for sub_ds in subdatasets_0:
        print('Calibration curve of proba predicted on matches where Played_matches_nb C [',sub_ds['Played_matchs_nb'].min(),',', sub_ds['Played_matchs_nb'].max(),']')
        
        sub_ds_info= sub_ds.drop(columns = ['Result', 'Proba pred'])
        
        prob_pred_01, prob_true_01 = plot_calibration_curve_2(
                                        Y_test_0 = sub_ds['Result'].copy(),
                                        X_train_0= sub_ds_info,              #this dataframe isn't used, it's just to put something
                                        proba_pred_0 = sub_ds['Proba pred'].copy(),
                                        n_bins_0 = nb_bins_01,
                                        strategy_0 = 'quantile',
                                        color_0 = 'red',
                                        GW_training_or_not = GW_training_or_not_0)
        #We display statistics on the pipeline probabilities deviation 
        print_calibration_stats(prob_pred_01.copy(),
                                prob_true_01.copy(),
                                'non calibrated')
        
        print('\n\n')
        
    #We plot the histograms corresponding to each subdataset proba_pred
    for sub_ds in subdatasets_0:
        histo_title = f'Predicted Proba on matches where Played_matches_nb C [{sub_ds["Played_matchs_nb"].min()},{sub_ds["Played_matchs_nb"].max()}]'
        plot_histo_predicted_proba(sub_ds['Proba pred'].copy(), histo_bars_nb_0, 'r', histo_title)
        
# Compare predicted probabilities and the probabilities of bookmakers
def compare_pred_proba_and_odds(proba_pred_0,X_info_0):
    """Returns a dataset that contains the differences between predicted proba and the proba corresponding to avg_odd and max_odd

    Args:
        proba_pred_0 (_type_): The predicted proba column we want to compare
        X_info_0 (_type_): The contextual features columns corresponding the the proba_pred_0

    Returns:
        _type_: _description_
    """
    
    #We concatenate the probabilities predicted and contextual features
    proba_pred_01 = pd.DataFrame(proba_pred_0, columns = ['Proba pred'])
    col_names = []
    col_names.extend(X_info_0.columns.tolist())
    col_names.extend(proba_pred_01.columns.tolist())
    dataset_concatenated = pd.concat([X_info_0, proba_pred_01] , axis=1, ignore_index=True)
    dataset_concatenated.columns = col_names
    
    #We add a column that contains the difference between predicted proba and the proba corresponding to average_victory_odd
    dataset_concatenated['Diff proba_pred avg_odd proba'] = dataset_concatenated['Proba pred'] - (1/dataset_concatenated['Avg_victory_odd'])
    #We add a column that contains the difference between predicted proba and the proba corresponding to Max_victory_odd
    dataset_concatenated['Diff proba_pred Max_odd proba'] = dataset_concatenated['Proba pred'] - (1/dataset_concatenated['Max_victory_odd'])
    
    return dataset_concatenated

# Simulate betting on a certain number of seasons
def betting_simulation(dataset_0, Y_0, proba_interval_0, min_diff_with_odd_proba_0, GW_interval_0, bet_0):
    #We concat proba_pred and contextual col
    col_names = []
    col_names.extend(dataset_0.columns.tolist())
    col_names.extend(Y_0.columns.tolist())
    dataset_concatenated = pd.concat([dataset_0, Y_0] , axis=1, ignore_index=True)
    dataset_concatenated.columns = col_names
    
    #We select only matches that respect the conditions of betting defined by the parameters inputted
    selected_dataset_1 = dataset_concatenated[(dataset_concatenated['Proba pred'] >= proba_interval_0[0]) & (dataset_concatenated['Proba pred'] <= proba_interval_0[1])].copy()
    
    selected_dataset_2 = selected_dataset_1[selected_dataset_1['Diff proba_pred Max_odd proba'] >= min_diff_with_odd_proba_0].copy()
    
    selected_dataset_3 = selected_dataset_2[(selected_dataset_2['Played_matchs_nb'] >= GW_interval_0[0]) & (selected_dataset_2['Played_matchs_nb'] <= GW_interval_0[1])].copy()
   
    #We compute bets results
    selected_dataset_3 = selected_dataset_3.assign(Gain = -bet_0)
    selected_dataset_3['Gain'] = selected_dataset_3['Gain'].astype(float)
                                                                   
    nb_of_bets = 0
    for index, row in selected_dataset_3.iterrows():
            nb_of_bets +=1
            
            if row['Result'] == 1:
                selected_dataset_3.at[index, 'Gain'] += row['Max_victory_odd'] * bet_0
    
    final_gain = selected_dataset_3['Gain'].sum()
    nb_of_matches = dataset_0.shape[0]
    ratio_gain_bet = final_gain/(nb_of_bets*bet_0)
    
    print('The final gain is', final_gain,'€ , betting on', nb_of_bets, 'matches, out of', nb_of_matches)
    print('This is equivalent to a ratio gain/bet =', ratio_gain_bet)
    
    return(selected_dataset_3)
    

            
    
    
        
        
        
    
    