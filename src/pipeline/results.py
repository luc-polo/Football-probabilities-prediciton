"""This module contains functions used to display the results of our pipeline"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from sklearn.utils import column_or_1d,  check_consistent_length
from sklearn.calibration import _check_pos_label_consistency
from dateutil.relativedelta import relativedelta
from tabulate import tabulate
import pickle



from configuration import constant_variables

# --------------------------------------------------------------
# GridSearchCV() Results
# --------------------------------------------------------------


# Displaying the results of the GridSearchCV() function that optimised pipeline parameters
def GridSearchCV_results(grid_search_0, X_train_0):
    """
    Displays the best parameters, score, and selected features from a GridSearchCV result.

    Args:
        grid_search (GridSearchCV): The fitted GridSearchCV object after hyperparameter tuning.
        X_train (DataFrame): The training dataset used in the GridSearchCV process.

    Returns:
        None: Prints the best parameters, score, and selected features directly.
    """
    # Extract best parameters and score
    best_params = grid_search_0.best_params_
    best_score = grid_search_0.best_score_

    # Print best parameters and score
    print("Best Parameters:", best_params)
    print("\nBest score with these hyperparameters:", best_score)

    # Extract selected features
    best_selector = grid_search_0.best_estimator_['features_selector']
    selected_indices = best_selector.get_support(indices=True)
    selected_features = X_train_0.columns[selected_indices]

    print("\nSelected Features:", list(selected_features))


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


def plot_calibration_curve_2(Y_test_0, X_train_0, proba_pred_0, n_bins_0, strategy_0, color_0, GW_training_or_not, best_model_plot=True):
    """
    Plots the calibration curve for predicted probabilities and annotates deviations.

    Args:
        y_true (DataFrame): True labels for the test set.
        X_train (DataFrame): Training set used for the predictions.
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins for the calibration curve.
        strategy (str): Strategy for binning; either "uniform" or "quantile".
        color (str): Color for the curve.
        retrained (bool): Indicates if the model was retrained per Game Week or Season.
        include_best_model (bool): Whether to include the best model's calibration curve for comparison.

    Returns:
        None: Displays the calibration plot.
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

    # Plot best model calibration curve if requested
    if best_model_plot:
       
       proba_pred_best, Y_test_best, _ = load_pred_proba("best_pipeline_results")
       prob_true_best, prob_pred_best, _ , _ = calibration_curve_bis(
           Y_test_best, proba_pred_best, n_bins=n_bins_0, strategy=strategy_0, pos_label=1
       )
       plt.plot(prob_pred_best, prob_true_best, linestyle='-', color='skyblue', alpha=1,
                label='Best model built')

    
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
         f'Diff with pred proba': np.round(differences, decimals=3),
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

    print('\nThe average deviation for this pipeline is', round(deviation*100, 2), "%")
    
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
    """
    Computes the ratio of predicted probabilities to true target sums for train and test sets.

    This diagnostic function assesses the coherence of the model's predictions
    by comparing the sum of predicted probabilities to the sum of true targets.

    Args:
        X_train (DataFrame): Training feature data.
        y_train (DataFrame): Training target labels.
        X_test (DataFrame): Test feature data.
        y_test (DataFrame): Test target labels.
        pipeline (Pipeline): Trained machine learning pipeline.

    Returns:
        None: Prints ratios for train and test sets.
    """
    proba_pred_train = pipeline_0.predict_proba(X_train_0)[:,1]
    proba_pred_test = pipeline_0.predict_proba(X_test_0)[:,1]
    
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
                                        GW_training_or_not = GW_training_or_not_0,
                                        best_model_plot = False)
        #We display statistics on the pipeline probabilities deviation 
        print_calibration_stats(prob_pred_01.copy(),
                                prob_true_01.copy(),
                                'non calibrated')
        
        print('\n\n')
        
    #We plot the histograms corresponding to each subdataset proba_pred
    for sub_ds in subdatasets_0:
        histo_title = f'Predicted Proba on matches where Played_matches_nb C [{sub_ds["Played_matchs_nb"].min()},{sub_ds["Played_matchs_nb"].max()}]'
        plot_histo_predicted_proba(sub_ds['Proba pred'].copy(), histo_bars_nb_0, 'r', histo_title)
        
def compare_pred_proba_and_odds(proba_pred_0, X_info_0):
    """
    Returns a dataset that contains the differences between predicted proba and the proba corresponding to avg_odd and max_odd,
    with probabilities displayed as percentages (1 decimal). Provides a preview of the table with the option to view the full dataset.

    Args:
        proba_pred_0 (_type_): The predicted proba column we want to compare.
        X_info_0 (_type_): The contextual features columns corresponding the proba_pred_0.

    Returns:
        tuple: (Styled full dataset, Styled preview dataset, Unstyled dataset for comparison).
    """

    # We concatenate the probabilities predicted and contextual features
    proba_pred_01 = pd.DataFrame(proba_pred_0, columns=['Proba pred'])
    col_names = []
    col_names.extend(X_info_0.columns.tolist())
    col_names.extend(proba_pred_01.columns.tolist())
    dataset_concatenated = pd.concat([X_info_0, proba_pred_01], axis=1, ignore_index=True)
    dataset_concatenated.columns = col_names

    # Add a column that contains the difference between predicted proba and the proba corresponding to average_victory_odd
    dataset_concatenated['Diff proba_pred avg_odd proba'] = dataset_concatenated['Proba pred'] - (1 / dataset_concatenated['Avg_victory_odd'])
    # Add a column that contains the difference between predicted proba and the proba corresponding to Max_victory_odd
    dataset_concatenated['Diff proba_pred Max_odd proba'] = dataset_concatenated['Proba pred'] - (1 / dataset_concatenated['Max_victory_odd'])

    # Keep the dataset for comparison before formatting probabilities
    comparison_table = dataset_concatenated.copy()

    # Format probabilities and differences as percentages with 1 decimal
    percentage_columns = ['Proba pred', 'Diff proba_pred avg_odd proba', 'Diff proba_pred Max_odd proba']
    for col in percentage_columns:
        dataset_concatenated[col] = (dataset_concatenated[col] * 100).round(1).astype(str) + '%'

    # Apply red text color to specific columns
    def apply_text_color(val):
        return 'color: red'

    styled_df = dataset_concatenated.style.map(
        apply_text_color, subset=['Proba pred', 'Diff proba_pred avg_odd proba']
    )

    # Display preview functionality
    preview_df = dataset_concatenated.head(10)  # Limit to first 10 rows for preview
    preview_styled = preview_df.style.map(
        apply_text_color, subset=['Proba pred', 'Diff proba_pred avg_odd proba']
    )

    # Return styled dataset, preview, and unstyled dataset for comparison
    return styled_df, preview_styled, comparison_table





# Compute our predicted proba deviation statistics with bookmakers proba
def compare_pred_proba_and_odds_stats(diff_dataset):
    """
    Calculate and present statistics comparing predicted probabilities and bookmaker probabilities based on absolute differences.

    Args:
        diff_dataset (DataFrame): Dataset containing differences between predicted and bookmaker probabilities.

    Returns:
        DataFrame: Table with mean and quantile statistics of absolute differences in percentage format.
    """
    # Calculate statistics on absolute differences
    mean_diff = diff_dataset[['Diff proba_pred avg_odd proba']].abs().mean()
    quantiles_diff = diff_dataset[['Diff proba_pred avg_odd proba']].abs().quantile([0.25, 0.5, 0.75])

    # Combine mean and quantiles into a single DataFrame
    stats_table = pd.DataFrame({
        'Mean Absolute Difference (%)': (mean_diff * 100).round(1),
        '25th Percentile (%)': (quantiles_diff.loc[0.25] * 100).round(1),
        'Median (%)': (quantiles_diff.loc[0.5] * 100).round(1),
        '75th Percentile (%)': (quantiles_diff.loc[0.75] * 100).round(1)
    })

    # Return the table
    display(stats_table)




# Simulate betting on a certain number of seasons
def betting_simulation(compa_dataset, Y_0, proba_interval_0, min_diff_with_odd_proba_0, GW_interval_0, bet_0):
    """
    Simulates a betting strategy on a given dataset based on specific conditions and parameters.

    Args:
        compa_dataset (pd.DataFrame): The dataset containing features, including predicted probabilities and odds.
        Y_0 (pd.DataFrame): The target dataset containing match results.
        proba_interval_0 (tuple): A tuple specifying the range of predicted probabilities for betting (min, max).
        min_diff_with_odd_proba_0 (float): Minimum difference between predicted probability and implied probability of the odds.
        GW_interval_0 (tuple): A tuple specifying the range of game weeks for betting (min, max).
        bet_0 (float): The amount of money to bet on each match.

    Returns:
        pd.DataFrame: A dataframe containing the matches that met the betting conditions and their corresponding gains.

    Description:
        1. Concatenates the dataset containing features (`compa_dataset`) and the target dataset (`Y_0`) into one dataframe.
        2. Filters the matches based on the specified betting conditions:
            - The predicted probability must fall within the `proba_interval_0` range.
            - The difference between the predicted probability and the implied probability of the odds must be at least `min_diff_with_odd_proba_0`.
            - The number of matches played must fall within the `GW_interval_0` range.
        3. Computes the betting outcomes for the selected matches. For each bet:
            - A loss of `bet_0` is registered initially.
            - If the match result indicates a win (`Result == 1`), the gain is calculated as the product of the maximum victory odds and the bet amount.
        4. Calculates the total gain, the number of bets placed, and the ratio of gain to bet.
        5. Prints the results, including the final gain, number of bets, total matches, and gain-to-bet ratio.

    Example:
        betting_simulation(compa_dataset, Y_0, (0.6, 0.8), 0.1, (5, 38), 10)
    """


    # Concatenate predicted probabilities and contextual columns
    col_names = []
    col_names.extend(compa_dataset.columns.tolist())
    col_names.extend(Y_0.columns.tolist())
    dataset_concatenated = pd.concat([compa_dataset, Y_0], axis=1, ignore_index=True)
    dataset_concatenated.columns = col_names

    # Filter matches based on betting conditions
    selected_dataset_1 = dataset_concatenated[
        (dataset_concatenated['Proba pred'] >= proba_interval_0[0]) &
        (dataset_concatenated['Proba pred'] <= proba_interval_0[1])
    ].copy()

    selected_dataset_2 = selected_dataset_1[
        selected_dataset_1['Diff proba_pred Max_odd proba'] >= min_diff_with_odd_proba_0
    ].copy()

    selected_dataset_3 = selected_dataset_2[
        (selected_dataset_2['Played_matchs_nb'] >= GW_interval_0[0]) &
        (selected_dataset_2['Played_matchs_nb'] <= GW_interval_0[1])
    ].copy()

    # Compute betting results
    selected_dataset_3 = selected_dataset_3.assign(Gain=-bet_0)
    selected_dataset_3['Gain'] = selected_dataset_3['Gain'].astype(float)

    nb_of_bets = 0
    for index, row in selected_dataset_3.iterrows():
        nb_of_bets += 1
        if row['Result'] == 1:
            selected_dataset_3.at[index, 'Gain'] += row['Max_victory_odd'] * bet_0

    final_gain = selected_dataset_3['Gain'].sum()
    nb_of_matches = compa_dataset.shape[0]
    ratio_gain_bet = final_gain / (nb_of_bets * bet_0)

    # Print summary of the betting simulation

    
    if ratio_gain_bet > 0:
        print("Congratulation, you beat the bookmakers!")
        print('The final gain is', round(final_gain,2), '€ , betting on', nb_of_bets, 'matches, out of', nb_of_matches)
        print('This is equivalent to a net gain-to-bet ratio =', round(ratio_gain_bet * 100, 2), '%')
    else:
        print("Unfortunatly your model and your stategy are not performant enough to beat the bookmakers...")
        print('The final loss is', round(final_gain,2), '€ , betting on', nb_of_bets, 'matches, out of', nb_of_matches)
        print('This is equivalent to a net loss-to-bet ratio =', round(ratio_gain_bet * 100, 2), '%')
        
    return selected_dataset_3
    

            
# Save the predicted probabilities predicted with "proba_prediction_retrained_each_seas()",  the corresponding Y and X_info

def save_pred_proba(proba_pred, Y_test, X_info, file_name):
    """
    Save the prediction probability, test labels, and contextual information to a specified location.

    Args:
        proba_pred (np.ndarray): The predicted probabilities.
        Y_test (pd.DataFrame): The true labels corresponding to the predictions.
        X_info (pd.DataFrame): Contextual information about the predictions.
        file_name (str): The name of the file (without extension) to save the datasets.

    Returns:
        None
    """
    # Define the absolute path for the datasets
    save_path = f"../models/results/{file_name}.pkl"

    # Prepare the data as a dictionary
    data_to_save = {
        "proba_pred": proba_pred,
        "Y_test": Y_test,
        "X_info": X_info
    }

    # Check if an old file exists and compare
    try:
        with open(save_path, 'rb') as file:
            old_data = pickle.load(file)
            same_data = (
                (old_data['proba_pred'] == proba_pred).all()
                and old_data['Y_test'].equals(Y_test)
                and old_data['X_info'].equals(X_info)
            )
            if same_data:
                print(f"The datasets ARE the same for both old and new {file_name}")
            else:
                print(f"The datasets ARE NOT the same for both old and new {file_name}")
    except FileNotFoundError:
        print(f"No {file_name} existing file to compare; treating as new datasets.")

    # Delete the old file if it exists
    try:
        os.remove(save_path)
        print(f"Successfully deleted the old file: {file_name}")
    except FileNotFoundError:
        print(f"No old file to delete at: {file_name}")
    except Exception as e:
        print(f"An error occurred while deleting the old file: {e}")

    # Save the new data
    try:
        with open(save_path, 'wb') as file:
            pickle.dump(data_to_save, file)
        print(f"Successfully saved the new datasets: {file_name}")
    except Exception as e:
        print(f"An error occurred while saving the new datasets: {e}")

    print("\n")

# Function to load the datasets
def load_pred_proba(file_name, print_msg = True):
    """
    Load the prediction probability, test labels, and contextual information from a specified location.

    Args:
        file_name (str): The name of the file (without extension) to load the datasets.

    Returns:
        tuple: (proba_pred, Y_test, X_info) if the file exists, else None.
    """
    # Define the absolute path for the datasets
    load_path = f"../models/results/{file_name}.pkl"

    # Load the data
    try:
        with open(load_path, 'rb') as file:
            data_loaded = pickle.load(file)
            if print_msg == True:
                print(f"Successfully loaded the datasets: {file_name}")
            return data_loaded["proba_pred"], data_loaded["Y_test"], data_loaded["X_info"]
    except FileNotFoundError:
        print(f"The file {file_name} does not exist.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred while loading the datasets: {e}")
        return None, None, None

        
        
        
    
    