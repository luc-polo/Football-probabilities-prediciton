


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