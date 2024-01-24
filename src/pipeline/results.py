"""This module contains functions used to display the results of our pipeline"""


#Displaying the results of the GridSearchCV() function that optimised pipeline parameters
def GridSearchCV_results(grid_search_0, X_train_0):
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