projet_Mbappe
==============================

Purpose and Scope:
-------------------
This project is a personal initiative that I am particularly proud of. Inspired by my professional aspirations in data science and my genuine interest in football and statistics, I decided to take on this instructive challenge. The project involves building a complex and comprehensive machine learning model to predict the probabilities of victory in football matches and identify value bets. It encompasses the entire process, from data extraction to betting strategy building.

```diff
- For a quick and clear overview of this project, open and follow the little tutorial in the user_test.ipynb file,
- located in the notebooks directory 
```
 
#### Project Description

- What my application does?

This project focuses on identifying value bets offered by bookmakers on football matches. A value bet occurs when bookmakers propose odds that underestimate a team's probability of winning, resulting in odds that are higher than they should be based on the actual probability

The program inputs teams stats since the beginning of the season. And outputs victories probabilities and Boolean value of whether the bet proposed by bookmaker is a value bet.

Where this bookmakers' flaw comes from?

This discrepancy between bookmakers' probabilities and the actual probabilities often arises because bookmakers factor in bettors' behaviors when calculating their odds. They do this to encourage balanced betting on both sides, minimizing their risk and ensuring a consistent profit margin regardless of the match outcome.
 

- How does it do it?

To achieve this, the program predicts the probabilities of football match outcomes and compares them to those provided by bookmakers. The bookmakers' probabilities are calculated as the inverse of their odds.

The models chosen for these predictions are Logistic Regression and Neural Networks.


- Why I used the technologies I used?

Logistic Regression proved to be an excellent model for this task as it computes probabilities for classification using a model function (the logistic function) that is particularly well-suited for predicting probabilities. Moreover, the scoring function is directly based on the predicted probabilities, penalizing deviations from accurate probability predictions

Neural Networks were also chosen for their ability to model complex, non-linear relationships in the data and their adaptability in probability prediction, supported by the wide range of activation functions available.

 
Repository Structure:
-----------------------

#### Brief explanation of project organisation:
In this project, we have a comprehensive model development component and a “user test” version. The comprehensive model development covers a range of tasks and analyses, whereas the “user test” is a concise version that lets new users quickly explore my work and results.

The comprehensive model development is split across several notebooks (all except user_test.ipynb). In these notebooks, we perform our analyses and model building, calling functions defined in the .py modules located in the src folder. I split the comprehensive model development across multiple notebooks to make the project easier to follow.

On the other hand, the user test version is compiled into a single notebook: user_test.ipynb. There, you can explore and interact with the key steps of my model development, tune a model, and even implement a small betting strategy.

#### Global structure:

- **data**: Folder containing the datasets at different stages of transformation. Only used for storage.
- **docs**: Contains files related to the Sphinx documentation generation process. Not interesting for those exploring the project.
- **env**: My virtual environment.
- **environment and requirements**: Includes `.yml` and `.txt` files listing the required dependencies to replicate my virtual environment.
- **models**: Stores the pipelines that performed well, along with their results. Primarily used for storage.
- **notebooks**: The most crucial part of the project. This folder contains the Jupyter notebooks where you can execute the comprehensive model development as well as the user test.
- **references**: Contains an excel file explaining what every feature represents
- **reports**: Contains my graphs. Only made for storage.
- **src**: Contains all the .py modules called in the notebooks. We stored functions there to make the notebooks shorter and easier to read.
- **setup.py**: Nothing intersting, just in the case I want to publish the project as a package.

#### Detailed repository structure:
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- This file
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   └── results        <- Results and outputs generated by the pipeline.
    │
    ├── docs               <- Projects relative documentation
    │
    ├── models             <- Trained and serialized models
    │   ├── chosen_pipeline.pkl
    │   ├── chosen_pipeline_trained.pkl
    │   └── .gitkeep       
    │
    ├── notebooks          <- The different Jupyter notebooks used to execute my code
    │   ├── data_preparation.ipynb
    │   ├── features_exploration.ipynb
    │   ├── pipeline_dev.ipynb
    │   ├── pipeline_test.ipynb
    │   ├── results_final.ipynb
    │   ├── support_analysis.ipynb
    │   ├── user_test.ipynb
    │   ├── archive        <- Archived notebooks for backup and reference.
    │   │   ├── data_archive_code.ipynb
    │   │   └── X)_Model_exec_curr_seas.ipynb
    │   │   └── __main__.ipynb
    │   └── dvclive 
    │
    ├── references  
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │       ├── all_features_density_estimate.png
    │       ├── all_features_histo.png
    │       └── .gitkeep
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── environment and requirements
    │   ├── environment_projet_mbappe.yml
    │   ├── requirements_proj_mbappe.txt
    │   ├── test_environment.py
    │   └── tox.ini
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── configuration  <- Configuration files and variables
    │   │   ├── constant_variables.py
    │   │   └── settings.py
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── data_combination.py
        │   │   ├── make_dataset.py
    │   │   └── preprocessing.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── features_selection.py
    │   │   ├── initialize_new_features_columns.py
    │   │   └── make_new_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── pipeline       <- Scripts to handle analysis and pipeline results
    │   │   ├── analysis.py
    │   │   ├── model.py
    │   │   └── results.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   ├── visualize.py
    │   │   └── learning_curves.py
    │   │
    │   ├── tests.py       <- Unit tests and testing utilities
    │   ├── useful_functions.py <- Utility functions for the project
    │   └── user_test.py   <- Scripts for user testing and validation
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



--------

How to install and run the project
----------------------------------
I have not solid knowledge on code’s surrounding tools and executors, so I do not know it there are specific software or others required for my project…

I only use VS Code and the modules/packages listed in my ‘requirements’ file. However, I plan to set authentication or protection on my project.
