projet_Mbappe
==============================

This project is a personal initiative that I am particularly proud of. Inspired by my professional aspirations in data science and my genuine interest in football and statistics, I decided to take on this instructive challenge. The project involves building a complex and comprehensive machine learning model to predict the probabilities of victory in football matches and identify value bets. It encompasses the entire process, from data extraction to betting strategy building.
 
Project Description
-------------------

- What my application does?

This project focuses on identifying value bets offered by bookmakers on football matches. A value bet occurs when bookmakers propose odds that underestimate a team's probability of winning, resulting in odds that are higher than they should be based on the actual probability

The program inputs teams stats since the beginning of the season. And outputs victories probabilities and Boolean value of whether the bet proposed by bookmaker is a value bet.

Where this bookmakers' flaw comes from?

This discrepancy between bookmakers' probabilities and the actual probabilities often arises because bookmakers factor in bettors' behaviors when calculating their odds. They do this to encourage balanced betting on both sides, minimizing their risk and ensuring a consistent profit margin regardless of the match outcome.
 

- How does it do it?

To do so the program tries to predict football matches outcomes (Win / Lose only for now)  probabilities and compare it to bookmakers one. The proba of bookmakers are the inverse of their odds.

The model chosen to do predictions is Logistic Regression. 


- Why I used the technologies I used?

Logistic Regression prooved to be the best model as it’s a model that computes probabilities to do classification. And the output needed for our model is probabilities. Moreover, the scoring function is directly based on the the proba predicted. Indeed, the more the predicted proba is far from the real outcome, the more the scoring function penalizes.


 

What problems I faced?
----------------------

The hardest steps in my project were (in increasing order of difficulty):

1- Data cleaning/verification and construction.

First of all I had to check that the data supplied by footystats were accurate. I did it manually and was very long. It turned out that a lot was not. So it was a hard decisions to decide which feature keeping or dropping (XG for ex).

Then came features creation. I had to compute almost all the features I used to feed my model, based on basic stats of footystats. And after each new feature creation, I had to check that my calculation was accurate. As I had no automated test, I had to manually check… This whole process took me a lot of time too.

2- Computing models performances.

There are no True labels for probabilities, we must bin/discretise the predictions to evaluate their calibration. That’s not extremely precise.


3-Another difficulty, quite personal as a beginner data scientist, was the organization of the project. Indeed, I struggled to find the right structure and to understand how this structure worked. I had no knowledge on python modules, VS Code, Code splitting in several files… When I started the project, I put all my code on a Jupiter Notebook. I had to run everything almost every time I changed a line. However, I had created a quite clear and rigorous structure with a lot of commentaries. That allowed me to have a quite clear vision on my project.

At the end of November 2023, I decided to switch to VS Code, to adopt the Cookiecutter project structure and the Sphinx documentation. That was a major change for me that required me around one month of adaptation.

 

How to install and run the project
----------------------------------
I have not solid knowledge on code’s surrounding tools and executors, so I do not know it there are specific software or others required for my project…

I only use VS Code and the modules/packages listed in my ‘requirements’ file. However, I plan to set authentication or protection on my project.

 

Licence:
--------
Thise project is striclty confidential. Its use is only reserved to his author.



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
