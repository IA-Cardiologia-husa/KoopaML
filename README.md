# KoopaML

Code to automatically analyze data with binary classification algorithms. To start using this code you need to:

- Create a new python environment and install the requirements
- Provide a database
- Edit user_data_utils.py and provide: load_database(), clean_database(), process_database(), fillna_database()
- Edit user_Workflow_info.py and provide: WF_info, a dictionary with information of the workflows
- Edit user_MLmodels_info.py and provide: ML_info, a dictionary with information of the ML algorithms to use
- Edit user_RiskScores_info.py and provide: RS_info, a dictionary with information of the risk scores (if any)
- Launch the luigi scheduler: luigid --background
- Launch the luigi pipeline: python -m luigi --module KoopaML AllTasks  (add --workers 4 to have 4 concurrent processes, for example)


This action will create several folders:
- report: Contains another folder for each of the workflows in WF_info with:
  - A graph with the ROC curve of all models and risk scores
  - A graph with the ROC curve of the best model and risk score
  - A report for the best ML model, including the AUC-ROC value with its confidence interval, best threshold points, and feature importances using MDA eli5
  - A report for the best Risk score, including the AUC-ROC value with its confidence interval and all the possible threshold points
  - A report of a corrected paired t-test for the AUC-ROC between all the models and risk scores.
  - An excel table with descriptive statistics of the database, classified using the value of the label in the given workflow.
- models: Contains another folder for each of the workflows in WF_info with:
  - The ML model trained with the full dataset (i.e. without dividing data into train and test) stored in a pickle format
  - An excel table with the results of the hyperparameter tuning, if the ML model has it (function cv_results_)
- log: log of the output of the different luigi tasks
- intermediate: intermediate files of the different luigi tasks

### user_data_utils.py

To be completed...
