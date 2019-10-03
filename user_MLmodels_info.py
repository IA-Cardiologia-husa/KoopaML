# In this archive we have to define the dictionary ml_info. This is a dictionary of dictionaries, that for each of the ML models we want
# assigns a dictionary that contains:
#
# clf: a scikit-learn classifier, or any object that implements the functions fit, and predict_proba or decision_function in the same way.
# formal_name: name to be used in plots and report
#
# In this archive we provide 4 examples:
# RF for Random Forest
# BT for Boosted Trees
# LR for Logistic Regression
# RF_pipeline for a Random Forest with hyperparameter tuning including the choice of feature selection strategy

import sklearn.ensemble as sk_en
import sklearn.linear_model as sk_lm
import sklearn.pipeline as sk_pl
import xgboost as xgb
from utils.featureselecter import FeatureSelecter

ML_info ={}

ML_info['RF'] = {'formal_name': 'Random Forest',
					'clf': sk_en.RandomForestClassifier(n_estimators = 1000, max_features = 'auto')}
ML_info['BT'] = {'formal_name': 'XGBoost',
					'clf': xgb.XGBClassifier(n_estimators=1000)}
ML_info['LR'] = {'formal_name': 'Logistic Regression',
					'clf': sk_lm.LogisticRegression()}

pipeline_rf = sk_pl.Pipeline(steps=[("fs",FeatureSelecter()),("rf",sk_en.RandomForestClassifier(n_estimators = 1000,  max_features = 'auto'))])
grid_params_rf=[{'fs__method':['eq','sfm_rf', 'skb_10'],'rf__n_estimators':[100,1000],
					'rf__max_features':[1,'auto'], 'rf__criterion':['gini','entropy'], 'rf__max_depth':[None, 1,2,5]}]
tuned_rf=sk_ms.GridSearchCV(pipeline_rf,grid_params_rf, cv=10,scoring ='roc_auc', return_train_score=False, verbose=1)

ML_info['RF_pipeline'] = {'formal_name': 'Random Forest (Hyperparameter Tuning)',
						  'clf': tuned_rf}
