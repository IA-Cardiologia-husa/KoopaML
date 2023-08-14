# In this archive we have to define the dictionary ml_info. This is a dictionary of dictionaries, that for each of the ML models we want
# assigns a dictionary that contains:
#
# clf: a scikit-learn classifier, or any object that implements the functions fit, and predict_proba or decision_function in the same way.
# formal_name: name to be used in plots and report
#
# In this archive we provide 4 examples:
# BT for Boosted Trees
# LR for Logistic Regression with Simple Imputer
# RF for Random Forest with KNN imputer
# LR_SCL_HypTuning for Logistic Regression with Hyperparameter tuning


import sklearn.ensemble as sk_en
import sklearn.linear_model as sk_lm
import sklearn.pipeline as sk_pl
import sklearn.preprocessing as sk_pp
import sklearn.impute as sk_im
import xgboost as xgb

ML_info ={}

ML_info['BT'] = {'formal_name': 'XGBoost',
				 'clf': xgb.XGBClassifier(n_estimators=100)}

pipeline_lr = sk_pl.Pipeline(steps=[("im",sk_im.SimpleImputer()),("lr",sk_lm.LogisticRegression())])

ML_info['LR'] = {'formal_name': 'Logistic Regression',
				 'clf': pipeline_lr}

pipeline_rf = sk_pl.Pipeline(steps=[("knn_im",sk_im.KNNImputer()),("rf",sk_en.RandomForestClassifier(n_estimators = 1000, max_features = 'auto'))])

ML_info['RF'] = {'formal_name': 'Random Forest',
				 'clf': pipeline_rf}

pipeline_lr = sk_pl.Pipeline(steps=[("im",sk_im.SimpleImputer()),
									('scl',sk_pp.StandardScaler()),
									('pt', sk_pp.PowerTransformer()),
									('lr', sk_lm.LogisticRegression())])
grid_params_lr=[{'lr__penalty':['l1', 'l2'], 'lr__C':[0.1,1.,10.,100.], 'lr__solver':['saga']},
				{'lr__penalty':['elasticnet'], 'lr__l1_ratio':[0.5], 'lr__C':[0.1,1.,10.,100.], 'lr__solver':['saga']},
				{'lr__penalty':['none'], 'lr__solver':['saga']}]
tuned_lr=sk_ms.GridSearchCV(pipeline_lr,grid_params_lr, cv=10,scoring ='roc_auc', return_train_score=False, verbose=1)

ML_info['LR_SCL_HypTuning'] = {'formal_name': 'LR (Standard Scaler, Power Transformer, and hyperparameters)',
							   'clf': tuned_lr}
