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

import sklearn.base as sk_ba
import sklearn.ensemble as sk_en
import sklearn.impute as sk_im
import sklearn.linear_model as sk_lm
import sklearn.model_selection as sk_ms
import sklearn.pipeline as sk_pl
import sklearn.preprocessing as sk_pp
import xgboost as xgb
import catboost

ML_info ={}

# CLASSIFICATION MODELS

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

class RiskScore(sk_ba.TransformerMixin, sk_ba.BaseEstimator):
	def __init__(self, feature_oddratio_dict, refit = False):
		self.feature_oddratio_dict = feature_oddratio_dict

	def fit(self, X, y):
		self.threshold = 0
		for feat in self.feature_oddratio_dict.keys():
			self.threshold += self.feature_oddratio_dict[feat]*(X.loc[y==0, feat].mean()/2.+X.loc[y==1, feat].mean()/2.)
		return self.threshold

	def decision_function(self, X):
		Y_prob = 0
		for feat in self.feature_oddratio_dict.keys():
			Y_prob += self.feature_oddratio_dict[feat]*X.loc[:,feat]
		return Y_prob

	def predict(self, X):
		return self.decision_function(X) > self.norm

# ML_info['RS'] = {'formal_name': 'Risk Score',
# 				 'clf': sk_pl.Pipeline(steps=[("im",sk_im.SimpleImputer(strategy='median').set_output(transform="pandas")),
# 				 							  ('rs',RiskScore({'Var1':1, 'Var2':2, 'Var3':-1}))])}

class VariableSelection(sk_ba.TransformerMixin):
	def __init__(self, variables):
		self.variables = variables

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return X.loc[:, self.variables]

class UnivariateSelection(sk_ba.TransformerMixin):
	def __init__(self, pvalue_threshold = 0.05):
		self.pvalue_threshold = pvalue_threshold

	def fit(self, X, y):
		self.pvalues_dict = {}
		for c in X.columns:
			unique_values = sorted(list(X.loc[X[c].notnull(),c].unique()))
			if(len(unique_values) == 2):
				negative_class = unique_values[0]
				positive_class = unique_values[1]
				f00 = list(X.loc[y==0,c]).count(negative_class)
				f01 = list(X.loc[y==0,c]).count(positive_class)
				f10 = list(X.loc[y==1,c]).count(negative_class)
				f11 = list(X.loc[y==1,c]).count(positive_class)
				pvalue = sc_st.fisher_exact([[f00,f01],[f10,f11]])[1]
				self.pvalues_dict[c] = pvalue
			else:
				pvalue = sc_st.mannwhitneyu(X.loc[y==0, c].astype(float), X.loc[y==1, c].astype(float), nan_policy='omit')[1]
				self.pvalues_dict[c] = pvalue
		self.selected_features = [c for c in self.pvalues_dict.keys() if self.pvalues_dict[c]<=self.pvalue_threshold]
		self._is_fitted = True
		return self

	def transform(self, X):
		if len(self.selected_features) == 0:
			return pd.DataFrame(np.ones([X.shape[0],1]))
		return X.loc[:,self.selected_features]


	def __sklearn_is_fitted__(self):
		"""
		Check fitted status and return a Boolean value.
		"""
		return hasattr(self, "_is_fitted") and self._is_fitted

	def get_feature_names_out(self, input_features=None):
		if self.__sklearn_is_fitted__(self):
			return self.selected_features
		else:
			raise Exception("UnivariateSelection not fitted")



class LRSignificanceSelection(sk_ba.TransformerMixin):
	def __init__(self, pvalue_threshold = 0.05):
		self.pvalue_threshold = pvalue_threshold

	def fit(self, X, y):
		clf = sk_lm.LogisticRegression(penalty=None)
		clf.fit(X,y)
		preds = clf.predict_proba(X)[:,1]
		X_int = np.hstack([np.ones([X.shape[0],1]), X.values])
		cov_matrix = np.linalg.inv(X_int.T@np.diag(preds*(1-preds))@X_int)
		self.pvalues_dict = {}
		for i in range(X.shape[1]):
			z = clf.coef_[0][i] / np.sqrt(cov_matrix[i+1,i+1])
			self.pvalues_dict[X.columns[i]] =  1-scipy.special.erf(np.abs(z)/np.sqrt(2))
		self.selected_features = [c for c in self.pvalues_dict.keys() if self.pvalues_dict[c]<=self.pvalue_threshold]
		self._is_fitted = True
		return self

	def transform(self, X):
		if len(self.selected_features) == 0:
			return pd.DataFrame(np.ones([X.shape[0],1]))
		return X.loc[:,self.selected_features]


	def __sklearn_is_fitted__(self):
		"""
		Check fitted status and return a Boolean value.
		"""
		return hasattr(self, "_is_fitted") and self._is_fitted

	def get_feature_names_out(self, input_features=None):
		if self.__sklearn_is_fitted__(self):
			return self.selected_features
		else:
			raise Exception("LRSignificanceSelection not fitted")

# ML_info['LR_classic_Median_FS'] = {
# 	'formal_name': 'LR (Median Imputer, Feature Selection, Classic)',
# 	'clf': sk_pl.Pipeline(steps=[("im",sk_im.SimpleImputer(strategy = 'median').set_output(transform="pandas")),
# 								 ('scl',sk_pp.StandardScaler().set_output(transform="pandas")),
# 								 ('uns',UnivariateSelection()),
# 								 ('mus',LRSignificanceSelection()),
# 								 ('lr', sk_lm.LogisticRegression(penalty='none'))])}


# REGRESSION MODELS

ML_info['R_BT'] = {'formal_name': 'XGBoost',
				 'clf': xgb.XGBRegressor(n_estimators=100)}

ML_info['R_CAT'] = {'formal_name': 'CatBoost',
				 'clf': catboost.CatBoostRegressor()}

ML_info['R_LGBM'] = {'formal_name': 'LightGBM',
				 'clf': sk_pl.Pipeline(steps=[("im",sk_im.SimpleImputer(strategy='median')), ('lgbm', lightgbm.LGBMRegressor())])}

ML_info['Ridge'] = {'formal_name': 'Ridge Regression',
					'clf': sk_pl.Pipeline(steps=[("im",sk_im.SimpleImputer(strategy='median').set_output(transform="pandas")),
												 ("std", sk_pp.StandardScaler()),
												 ("rr",sk_lm.Ridge())])}

ML_info['Lasso'] = {'formal_name': 'Lasso Regression',
					'clf': sk_pl.Pipeline(steps=[("im",sk_im.SimpleImputer(strategy='median').set_output(transform="pandas")),
												 ("std", sk_pp.StandardScaler()),
												 ("ls",sk_lm.Lasso(alpha = 0.1))])}

pipeline_lasso = sk_pl.Pipeline(steps=[("im",sk_im.SimpleImputer(strategy='median').set_output(transform="pandas")),
									   ("std", sk_pp.StandardScaler()),
									   ("ls",sk_lm.Lasso())])
grid_params_lasso=[{'ls__alpha':[10**n for n in range(-3,4)]}]
tuned_lasso=sk_ms.GridSearchCV(pipeline_lasso,grid_params_lasso, cv=10,scoring ='r2', return_train_score=False, verbose=1)

ML_info['HypLasso'] = {'formal_name': 'Lasso Regression (Hyperparameter Tuning)',
					   'clf': tuned_lasso}

pipeline_ridge = sk_pl.Pipeline(steps=[("im",sk_im.SimpleImputer(strategy='median').set_output(transform="pandas")),
									   ("std", sk_pp.StandardScaler()),
									   ("rr",sk_lm.Ridge())])
grid_params_ridge=[{'rr__alpha':[10**n for n in range(-3,4)]}]
tuned_ridge=sk_ms.GridSearchCV(pipeline_ridge,grid_params_ridge, cv=10,scoring ='r2', return_train_score=False, verbose=1)

ML_info['HypRidge'] = {'formal_name': 'Ridge Regression (Hyperparameter Tuning)',
					   'clf': tuned_ridge}

ML_info['R_Linear'] = {'formal_name': 'Linear Regression',
					'clf': sk_pl.Pipeline(steps=[("im",sk_im.SimpleImputer(strategy='median').set_output(transform="pandas")),
												 ("lr",sk_lm.LinearRegression ())])}

# SUPERVIVENCIA
