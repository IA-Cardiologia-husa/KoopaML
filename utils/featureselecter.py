import numpy as np

import sklearn.ensemble as sk_en
import sklearn.naive_bayes as sk_nb
import sklearn.linear_model as sk_lm
import sklearn.model_selection as sk_ms
import sklearn.feature_selection as sk_fs
import xgboost as xgb
import eli5


class FeatureSelecter:
	def __init__(self,method='sfm_rf',clf=None, n_vars = None):
		self.n_vars = n_vars
		self.method = method
		#print("Metodo elegido:", self.method)
		if(method=='sfm_rf'):
			self.clf = sk_en.RandomForestClassifier(n_estimators = 100,  max_features = 'auto')
		elif(method=='sfm_xgb'):
			self.clf = xgb.XGBClassifier(n_estimators=100)
		elif(self.method=='sv_gnb'):
			self.clf = sk_nb.GaussianNB()
		elif(self.method=='skb'):
			self.clf = sk_fs.SelectKBest(score_func=sk_fs.f_classif, k=self.n_vars)
		elif(self.method=='eli5_rfe'):
			if(self.n_vars is None):
				self.n_vars=10
			if(clf==None):
				base_clf = sk_lm.LogisticRegression()
			else:
				base_clf = clf
			eli5_estimator = eli5.sklearn.PermutationImportance(base_clf, cv=10)
			self.clf = sk_fs.RFE(eli5_estimator, n_features_to_select=self.n_vars, step=1)
	def transform(self,X):
		if((self.method=='sfm_rf')or(self.method=='sfm_xgb')):
			if(self.n_vars is None):
				return sk_fs.SelectFromModel(self.clf, prefit=True).transform(X)
			else:
				return sk_fs.SelectFromModel(self.clf, max_features=self.nvars, thresholds=-np.inf, prefit=True).transform(X)
		elif(self.method=='sv_gnb'):
			return X[self.X_columns]
		elif(self.method=='PCA'):
			return self.clf.transform(X)
		elif(self.method=='skb'):
			return self.clf.transform(X)
		elif(self.method=='eq'):
			return X
		elif(self.method=='eli5_rfe'):
			return self.clf.transform(X)
	def fit(self,X,y):
		if(self.method=='eq'):
			return self
		elif(self.method=='sfm_rf'):
			self.clf = self.clf.fit(X,y)
		elif(self.method=='sfm_xgb'):
			self.clf = self.clf.fit(X,y)
		elif(self.method=='sv_gnb'):
			gnb = sk_nb.GaussianNB()
			#self.X_columns = seleccion_variables(X,y, 3, 3, self.clf)
			self.X_columns = seleccion_variables(X,y, 3, 10, gnb)

		elif(self.method=='skb'):
			return self.clf.fit(X,y)
		elif(self.method=='eli5_rfe'):
			return self.clf.fit(X,y)
		return self
	def set_params(self, method, clf=None, n_vars = None):
		self.method=method
		self.n_vars=n_vars
		#print("Metodo elegido:", self.method)
		if(method=='sfm_rf'):
			self.clf = sk_en.RandomForestClassifier(n_estimators = 1000,  max_features = 'auto')
		elif(method=='sfm_xgb'):
			self.clf = xgb.XGBClassifier(n_estimators=10000)
		elif(self.method=='sv_gnb'):
			self.clf = sk_nb.GaussianNB()
		elif(self.method=='skb'):
			self.clf = sk_fs.SelectKBest(score_func=sk_fs.f_classif, k=self.n_vars)
		elif(self.method=='eli5_rfe'):
			if(self.n_vars is None):
				self.n_vars=10
			if(clf==None):
				base_clf = sk_lm.LogisticRegression()
			else:
				base_clf = clf
			eli5_estimator = eli5.sklearn.PermutationImportance(base_clf, cv=10)
			self.clf = sk_fs.RFE(eli5_estimator, n_features_to_select=self.n_vars, step=1)

def seleccion_variables(X,Y, num_splits, num_repeat, clf):
	X=pd.DataFrame(X)
	Y=pd.DataFrame(Y)
	Yy=Y.astype(bool).values.ravel()

	X_columns = list(X.columns)
	seed = 1

	bestscore=0

	for j in range(0, 5):
		#print(f'Ordenacion{j}/5', flush=True)
		sc_columns=[]
		score=0.5


		for i in range(1,len(X_columns)+1):
			var_sel= X_columns[0:i]
			score_old = score

			probas = []
			respuestas =[]
			for j in range(0,num_repeat):
				skf = sk_ms.KFold(n_splits=num_splits, random_state=j, shuffle=True)
				y_prob = sk_ms.cross_val_predict(clf, X[var_sel], y=Yy, cv=skf, method='predict_proba')
				probas+=list(y_prob[:,1])
				respuestas+=list(Y.values)
			probas=pd.Series(probas).fillna(0).tolist()
			score = sk_m.roc_auc_score(respuestas,probas)


			if(score > bestscore):
				bestscore = score
				bestcolumns = X_columns[:i]
			sc_columns.append(score-score_old)
			#print(var_sel[-1], score-score_old)

		keydict = dict(zip(X_columns, sc_columns))
		while(keydict[X_columns[-1]]<0): X_columns.pop()
		X_columns.sort(key=keydict.get, reverse=True)

	print(f'SelecciÃ³n({bestscore}):', bestcolumns)
	return bestcolumns
