import numpy as np
import pandas as pd
import sklearn.model_selection as sk_ms
import time

from .stratifiedgroupkfold import StratifiedGroupKFold

def predict_kfold_ML(data, label, features, clf, clf_name, seed, cvfolds):


	X = data.loc[:,features]
	Y = data.loc[:,[label]]

	skf = sk_ms.StratifiedKFold(cvfolds, random_state=seed, shuffle=True)
	predicted_probability = []
	true_label = []

	for train_index, test_index in skf.split(X,Y):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index].astype(bool), Y.iloc[test_index].astype(bool)
		try:
			Y_prob = clf.fit(X_train, Y_train).predict_proba(X_test)
			predicted_probability.append(Y_prob[:,1])
		except:
			Y_prob = clf.fit(X_train, Y_train).decision_function(X_test)
			predicted_probability.append(Y_prob)
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict


def predict_kfold_RS(data, label, features, sign, score_name,seed, cvfolds):

	X = data.loc[:, :]
	Y = data.loc[:,[label]]

	skf = sk_ms.StratifiedKFold(cvfolds, random_state=seed, shuffle=True)
	predicted_probability = []
	true_label = []

	for train_index, test_index in skf.split(X,Y):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index].astype(bool), Y.iloc[test_index].astype(bool)
		Y_prob = sign*X_test.loc[:,score_name]
		predicted_probability.append(list(Y_prob.values.flat))
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict

def predict_groupkfold_ML(data, label, features, group_label, clf, clf_name, seed, cvfolds):

	X = data.loc[:,features]
	Y = data.loc[:,[label]]
	G = data.loc[:, group_label]

	gkf = StratifiedGroupKFold(cvfolds)

	predicted_probability = []
	true_label = []

	for train_index, test_index in gkf.split(X,Y,G):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index].astype(bool), Y.iloc[test_index].astype(bool)
		G_train, G_test = G.iloc[train_index], G.iloc[test_index]
		try:
			try:
				Y_prob = clf.fit(X_train, Y_train, groups=G_train).predict_proba(X_test)
			except:
				Y_prob = clf.fit(X_train, Y_train).predict_proba(X_test)
			predicted_probability.append(Y_prob[:,1])
		except:
			try:
				Y_prob = clf.fit(X_train, Y_train, groups=G_train).decision_function(X_test)
			except:
				Y_prob = clf.fit(X_train, Y_train).decision_function(X_test)
			predicted_probability.append(Y_prob)
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict

def predict_groupkfold_RS(data, label, features, sign, score_name,seed, cvfolds):

	X = data.loc[:,features]
	Y = data.loc[:,[label]]
	G = data.loc[:, group_label]

	gkf = StratifiedGroupKFold(cvfolds)

	predicted_probability = []
	true_label = []

	for train_index, test_index in gkf.split(X,Y,G):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index].astype(bool), Y.iloc[test_index].astype(bool)

		Y_prob = sign*X_test.loc[:,score_name]
		predicted_probability.append(list(Y_prob.values.flat))
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict
