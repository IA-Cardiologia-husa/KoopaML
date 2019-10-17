import numpy as np
import pandas as pd
import sklearn.model_selection as sk_ms
import sklearn.utils as sk_u
import time

from .stratifiedgroupkfold import StratifiedGroupKFold

def external_validation_RS(data, external_data, label, score_label, sign):

	Y = external_data.loc[:,[label]]
	S = sign*external_data.loc[:,[score_label]]

	#Saved as a list of lists because of compatibility with predict_kfold
	tl_pp_dict={"true_label":[list(Y.values)], "pred_prob":[list(S.values)]}

	return tl_pp_dict

def external_validation(data, external_data, label, features, clf):
	X = data.loc[:,features]
	Y = data.loc[:,[label]]

	external_X = external_data.loc[:,features]
	external_Y = external_data.loc[:,[label]]

	external_Y_prob = clf.fit(X, Y).predict_proba(external_X)[:,1]

	#Saved as a list of lists because of compatibility with predict_kfold
	tl_pp_dict={"true_label":[list(external_Y.values)], "pred_prob":[list(external_Y_prob)]}

	return tl_pp_dict

def predict_kfold_ML(data, label, features, cv_type, group_label, clf, seed, cvfolds):


	X = data.loc[:,features]
	Y = data.loc[:,[label]].astype(bool)

	if(cv_type == 'stratifiedkfold'):
		skf = sk_ms.StratifiedKFold(cvfolds, random_state=seed, shuffle=True)
	elif(cv_type == 'kfold'):
		skf = sk_ms.KFold(cvfolds, random_state=seed, shuffle=True)
	else:
		raise('incompatible crossvalidation type')

	predicted_probability = []
	true_label = []

	for train_index, test_index in skf.split(X,Y):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
		try:
			Y_prob = clf.fit(X_train, Y_train).predict_proba(X_test)
			predicted_probability.append(Y_prob[:,1])
		except:
			Y_prob = clf.fit(X_train, Y_train).decision_function(X_test)
			predicted_probability.append(Y_prob)
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict


def predict_kfold_RS(data, label, features, cv_type, sign, score_name,seed, cvfolds):

	X = data.loc[:, :]
	Y = data.loc[:,[label]].astype(bool)

	if(cv_type == 'stratifiedkfold'):
		skf = sk_ms.StratifiedKFold(cvfolds, random_state=seed, shuffle=True)
	elif(cv_type == 'kfold'):
		skf = sk_ms.KFold(cvfolds, random_state=seed, shuffle=True)
	else:
		raise('incompatible crossvalidation type')

	predicted_probability = []
	true_label = []

	for train_index, test_index in skf.split(X,Y):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
		Y_prob = sign*X_test.loc[:,score_name]
		predicted_probability.append(list(Y_prob.values.flat))
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict

def predict_groupkfold_ML(data, label, features, group_label, cv_type, clf, seed, cvfolds):

	X = data.loc[:,features]
	Y = data.loc[:,[label]].astype(bool)
	G = data.loc[:, group_label]

	if (cv_type == 'stratifiedgroupkfold'):
		gkf = StratifiedGroupKFold(cvfolds, random_state=seed, shuffle=True)
	elif (cv_type == 'groupkfold'):
		X, Y, G = sk_u.shuffle(X,Y,G, random_state=seed)
		gkf = GroupKFold(cvfolds)
	else:
		raise('incompatible crossvalidation type')

	predicted_probability = []
	true_label = []

	for train_index, test_index in gkf.split(X,Y,G):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
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

def predict_groupkfold_RS(data, label, features, group_label, cv_type, sign, score_name,seed, cvfolds):

	X = data.loc[:,:]
	Y = data.loc[:,[label]].astype(bool)
	G = data.loc[:, group_label]

	if (cv_type == 'stratifiedgroupkfold'):
		gkf = StratifiedGroupKFold(cvfolds, random_state=seed, shuffle=True)
	elif (cv_type == 'groupkfold'):
		X, Y, G = sk_u.shuffle(X,Y,G, random_state=seed)
		gkf = GroupKFold(cvfolds)
	else:
		raise('incompatible crossvalidation type')

	predicted_probability = []
	true_label = []

	for train_index, test_index in gkf.split(X,Y,G):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

		Y_prob = sign*X_test.loc[:,score_name]
		predicted_probability.append(list(Y_prob.values.flat))
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict
