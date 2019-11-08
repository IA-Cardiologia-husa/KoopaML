import numpy as np
import pandas as pd
import sklearn.model_selection as sk_ms
import sklearn.utils as sk_u
import time

from .stratifiedgroupkfold import StratifiedGroupKFold

def external_validation_RS(external_data, label, score_label, sign):

	Y = external_data.loc[:,[label]]
	S = sign*external_data.loc[:,[score_label]]

	#Saved as a list of lists because of compatibility with predict_kfold
	tl_pp_dict={"true_label":[list(Y.values.flat)], "pred_prob":[list(S.values.flat)]}

	return tl_pp_dict

def external_validation(external_data, label, features, clf):
	X = external_data.loc[:,features]
	Y = external_data.loc[:,[label]]


	Y_prob = clf.predict_proba(X)[:,1]

	#Saved as a list of lists because of compatibility with predict_kfold
	tl_pp_dict={"true_label":[list(Y.values.flat)], "pred_prob":[list(Y_prob)]}

	return tl_pp_dict

def predict_filter_kfold_ML(data, label, features, filter_function, clf, seed, cvfolds):

	kf = sk_ms.KFold(cvfolds, random_state=seed, shuffle=True)

	predicted_probability = []
	true_label = []

	for train_index, test_index in kf.split(data):
		data_train, data_test = data.iloc[train_index], data.iloc[test_index]

		X_train = filter_function(data_train).loc[:,features]
		Y_train = filter_function(data_train).loc[:,[label]]

		X_train = X_train.loc[~Y_train[label].isnull()]
		Y_train = Y_train.loc[~Y_train[label].isnull()]

		X_test = data_test.loc[:,features]
		Y_test = data_test.loc[:,[label]]

		try:
			Y_prob = clf.fit(X_train, Y_train).predict_proba(X_test)
			predicted_probability.append(Y_prob[:,1])
		except:
			Y_prob = clf.fit(X_train, Y_train).decision_function(X_test)
			predicted_probability.append(Y_prob)
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict

def predict_filter_kfold_RS(data, label, features, filter_function, sign, score_name, seed, cvfolds):

	kf = sk_ms.KFold(cvfolds, random_state=seed, shuffle=True)

	predicted_probability = []
	true_label = []

	for train_index, test_index in kf.split(data):
		data_train, data_test = data.iloc[train_index], data.iloc[test_index]

		X_train = filter_function(data_train).loc[:,features]
		Y_train = filter_function(data_train).loc[:,[label]]

		X_train = X_train.loc[~Y_train[label].isnull()]
		Y_train = Y_train.loc[~Y_train[label].isnull()]

		X_test = data_test
		Y_test = data_test.loc[:,[label]]

		Y_prob = sign*X_test.loc[:,score_name]

		predicted_probability.append(Y_prob)
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict

def predict_kfold_ML(data, label, features, cv_type, clf, seed, cvfolds):


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
