import os
import numpy as np
import pandas as pd
import datetime as dt
import pickle
import sklearn.metrics as sk_m
import scipy.stats as sc_st
import logging
import sys
import shutil
import shap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import luigi
import contextlib

from utils.crossvalidation import *
from utils.analysis import *
from user_data_utils import *
from user_MLmodels_info import ML_info
from user_RiskScores_info import RS_info
from user_Workflow_info import WF_info

# Global variables for path folders

log_path = os.path.abspath("log")
tmp_path = os.path.abspath("intermediate")
model_path = os.path.abspath("models")
report_path = os.path.abspath(f"report")

def setupLog(name):
	try:
		os.makedirs(log_path)
	except:
		pass
	logging.basicConfig(
		level=logging.DEBUG,
		format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
		filename=os.path.join(log_path, f'{name}.log'),
		filemode='a'
		)

	stdout_logger = logging.getLogger(f'STDOUT_{name}')
	sl = StreamToLogger(stdout_logger, logging.INFO)
	sys.stdout = sl

	stderr_logger = logging.getLogger(f'STDERR_{name}')
	sl = StreamToLogger(stderr_logger, logging.ERROR)
	sys.stderr = sl

class StreamToLogger(object):
	"""
	Fake file-like stream object that redirects writes to a logger instance.
	"""
	def __init__(self, logger, log_level=logging.INFO):
		self.logger = logger
		self.log_level = log_level
		self.linebuf = ''

	def write(self, buf):
		for line in buf.rstrip().splitlines():
			self.logger.log(self.log_level, line.rstrip())

	def flush(self):
		pass

#Luigi Tasks
class LoadDatabase(luigi.Task):
	def run(self):
		setupLog(self.__class__.__name__)
		df_input = load_database()
		df_input.to_pickle(self.output()["pickle"].path)
		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
		df_input.to_excel(writer, sheet_name='Sheet1')
		writer.close()

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_loaded.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_loaded.xlsx"))}

class CleanDatabase(luigi.Task):
	def requires(self):
		return LoadDatabase()
	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_output = clean_database(df_input)
		df_output.to_pickle(self.output()["pickle"].path)
		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
		df_output.to_excel(writer, sheet_name='Sheet1')
		writer.close()


	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_clean.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_clean.xlsx"))}

class ProcessDatabase(luigi.Task):
	wf_name = luigi.Parameter()

	def requires(self):
		return CleanDatabase()
	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_output = process_database(df_input, self.wf_name)
		df_output.to_pickle(self.output()["pickle"].path)
		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
		df_output.to_excel(writer, sheet_name='Sheet1')
		writer.close()


	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"df_processed_{self.wf_name}.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"df_processed_{self.wf_name}.xlsx"))}

# class FillnaDatabase(luigi.Task):
# 	def requires(self):
# 		return ProcessDatabase()
#
# 	def run(self):
# 		setupLog(self.__class__.__name__)
# 		df_input = pd.read_pickle(self.input()["pickle"].path)
# 		df_output = fillna_database(df_input)
# 		df_output.to_pickle(self.output()["pickle"].path)
# 		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
# 		df_output.to_excel(writer, sheet_name='Sheet1')
# 		writer.close()
#
#
# 	def output(self):
# 		try:
# 			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
# 		except:
# 			pass
# 		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_fillna.pickle")),
# 				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_fillna.xlsx"))}
#
# class FilterPreprocessDatabase(luigi.Task):
# 	wf_name = luigi.Parameter()
#
# 	def requires(self):
# 		return FillnaDatabase()
#
# 	def run(self):
# 		setupLog(self.__class__.__name__)
# 		df_input = pd.read_pickle(self.input()["pickle"].path)
# 		filter_function = WF_info[self.wf_name]["filter_function"]
# 		df_filtered = filter_function(df_input)
# 		df_preprocessed = preprocess_filtered_database(df_filtered, self.wf_name)
# 		df_preprocessed.to_pickle(self.output()["pickle"].path)
# 		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
# 		df_preprocessed.to_excel(writer, sheet_name='Sheet1')
# 		writer.close()
#
# 	def output(self):
# 		try:
# 			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
# 		except:
# 			pass
# 		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"df_filtered_preprocessed_{self.wf_name}.pickle")),
# 				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"df_filtered_preprocessed_{self.wf_name}.xlsx"))}

class LoadExternalDatabase(luigi.Task):
	def run(self):
		setupLog(self.__class__.__name__)
		df_input = load_external_database()
		df_input.to_pickle(self.output()["pickle"].path)
		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
		df_input.to_excel(writer, sheet_name='Sheet1')
		writer.close()

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "external_df_loaded.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "external_df_loaded.xlsx"))}

class CleanExternalDatabase(luigi.Task):
	def requires(self):
		return LoadExternalDatabase()
	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_output = clean_external_database(df_input)
		df_output.to_pickle(self.output()["pickle"].path)
		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
		df_output.to_excel(writer, sheet_name='Sheet1')
		writer.close()


	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "external_df_clean.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "external_df_clean.xlsx"))}

class ProcessExternalDatabase(luigi.Task):
	wf_name = luigi.Parameter()
	def requires(self):
		return CleanExternalDatabase()
	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_output = process_external_database(df_input, self.wf_name)
		df_output.to_pickle(self.output()["pickle"].path)
		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
		df_output.to_excel(writer, sheet_name='Sheet1')
		writer.close()


	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"external_df_processed_{self.wf_name}.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"external_df_processed_{self.wf_name}.xlsx"))}

# class FillnaExternalDatabase(luigi.Task):
# 	def requires(self):
# 		return ProcessExternalDatabase()
#
# 	def run(self):
# 		setupLog(self.__class__.__name__)
# 		df_input = pd.read_pickle(self.input()["pickle"].path)
# 		df_output = fillna_external_database(df_input)
# 		df_output.to_pickle(self.output()["pickle"].path)
# 		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
# 		df_output.to_excel(writer, sheet_name='Sheet1')
# 		writer.close()
#
#
# 	def output(self):
# 		try:
# 			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
# 		except:
# 			pass
# 		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "external_df_fillna.pickle")),
# 				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "external_df_fillna.xlsx"))}
#
# class FilterPreprocessExternalDatabase(luigi.Task):
# 	wf_name = luigi.Parameter()
#
# 	def requires(self):
# 		return FillnaExternalDatabase()
#
# 	def run(self):
# 		setupLog(self.__class__.__name__)
# 		df_input = pd.read_pickle(self.input()["pickle"].path)
# 		filter_function = WF_info[self.wf_name]["filter_external_validation"]
# 		df_filtered = filter_function(df_input)
# 		df_preprocessed = preprocess_filtered_external_database(df_filtered, self.wf_name)
# 		df_preprocessed .to_pickle(self.output()["pickle"].path)
# 		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
# 		df_preprocessed.to_excel(writer, sheet_name='Sheet1')
# 		writer.close()
#
#
# 	def output(self):
# 		try:
# 			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
# 		except:
# 			pass
# 		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"external_df_filtered_preprocessed_{self.wf_name}.pickle")),
# 				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"external_df_filtered_preprocessed_{self.wf_name}.xlsx"))}

class ExternalValidation(luigi.Task):
	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return {'data': ProcessExternalDatabase(wf_name = self.wf_name),
				'clf': FinalModelAndHyperparameterResults(wf_name = self.wf_name, clf_name = self.clf_name)
				}

	def run(self):
		setupLog(self.__class__.__name__)
		external_data = pd.read_pickle(self.input()["data"]["pickle"].path)
		features = WF_info[self.wf_name]["feature_list"]
		label = WF_info[self.wf_name]["label_name"]
		filter = WF_info[self.wf_name]["filter_external_validation"]
		external_data = filter(external_data)
		with open(self.input()["clf"].path, 'rb') as f:
			clf = pickle.load(f)

		X = external_data.loc[:,features]
		Y = external_data.loc[:,[label]]
		try:
			Y_prob = clf.predict_proba(X)[:,1]
		except:
			Y_prob = clf.decision_function(X)

		X['True Label'] = Y
		X['Predicted Probability'] = Y_prob
		X.to_excel(self.output()["xls"].path)
		X.to_pickle(self.output()["pickle"].path)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"ExternalValidation_PredProb_{self.wf_name}_{self.clf_name}.xlsx")),
				"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"ExternalValidation_PredProb_{self.wf_name}_{self.clf_name}.pickle"))}

class ExternalValidationRS(luigi.Task):
	score_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return ProcessExternalDatabase(wf_name = self.wf_name)

	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		label = WF_info[self.wf_name]["label_name"]
		feature_oddratio_dict = RS_info[self.score_name]["feature_oddratio"]

		tl_pp_dict = external_validation_RS(df_input, label, feature_oddratio_dict)

		with open(self.output().path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(tl_pp_dict, f, pickle.HIGHEST_PROTOCOL)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"ExternalValidation_PredProb_{self.wf_name}_{self.score_name}.dict"))

class ExternalValidationRefittedRS(luigi.Task):
	score_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return {'df': ProcessExternalDatabase(wf_name = self.wf_name),
				'feature_oddratio': FinalRefittedRSAndOddratios(wf_name = self.wf_name, score_name=self.score_name)}

	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()['df']["pickle"].path)
		label = WF_info[self.wf_name]["label_name"]
		with open(self.input()["feature_oddratio"]['pickle'].path, 'rb') as f:
			feature_oddratio_dict=pickle.load(f)


		tl_pp_dict = external_validation_RS(df_input, label, feature_oddratio_dict)

		with open(self.output().path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(tl_pp_dict, f, pickle.HIGHEST_PROTOCOL)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"ExternalValidation_PredProb_{self.wf_name}_{self.score_name}.dict"))

# class CreateFolds(luigi.Task):
# 	seed = luigi.IntParameter()
# 	wf_name = luigi.Parameter()
#
# 	def requires(self):
# 		return ProcessDatabase(wf_name = self.wf_name)
#
# 	def run(self):
# 		setupLog(self.__class__.__name__)
#
# 		filter_function = WF_info[self.wf_name]["filter_function"]
# 		features = WF_info[self.wf_name]["feature_list"]
# 		label = WF_info[self.wf_name]["label_name"]
# 		group_label = WF_info[self.wf_name]["group_label"]
# 		cv_type = WF_info[self.wf_name]["validation_type"]
# 		cv_folds = WF_info[self.wf_name]["cv_folds"]
#
# 		df_input = pd.read_pickle(self.input()["pickle"].path)
#
# 		if(cv_type == 'kfold'):
# 			data = filter_function(df_input)
# 			X = data.loc[:,features]
# 			Y = data.loc[:,[label]].astype(bool)
# 			kf = sk_ms.KFold(cv_folds, random_state=self.seed, shuffle=True)
# 			i=0
# 			for train_index, test_index in kf.split(X,Y):
# 				data_train, data_test = data.iloc[train_index], data.iloc[test_index]
# 				data_train.to_excel(self.output()[f'Train_{i}_excel'].path)
# 				data_test.to_excel(self.output()[f'Test_{i}_excel'].path)
# 				data_train.to_pickle(self.output()[f'Train_{i}'].path)
# 				data_test.to_pickle(self.output()[f'Test_{i}'].path)
# 				i+=1
# 		elif(cv_type == 'unfilteredkfold'):
# 			data = df_input
# 			X = data.loc[:,features]
# 			Y = data.loc[:,[label]].astype(bool)
# 			kf = sk_ms.KFold(cv_folds, random_state=self.seed, shuffle=True)
# 			i=0
# 			for train_index, test_index in kf.split(X,Y):
# 				data_train, data_test = data.iloc[train_index], data.iloc[test_index]
# 				data_train.to_excel(self.output()[f'Train_{i}_excel'].path)
# 				data_test.to_excel(self.output()[f'Test_{i}_excel'].path)
# 				data_train.to_pickle(self.output()[f'Train_{i}'].path)
# 				data_test.to_pickle(self.output()[f'Test_{i}'].path)
# 				i+=1
# 		elif(cv_type == 'stratifiedkfold'):
# 			data = filter_function(df_input)
# 			X = data.loc[:,features]
# 			Y = data.loc[:,[label]].astype(bool)
# 			skf = sk_ms.StratifiedKFold(cv_folds, random_state=self.seed, shuffle=True)
# 			i=0
# 			for train_index, test_index in skf.split(X,Y):
# 				data_train, data_test = data.iloc[train_index], data.iloc[test_index]
# 				data_train.to_excel(self.output()[f'Train_{i}_excel'].path)
# 				data_test.to_excel(self.output()[f'Test_{i}_excel'].path)
# 				data_train.to_pickle(self.output()[f'Train_{i}'].path)
# 				data_test.to_pickle(self.output()[f'Test_{i}'].path)
# 				i+=1
# 		elif (cv_type == 'groupkfold'):
# 			data = filter_function(df_input)
# 			data = sk_u.shuffle(data, random_state=self.seed)
# 			X = data.loc[:,features]
# 			Y = data.loc[:,[label]].astype(bool)
# 			G = data.loc[:, group_label]
#
# 			gkf = sk_ms.GroupKFold(cv_folds)
# 			i=0
# 			for train_index, test_index in gkf.split(X,Y,G):
# 				data_train, data_test = data.iloc[train_index], data.iloc[test_index]
# 				data_train.to_excel(self.output()[f'Train_{i}_excel'].path)
# 				data_test.to_excel(self.output()[f'Test_{i}_excel'].path)
# 				data_train.to_pickle(self.output()[f'Train_{i}'].path)
# 				data_test.to_pickle(self.output()[f'Test_{i}'].path)
# 				i+=1
# 		elif (cv_type == 'stratifiedgroupkfold'):
# 			data = filter_function(df_input)
# 			X = data.loc[:,features]
# 			Y = data.loc[:,[label]].astype(bool)
# 			G = data.loc[:, group_label]
# 			sgkf = sk_ms.StratifiedGroupKFold(cv_folds, random_state=self.seed, shuffle=True)
# 			i=0
# 			for train_index, test_index in sgkf.split(X,Y,G):
# 				data_train, data_test = data.iloc[train_index], data.iloc[test_index]
# 				data_train.to_excel(self.output()[f'Train_{i}_excel'].path)
# 				data_test.to_excel(self.output()[f'Test_{i}_excel'].path)
# 				data_train.to_pickle(self.output()[f'Train_{i}'].path)
# 				data_test.to_pickle(self.output()[f'Test_{i}'].path)
# 				i+=1
# 		else:
# 			raise('incompatible crossvalidation type')
#
# 	def output(self):
# 		try:
# 			os.makedirs(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"RepetitionNo{self.seed:03d}"))
# 		except:
# 			pass
# 		dic = {}
# 		for i in range(WF_info[self.wf_name]["cv_folds"]):
# 			dic[f"Train_{i}"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"RepetitionNo{self.seed:03d}",f"Train_Fold_{i:02d}.pickle"))
# 			dic[f"Test_{i}"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"RepetitionNo{self.seed:03d}",f"Test_Fold_{i:02d}.pickle"))
# 			dic[f"Train_{i}_excel"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"RepetitionNo{self.seed:03d}",f"Train_Fold_{i:02d}.xlsx"))
# 			dic[f"Test_{i}_excel"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"RepetitionNo{self.seed:03d}",f"Test_Fold_{i:02d}.xlsx"))
# 		return dic
#
# class CalculateKFold(luigi.Task):
#
# 	seed = luigi.IntParameter()
# 	clf_name = luigi.Parameter()
# 	wf_name = luigi.Parameter()
#
# 	def requires(self):
# 		return CreateFolds(seed=self.seed, wf_name = self.wf_name)
#
# 	def run(self):
# 		setupLog(self.__class__.__name__)
#
# 		filter_function = WF_info[self.wf_name]["filter_function"]
# 		features = WF_info[self.wf_name]["feature_list"]
# 		label = WF_info[self.wf_name]["label_name"]
# 		group_label = WF_info[self.wf_name]["group_label"]
# 		cv_type = WF_info[self.wf_name]["validation_type"]
# 		folds = WF_info[self.wf_name]["cv_folds"]
# 		clf = ML_info[self.clf_name]["clf"]
#
# 		for i in range(folds):
# 			df_train = pd.read_pickle(self.input()[f'Train_{i}'].path)
# 			df_test = pd.read_pickle(self.input()[f'Test_{i}'].path)
# 			X_train, X_test = df_train.loc[:,features], df_test.loc[:,features]
# 			Y_train, Y_test = df_train.loc[:,[label]].astype(bool), df_test.loc[:,[label]].astype(bool)
# 			if ((cv_type == 'groupkfold') or (cv_type=='stratifiedgroupkfold')):
# 				G_train, G_test = df_train.loc[:,group_label], df_test.loc[:,group_label]
#
# 			try:
# 				clf.fit(X_train, Y_train.values.ravel().astype(int), groups=G_train)
# 			except:
# 				clf.fit(X_train, Y_train.values.ravel().astype(int))
#
# 			try:
# 				Y_prob_test = clf.predict_proba(X_test)[:,1]
# 				Y_prob_train = clf.predict_proba(X_train)[:,1]
# 			except:
# 				Y_prob_test = clf.decision_function(X_test)
# 				Y_prob_train = clf.decision_function(X_train)
# 			true_label_test = Y_test.values.flat
# 			if ((cv_type == 'groupkfold') or (cv_type=='stratifiedgroupkfold')):
# 				X_train[f'Group_label: {group_label}'] = G_train
# 				X_test[f'Group_label: {group_label}'] = G_test
# 			X_train['Repetition'] = self.seed
# 			X_test['Repetition'] = self.seed
# 			X_train['Fold'] = i
# 			X_test['Fold'] = i
# 			X_train['True Label'] = Y_train.astype(int)
# 			X_test['True Label'] = Y_test.astype(int)
# 			X_train['Predicted Probability'] = Y_prob_train
# 			X_test['Predicted Probability'] = Y_prob_test
#
# 			X_train.to_excel(self.output()[f"Train_{i}_excel"].path)
# 			X_test.to_excel(self.output()[f"Test_{i}_excel"].path)
# 			X_train.to_pickle(self.output()[f"Train_{i}"].path)
# 			X_test.to_pickle(self.output()[f"Test_{i}"].path)
#
# 			with open(self.output()[f"Model_{i}"].path,'wb') as f:
# 				pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
#
# 	def output(self):
# 		try:
# 			os.makedirs(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.clf_name,f"RepetitionNo{self.seed:03d}"))
# 		except:
# 			pass
# 		dic = {}
# 		for i in range(WF_info[self.wf_name]["cv_folds"]):
# 			dic[f"Train_{i}"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.clf_name,f"RepetitionNo{self.seed:03d}",f"Train_Results_{i:02d}.pickle"))
# 			dic[f"Test_{i}"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.clf_name,f"RepetitionNo{self.seed:03d}",f"Test_Results_{i:02d}.pickle"))
# 			dic[f"Train_{i}_excel"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.clf_name,f"RepetitionNo{self.seed:03d}",f"Train_Results_{i:02d}.xlsx"))
# 			dic[f"Test_{i}_excel"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.clf_name,f"RepetitionNo{self.seed:03d}",f"Test_Results_{i:02d}.xlsx"))
# 			dic[f"Model_{i}"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.clf_name,f"RepetitionNo{self.seed:03d}",f"{self.clf_name}_r{self.seed}_f{i}.pickle"))
# 		return dic

class CreateFolds(luigi.Task):
	wf_name = luigi.Parameter()

	def requires(self):
		return ProcessDatabase(wf_name = self.wf_name)

	def run(self):
		setupLog(self.__class__.__name__)

		filter_function = WF_info[self.wf_name]["filter_function"]
		features = WF_info[self.wf_name]["feature_list"]
		label = WF_info[self.wf_name]["label_name"]
		group_label = WF_info[self.wf_name]["group_label"]
		cv_type = WF_info[self.wf_name]["validation_type"]
		cv_folds = WF_info[self.wf_name]["cv_folds"]
		cv_repetitions = WF_info[self.wf_name]["cv_repetitions"]

		df_input = pd.read_pickle(self.input()["pickle"].path)

		if(cv_type == 'kfold'):
			data = filter_function(df_input).reset_index(drop=True)
			X = data.loc[:,features]
			Y = data.loc[:,[label]].astype(bool)
			data['CustomIndex'] = data.index
			for rep in range(cv_repetitions):
				kf = sk_ms.KFold(cv_folds, random_state=rep, shuffle=True)
				fold=0
				data[f'Repetition_{rep}_folds'] = np.nan
				for train_index, test_index in kf.split(X,Y):
					data.loc[test_index, f'Repetition_{rep}_folds'] = fold
					fold+=1
			data.to_excel(self.output()["xls"].path, index=False)
			data.to_pickle(self.output()["pickle"].path)

		elif(cv_type == 'unfilteredkfold'):
			data = df_input.reset_index(drop=True)
			X = data.loc[:,features]
			Y = data.loc[:,[label]].astype(bool)
			data['CustomIndex'] = data.index
			for rep in range(cv_repetitions):
				kf = sk_ms.KFold(cv_folds, random_state=rep, shuffle=True)
				fold=0
				data[f'Repetition_{rep}_folds'] = np.nan
				for train_index, test_index in kf.split(X,Y):
					data.loc[test_index, f'Repetition_{rep}_folds'] = fold
					fold+=1
			data.to_excel(self.output()["xls"].path, index=False)
			data.to_pickle(self.output()["pickle"].path)
		elif(cv_type == 'stratifiedkfold'):
			data = filter_function(df_input).reset_index(drop=True)
			X = data.loc[:,features]
			Y = data.loc[:,[label]].astype(bool)
			data['CustomIndex'] = data.index
			for rep in range(cv_repetitions):
				kf = sk_ms.StratifiedKFold(cv_folds, random_state=rep, shuffle=True)
				fold=0
				data[f'Repetition_{rep}_folds'] = np.nan
				for train_index, test_index in kf.split(X,Y):
					data.loc[test_index, f'Repetition_{rep}_folds'] = fold
					fold+=1
			data.to_excel(self.output()["xls"].path, index=False)
			data.to_pickle(self.output()["pickle"].path)

		elif (cv_type == 'groupkfold'):
			data = filter_function(df_input).reset_index(drop = True)
			data['CustomIndex'] = data.index

			for rep in range(cv_repetitions):
				# data = sk_u.shuffle(data, random_state=rep).reset_index(drop = True)
				X = data.loc[:,features]
				Y = data.loc[:,[label]].astype(bool)
				G = data.loc[:, group_label]
				# n = G.astype('category').cat.codes.max()+1
				# np.random.seed(rep)
				# dict_transform = dict(zip(np.arange(n), np.random.choice(np.arange(n), n, replace=False)))
				# G.astype('category').cat.codes.apply(lambda x: dict_transform[x])
				gkf = sk_ms.GroupKFold(cv_folds)
				fold=0
				for train_index, test_index in gkf.split(X,Y,G):
					data.loc[data['CustomIndex'].isin(data.loc[test_index,'CustomIndex'].values),f'Repetition_{rep}_folds'] = fold
					fold+=1
			data = data.sort_values('CustomIndex').reset_index()
			data.to_excel(self.output()["xls"].path, index=False)
			data.to_pickle(self.output()["pickle"].path)

		elif (cv_type == 'stratifiedgroupkfold'):
			data = filter_function(df_input).reset_index(drop=True)
			X = data.loc[:,features]
			Y = data.loc[:,[label]].astype(bool)
			G = data.loc[:, group_label]
			data['CustomIndex'] = data.index

			for rep in range(cv_repetitions):
				sgkf = sk_ms.StratifiedGroupKFold(cv_folds, random_state=rep, shuffle=True)
				fold=0
				for train_index, test_index in sgkf.split(X,Y,G):
					data.loc[test_index, f'Repetition_{rep}_folds'] = fold
					fold+=1
			data.to_excel(self.output()["xls"].path, index=False)
			data.to_pickle(self.output()["pickle"].path)
		else:
			raise('incompatible crossvalidation type')

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__,self.wf_name))
		except:
			pass
		dic = {}
		for i in range(WF_info[self.wf_name]["cv_folds"]):
			dic[f"pickle"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"df_folded.pickle"))
			dic[f"xls"] =luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"df_folded.xlsx"))
		return dic

class CalculateKFold(luigi.Task):

	seed = luigi.IntParameter()
	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return CreateFolds(wf_name = self.wf_name)

	def run(self):
		setupLog(self.__class__.__name__)

		filter_function = WF_info[self.wf_name]["filter_function"]
		features = WF_info[self.wf_name]["feature_list"]
		label = WF_info[self.wf_name]["label_name"]
		group_label = WF_info[self.wf_name]["group_label"]
		cv_type = WF_info[self.wf_name]["validation_type"]
		folds = WF_info[self.wf_name]["cv_folds"]
		clf = ML_info[self.clf_name]["clf"]

		data = pd.read_pickle(self.input()[f'pickle'].path)
		for i in range(folds):
			data_train, data_test = data.loc[data[f'Repetition_{self.seed}_folds'] != i], data.loc[data[f'Repetition_{self.seed}_folds'] == i]
			X_train, X_test = data_train.loc[:,features], data_test.loc[:,features]
			Y_train, Y_test = data_train.loc[:,[label]].astype(bool), data_test.loc[:,[label]].astype(bool)
			if ((cv_type == 'groupkfold') or (cv_type=='stratifiedgroupkfold')):
				G_train, G_test = data_train.loc[:,group_label], data_test.loc[: ,group_label]

			try:
				clf.fit(X_train, Y_train.values.ravel().astype(int), groups=G_train)
			except:
				clf.fit(X_train, Y_train.values.ravel().astype(int))

			try:
				Y_prob_test = clf.predict_proba(X_test)[:,1]
				Y_prob_train = clf.predict_proba(X_train)[:,1]
			except:
				Y_prob_test = clf.decision_function(X_test)
				Y_prob_train = clf.decision_function(X_train)
			true_label_test = Y_test.values.flat
			if ((cv_type == 'groupkfold') or (cv_type=='stratifiedgroupkfold')):
				data_train[f'Group_label: {group_label}'] = G_train
				data_test[f'Group_label: {group_label}'] = G_test
			data_train['Repetition'] = self.seed
			data_test['Repetition'] = self.seed
			data_train['Fold'] = i
			data_test['Fold'] = i
			data_train['True Label'] = Y_train.astype(int)
			data_test['True Label'] = Y_test.astype(int)
			data_train['Predicted Probability'] = Y_prob_train
			data_test['Predicted Probability'] = Y_prob_test

			saved_columns = ['CustomIndex', 'Repetition', 'Fold', 'True Label','Predicted Probability']
			if ((cv_type == 'groupkfold') or (cv_type=='stratifiedgroupkfold')):
				saved_columns.insert(1, f'Group_label: {group_label}')

			data_train.loc[:,saved_columns].to_excel(self.output()[f"Train_{i}_excel"].path, index=False)
			data_test.loc[:,saved_columns].to_excel(self.output()[f"Test_{i}_excel"].path, index=False)
			data_train.loc[:,saved_columns].to_pickle(self.output()[f"Train_{i}"].path)
			data_test.loc[:,saved_columns].to_pickle(self.output()[f"Test_{i}"].path)

			with open(self.output()[f"Model_{i}"].path,'wb') as f:
				pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.clf_name,f"RepetitionNo{self.seed:03d}"))
		except:
			pass
		dic = {}
		for i in range(WF_info[self.wf_name]["cv_folds"]):
			dic[f"Train_{i}"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.clf_name,f"RepetitionNo{self.seed:03d}",f"Train_Results_{i:02d}.pickle"))
			dic[f"Test_{i}"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.clf_name,f"RepetitionNo{self.seed:03d}",f"Test_Results_{i:02d}.pickle"))
			dic[f"Train_{i}_excel"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.clf_name,f"RepetitionNo{self.seed:03d}",f"Train_Results_{i:02d}.xlsx"))
			dic[f"Test_{i}_excel"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.clf_name,f"RepetitionNo{self.seed:03d}",f"Test_Results_{i:02d}.xlsx"))
			dic[f"Model_{i}"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.clf_name,f"RepetitionNo{self.seed:03d}",f"{self.clf_name}_r{self.seed}_f{i}.pickle"))
		return dic


class RiskScore_KFold(luigi.Task):

	seed = luigi.IntParameter()
	score_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return CreateFolds(wf_name = self.wf_name)

	def run(self):
		setupLog(self.__class__.__name__)

		label = WF_info[self.wf_name]["label_name"]
		folds = WF_info[self.wf_name]["cv_folds"]
		feature_oddratio_dict = RS_info[self.score_name]["feature_oddratio"]


		data = pd.read_pickle(self.input()[f'pickle'].path)
		for i in range(folds):
			df_train = data.loc[data[f'Repetition_{self.seed}_folds'] != i]
			df_test  = data.loc[data[f'Repetition_{self.seed}_folds'] == i]

			Y_prob = pd.Series(0, index=df_test.index)
			for feat in feature_oddratio_dict.keys():
				Y_prob += feature_oddratio_dict[feat]*df_test.loc[:,feat].fillna(pd.concat([df_test, df_train])[feat].median())
				# Y_prob += feature_oddratio_dict[feat]*df_test.loc[:,feat].fillna(df_train[feat].median())

			df_test['True Label'] = df_test[label].astype(int)
			df_test['Predicted Probability'] = Y_prob
			df_test['Repetition'] = self.seed
			df_test['Fold'] = i

			saved_columns = ['CustomIndex', 'Repetition', 'Fold', 'True Label','Predicted Probability']

			df_test.loc[:,saved_columns].to_excel(self.output()[f"Test_{i}_excel"].path, index=False)
			df_test.loc[:,saved_columns].to_pickle(self.output()[f"Test_{i}"].path)

			df_test.to_excel(self.output()[f"Test_{i}_excel"].path)
			df_test.to_pickle(self.output()[f"Test_{i}"].path)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.score_name,f"RepetitionNo{self.seed:03d}"))
		except:
			pass
		dic = {}
		for i in range(WF_info[self.wf_name]["cv_folds"]):
			dic[f"Test_{i}_excel"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.score_name,f"RepetitionNo{self.seed:03d}",f"Test_Results_{i:02d}.xlsx"))
			dic[f"Test_{i}"] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.score_name,f"RepetitionNo{self.seed:03d}",f"Test_Results_{i:02d}.pickle"))
		return dic

# class RefittedRiskScore_KFold(luigi.Task):
# 	seed = luigi.IntParameter()
# 	score_name = luigi.Parameter()
# 	wf_name = luigi.Parameter()
#
# 	def requires(self):
# 		return {'data':FilterPreprocessDatabase(self.wf_name),
# 				'unfiltered_data': FillnaDatabase()}
#
# 	def run(self):
# 		setupLog(self.__class__.__name__)
#
# 		df_input = pd.read_pickle(self.input()["unfiltered_data"]["pickle"].path)
# 		df_filtered = pd.read_pickle(self.input()["data"]["pickle"].path)
# 		label = WF_info[self.wf_name]["label_name"]
# 		filter_function = WF_info[self.wf_name]["filter_function"]
# 		features = WF_info[self.wf_name]["feature_list"]
# 		group_label = WF_info[self.wf_name]["group_label"]
# 		cv_type = WF_info[self.wf_name]["validation_type"]
# 		folds = WF_info[self.wf_name]["cv_folds"]
# 		feature_oddratio_dict = RS_info[self.score_name]["feature_oddratio"]
#
# 		if ((cv_type == 'kfold') or (cv_type=='stratifiedkfold')):
# 			tl_pp_dict = predict_kfold_refitted_RS(df_filtered, label, features, feature_oddratio_dict,  self.seed, folds)
# 		elif ((cv_type == 'groupkfold') or (cv_type=='stratifiedgroupkfold')):
# 			tl_pp_dict = predict_groupkfold_refitted_RS(df_filtered, label, features, group_label, cv_type, feature_oddratio_dict, self.seed, folds)
# 		elif (cv_type == 'unfilteredkfold'):
# 			tl_pp_dict = predict_filter_kfold_refitted_RS(df_input, label, features, filter_function, feature_oddratio_dict,  self.seed, folds)
# 		else:
# 			raise('cv_type not recognized')
#
#
# 		with open(self.output().path, 'wb') as f:
# 			# Pickle the 'data' dictionary using the highest protocol available.
# 			pickle.dump(tl_pp_dict, f, pickle.HIGHEST_PROTOCOL)
#
# 	def output(self):
# 		try:
# 			os.makedirs(os.path.join(tmp_path,self.__class__.__name__, self.wf_name, f'REFITTED_{self.score_name}'))
# 		except:
# 			pass
# 		return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, self.wf_name, f'REFITTED_{self.score_name}', f"TrueLabel_PredProb_{self.wf_name}_REFITTED_{self.score_name}_{self.seed}.dict"))


class Evaluate_ML(luigi.Task):

	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()
	ext_val = luigi.Parameter(default='No')

	def requires(self):
		if self.ext_val == 'Yes':
			yield ExternalValidation(wf_name=self.wf_name,clf_name=self.clf_name)
		else:
			for i in range(WF_info[self.wf_name]['cv_repetitions']):
				yield CalculateKFold(wf_name=self.wf_name, seed=i,clf_name=self.clf_name)

	def run(self):
		setupLog(self.__class__.__name__)

		if (self.ext_val == 'No'):
			df_aux = pd.read_pickle(self.input()[0][f'Test_{0}'].path)
			df = pd.DataFrame(columns = df_aux.columns)
			for repetition in range(len(self.input())):
				for fold in range(WF_info[self.wf_name]["cv_folds"]):
					df_aux = pd.read_pickle(self.input()[repetition][f'Test_{fold}'].path)
					df = pd.concat([df, df_aux])
			df = df.infer_objects()
		elif (self.ext_val == 'Yes'):
			df = pd.read_pickle(self.input()[0]["pickle"].path)
			df['Repetition'] = 0
			df['Fold'] = 0

		n_reps = df['Repetition'].max()+1
		n_folds = df['Fold'].max()+1
		n_repfolds = n_reps*n_folds

		aucroc_score = 0
		aucroc_score2 = 0
		aucpr_score = 0
		aucpr_score2 = 0
		for rep in range(n_reps):
			for fold in range(n_folds):
				true_label = df.loc[(df['Repetition']==rep)&(df['Fold']==fold), "True Label"]
				pred_prob = df.loc[(df['Repetition']==rep)&(df['Fold']==fold), "Predicted Probability"]
				repfold_aucroc = sk_m.roc_auc_score(true_label.loc[true_label.notnull()].astype(bool),pred_prob.loc[true_label.notnull()])
				aucroc_score+=repfold_aucroc
				aucroc_score2+=repfold_aucroc**2
				repfold_aucpr = sk_m.average_precision_score(true_label.loc[true_label.notnull()].astype(bool),pred_prob.loc[true_label.notnull()])
				aucpr_score+=repfold_aucpr
				aucpr_score2+=repfold_aucpr**2

		unfolded_true_label = df.loc[:, "True Label"]
		unfolded_pred_prob = df.loc[:, "Predicted Probability"]
		pooling_aucroc = sk_m.roc_auc_score(unfolded_true_label.loc[unfolded_true_label.notnull()].astype(bool),unfolded_pred_prob.loc[unfolded_true_label.notnull()])
		averaging_aucroc = aucroc_score/(n_repfolds)
		averaging_sample_variance_aucroc = (aucroc_score2-aucroc_score**2/n_repfolds)/(n_repfolds-1)

		pooling_aucpr = sk_m.average_precision_score(unfolded_true_label.loc[unfolded_true_label.notnull()].astype(bool),unfolded_pred_prob.loc[unfolded_true_label.notnull()])
		averaging_aucpr = aucpr_score/n_repfolds
		averaging_sample_variance_aucpr = (aucpr_score2-aucpr_score**2/n_repfolds)/(n_repfolds-1)

		if(n_folds>1):
			critical_pvalue=0.05
			c = sc_st.t.ppf(1-critical_pvalue/2, df= n_repfolds-1)
			std_error_aucroc = np.sqrt(averaging_sample_variance_aucroc*(1/n_repfolds+1/(n_folds-1)))
			std_error_aucpr = np.sqrt(averaging_sample_variance_aucpr*(1/n_repfolds+1/(n_folds-1)))
		else:
			m = (unfolded_true_label==0).sum()
			n = (unfolded_true_label==1).sum()
			auc = pooling_aucroc
			pxxy = auc/(2-auc)
			pxyy = 2*auc**2/(1+auc)
			variance = (auc*(1-auc)+(m-1)*(pxxy-auc**2)+(n-1)*(pxyy-auc**2))/(m*n)
			std_error_aucroc = np.sqrt(variance)
			c=1
			auc = pooling_aucpr
			pxxy = auc/(2-auc)
			pxyy = 2*auc**2/(1+auc)
			variance = (auc*(1-auc)+(m-1)*(pxxy-auc**2)+(n-1)*(pxyy-auc**2))/(m*n)
			std_error_aucpr = np.sqrt(variance)

		results_dict = {"pool_aucroc": pooling_aucroc,
						"avg_aucroc": averaging_aucroc,
						"avg_aucroc_stderr": std_error_aucroc,
						"aucroc_95ci_low": averaging_aucroc - c*std_error_aucroc,
						"aucroc_95ci_high": averaging_aucroc+c*std_error_aucroc,
						"pool_aucpr": pooling_aucpr,
						"avg_aucpr": averaging_aucpr,
						"avg_aucpr_stderr": std_error_aucpr,
						"aucpr_95ci_low": averaging_aucpr - c*std_error_aucpr,
						"aucpr_95ci_high": averaging_aucpr+c*std_error_aucpr}

		df.to_excel(self.output()["xls"].path)
		df.to_pickle(self.output()["pickle"].path)

		with open(self.output()["auc_results"].path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)

		with open(self.output()["auc_results_txt"].path, 'w') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			print(results_dict, file=f)


	def output(self):
		if(self.ext_val == 'Yes'):
			prefix = 'EXT_'
		else:
			prefix = ''
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name))
		except:
			pass

		return {"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name,f"Unfolded_df_{prefix}{self.clf_name}.xlsx")),
				"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name,f"Unfolded_df_{prefix}{self.clf_name}.pickle")),
				"auc_results": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name,f"AUC_results_{prefix}{self.clf_name}.pickle")),
				"auc_results_txt": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name,f"AUC_results_{prefix}{self.clf_name}.txt"))}

class EvaluateRiskScore(luigi.Task):
	wf_name = luigi.Parameter()
	score_name = luigi.Parameter()
	ext_val = luigi.Parameter(default='No')

	def requires(self):
		if self.ext_val == 'Yes':
			if(RS_info[self.score_name]['refit_oddratios'] == 'No'):
				yield ExternalValidationRS(wf_name=self.wf_name,score_name=self.score_name)
			elif(RS_info[self.score_name]['refit_oddratios'] == 'Yes'):
				yield ExternalValidationRefittedRS(wf_name=self.wf_name, score_name=self.score_name)
			else:
				raise(f"invalid 'refit_or' value in score {score_name}")
		else:
			for i in range(WF_info[self.wf_name]['cv_repetitions']):
				if(RS_info[self.score_name]['refit_oddratios'] == 'No'):
					yield RiskScore_KFold(wf_name=self.wf_name, score_name=self.score_name, seed=i)
				elif(RS_info[self.score_name]['refit_oddratios'] == 'Yes'):
					yield RefittedRiskScore_KFold(wf_name=self.wf_name, score_name=self.score_name, seed=i)
				else:
					raise(f"invalid 'refit_or' value in score {score_name}")

	def run(self):
		setupLog(self.__class__.__name__)

		if (self.ext_val == 'No'):
			df_aux = pd.read_pickle(self.input()[0][f'Test_{0}'].path)
			df = pd.DataFrame(columns = df_aux.columns)
			for repetition in range(len(self.input())):
				for fold in range(WF_info[self.wf_name]["cv_folds"]):
					df_aux = pd.read_pickle(self.input()[repetition][f'Test_{fold}'].path)
					df = pd.concat([df, df_aux])
		elif (self.ext_val == 'Yes'):
			df = pd.read_pickle(self.input()[0].path)
			df['Repetition'] = 0
			df['Fold'] = 0

		n_reps = df['Repetition'].max()+1
		n_folds = df['Fold'].max()+1
		n_repfolds = n_reps*n_folds

		aucroc_score = 0
		aucroc_score2 = 0
		aucpr_score = 0
		aucpr_score2 = 0
		for rep in range(n_reps):
			for fold in range(n_folds):
				true_label = df.loc[(df['Repetition']==rep)&(df['Fold']==fold), "True Label"]
				pred_prob = df.loc[(df['Repetition']==rep)&(df['Fold']==fold), "Predicted Probability"]
				repfold_aucroc = sk_m.roc_auc_score(true_label.loc[true_label.notnull()].astype(bool),pred_prob.loc[true_label.notnull()])
				aucroc_score+=repfold_aucroc
				aucroc_score2+=repfold_aucroc**2
				repfold_aucpr = sk_m.average_precision_score(true_label.loc[true_label.notnull()].astype(bool),pred_prob.loc[true_label.notnull()])
				aucpr_score+=repfold_aucpr
				aucpr_score2+=repfold_aucpr**2

		unfolded_true_label = df.loc[:, "True Label"]
		unfolded_pred_prob = df.loc[:, "Predicted Probability"]
		pooling_aucroc = sk_m.roc_auc_score(unfolded_true_label.loc[unfolded_true_label.notnull()].astype(bool),unfolded_pred_prob.loc[unfolded_true_label.notnull()])
		averaging_aucroc = aucroc_score/(n_repfolds)
		averaging_sample_variance_aucroc = (aucroc_score2-aucroc_score**2/n_repfolds)/(n_repfolds-1)

		pooling_aucpr = sk_m.average_precision_score(unfolded_true_label.loc[unfolded_true_label.notnull()].astype(bool),unfolded_pred_prob.loc[unfolded_true_label.notnull()])
		averaging_aucpr = aucpr_score/n_repfolds
		averaging_sample_variance_aucpr = (aucpr_score2-aucpr_score**2/n_repfolds)/(n_repfolds-1)

		if(n_folds>1):
			critical_pvalue=0.05
			c = sc_st.t.ppf(1-critical_pvalue/2, df= n_repfolds-1)
			std_error_aucroc = np.sqrt(averaging_sample_variance_aucroc*(1/n_repfolds+1/(n_folds-1)))
			std_error_aucpr = np.sqrt(averaging_sample_variance_aucpr*(1/n_repfolds+1/(n_folds-1)))
		else:
			m = (unfolded_true_label==0).sum()
			n = (unfolded_true_label==1).sum()
			auc = pooling_aucroc
			pxxy = auc/(2-auc)
			pxyy = 2*auc**2/(1+auc)
			variance = (auc*(1-auc)+(m-1)*(pxxy-auc**2)+(n-1)*(pxyy-auc**2))/(m*n)
			std_error_aucroc = np.sqrt(variance)
			c=1
			auc = pooling_aucpr
			pxxy = auc/(2-auc)
			pxyy = 2*auc**2/(1+auc)
			variance = (auc*(1-auc)+(m-1)*(pxxy-auc**2)+(n-1)*(pxyy-auc**2))/(m*n)
			std_error_aucpr = np.sqrt(variance)

		results_dict = {"pool_aucroc": pooling_aucroc,
						"avg_aucroc": averaging_aucroc,
						"avg_aucroc_stderr": std_error_aucroc,
						"aucroc_95ci_low": averaging_aucroc - c*std_error_aucroc,
						"aucroc_95ci_high": averaging_aucroc+c*std_error_aucroc,
						"pool_aucpr": pooling_aucpr,
						"avg_aucpr": averaging_aucpr,
						"avg_aucpr_stderr": std_error_aucpr,
						"aucpr_95ci_low": averaging_aucpr - c*std_error_aucpr,
						"aucpr_95ci_high": averaging_aucpr+c*std_error_aucpr}

		df.to_excel(self.output()["xls"].path)
		df.to_pickle(self.output()["pickle"].path)

		with open(self.output()["auc_results"].path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)

		with open(self.output()["auc_results_txt"].path, 'w') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			print(results_dict, file=f)


	def output(self):
		if(self.ext_val == 'Yes'):
			prefix = 'EXT_'
		else:
			prefix = ''
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name))
		except:
			pass

		return {"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name,f"Unfolded_df_{prefix}{self.score_name}.xlsx")),
				"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name,f"Unfolded_df_{prefix}{self.score_name}.pickle")),
				"auc_results": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name,f"AUC_results_{prefix}{self.score_name}.pickle")),
				"auc_results_txt": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name,f"AUC_results_{prefix}{self.score_name}.txt"))}

class ConfidenceIntervalHanleyRS(luigi.Task):
	wf_name = luigi.Parameter()
	score_name = luigi.Parameter()
	ext_val = luigi.Parameter(default = 'No')

	def requires(self):
		if (self.ext_val == 'No'):
			return ProcessDatabase(wf_name = self.wf_name)
		elif (self.ext_val == 'Yes'):
			return ProcessExternalDatabase(wf_name = self.wf_name)

	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		if self.ext_val == 'No' :
			df_filtered = WF_info[self.wf_name]["filter_function"](df_input)
		else:
			df_filtered = WF_info[self.wf_name]["filter_external_validation"](df_input)
		(auc, stderr) = AUC_stderr_classic(df_input, label_name=WF_info[self.wf_name]["label_name"], feature_oddratio=RS_info[self.score_name]["feature_oddratio"])
		ci95_low= auc-1.96*stderr
		ci95_high= auc+1.96*stderr


		with open(self.output().path,'w') as f:
			f.write(f"Confidence Interval Upper Bound Classic(95%): {auc} ({ci95_low}-{ci95_high})\n")

			(auc, stderr) = AUC_stderr_hanley(df_input , label_name=WF_info[self.wf_name]["label_name"], feature_oddratio=RS_info[self.score_name]["feature_oddratio"])
			ci95_low= auc-1.96*stderr
			ci95_high= auc+1.96*stderr
			f.write(f"Confidence Interval Hanley(95%): {ci95_low}-{ci95_high}\n")
	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path, self.__class__.__name__, self.wf_name))
		except:
			pass
		if(self.ext_val == 'Yes'):
			prefix = 'EXT_'
		else:
			prefix = ''
		return luigi.LocalTarget(os.path.join(tmp_path, self.__class__.__name__, self.wf_name, f"CI_{prefix}{self.score_name}.txt"))


class DescriptiveXLS(luigi.Task):
	wf_name = luigi.Parameter()
	ext_val = luigi.Parameter(default='No')

	def requires(self):
		if(self.ext_val == 'Yes'):
			return ProcessExternalDatabase(wf_name = self.wf_name)
		else:
			return ProcessDatabase(wf_name = self.wf_name)

	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		if self.ext_val == 'No':
			df_filtered = WF_info[self.wf_name]["filter_function"](df_input)
		else:
			df_filtered = WF_info[self.wf_name]["filter_external_validation"](df_input)
		label = WF_info[self.wf_name]["label_name"]

		df_output=create_descriptive_xls(df_filtered, self.wf_name, label)
		writer = pd.ExcelWriter(self.output().path, engine='xlsxwriter')
		df_output.to_excel(writer, sheet_name='Sheet1')
		writer.close()

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path, self.__class__.__name__))
		except:
			pass

		if(self.ext_val == 'No'):
			return luigi.LocalTarget(os.path.join(tmp_path, self.__class__.__name__, f"{self.wf_name}_descriptivo.xlsx"))
		elif(self.ext_val == 'Yes'):
			return luigi.LocalTarget(os.path.join(tmp_path, self.__class__.__name__, f"{self.wf_name}_descriptivo_EXT.xlsx"))


class HistogramsPDF(luigi.Task):
	wf_name = luigi.Parameter()
	ext_val = luigi.Parameter(default='No')

	def requires(self):
		if(self.ext_val == 'Yes'):
			return ProcessExternalDatabase(wf_name = self.wf_name)
		else:
			return ProcessDatabase(wf_name = self.wf_name)

	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		if self.ext_val == 'No':
			df_filtered = WF_info[self.wf_name]["filter_function"](df_input)
		else:
			df_filtered = WF_info[self.wf_name]["filter_external_validation"](df_input)
		label = WF_info[self.wf_name]["label_name"]
		features = WF_info[self.wf_name]["feature_list"]

		file_path = os.path.join(tmp_path, self.__class__.__name__, f"histograma_temporal_{self.wf_name}.pdf")
		pp = PdfPages(file_path)
		for f in features:
			a = np.random.random(200)
			fig, ax= plt.subplots(figsize=(10,10))
			f_min = df_filtered.loc[df_filtered[f].notnull(), f].min()
			f_max = df_filtered.loc[df_filtered[f].notnull(), f].max()
			f_std = df_filtered.loc[df_filtered[f].notnull(), f].std()
			if (f_std != 0) & (np.isnan(f_std)==False):
				ax.hist(df_filtered.loc[df_filtered[f].notnull()&(df_filtered[label]==0), f],
						bins = np.arange(f_min, f_max + f_std/4., f_std/4.),
						label = f"{label}=0")
				ax.hist(df_filtered.loc[df_filtered[f].notnull()&(df_filtered[label]==1), f],
						bins = np.arange(f_min, f_max + f_std/4., f_std/4.),
						label = f"{label}=1", alpha = 0.5)
				ax.set_title(f)
				ax.legend()
			else:
				ax.hist(df_filtered.loc[df_filtered[f].notnull()&(df_filtered[label]==0), f],
						label = f"{label}=0")
				ax.hist(df_filtered.loc[df_filtered[f].notnull()&(df_filtered[label]==1), f],
						label = f"{label}=1", alpha = 0.5)
				ax.set_title(f)
				ax.legend()
			pp.savefig(fig)
		pp.close()

		os.rename(file_path, self.output().path)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path, self.__class__.__name__))
		except:
			pass
		if(self.ext_val == 'No'):
			return luigi.LocalTarget(os.path.join(tmp_path, self.__class__.__name__, f"{self.wf_name}_histogramas.pdf"))
		elif(self.ext_val == 'Yes'):
			return luigi.LocalTarget(os.path.join(tmp_path, self.__class__.__name__, f"{self.wf_name}_histogramas_EXT.pdf"))


class FinalModelAndHyperparameterResults(luigi.Task):
	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return ProcessDatabase(wf_name = self.wf_name)

	def run(self):
		setupLog(self.__class__.__name__)

		label = WF_info[self.wf_name]["label_name"]
		features = WF_info[self.wf_name]["feature_list"]
		group_label = WF_info[self.wf_name]["group_label"]
		filter_function = WF_info[self.wf_name]["filter_function"]

		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_filtered = filter_function(df_input)

		self.clf=ML_info[self.clf_name]["clf"]

		X_full = df_filtered.loc[:,features]
		Y_full = df_filtered.loc[:,[label]]

		X = X_full.loc[~Y_full[label].isnull()]
		Y = Y_full.loc[~Y_full[label].isnull()]

		if(group_label is None):
			self.clf.fit(X,Y.values.ravel().astype(int))
		else:
			G_full = df_filtered.loc[:,[group_label]]
			G = G_full.loc[~Y_full[label].isnull()]
			try:
				self.clf.fit(X,Y.values.ravel().astype(int),groups=G)
			except:
				self.clf.fit(X,Y.values.ravel().astype(int))

		if hasattr(self.clf, 'best_estimator_'):
			writer = pd.ExcelWriter(os.path.join(model_path,self.wf_name,f"HyperparameterResults_{self.wf_name}_{self.clf_name}.xlsx"), engine='xlsxwriter')
			pd.DataFrame(self.clf.cv_results_).to_excel(writer, sheet_name='Sheet1')
			writer.close()
		elif hasattr(self.clf, '__getitem__'):
			if hasattr(self.clf[-1], 'best_estimator_'):
				writer = pd.ExcelWriter(os.path.join(model_path,self.wf_name,f"HyperparameterResults_{self.wf_name}_{self.clf_name}.xlsx"), engine='xlsxwriter')
				pd.DataFrame(self.clf[-1].cv_results_).to_excel(writer, sheet_name='Sheet1')
				writer.close()

		with open(self.output().path,'wb') as f:
			pickle.dump(self.clf, f, pickle.HIGHEST_PROTOCOL)


	def output(self):
		try:
			os.makedirs(os.path.join(model_path,self.wf_name, self.clf_name))
		except:
			pass
		return luigi.LocalTarget(os.path.join(model_path,self.wf_name,self.clf_name,f"ML_model_{self.wf_name}_{self.clf_name}.pickle"))

class FinalRefittedRSAndOddratios(luigi.Task):
	score_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return ProcessDatabase(self.wf_name)

	def run(self):
		setupLog(self.__class__.__name__)

		label = WF_info[self.wf_name]["label_name"]
		features = WF_info[self.wf_name]["feature_list"]
		feature_oddratio_dict = RS_info[self.score_name]["feature_oddratio"]
		filter_function = WF_info[self.wf_name]["filter_function"]

		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_filtered = filter_function(df_input)

		refitted_or =  refitted_oddratios(df_filtered, label, feature_oddratio_dict)

		with open(self.output()['pickle'].path,'wb') as f:
			pickle.dump(refitted_or, f, pickle.HIGHEST_PROTOCOL)

		with open(self.output()['txt'].path,'w') as f:
			for feat in refitted_or.keys():
				f.write(f'{feat}: {refitted_or[feat]}\n')

	def output(self):
		try:
			os.makedirs(os.path.join(model_path,self.wf_name,self.score_name))
		except:
			pass
		return {'pickle': luigi.LocalTarget(os.path.join(model_path,self.wf_name,self.score_name,f"Refitted_Oddratios_{self.wf_name}_{self.score_name}.pickle")),
				'txt': luigi.LocalTarget(os.path.join(model_path,self.wf_name,self.score_name,f"Refitted_Oddratios_{self.wf_name}_{self.score_name}.txt"))}

# class AllModels_PairedTTest(luigi.Task):
# 	wf_name = luigi.Parameter()
# 	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
# 	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
# 	ext_val = luigi.Parameter(default = 'No')
#
# 	def requires(self):
# 		requirements={}
# 		for clf_or_score1 in self.list_RS:
# 			requirements[clf_or_score1] = EvaluateRiskScore(wf_name=self.wf_name, score_name = clf_or_score1, ext_val=self.ext_val)
# 		for clf_or_score2 in self.list_ML:
# 			requirements[clf_or_score2] = Evaluate_ML(wf_name=self.wf_name, clf_name=clf_or_score2, ext_val=self.ext_val)
# 		return requirements
#
# 	def run(self):
# 		setupLog(self.__class__.__name__)
# 		with open(self.output().path,'w') as f:
# 			for clf_or_score1 in self.list_ML+self.list_RS:
# 				for clf_or_score2 in self.list_ML+self.list_RS:
# 					if (clf_or_score1 != clf_or_score2):
# 						df1 = pd.read_pickle(self.input()[clf_or_score1]['pickle'].path)
# 						df2 = pd.read_pickle(self.input()[clf_or_score2]['pickle'].path)
# 						score=0
# 						score2=0
#
# 						n_reps = WF_info[self.wf_name]["cv_repetitions"]
# 						n_folds = WF_info[self.wf_name]["cv_folds"]
# 						n_repfolds = n_reps*n_folds
#
# 						for rep in range(n_reps):
# 							for fold in range(n_folds):
# 								true_label1 = df1.loc[(df1['Repetition']==rep)&(df1['Fold']==fold), 'True Label'].values
# 								pred_prob1 = df1.loc[(df1['Repetition']==rep)&(df1['Fold']==fold), 'Predicted Probability'].values
# 								tl1 = true_label1[~np.isnan(true_label1)]
# 								pp1 = pred_prob1[~np.isnan(true_label1)]
# 								auc1 = sk_m.roc_auc_score(tl1,pp1)
#
# 								#True labels for the same workflow should be the same and there is no need to load the ones from the second
# 								pred_prob2 = df2.loc[(df2['Repetition']==rep)&(df1['Fold']==fold), 'Predicted Probability'].values
# 								pp2 = pred_prob2[~np.isnan(true_label1)]
# 								auc2 = sk_m.roc_auc_score(tl1,pp2)
#
# 								score+= auc1-auc2
# 								score2+=(auc1-auc2)**2
#
# 						averaging_diff = score/n_repfolds
# 						averaging_sample_variance = (score2-score**2/n_repfolds)/(n_repfolds-1)
# 						if(n_folds>1):
# 							std_error = np.sqrt(averaging_sample_variance*(1/n_repfolds+1/(n_folds-1)))
# 						else:
# 							std_error = 1e100
#
# 						t_statistic = averaging_diff/std_error
# 						pvalue = sc_st.t.sf(np.absolute(t_statistic), df= n_repfolds-1)
#
# 						if clf_or_score1 in self.list_ML:
# 							formal_name1 = ML_info[clf_or_score1]["formal_name"]
# 						else:
# 							formal_name1 = RS_info[clf_or_score1]["formal_name"]
# 						if clf_or_score2 in self.list_ML:
# 							formal_name2 = ML_info[clf_or_score2]["formal_name"]
# 						else:
# 							formal_name2 = RS_info[clf_or_score2]["formal_name"]
# 						wf_formal_title = WF_info[self.wf_name]["formal_title"]
# 						f.write(f"{wf_formal_title}: {formal_name1}-{formal_name2}, Avg Diff: {averaging_diff}, p-value: {pvalue}\n")
#
#
# 	def output(self):
# 		if(self.ext_val == 'Yes'):
# 			prefix = 'EXT_'
# 		else:
# 			prefix = ''
# 		try:
# 			os.makedirs(os.path.join(report_path, self.wf_name))
# 		except:
# 			pass
# 		return luigi.LocalTarget(os.path.join(report_path, self.wf_name,f"AllModelsPairedTTest_{prefix}{self.wf_name}.txt"))

class GraphsWF(luigi.Task):

	wf_name = luigi.Parameter()
	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	n_best_ML = luigi.IntParameter(default=1)
	n_best_RS = luigi.IntParameter(default=2)
	ext_val = luigi.Parameter(default='No')
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	def requires(self):
		requirements = {}
		for i in self.list_ML:
			requirements[i] = Evaluate_ML(clf_name = i, wf_name = self.wf_name, ext_val=self.ext_val)
		for i in self.list_RS:
			requirements[i] = EvaluateRiskScore(score_name = i, wf_name = self.wf_name, ext_val=self.ext_val)
		return requirements

	def run(self):
		setupLog(self.__class__.__name__)

		# First we plot every ML model and risk score
		plot_all_rocs(task_requires=self.input(), fig_path= self.output()["roc_all"].path,title=WF_info[self.wf_name]["formal_title"])
		plot_all_prs(task_requires=self.input(), fig_path= self.output()["pr_all"].path,title=WF_info[self.wf_name]["formal_title"])

		# Second we open the results dictionary for every ML model and Risk Score in the workflow wf_name to determine
		# the best ML model and the best risk score
		if((len(self.list_RS) > 0) & (len(self.list_ML) > 0)):
			if(self.n_best_ML > len(self.list_ML)):
				self.n_best_ML = len(self.list_ML)
			if(self.n_best_RS > len(self.list_RS)):
				self.n_best_RS = len(self.list_RS)
			auc_ml = {}
			for i in self.list_ML:
				with open(self.input()[i]["auc_results"].path, 'rb') as f:
					results_dict=pickle.load(f)
					auc_ml[i]=results_dict["avg_aucroc"]

			sorted_ml = sorted(auc_ml.keys(), key=lambda x: auc_ml[x], reverse=True)

			auc_rs = {}
			for i in self.list_RS:
				with open(self.input()[i]["auc_results"].path, 'rb') as f:
						results_dict=pickle.load(f)
						auc_rs[i]=results_dict["avg_aucroc"]

			sorted_rs = sorted(auc_rs.keys(), key=lambda x: auc_rs[x], reverse=True)

			# We plot the ROC of the best ML model and the best risk score
			tasks={}
			for ml in sorted_ml[0:self.n_best_ML]:
				tasks[ml]=self.input()[ml]
			for rs in sorted_rs[0:self.n_best_RS]:
				tasks[rs]=self.input()[rs]

			plot_all_rocs(task_requires=tasks , fig_path= self.output()["roc_best"].path,title=WF_info[self.wf_name]["formal_title"])
			plot_all_prs(task_requires=tasks , fig_path= self.output()["pr_best"].path,title=WF_info[self.wf_name]["formal_title"])

	def output(self):
		try:
			os.makedirs(os.path.join(report_path+f"-{self.datestring}",self.wf_name))
		except:
			pass
		if(self.ext_val == 'Yes'):
			prefix = 'EXT_'
		else:
			prefix = ''
		if((len(self.list_RS) > 0) & (len(self.list_ML) > 0)):
			return {"roc_all": luigi.LocalTarget(os.path.join(report_path+f"-{self.datestring}",self.wf_name, f"AllModelsROC_{prefix}{self.wf_name}.png")),
					"roc_best": luigi.LocalTarget(os.path.join(report_path+f"-{self.datestring}",self.wf_name, f"BestModelsROC_{prefix}{self.wf_name}.png")),
					"pr_all": luigi.LocalTarget(os.path.join(report_path+f"-{self.datestring}",self.wf_name, f"AllModelsPR_{prefix}{self.wf_name}.png")),
					"pr_best": luigi.LocalTarget(os.path.join(report_path+f"-{self.datestring}",self.wf_name, f"BestModelsPR_{prefix}{self.wf_name}.png"))}
		else:
			return {"roc_all": luigi.LocalTarget(os.path.join(report_path+f"-{self.datestring}",self.wf_name, f"AllModelsROC_{prefix}{self.wf_name}.png")),
					"pr_all": luigi.LocalTarget(os.path.join(report_path+f"-{self.datestring}",self.wf_name, f"AllModelsPR_{prefix}{self.wf_name}.png"))}



class ThresholdPoints(luigi.Task):
	clf_or_score=luigi.Parameter()
	wf_name = luigi.Parameter()
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	ext_val = luigi.Parameter(default = 'No')


	def requires(self):
		if (self.clf_or_score in self.list_RS):
			return EvaluateRiskScore(wf_name=self.wf_name, score_name = self.clf_or_score, ext_val = self.ext_val)
		elif (self.clf_or_score in self.list_ML):
			return Evaluate_ML(wf_name=self.wf_name, clf_name=self.clf_or_score, ext_val=self.ext_val)
		else:
			raise Exception(f"{self.clf_score} not in list_ristkscores or list_MLmodels")

	def run(self):
		setupLog(self.__class__.__name__)

		df = pd.read_pickle(self.input()["pickle"].path)
		true_label = df['True Label'].values
		pred_prob = df['Predicted Probability'].values

		with open(self.output().path,'w') as f:
			(best_threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = cutoff_threshold_accuracy(pred_prob, true_label)
			f.write(f'Threshold: {best_threshold} Optimum for accuracy\n')
			f.write(f'TP:{tprate*100:.1f} FP:{fprate*100:.1f} TN:{tnrate*100:.1f} FN:{fnrate*100:.1f}\n')
			f.write(f'Sensitivity:{sens*100:.1f} Specificity:{spec*100:.1f} Precision:{prec*100:.1f} NPRv:{nprv*100:.1f}\n')
			f.write("\n")

			(best_threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = cutoff_threshold_single(pred_prob, true_label)
			f.write(f'Threshold: {best_threshold} Optimum for single point AUC\n')
			f.write(f'TP:{tprate*100:.1f} FP:{fprate*100:.1f} TN:{tnrate*100:.1f} FN:{fnrate*100:.1f}\n')
			f.write(f'Sensitivity:{sens*100:.1f} Specificity:{spec*100:.1f} Precision:{prec*100:.1f} NPRv:{nprv*100:.1f}\n')
			f.write("\n")

			threshold_dict = cutoff_threshold_double(pred_prob, true_label)
			(best_threshold1, tprate1, fprate1, tnrate1, fnrate1, sens1, spec1, prec1, nprv1) = threshold_dict["threshold1"]
			(best_threshold2, tprate2, fprate2, tnrate2, fnrate2, sens2, spec2, prec2, nprv2) = threshold_dict["threshold2"]

			f.write('Optimum for double point AUC\n')
			f.write(f'Threshold: {best_threshold1}\n')
			f.write(f'TP:{tprate1*100:.1f} FP:{fprate1*100:.1f} TN:{tnrate1*100:.1f} FN:{fnrate1*100:.1f}\n')
			f.write(f'Sensitivity:{sens1*100:.1f} Specificity:{spec1*100:.1f} Precision:{prec1*100:.1f} NPRv:{nprv1*100:.1f}\n')
			f.write(f'Threshold: {best_threshold2}\n')
			f.write(f'TP:{tprate2*100:.1f} FP:{fprate2*100:.1f} TN:{tnrate2*100:.1f} FN:{fnrate2*100:.1f}\n')
			f.write(f'Sensitivity:{sens2*100:.1f} Specificity:{spec2*100:.1f} Precision:{prec2*100:.1f} NPRv:{nprv2*100:.1f}\n')
			f.write("\n")

			threshold_dict = cutoff_threshold_triple(pred_prob, true_label)
			(best_threshold1, tprate1, fprate1, tnrate1, fnrate1, sens1, spec1, prec1, nprv1) = threshold_dict["threshold1"]
			(best_threshold2, tprate2, fprate2, tnrate2, fnrate2, sens2, spec2, prec2, nprv2) = threshold_dict["threshold2"]
			(best_threshold3, tprate3, fprate3, tnrate3, fnrate3, sens3, spec3, prec3, nprv3) = threshold_dict["threshold3"]

			f.write('Optimum for triple point AUC\n')
			f.write(f'Threshold: {best_threshold1}\n')
			f.write(f'TP:{tprate1*100:.1f} FP:{fprate1*100:.1f} TN:{tnrate1*100:.1f} FN:{fnrate1*100:.1f}\n')
			f.write(f'Sensitivity:{sens1*100:.1f} Specificity:{spec1*100:.1f} Precision:{prec1*100:.1f} NPRv:{nprv1*100:.1f}\n')
			f.write(f'Threshold: {best_threshold2}\n')
			f.write(f'TP:{tprate2*100:.1f} FP:{fprate2*100:.1f} TN:{tnrate2*100:.1f} FN:{fnrate2*100:.1f}\n')
			f.write(f'Sensitivity:{sens2*100:.1f} Specificity:{spec2*100:.1f} Precision:{prec2*100:.1f} NPRv:{nprv2*100:.1f}\n')
			f.write(f'Threshold: {best_threshold3}\n')
			f.write(f'TP:{tprate3*100:.1f} FP:{fprate3*100:.1f} TN:{tnrate3*100:.1f} FN:{fnrate3*100:.1f}\n')
			f.write(f'Sensitivity:{sens3*100:.1f} Specificity:{spec3*100:.1f} Precision:{prec3*100:.1f} NPRv:{nprv3*100:.1f}\n')
			f.write("\n")

			for beta in [0.5,1,2]:
				(max_f1_threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = cutoff_threshold_maxfbeta(pred_prob, true_label, beta)
				f.write(f'Threshold: {max_f1_threshold} Optimum for f{beta}\n')
				f.write(f'TP:{tprate*100:.1f} FP:{fprate*100:.1f} TN:{tnrate*100:.1f} FN:{fnrate*100:.1f}\n')
				f.write(f'Sensitivity:{sens*100:.1f} Specificity:{spec*100:.1f} Precision:{prec*100:.1f} NPRv:{nprv*100:.1f}\n')
				f.write("\n")

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		if self.ext_val == 'No':
			return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}.txt"))
		elif self.ext_val == 'Yes':
			return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}_EXT.txt"))

class BestMLModelReport(luigi.Task):
	wf_name = luigi.Parameter()
	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	ext_val = luigi.Parameter(default='No')
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	def requires(self):
		requirements = {}
		for i in self.list_ML:
			requirements[i] = Evaluate_ML(clf_name = i, wf_name = self.wf_name, ext_val=self.ext_val)
			requirements[i+'_threshold'] = ThresholdPoints(clf_or_score = i, wf_name = self.wf_name, list_ML = self.list_ML, ext_val=self.ext_val)
		return requirements

	def run(self):
		setupLog(self.__class__.__name__)
		# First we open the results dictionary for every ML model in the workflow wf_name to determine
		# the best ML model
		auc_ml = {}
		for i in self.list_ML:
			with open(self.input()[i]["auc_results"].path, 'rb') as f:
				results_dict=pickle.load(f)
				auc_ml[i]=results_dict["avg_aucroc"]

		best_ml = max(auc_ml.keys(), key=(lambda k: auc_ml[k]))

		with open(self.input()[best_ml]["auc_results"].path, 'rb') as f:
			best_ml_results_dict=pickle.load(f)

		with open(self.output().path,'w') as f:
			f.write(f"Model name: {best_ml}\n")
			f.write(f"AUC ROC: {best_ml_results_dict['avg_aucroc']}\n")
			f.write(f"AUC ROC stderr: {best_ml_results_dict['avg_aucroc_stderr']}\n")
			f.write(f"AUC ROC Confidence Interval (95%): {best_ml_results_dict['aucroc_95ci_low']}-{best_ml_results_dict['aucroc_95ci_high']}\n")
			f.write(f"AUC PR: {best_ml_results_dict['avg_aucpr']}\n")
			f.write(f"AUC PR stderr: {best_ml_results_dict['avg_aucpr_stderr']}\n")
			f.write(f"AUC PR Confidence Interval (95%): {best_ml_results_dict['aucpr_95ci_low']}-{best_ml_results_dict['aucpr_95ci_high']}\n")
			f.write("\n")
			with open(self.input()[best_ml+'_threshold'].path, 'r') as f2:
				for line in f2.readlines():
					f.write(line)

	def output(self):
		try:
			os.makedirs(os.path.join(report_path+f'-{self.datestring}',self.wf_name))
		except:
			pass
		if self.ext_val == 'No':
			return luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}',self.wf_name,f"BestML_Model_report_{self.wf_name}.txt"))
		elif self.ext_val == 'Yes':
			return luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}',self.wf_name,f"BestML_Model_report_{self.wf_name}_EXT.txt"))

class BestRSReport(luigi.Task):
	wf_name = luigi.Parameter()
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	ext_val = luigi.Parameter(default = 'No')
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	def requires(self):
		requirements = {}
		for rs in self.list_RS:
			requirements[rs] = EvaluateRiskScore(score_name = rs, wf_name = self.wf_name, ext_val = self.ext_val)
			requirements[rs+'_threshold'] = AllThresholds(clf_or_score = rs, wf_name = self.wf_name, list_RS=self.list_RS, ext_val = self.ext_val)
			if (RS_info[rs]['refit_oddratios']=='No'):
				requirements[rs+'_hanley'] = ConfidenceIntervalHanleyRS(score_name = rs, wf_name = self.wf_name, ext_val = self.ext_val)
		return requirements

	def run(self):
		setupLog(self.__class__.__name__)
		# First we open the results dictionary for every ML model in the workflow wf_name to determine
		# the best risk score
		auc_rs = {}
		for i in self.list_RS:
			with open(self.input()[i]["auc_results"].path, 'rb') as f:
				results_dict=pickle.load(f)
				auc_rs[i]=results_dict["avg_aucroc"]

		best_rs = max(auc_rs.keys(), key=(lambda k: auc_rs[k]))

		with open(self.input()[best_rs]["auc_results"].path, 'rb') as f:
			best_rs_results_dict=pickle.load(f)

		with open(self.output().path,'w') as f:
			f.write(f"Score name: {RS_info[best_rs]['formal_name']}\n")
			f.write(f"AUC ROC: {best_rs_results_dict['avg_aucroc']}\n")
			f.write(f"AUC ROC stderr: {best_rs_results_dict['avg_aucroc_stderr']}\n")
			f.write(f"AUC ROC Confidence Interval Subsampling(95%): {best_rs_results_dict['aucroc_95ci_low']}- {best_rs_results_dict['aucroc_95ci_high']}\n")
			if (RS_info[best_rs]['refit_oddratios'] == 'No'):
				with open(self.input()[best_rs+'_hanley'].path, 'r') as f2:
					for line in f2.readlines():
						f.write(line)
			f.write("\n")
			f.write(f"AUC PR: {best_rs_results_dict['avg_aucpr']}\n")
			f.write(f"AUC PR stderr: {best_rs_results_dict['avg_aucpr_stderr']}\n")
			f.write(f"AUC PR Confidence Interval Subsampling(95%): {best_rs_results_dict['aucpr_95ci_low']}- {best_rs_results_dict['aucpr_95ci_high']}\n")
			with open(self.input()[best_rs+'_threshold']['txt'].path, 'r') as f3:
				for line in f3.readlines():
					f.write(line)
	def output(self):
		try:
			os.makedirs(os.path.join(report_path+f'-{self.datestring}',self.wf_name))
		except:
			pass
		if self.ext_val == 'No':
			return luigi.LocalTarget(os.path.join(report_path+f"-{self.datestring}",self.wf_name,f"BestRS_report_{self.wf_name}.txt"))
		elif self.ext_val  == 'Yes':
			return luigi.LocalTarget(os.path.join(report_path+f"-{self.datestring}",self.wf_name,f"BestRS_report_{self.wf_name}_EXT.txt"))

class AllThresholds(luigi.Task):
	clf_or_score=luigi.Parameter()
	wf_name = luigi.Parameter()
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	ext_val = luigi.Parameter(default='No')

	def requires(self):
		if (self.clf_or_score in self.list_RS):
			return EvaluateRiskScore(wf_name=self.wf_name, score_name = self.clf_or_score, ext_val = self.ext_val)
		elif (self.clf_or_score in self.list_ML):
			return Evaluate_ML(wf_name=self.wf_name, clf_name=self.clf_or_score, ext_val = self.ext_val)
		else:
			raise Exception(f"{self.clf_score} not in list_ristkscores or list_MLmodels")

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.input()["pred_prob"].path, 'rb') as f:
			pred_prob=pickle.load(f)
		with open(self.input()["true_label"].path, 'rb') as f:
			true_label=pickle.load(f)

		list_thresholds = all_thresholds(pred_prob, true_label)

		with open(self.output()['txt'].path,'w') as f:
			rows = []
			for i in list_thresholds:
				(threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = i
				f.write(f'Threshold: {threshold}\n')
				f.write(f'TP:{tprate*100:.1f} FP:{fprate*100:.1f} TN:{tnrate*100:.1f} FN:{fnrate*100:.1f}\n')
				f.write(f'Sensitivity:{sens*100:.1f} Specificity:{spec*100:.1f} Precision:{prec*100:.1f} NPRv:{nprv*100:.1f}\n')
				rows.append([threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv])
		df_thr = pd.DataFrame(rows, columns=['Threshold','TP','FP','TN','FN', 'sensitivity','specificity','precision','nprv'])
		with open(self.output()['df'].path,'w') as f:
			df_thr.to_csv(f)
	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		if self.ext_val == 'No':
			return {'txt': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}.txt")),
					'df': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}.csv"))}
		elif self.ext_val == 'Yes':
			return {'txt': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}_EXT.txt")),
					'df': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}_EXT.csv"))}

class ShapleyValues(luigi.Task):
	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()
	ext_val = luigi.Parameter(default='Yes')

	def requires(self):
		if self.ext_val == 'No':
			requirements = {'CreateFolds': CreateFolds(wf_name = self.wf_name)}
			for rep in range(WF_info[self.wf_name]["cv_repetitions"]):
				requirements[f'CalculateKFold_{rep}'] =  CalculateKFold(clf_name = self.clf_name, wf_name = self.wf_name, seed = rep)
			return requirements
		elif self.ext_val == 'Yes':
			return {"model":FinalModelAndHyperparameterResults(wf_name = self.wf_name, clf_name = self.clf_name),
					"test_data":ProcessExternalDatabase(self.wf_name),
					"train_data":ProcessDatabase(self.wf_name)}

	def run(self):
		setupLog(self.__class__.__name__)

		feature_list = WF_info[self.wf_name]['feature_list']

		list_shap_values = []

		if self.ext_val == 'No':
			df_test_total = pd.DataFrame()
			df_folded =  pd.read_pickle(self.input()['CreateFolds']['pickle'].path)
			for rep in range(WF_info[self.wf_name]["cv_repetitions"]):
				for fold in range(WF_info[self.wf_name]["cv_folds"]):
					df_train = df_folded.loc[df_folded[f'Repetition_{rep}_folds']!=fold, feature_list]
					#If the dataset is too big (we have considered greater than 500), we select a random sample of it
					ind = df_train.index
					if len(ind) > 500:
						ind = np.random.choice(ind, 500, replace = False)
						df_train = df_train.loc[ind]

					df_test = df_folded.loc[df_folded[f'Repetition_{rep}_folds']==fold, feature_list]
					#If the test dataset is too big (we have considered greater than 100), we select a random sample of it
					ind = df_test.index
					if len(ind) > 100:
						ind = np.random.choice(ind, 100, replace = False)
						df_test = df_test.loc[ind]

					df_test_total = pd.concat([df_test_total, df_test])
					with open(self.input()[f'CalculateKFold_{rep}'][f"Model_{fold}"].path, "rb") as f:
						model = pickle.load(f)

					masker = shap.maskers.Independent(data =  df_train.loc[:,feature_list].astype(float).values)
					explainer = shap.PermutationExplainer(lambda x: model.predict_proba(x)[:,1], masker, link = shap.links.logit)
					shap_values = explainer(df_test.astype(float).values).values

					# try:
					# 	if hasattr(model, '__getitem__'):
					# 		data = df_train.loc[:,feature_list].copy()
					# 		for i in range(len(model)-1):
					# 			data = model[i].transform(data)
					# 		masker = shap.maskers.Independent(data = data)
					# 		explainer = shap.Explainer(model[-1], masker)
					# 		data_test = df_test.copy()
					# 		for i in range(len(model)-1):
					# 			data_test = model[i].transform(data_test)
					# 		shap_values = explainer.shap_values(data_test)
					# 	else:
					# 		masker = shap.maskers.Independent(data = df_train.loc[:,feature_list])
					# 		explainer = shap.Explainer(model, masker)
					# 		shap_values = explainer.shap_values(df_test)
					# except:
					# 	explainer = shap.KernelExplainer(model = lambda x: model.predict_proba(x)[:,1], data = df_train.loc[:,feature_list], link = "identity")
					# 	shap_values = explainer.shap_values(df_test)
					list_shap_values.append(shap_values)


			#combining results from all iterations
			shap_values = np.array(list_shap_values[0])
			for i in range(1,len(list_shap_values)):
				shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=0)

			shap.summary_plot(shap_values, df_test_total, max_display = 100, show=False)
			plt.savefig(self.output()["png"].path, bbox_inches='tight', dpi=300)
			plt.close()
			df_test_total.to_pickle(self.output()["pickle_dftest"].path)
			with open(self.output()["pickle_values"].path, 'wb') as f:
				pickle.dump(shap_values, f, pickle.HIGHEST_PROTOCOL)

		elif self.ext_val == 'Yes':
			filter_function = WF_info[self.wf_name]["filter_external_validation"]

			df_input_train = pd.read_pickle(self.input()["train_data"]["pickle"].path)
			df_input_test = pd.read_pickle(self.input()["test_data"]["pickle"].path)
			df_filtered_train = filter_function(df_input_train)
			df_filtered_test = filter_function(df_input_test)

			df_train = df_filtered_train.loc[:,feature_list]
			df_test = df_filtered_test.loc[:,feature_list]

			with open(self.input()["model"].path, "rb") as f:
				model = pickle.load(f)

			try:
				if hasattr(model, '__getitem__'):
					data = df_train.loc[:,feature_list]
					for i in range(len(model)-1):
						data = model[i].transform(data)
					masker = shap.maskers.Independent(data = data)
					explainer = shap.Explainer(model, masker)
					data_test = df_test
					for i in range(len(model)-1):
						data_test = model[i].transform(data_test)
					shap_values = explainer.shap_values(data_test)
				else:
					masker = shap.maskers.Independent(data = df_train.loc[:,feature_list])
					explainer = shap.Explainer(model, masker)
					shap_values = explainer.shap_values(df_test)
			except:
				explainer = shap.KernelExplainer(model = lambda x: model.predict_proba(x)[:,1], data = df_train.loc[:,feature_list], link = "identity")
				shap_values = explainer.shap_values(df_test)
			shap.summary_plot(shap_values, df_test, max_display = 100, show=False)
			plt.savefig(self.output()["png"].path, bbox_inches='tight', dpi=300)
			plt.close()
			df_test.to_pickle(self.output()["pickle_dftest"].path)
			with open(self.output()["pickle_values"].path, 'wb') as f:
				pickle.dump(shap_values, f, pickle.HIGHEST_PROTOCOL)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__,self.wf_name))
		except:
			pass

		if self.ext_val == 'No':
			return {"png": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"ShapleyValues_{self.wf_name}_{self.clf_name}.png")),
					"pickle_values": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"ShapleyValues_{self.wf_name}_{self.clf_name}.pickle")),
					"pickle_dftest": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"dftest_{self.wf_name}_{self.clf_name}.pickle"))}
		elif self.ext_val == 'Yes':
			return {"png": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"ShapleyValues_{self.wf_name}_{self.clf_name}_EXT.png")),
					"pickle_values": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"ShapleyValues_{self.wf_name}_{self.clf_name}_EXT.pickle")),
					"pickle_dftest": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"dftest_{self.wf_name}_{self.clf_name}.pickle"))}


class MDAFeatureImportances(luigi.Task):
	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()
	ext_val = luigi.Parameter(default='No')
	n_iterations = luigi.IntParameter(default=5)

	def requires(self):
		# if self.ext_val == 'No':
		# 	return {'df': FilterPreprocessDatabase(self.wf_name)}
		# elif self.ext_val == 'Yes':
		# 	return {'df': FilterPreprocessExternalDatabase(self.wf_name),
		# 			'clf': FinalModelAndHyperparameterResults(clf_name=self.clf_name, wf_name=self.wf_name)}
		if self.ext_val == 'No':
			requirements = {'CreateFolds': CreateFolds(wf_name = self.wf_name)}
			for rep in range(WF_info[self.wf_name]["cv_repetitions"]):
				requirements[f'CalculateKFold_{rep}'] =  CalculateKFold(clf_name = self.clf_name, wf_name = self.wf_name, seed = rep)
		elif self.ext_val == 'Yes':
			requirements = {"model":FinalModelAndHyperparameterResults(wf_name = self.wf_name, clf_name = self.clf_name),
					"test_data":ProcessExternalDatabase(self.wf_name),
					"train_data":ProcessDatabase(self.wf_name)}
		return requirements

	def run(self):
		setupLog(self.__class__.__name__)

		feature_list = WF_info[self.wf_name]['feature_list']
		mda = {}
		mda2 = {}
		for feat in feature_list:
			mda[feat] = 0
			mda2[feat] = 0

		rows = []
		if self.ext_val == 'No':
			df_folded = pd.read_pickle(self.input()['CreateFolds']['pickle'].path)
			for rep in range(WF_info[self.wf_name]["cv_repetitions"]):
				for fold in range(WF_info[self.wf_name]["cv_folds"]):
					# df_train = pd.read_pickle(self.input()[rep][f"Train_{fold}"].path)
					# df_test = pd.read_pickle(self.input()[rep][f"Test_{fold}"].path)
					df_test = df_folded.loc[df_folded[f"Repetition_{rep}_folds"]==fold]
					df_results = pd.read_pickle(self.input()[f'CalculateKFold_{rep}'][f"Test_{fold}"].path)
					df_test.set_index("CustomIndex")
					df_results.set_index("CustomIndex")
					df_test["Predicted Probability"] = df_results["Predicted Probability"]

					with open(self.input()[f'CalculateKFold_{rep}'][f"Model_{fold}"].path, "rb") as f:
						model = pickle.load(f)

					for feat in feature_list:
						df_shuffled = df_test.copy()
						label = WF_info[self.wf_name]["label_name"]
						true_label = df_shuffled[label].astype(int)
						pred_prob_original = df_shuffled["Predicted Probability"]
						aucroc_original = sk_m.roc_auc_score(true_label.loc[true_label.notnull()].astype(bool),pred_prob_original.loc[true_label.notnull()])

						for i in range(self.n_iterations):
							df_shuffled[feat] = np.random.permutation(df_test[feat].values)
							try:
								pred_prob = model.predict_proba(df_shuffled.loc[:, feature_list])[:,1]
							except:
								pred_prob = model.decision_function(df_shuffled.loc[:, feature_list])
							aucroc_shuffled = sk_m.roc_auc_score(true_label.loc[true_label.notnull()].astype(bool),pred_prob[true_label.notnull()])
							mda[feat] += aucroc_original - aucroc_shuffled
							mda2[feat] += (aucroc_original - aucroc_shuffled)**2
							row = {'feat':feat,
								   'rep':rep,
								   'fold':fold,
								   'iter':i,
								   'aucroc_original':aucroc_original,
								   'aucroc_shuffled':aucroc_shuffled,
								   'mda':aucroc_original-aucroc_shuffled}
							rows.append(row)
			for feat in feature_list:
				mda[feat] = mda[feat]/(WF_info[self.wf_name]["cv_repetitions"]*WF_info[self.wf_name]["cv_folds"]*self.n_iterations)
				mda2[feat] = mda2[feat]/(WF_info[self.wf_name]["cv_repetitions"]*WF_info[self.wf_name]["cv_folds"]*self.n_iterations)
		elif self.ext_val == 'Yes':

			label = WF_info[self.wf_name]["label_name"]
			filter_function = WF_info[self.wf_name]["filter_external_validation"]

			df_input_test = pd.read_pickle(self.input()["test_data"]["pickle"].path)
			df_filtered_test = filter_function(df_input_test)

			true_label_original = df_filtered_test[label].values

			with open(self.input()["model"].path, "rb") as f:
				model = pickle.load(f)

			try:
				pred_prob_original = model.predict_proba(df_filtered_test.loc[:,feature_list])[:,1]
			except:
				pred_prob_original = model.decision_function(df_filtered_test.loc[:,feature_list])
			aucroc_original = sk_m.roc_auc_score(true_label_original[~np.isnan(true_label_original)].astype(bool),pred_prob_original[~np.isnan(true_label_original)])
			for feat in feature_list:
				df_shuffled = df_filtered_test.copy()
				true_label = df_shuffled[label]

				for i in range(self.n_iterations):
					df_shuffled[feat] = np.random.permutation(df_filtered_test[feat].values)
					try:
						pred_prob = model.predict_proba(df_shuffled.loc[:, feature_list])[:,1]
					except:
						pred_prob = model.decision_function(df_shuffled.loc[:, feature_list])
					aucroc_shuffled = sk_m.roc_auc_score(true_label[~np.isnan(true_label)].astype(bool),pred_prob[~np.isnan(true_label)])
					mda[feat] += aucroc_original - aucroc_shuffled
					mda2[feat] += (aucroc_original - aucroc_shuffled)**2
					row = {'feat':feat,
						   'rep':0,
						   'fold':0,
						   'iter':i,
						   'aucroc_original':aucroc_original,
						   'aucroc_shuffled':aucroc_shuffled,
						   'mda':aucroc_original-aucroc_shuffled}
					rows.append(row)
			for feat in feature_list:
				mda[feat] = mda[feat]/(self.n_iterations)
				mda2[feat] = mda2[feat]/(self.n_iterations)

		pd.DataFrame(rows).to_pickle(self.output()['iter'].path)
		# sorted_feats = sorted(feature_list, key= lambda x: mda[feat]/(np.sqrt(mda2[feat]-mda[feat]**2)+1e-14))
		sorted_feats = sorted(feature_list, key= lambda x: mda[x], reverse=True)

		with open(self.output()['txt'].path,'w') as f:
			print(f"{'Feature':30.30} {'MDA_norm':10.10} {'MDA':10.10} {'Variation':10.10} {'z-score':10.10}", file=f)
			for feat in sorted_feats:
				print(f"{feat:30.30} {mda[feat]/mda[sorted_feats[0]]:0.4e} {mda[feat]:0.4e} {np.sqrt(mda2[feat]-mda[feat]**2):0.4e} {mda[feat]/(np.sqrt(mda2[feat]-mda[feat]**2)+1e-14):0.4e}", file=f)
		rows = []
		for feat in sorted_feats:
			row = {"Feature":feat,
				   "MDA_norm":mda[feat]/mda[sorted_feats[0]],
				   "MDA": mda[feat],
				   "stdev": np.sqrt(mda2[feat]-mda[feat]**2),
				   "z-score": mda[feat]/(np.sqrt(mda2[feat]-mda[feat]**2)+1e-14)}
			rows.append(row)
		pd.DataFrame(rows).to_csv(self.output()['csv'].path)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass

		if self.ext_val == 'No':
			return {'txt': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"MDAeli5_Log_{self.wf_name}_{self.clf_name}.txt")),
					'csv': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"MDAeli5_Log_{self.wf_name}_{self.clf_name}.csv")),
					'iter': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"df_iter_{self.wf_name}_{self.clf_name}.pickle"))}
		elif self.ext_val == 'Yes':
			return {'txt': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"MDAeli5_Log_{self.wf_name}_{self.clf_name}_EXT.txt")),
					'csv': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"MDAeli5_Log_{self.wf_name}_{self.clf_name}_EXT.csv")),
					'iter': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"df_iter_{self.wf_name}_{self.clf_name}_EXT.pickle"))}


# class FeatureScorer(luigi.Task):
# 	wf_name = luigi.Parameter()
# 	fs_name = luigi.Parameter()
#
# 	def requires(self):
# 		for i in range(1,WF_info[self.wf_name]['cv_repetitions']+1):
# 			yield FeatureScoringFolds(seed = i, cvfolds = WF_info[self.wf_name]['cvfolds'], wf_name = self.wf_name, fs_name = self.fs_name)
#
# 	def run(self):
# 		setupLog(self.__class__.__name__)
# 		pass
#
# 	def output(self):
# 		pass
#
# class FeatureScoringFolds(luigi.Task):
# 	seed = luigi.IntParameter()
# 	cvfolds = luigi.IntParameter()
# 	wf_name = luigi.Parameter()
# 	fs_name = luigi.Parameter()
#
# 	def requires(self):
# 		return FillnaDatabase()
#
# 	def run(self):
# 		setupLog(self.__class__.__name__)
#
# 		df_input = pd.read_pickle(self.input()["pickle"].path)
# 		filter_function = WF_info[self.wf_name]["filter_function"]
# 		df_filtered = filter_function(df_input)
# 		features = WF_info[self.wf_name]["feature_list"]
# 		label = WF_info[self.wf_name]["label_name"]
# 		group_label = WF_info[self.wf_name]["group_label"]
# 		cv_type = WF_info[self.wf_name]["validation_type"]
# 		fs_function = FS_info[self.fs_name]["scorer_function"]
#
# 		X = df_filtered.loc[:, features]
# 		Y = df_filtered.loc[:,[label]].astype(bool)
#
# 		if (cv_type == 'kfold'):
# 			kf = sk_ms.KFold(cvfolds, random_state=seed, shuffle=True)
# 		elif(cv_type == 'stratifiedkfold'):
# 			kf = sk_ms.StratifiedKFold(cvfolds, random_state=seed, shuffle=True)
# 		elif(cv_type == 'groupkfold'):
# 			kf = GroupKFold(cvfolds)
# 		elif(cv_type == 'stratifiedgroupkfold'):
# 			kf = StratifiedGroupKFold(cvfolds, random_state=seed, shuffle=True)
# 		elif (cv_type == 'unfilteredkfold'):
# 			kf = sk_ms.KFold(cvfolds, random_state=seed, shuffle=True)
# 		else:
# 			raise('cv_type not recognized')
#
# 		if ((cv_type == 'kfold') or (cv_type == 'stratifiedkfold') or (cv_type == 'unfilteredkfold')):
# 			fold = 0
# 			for train_index, test_index in kf.split(X,Y):
# 				X_train, X_test = X.iloc[train_index], X.iloc[test_index]
# 				Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
#
# 				X_train = X_train[~np.isnan(Y_train)]
# 				Y_train = Y_train[~np.isnan(Y_train)].astype(bool)
#
# 				feature_scores = fs_function(X_train,Y_train)
#
# 				scores_dict = dict(zip(X_train.columns, feature_scores))
# 				with open(self.output()[fold]["pickle"].path, 'wb') as f:
# 					# Pickle the 'data' dictionary using the highest protocol available.
# 					pickle.dump(scores_dict, f, pickle.HIGHEST_PROTOCOL)
#
# 				with open(self.output()[fold]["pickle"].path, 'w') as f:
# 					# Pickle the 'data' dictionary using the highest protocol available.
# 					for (feature, score) in zip(X_train.columns, feature_scores):
# 						f.write(f"{feature}, {score}\n")
# 				fold+=1
#
# 		if ((cv_type == 'groupkfold') or (cv_type == 'stratifiedgroupkfold')):
# 			G = df_filtered.loc[:,[group_label]]
# 			fold = 1
# 			for train_index, test_index in kf.split(X,Y,G):
# 				X_train, X_test = X.iloc[train_index], X.iloc[test_index]
# 				Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
#
# 				X_train = X_train[~np.isnan(Y_train)]
# 				Y_train = Y_train[~np.isnan(Y_train)].astype(bool)
#
# 				feature_scores = fs_function(X_train,Y_train)
#
# 				scores_dict = dict(zip(X_train.columns, feature_scores))
# 				with open(self.output()[fold]["pickle"].path, 'wb') as f:
# 					# Pickle the 'data' dictionary using the highest protocol available.
# 					pickle.dump(scores_dict, f, pickle.HIGHEST_PROTOCOL)
#
# 				with open(self.output()[fold]["pickle"].path, 'w') as f:
# 					# Pickle the 'data' dictionary using the highest protocol available.
# 					for (feature, score) in zip(X_train.columns, feature_scores):
# 						f.write(f"{feature}, {score}\n")
# 				fold+=1
#
# 		pass
# 	def output(self):
# 		try:
# 			os.makedirs(os.path.join(tmp_path,self.__class__.__name__,self.wf_name))
# 		except:
# 			pass
# 		for i in range(1,cvfolds+1):
# 			yield {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name, f"FeatureScores_{self.FS_name}_r{self.seed}_f{i}.pickle")),
# 					"txt": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name, f"FeatureScores_{self.FS_name}_r{self.seed}_f{i}.txt"))}
#

class InterpretationReport(luigi.Task):
	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	wf_name = luigi.Parameter()
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))
	ext_val = luigi.Parameter(default = 'No')

	best_MDA = luigi.Parameter(default = 'Yes')
	all_MDA = luigi.Parameter(default = 'Yes')
	best_shap = luigi.Parameter(default = 'No')
	all_shap = luigi.Parameter(default = 'No')

	def requires(self):
		requirements = {}
		for i in self.list_ML:
			if (self.best_MDA == 'Yes') or (self.best_shap=='Yes'):
				requirements[i] = Evaluate_ML(clf_name = i, wf_name = self.wf_name, ext_val=self.ext_val)
			if self.all_MDA == 'Yes':
				requirements[i+'_mda'] = MDAFeatureImportances(clf_name = i, wf_name = self.wf_name, ext_val=self.ext_val)
			if self.all_shap == 'Yes':
				requirements[i+'_shap'] = ShapleyValues(clf_name = i, wf_name = self.wf_name, ext_val=self.ext_val)
		return requirements


	def run(self):
		setupLog(self.__class__.__name__)

		if (self.best_MDA == 'Yes') or (self.best_shap == 'Yes'):
			auc_ml = {}
			for i in self.list_ML:
				with open(self.input()[i]["auc_results"].path, 'rb') as f:
					results_dict=pickle.load(f)
					auc_ml[i]=results_dict["avg_aucroc"]
			best_ml = max(auc_ml.keys(), key=(lambda k: auc_ml[k]))
			if self.best_MDA == 'Yes':
				prerequisite = MDAFeatureImportances(clf_name = best_ml, wf_name = self.wf_name, ext_val = self.ext_val)
				luigi.build([prerequisite], local_scheduler = False)
				shutil.copy(prerequisite.output()['txt'].path, self.output()['best_mda'].path)
			if self.best_shap == 'Yes':
				prerequisite = ShapleyValues(clf_name = best_ml, wf_name = self.wf_name, ext_val = self.ext_val)
				luigi.build([prerequisite], local_scheduler = False)
				shutil.copy(prerequisite.output()["png"].path, self.output()['best_shap'].path)
		if self.all_MDA == 'Yes':
			for i in self.list_ML:
				shutil.copy(self.input()[i+'_mda']['txt'].path, self.output()[i+'_mda'].path)
		if self.all_shap == 'Yes':
			for i in self.list_ML:
				shutil.copy(self.input()[i+'_shap']["png"].path, self.output()[i+'_shap'].path)

	def output(self):
		outputs = {}
		try:
			os.makedirs(os.path.join(report_path+f'-{self.datestring}', self.wf_name, "Interpretation"))
		except:
			pass
		if self.best_MDA == 'Yes':
			outputs['best_mda'] = luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}', self.wf_name, f"MDA_Importances_BestModel{'_EXT' if self.ext_val == 'Yes' else ''}.txt"))
		if self.best_shap == 'Yes':
			outputs['best_shap'] = luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}', self.wf_name, f"Shap_Values_BestModel{'_EXT' if self.ext_val == 'Yes' else ''}.png"))
		for i in self.list_ML:
			if self.all_MDA == 'Yes':
				outputs[i+'_mda'] = luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}', self.wf_name, "Interpretation", f"MDA_Importances_{i}{'_EXT' if self.ext_val == 'Yes' else ''}.txt"))
			if self.all_shap == 'Yes':
				outputs[i+'_shap'] = luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}', self.wf_name, "Interpretation", f"Shap_Values_{i}{'_EXT' if self.ext_val == 'Yes' else ''}.png"))
		return outputs


class AllInterpretationReports(luigi.Task):
	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	list_WF = luigi.ListParameter(default=list(WF_info.keys()))
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	best_MDA = luigi.Parameter(default = 'Yes')
	all_MDA = luigi.Parameter(default = 'Yes')
	best_shap = luigi.Parameter(default = 'No')
	all_shap = luigi.Parameter(default = 'No')

	def requires(self):
		for it_wf_name in self.list_WF:
			yield InterpretationReport(wf_name = it_wf_name, list_ML=self.list_ML, datestring=self.datestring,
										best_MDA = self.best_MDA, best_shap = self.best_shap, all_MDA = self.all_MDA, all_shap = self.all_shap)
			if(WF_info[it_wf_name]['external_validation'] == 'Yes'):
				yield InterpretationReport(wf_name = it_wf_name, list_ML=self.list_ML, ext_val = 'Yes', datestring=self.datestring,
											best_MDA = self.best_MDA, best_shap = self.best_shap, all_MDA = self.all_MDA, all_shap = self.all_shap)

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write("prueba\n")

	def output(self):
		return luigi.LocalTarget(os.path.join(log_path, f"AllInterpretationReports_Log-{self.datestring}.txt"))

class AllPerformanceReports(luigi.Task):

	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	list_WF = luigi.ListParameter(default=list(WF_info.keys()))
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	def requires(self):
		for it_wf_name in self.list_WF:
			if(len(self.list_ML) > 0):
				yield BestMLModelReport(wf_name = it_wf_name, list_ML=self.list_ML, datestring=self.datestring)
			if(len(self.list_RS) > 0):
				yield BestRSReport(wf_name = it_wf_name, list_RS=self.list_RS, datestring=self.datestring)
			if(WF_info[it_wf_name]['external_validation'] == 'Yes'):
				if(len(self.list_ML) > 0):
					yield BestMLModelReport(wf_name = it_wf_name, list_ML=self.list_ML, datestring=self.datestring, ext_val = 'Yes')
				if(len(self.list_RS) > 0):
					yield BestRSReport(wf_name = it_wf_name, list_RS=self.list_RS, datestring=self.datestring, ext_val = 'Yes')

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write("prueba\n")

	def output(self):
		return luigi.LocalTarget(os.path.join(log_path, f"AllPerformanceReports_Log-{self.datestring}.txt"))

class AllModels(luigi.Task):

	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	list_WF = luigi.ListParameter(default=list(WF_info.keys()))
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	def requires(self):
		for it_wf_name in self.list_WF:
			for it_clf_name in self.list_ML:
				yield FinalModelAndHyperparameterResults(wf_name = it_wf_name, clf_name = it_clf_name)
			for it_rs_name in self.list_RS:
				if(RS_info[it_rs_name]['refit_oddratios']=='Yes'):
					yield FinalRefittedRSAndOddratios(wf_name = it_wf_name, score_name = it_rs_name)

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write("prueba\n")

	def output(self):
		return luigi.LocalTarget(os.path.join(log_path, f"AllModels_Log-{self.datestring}.txt"))

class AllHistograms(luigi.Task):
	list_WF = luigi.ListParameter(default=list(WF_info.keys()))
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	def requires(self):
		for it_wf_name in self.list_WF:
			yield HistogramsPDF(wf_name = it_wf_name)
			if(WF_info[it_wf_name]['external_validation'] == 'Yes'):
				yield HistogramsPDF(wf_name = it_wf_name, ext_val = 'Yes')

	def run(self):
		setupLog(self.__class__.__name__)
		for input, output in zip(self.input(), self.output()):
			shutil.copy(input.path, output.path)

	def output(self):
		try:
			os.makedirs(os.path.join(report_path+f"-{self.datestring}",self.wf_name))
		except:
			pass
		for it_wf_name in self.list_WF:
			yield luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}', it_wf_name, f"{it_wf_name}_histogramas.pdf"))
			if(WF_info[it_wf_name]['external_validation'] == 'Yes'):
				yield luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}', it_wf_name, f"{it_wf_name}_histogramas_EXT.pdf"))

class AllDescriptiveReports(luigi.Task):
	list_WF = luigi.ListParameter(default=list(WF_info.keys()))
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	def requires(self):
		for it_wf_name in self.list_WF:
			yield DescriptiveXLS(wf_name = it_wf_name)
			if(WF_info[it_wf_name]['external_validation'] == 'Yes'):
				yield DescriptiveXLS(wf_name = it_wf_name, ext_val = 'Yes')

	def run(self):
		setupLog(self.__class__.__name__)
		for input, output in zip(self.input(), self.output()):
			shutil.copy(input.path, output.path)

	def output(self):
		try:
			os.makedirs(os.path.join(report_path+f"-{self.datestring}",self.wf_name))
		except:
			pass
		for it_wf_name in self.list_WF:
			yield luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}', it_wf_name, f"{it_wf_name}_descriptivo.xlsx"))
			if(WF_info[it_wf_name]['external_validation'] == 'Yes'):
				yield luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}', it_wf_name, f"{it_wf_name}_descriptivo_EXT.xlsx"))

class AllGraphs(luigi.Task):
	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	list_WF = luigi.ListParameter(default=list(WF_info.keys()))
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	def requires(self):
		for it_wf_name in self.list_WF:
			yield GraphsWF(wf_name = it_wf_name, list_ML=self.list_ML, list_RS=self.list_RS, datestring=self.datestring)
			if(WF_info[it_wf_name]['external_validation'] == 'Yes'):
				yield GraphsWF(wf_name = it_wf_name, list_ML=self.list_ML, list_RS=self.list_RS, ext_val = 'Yes', datestring=self.datestring)

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write("prueba\n")

	def output(self):
		return luigi.LocalTarget(os.path.join(log_path, f"AllGraphs_Log-{self.datestring}.txt"))

class AllTasks(luigi.Task):

	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	list_WF = luigi.ListParameter(default=list(WF_info.keys()))
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))
	best_MDA = luigi.Parameter(default = 'Yes')
	all_MDA = luigi.Parameter(default = 'Yes')
	best_shap = luigi.Parameter(default = 'No')
	all_shap = luigi.Parameter(default = 'No')


	def __init__(self, *args, **kwargs):
		super(AllTasks, self).__init__(*args, **kwargs)

	def requires(self):

		return [AllGraphs(list_ML = self.list_ML, list_RS = self.list_RS, list_WF = self.list_WF, datestring=self.datestring),
				AllDescriptiveReports(list_WF = self.list_WF, datestring=self.datestring),
				AllHistograms(list_WF = self.list_WF, datestring=self.datestring),
				AllModels(list_ML = self.list_ML, list_RS = self.list_RS, list_WF = self.list_WF, datestring=self.datestring),
				AllPerformanceReports(list_ML = self.list_ML, list_RS = self.list_RS, list_WF = self.list_WF, datestring=self.datestring),
				AllInterpretationReports(list_ML = self.list_ML, list_WF = self.list_WF, datestring=self.datestring,
										best_MDA = self.best_MDA, best_shap = self.best_shap, all_MDA = self.all_MDA, all_shap = self.all_shap)]

		# for it_wf_name in self.list_WF:
		# 	yield DescriptiveReport(wf_name = it_wf_name, datestring = self.datestring)
		# 	yield HistogramsReport(wf_name = it_wf_name, datestring = self.datestring)
		# 	yield GraphsWF(wf_name = it_wf_name, list_ML=self.list_ML, list_RS=self.list_RS, datestring=self.datestring)
		# 	if(len(self.list_ML) > 0):
		# 		yield BestMLModelReport(wf_name = it_wf_name, list_ML=self.list_ML, datestring=self.datestring)
		# 	if(len(self.list_RS) > 0):
		# 		yield BestRSReport(wf_name = it_wf_name, list_RS=self.list_RS, datestring=self.datestring)
		# 	yield AllModels_PairedTTest(wf_name = it_wf_name, list_ML=self.list_ML, list_RS=self.list_RS, datestring=self.datestring)
		# 	for it_clf_name in self.list_ML:
		# 		yield FinalModelAndHyperparameterResults(wf_name = it_wf_name, clf_name = it_clf_name)
		# 	for it_rs_name in self.list_RS:
		# 		if(RS_info[it_rs_name]['refit_oddratios']=='Yes'):
		# 			yield FinalRefittedRSAndOddratios(wf_name = it_wf_name, score_name = it_rs_name)
		# 	if(WF_info[it_wf_name]['external_validation'] == 'Yes'):
		# 		yield DescriptiveReport(wf_name = it_wf_name, ext_val = 'Yes', datestring=self.datestring)
		# 		yield HistogramsReport(wf_name = it_wf_name, ext_val = 'Yes', datestring=self.datestring)
		# 		yield GraphsWF(wf_name = it_wf_name, list_ML=self.list_ML, list_RS=self.list_RS, ext_val = 'Yes', datestring=self.datestring)
		# 		if(len(self.list_ML) > 0):
		# 			yield BestMLModelReport(wf_name = it_wf_name, list_ML=self.list_ML, ext_val = 'Yes', datestring=self.datestring)
		# 		if(len(self.list_RS) > 0):
		# 			yield BestRSReport(wf_name = it_wf_name, list_RS=self.list_RS, ext_val = 'Yes', datestring=self.datestring)
		# 		yield AllModels_PairedTTest(wf_name = it_wf_name, list_ML=self.list_ML, list_RS=self.list_RS,ext_val = 'Yes', datestring=self.datestring)

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write("prueba\n")

	def output(self):
		return luigi.LocalTarget(os.path.join(log_path, f"AllTask_Log-{self.datestring}.txt"))
