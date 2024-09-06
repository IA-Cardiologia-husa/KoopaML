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
from utils.metrics import metrics_list
from user_data_utils import *
from user_MLmodels_info import ML_info
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

		if WF_info[self.wf_name]["type"] == 'classification':
			try:
				Y_pred = clf.predict_proba(X)[:,1]
			except:
				Y_pred = clf.decision_function(X)
			X['True Label'] = external_data.loc[:,label]
		elif WF_info[self.wf_name]["type"] == 'regression':
			Y_pred = clf.predict(X)
			X['True Label'] = external_data.loc[:,label]
		elif WF_info[self.wf_name]["type"] == 'survival':
			time = WF_info[self.wf_name]["label_time"]
			Y_pred = clf.predict(X)
			X['True Label'] = external_data.loc[:,label]
			X['Time'] = external_data.loc[:,time]

		X['Prediction'] = Y_pred
		X.to_excel(self.output()["xls"].path)
		X.to_pickle(self.output()["pickle"].path)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"ExternalValidation_PredProb_{self.wf_name}_{self.clf_name}.xlsx")),
				"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"ExternalValidation_PredProb_{self.wf_name}_{self.clf_name}.pickle"))}


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
		if WF_info[self.wf_name]['type'] == 'survival':
			time = WF_info[self.wf_name]["label_time"]


		data = pd.read_pickle(self.input()[f'pickle'].path)
		for i in range(folds):
			data_train, data_test = data.loc[data[f'Repetition_{self.seed}_folds'] != i], data.loc[data[f'Repetition_{self.seed}_folds'] == i]
			X_train, X_test = data_train.loc[:,features], data_test.loc[:,features]
			Y_train, Y_test = data_train.loc[:,label], data_test.loc[:,label]
			if WF_info[self.wf_name]['type'] == 'survival':
				T_train, T_test = data_train.loc[:,time], data_test.loc[:,time]

			if ((cv_type == 'groupkfold') or (cv_type=='stratifiedgroupkfold')):
				G_train, G_test = data_train.loc[:,group_label], data_test.loc[: ,group_label]

			if WF_info[self.wf_name]['type'] == 'classification':
				try:
					clf.fit(X_train, Y_train.astype("Int64"), groups=G_train)
				except:
					clf.fit(X_train, Y_train.astype("Int64"))
			elif WF_info[self.wf_name]['type'] == 'regression':
				clf.fit(X_train, Y_train)
			elif WF_info[self.wf_name]['type'] == 'survival':
				Y_train = Y_train.astype(bool)
				R_train = pd.DataFrame({'label':Y_train, 'time':T_train}).to_records(index=False)
				clf.fit(X_train, R_train)

			if WF_info[self.wf_name]['type'] == 'classification':
				try:
					Y_pred_test = clf.predict_proba(X_test)[:,1]
					Y_pred_train = clf.predict_proba(X_train)[:,1]
				except:
					Y_pred_test = clf.decision_function(X_test)
					Y_pred_train = clf.decision_function(X_train)
			else:
				Y_pred_test = clf.predict(X_test)
				Y_pred_train = clf.predict(X_train)

			if ((cv_type == 'groupkfold') or (cv_type=='stratifiedgroupkfold')):
				data_train[f'Group_label: {group_label}'] = G_train
				data_test[f'Group_label: {group_label}'] = G_test
			data_train['Repetition'] = self.seed
			data_test['Repetition'] = self.seed
			data_train['Fold'] = i
			data_test['Fold'] = i
			data_train['True Label'] = Y_train
			data_test['True Label'] = Y_test
			if WF_info[self.wf_name]['type'] == 'survival':
				data_train['Time'] = T_train
				data_test['Time'] = T_test
			data_train['Prediction'] = Y_pred_train
			data_test['Prediction'] = Y_pred_test

			saved_columns = ['CustomIndex', 'Repetition', 'Fold', 'True Label','Prediction']
			if WF_info[self.wf_name]['type'] == 'survival':
				saved_columns.insert(4, 'Time')
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

class InterpretationShapFairness(luigi.Task):
	wf_name = luigi.Parameter()
	clf_name = luigi.Parameter()
	ext_val = luigi.Parameter(default='No')

	def requires(self):
		requirements = {}
		requirements['shap'] = FinalModelTrainResults(wf_name = self.wf_name, clf_name = self.clf_name)
		if(self.ext_val == 'Yes'):
			requirements['data'] = ProcessExternalDatabase(wf_name = self.wf_name)
		else:
			requirements['data'] = ProcessDatabase(wf_name = self.wf_name)
		return requirements

	def run(self):
		setupLog(self.__class__.__name__)

		df = pd.read_pickle(self.input()["data"]["pickle"].path)
		filter_function = WF_info[self.wf_name]["filter_function"]
		features = WF_info[self.wf_name]["feature_list"]
		# label = WF_info[self.wf_name]["label_name"]
		# group_label = WF_info[self.wf_name]["group_label"]
		fairness_label = WF_info[self.wf_name]["fairness_label"]
		# if WF_info[self.wf_name]['type'] == 'survival':
		# 	time = WF_info[self.wf_name]["label_time"]

		df = filter_function(df).reset_index(drop=True)
		X = df.loc[:,features]

		with open(self.input()["shap"]["shapvalues"].path, 'rb') as f:
			shap_values = pickle.load(f)

		cats = X[fairness_label].unique()
		plt.figure(figsize=(5*len(cats),5))
		for i in range(len(cats)):
			c = cats[i]
			X_cat = X.loc[X[fairness_label]==c]
			shap_cat = shap_values[X[fairness_label]==c]
			shap_cat = shap_cat - shap_cat.mean(axis = 0)[np.newaxis,...]
			ax = plt.subplot(1,len(cats), i+1)
			shap.summary_plot(shap_cat, X_cat, max_display = 20, show=False)
			ax.set_title(f"{fairness_label} = {c}")
		plt.savefig(self.output().path, bbox_inches='tight', dpi=300)
		plt.close()

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path, self.__class__.__name__,self.wf_name))
		except:
			pass
		return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"{'EXT_' if self.ext_val == 'Yes' else ''}Evaluation_Fairness_{self.clf_name}.png"))

class DescriptiveFairness(luigi.Task):
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
		label = WF_info[self.wf_name]["fairness_label"]

		df_output=create_descriptive_comparation(df_filtered, self.wf_name, label)

		writer = pd.ExcelWriter(self.output().path, engine='xlsxwriter')
		df_output.to_excel(writer, sheet_name='Sheet1')
		writer.close()

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path, self.__class__.__name__))
		except:
			pass

		if(self.ext_val == 'No'):
			return luigi.LocalTarget(os.path.join(tmp_path, self.__class__.__name__, f"{self.wf_name}_descriptivo_fairness_label.xlsx"))
		elif(self.ext_val == 'Yes'):
			return luigi.LocalTarget(os.path.join(tmp_path, self.__class__.__name__, f"{self.wf_name}_descriptivo_fairness_label_EXT.xlsx"))

class Evaluate_Fairness(luigi.Task):
	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()
	ext_val = luigi.Parameter(default='No')

	def requires(self):
		return {"Evaluate_ML": Evaluate_ML(clf_name = self.clf_name, wf_name = self.wf_name, ext_val = self.ext_val),
				"CreateFolds": CreateFolds(wf_name = self.wf_name)}

	def run(self):
		setupLog(self.__class__.__name__)
		fairness_label =  WF_info[self.wf_name]['fairness_label']

		df = pd.read_pickle(self.input()[f"Evaluate_ML"]["pickle"].path)
		df_folds = pd.read_pickle(self.input()[f"CreateFolds"]["pickle"].path)
		df = df.merge(df_folds, on='CustomIndex')

		n_reps = df['Repetition'].max()+1
		n_folds = df['Fold'].max()+1
		n_repfolds = n_reps*n_folds
		critical_pvalue=0.05
		results_dict = {}

		dict_df = {}
		for category in df[fairness_label].unique():
			dict_df[category] = {}
			for metric in WF_info[self.wf_name]['metrics']:
				m = metrics_list[metric]()
				score = []
				for rep in range(n_reps):
					for fold in range(n_folds):
						df_aux = df.loc[(df['Repetition']==rep)&(df['Fold']==fold)&(df[fairness_label]==category)]
						df_aux2 = df.loc[(df['Repetition']==rep)&(df['Fold']!=fold)&(df[fairness_label]==category)]
						if len(df_aux) == 0:
							score.append(np.nan)
						else:
							try:
								score.append(m(df_aux, df_aux2))
							except:
								score.append(np.nan)
				score = np.array(score)

				dict_df[category][f'avg_{m.name}'] = score.mean(where=~np.isnan(score))
				if(n_folds>1):
					stderr = score.std(ddof = 1)*np.sqrt(1/n_repfolds+1/(n_folds-1))
					dict_df[category][f'avg_{m.name}_stderr'] = stderr
					c = sc_st.t.ppf(1-critical_pvalue/2, df= n_repfolds-1)
					dict_df[category][f'{m.name}_95ci_low'] = score.mean() - c*stderr
					dict_df[category][f'{m.name}_95ci_high'] = score.mean() + c*stderr
					dict_df[category][f'pool_{m.name}'] = m(df.loc[df[fairness_label]==category], df.loc[df[fairness_label]==category])
				else:
					if m.variance is not None:
						c = sc_st.t.ppf(1-critical_pvalue/2, df= n_repfolds-1)
						dict_df[category][f'avg_{m.name}_stderr'] = np.sqrt(m.variance)
						dict_df[category][f'{m.name}_95ci_low'] = score.mean() - c*np.sqrt(m.variance)
						dict_df[category][f'{m.name}_95ci_high'] = score.mean() + c*np.sqrt(m.variance)
					else:
						dict_df[category][f'avg_{m.name}_stderr'] = np.nan
						dict_df[category][f'{m.name}_95ci_low'] = np.nan
						dict_df[category][f'{m.name}_95ci_high'] = np.nan
					dict_df[category][f'pool_{m.name}'] = m(df.loc[df[fairness_label]==category], df.loc[df[fairness_label]==category])
		df_xls = pd.DataFrame().from_dict(dict_df, orient = 'index')
		df_xls = df_xls.reset_index()
		df_xls = df_xls.rename({'index':fairness_label}, axis = 'columns')
		df_xls.to_excel(self.output()["xls"].path, index=False)
		df_xls.to_pickle(self.output()["pickle"].path)

	def output(self):
		if(self.ext_val == 'Yes'):
			prefix = 'EXT_'
		else:
			prefix = ''
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__,self.wf_name))
		except:
			pass
		return {"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"Evaluation_Fairness_{prefix}{self.clf_name}.xlsx")),
				"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"Evaluation_Fairness_{prefix}{self.clf_name}.pickle"))}


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

		# if (WF_info[self.wf_name]['type']=='classification'):
		# 	metrics = ['aucroc', 'aucpr']
		# elif (WF_info[self.wf_name]['type']=='regression'):
		# 	metrics = ['rmse']
		# elif (WF_info[self.wf_name]['type']=='survival'):
		# 	metrics = ['cindex_censored', 'cumulative_auc']

		n_reps = df['Repetition'].max()+1
		n_folds = df['Fold'].max()+1
		n_repfolds = n_reps*n_folds
		critical_pvalue=0.05
		results_dict = {}

		for metric in WF_info[self.wf_name]['metrics']:
			m = metrics_list[metric]()
			score = []
			for rep in range(n_reps):
				for fold in range(n_folds):
					df_aux = df.loc[(df['Repetition']==rep)&(df['Fold']==fold)]
					df_aux2 = df.loc[(df['Repetition']==rep)&(df['Fold']!=fold)]
					score.append(m(df_aux, df_aux2))
			score = np.array(score)
			results_dict[f'avg_{m.name}'] = score.mean()
			if(n_folds>1):
				stderr = score.std(ddof = 1)*np.sqrt(1/n_repfolds+1/(n_folds-1))
				results_dict[f'avg_{m.name}_stderr'] = stderr
				c = sc_st.t.ppf(1-critical_pvalue/2, df= n_repfolds-1)
				results_dict[f'{m.name}_95ci_low'] = score.mean() - c*stderr
				results_dict[f'{m.name}_95ci_high'] = score.mean() + c*stderr
				# Esto del bootstrap no tiene mucho sentido y parece que da resultados peores según añades folds
				# bootstrap_scores = [np.random.choice(score[rep*n_folds:(rep+1)*n_folds], n_folds, replace = True).mean() for rep in range(n_reps) for b in range(200)]
				# results_dict[f'{m.name}_95ci_low'] = np.quantile(bootstrap_scores, critical_pvalue/2)
				# results_dict[f'{m.name}_95ci_high'] = np.quantile(bootstrap_scores, 1-critical_pvalue/2)
				results_dict[f'pool_{m.name}'] = m(df, df)
			else:
				if m.variance is not None:
					c = sc_st.t.ppf(1-critical_pvalue/2, df= n_repfolds-1)
					results_dict[f'avg_{m.name}_stderr'] = np.sqrt(m.variance)
					results_dict[f'{m.name}_95ci_low'] = score.mean() - c*np.sqrt(m.variance)
					results_dict[f'{m.name}_95ci_high'] = score.mean() + c*np.sqrt(m.variance)
				else:
					results_dict[f'avg_{m.name}_stderr'] = np.nan
					results_dict[f'{m.name}_95ci_low'] = np.nan
					results_dict[f'{m.name}_95ci_high'] = np.nan
				results_dict[f'pool_{m.name}'] = m(df, df)

		df.to_excel(self.output()["xls"].path)
		df.to_pickle(self.output()["pickle"].path)

		with open(self.output()[f"results"].path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)

		with open(self.output()["results_txt"].path, 'w') as f:
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
				"results": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name,f"results_{prefix}{self.clf_name}.pickle")),
				"results_txt": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name,f"results_{prefix}{self.clf_name}.txt"))}


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

		if (WF_info[self.wf_name]['type'] == 'classification') or (WF_info[self.wf_name]['type'] == 'survival'):
			df_output=create_descriptive_comparation(df_filtered, self.wf_name, label)
		elif WF_info[self.wf_name]['type'] == 'regression':
			df_output=create_descriptive_correlation(df_filtered, self.wf_name, label)
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
	label_name = luigi.Parameter()

	def requires(self):
		if(self.ext_val == 'Yes'):
			return ProcessExternalDatabase(wf_name = self.wf_name)
		else:
			return ProcessDatabase(wf_name = self.wf_name)

	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		if self.ext_val == 'No':
			df = WF_info[self.wf_name]["filter_function"](df_input)
		else:
			df = WF_info[self.wf_name]["filter_external_validation"](df_input)
		label = WF_info[self.wf_name]["label_name"]
		features = WF_info[self.wf_name]["feature_list"]

		file_path = os.path.join(tmp_path, self.__class__.__name__, f"histograma_temporal_{self.wf_name}.pdf")
		pp = PdfPages(file_path)
		for f in features:
			fig, ax= plt.subplots(3,1,figsize=(10,15), height_ratios = [1,3,1])
			f_min = df.loc[df[f].notnull(), f].min()
			f_max = df.loc[df[f].notnull(), f].max()
			f_std = df.loc[df[f].notnull(), f].std()
			if f_min != f_max:
				ax[0].hist(df.loc[df[f].notnull(),f], bins = np.arange(f_min -f_std/8., f_max+f_std/8., f_std/4.))
			else:
				ax[0].hist(df.loc[df[f].notnull(),f])
			ax[0].set_title("Histogram")
			if len(df[self.label_name].unique()) > 5:
				ax[1].scatter(df.loc[df[f].notnull(), self.label_name], df.loc[df[f].notnull(),f])
				ax[1].set_xlabel(self.label_name)
				ax[1].set_ylabel(f)

				quantiles = [df[self.label_name].min(),
							df[self.label_name].quantile(0.25),
							df[self.label_name].quantile(0.5),
							df[self.label_name].quantile(0.75),
							df[self.label_name].max()]
				labels = []
				intervals = []
				for l in range(len(quantiles)-1):
					l_min = quantiles[l]
					l_max = quantiles[l+1]
					if l != len(quantiles)-2:
						interval = (df[self.label_name]>=l_min) & (df[self.label_name]<l_max)
						label_plot = f"{self.label_name} [{l_min:.3g}-{l_max:.3g})"
					else:
						interval = (df[self.label_name]>=l_min) & (df[self.label_name]<=l_max)
						label_plot = f"{self.label_name} [{l_min:.3g}-{l_max:.3g}]"
					labels.append(label_plot)
					intervals.append(interval)
				ax[2].barh([label_plot for label_plot in labels],
							[(df[f].notnull() & interval).sum() for interval in intervals],
							label = 'Filled')
				ax[2].barh([label_plot for label_plot in labels],
							[(df[f].isnull() & interval).sum() for interval in intervals],
							left = [(df[f].notnull() & interval).sum() for interval in intervals],
							label = 'Missing')
				ax[2].legend()

			else:
				if len(df[f].unique()) <=5:
					table = []
					for v in df[f].unique():
						table.append([((df[f]==v) & (df[self.label_name]==l)).sum() for l in df[self.label_name].unique()])
					ax[1].table(cellText=table,
								colLabels = [f"{self.label_name}={l}" for l in df[self.label_name].unique()],
								rowLabels = [f"{f}={v}" for v in df[f].unique()],
								rowLoc = 'right',
								bbox = [0.3,0.2,0.6,0.6])
				else:
					ax[1].boxplot([df.loc[df[f].notnull() & (df[self.label_name]==l), f] for l in df[self.label_name].unique()])
					ax[1].set_xticklabels([f"{self.label_name}={l}" for l in df[self.label_name].unique()], rotation=10, ha='right')
					ax[1].set_ylabel(f)

				ax[2].barh([f"{self.label_name}={l}" for l in df[self.label_name].unique()],
							[(df[f].notnull() & (df[self.label_name]==l)).sum() for l in df[self.label_name].unique()],
							label = 'Filled')
				ax[2].barh([f"{self.label_name}={l}" for l in df[self.label_name].unique()],
							[(df[f].isnull() & (df[self.label_name]==l)).sum() for l in df[self.label_name].unique()],
							left = [(df[f].notnull() & (df[self.label_name]==l)).sum() for l in df[self.label_name].unique()],
							label = 'Missing')

				ax[2].legend()

			# fig, ax= plt.subplots(figsize=(10,10))
			# f_min = df_filtered.loc[df_filtered[f].notnull(), f].min()
			# f_max = df_filtered.loc[df_filtered[f].notnull(), f].max()
			# f_std = df_filtered.loc[df_filtered[f].notnull(), f].std()
			# if (f_std != 0) & (np.isnan(f_std)==False):
			# 	ax.hist(df_filtered.loc[df_filtered[f].notnull()&(df_filtered[label]==0), f],
			# 			bins = np.arange(f_min, f_max + f_std/4., f_std/4.),
			# 			label = f"{label}=0")
			# 	ax.hist(df_filtered.loc[df_filtered[f].notnull()&(df_filtered[label]==1), f],
			# 			bins = np.arange(f_min, f_max + f_std/4., f_std/4.),
			# 			label = f"{label}=1", alpha = 0.5)
			# 	ax.set_title(f)
			# 	ax.legend()
			# else:
			# 	ax.hist(df_filtered.loc[df_filtered[f].notnull()&(df_filtered[label]==0), f],
			# 			label = f"{label}=0")
			# 	ax.hist(df_filtered.loc[df_filtered[f].notnull()&(df_filtered[label]==1), f],
			# 			label = f"{label}=1", alpha = 0.5)
			# 	ax.set_title(f)
			# 	ax.legend()
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

		clf=ML_info[self.clf_name]["clf"]

		X_full = df_filtered.loc[:,features]
		Y_full = df_filtered.loc[:,label]

		X = X_full.loc[~Y_full.isnull()]
		Y = Y_full.loc[~Y_full.isnull()]

		if(group_label is not None):
			G_full = df_filtered.loc[:,[group_label]]
			G = G_full.loc[~Y_full.isnull()]

		if WF_info[self.wf_name]['type'] == 'classification':
			try:
				clf.fit(X, Y.astype("Int64"), groups=G)
			except:
				clf.fit(X, Y.astype("Int64"))
		elif WF_info[self.wf_name]['type'] == 'regression':
			try:
				clf.fit(X, Y, groups=G)
			except:
				clf.fit(X, Y)
		elif WF_info[self.wf_name]['type'] == 'survival':
			time = WF_info[self.wf_name]["label_time"]
			T_full = df_filtered.loc[:,time]
			T = T_full.loc[~Y_full.isnull()]
			R = pd.DataFrame({'label':Y, 'time':T}).to_records(index=False)
			try:
				clf.fit(X, R, groups=G)
			except:
				clf.fit(X, R)

		if hasattr(clf, 'best_estimator_'):
			writer = pd.ExcelWriter(os.path.join(model_path,self.wf_name,f"HyperparameterResults_{self.wf_name}_{self.clf_name}.xlsx"), engine='xlsxwriter')
			pd.DataFrame(clf.cv_results_).to_excel(writer, sheet_name='Sheet1')
			writer.close()
		elif hasattr(clf, '__getitem__'):
			if hasattr(clf[-1], 'best_estimator_'):
				writer = pd.ExcelWriter(os.path.join(model_path,self.wf_name,f"HyperparameterResults_{self.wf_name}_{self.clf_name}.xlsx"), engine='xlsxwriter')
				pd.DataFrame(clf[-1].cv_results_).to_excel(writer, sheet_name='Sheet1')
				writer.close()

		with open(self.output().path,'wb') as f:
			pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)



	def output(self):
		try:
			os.makedirs(os.path.join(model_path,self.wf_name, self.clf_name))
		except:
			pass
		return luigi.LocalTarget(os.path.join(model_path,self.wf_name,self.clf_name,f"ML_model_{self.wf_name}_{self.clf_name}.pickle"))


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
# 								pred_prob1 = df1.loc[(df1['Repetition']==rep)&(df1['Fold']==fold), 'Prediction'].values
# 								tl1 = true_label1[~np.isnan(true_label1)]
# 								pp1 = pred_prob1[~np.isnan(true_label1)]
# 								auc1 = sk_m.roc_auc_score(tl1,pp1)
#
# 								#True labels for the same workflow should be the same and there is no need to load the ones from the second
# 								pred_prob2 = df2.loc[(df2['Repetition']==rep)&(df1['Fold']==fold), 'Prediction'].values
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
	ext_val = luigi.Parameter(default='No')
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))
	metric = luigi.Parameter()

	def requires(self):
		requirements = {}
		for i in self.list_ML:
			requirements[i] = Evaluate_ML(clf_name = i, wf_name = self.wf_name, ext_val=self.ext_val)
		return requirements

	def run(self):
		setupLog(self.__class__.__name__)

		m = metrics_list[self.metric]()
		fig, ax = m.figure(len(self.input().keys()))
		fig.suptitle(WF_info[self.wf_name]["formal_title"], fontsize = 14)
		for score in self.input().keys():
			df = pd.read_pickle(self.input()[score]["pickle"].path)
			with open(self.input()[score]["results"].path, 'rb') as f:
				results_dict=pickle.load(f)
			m.plot(df, results_dict, ax, ML_info[score]["formal_name"])
		m.save_figure(fig, ax, self.output().path)


	def output(self):
		try:
			os.makedirs(os.path.join(report_path+f"-{self.datestring}",self.wf_name))
		except:
			pass
		if(self.ext_val == 'Yes'):
			prefix = 'EXT_'
		else:
			prefix = ''
		return luigi.LocalTarget(os.path.join(report_path+f"-{self.datestring}",self.wf_name, f"AllModels{self.metric}_{prefix}{self.wf_name}.png"))

class ThresholdPoints(luigi.Task):
	clf_or_score=luigi.Parameter()
	wf_name = luigi.Parameter()
	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	ext_val = luigi.Parameter(default = 'No')


	def requires(self):
		return Evaluate_ML(wf_name=self.wf_name, clf_name=self.clf_or_score, ext_val=self.ext_val)

	def run(self):
		setupLog(self.__class__.__name__)

		df = pd.read_pickle(self.input()["pickle"].path)
		true_label = df['True Label'].values
		pred_prob = df['Prediction'].values

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
			if WF_info[self.wf_name]['type'] == 'classification':
				requirements[i+'_threshold'] = ThresholdPoints(clf_or_score = i, wf_name = self.wf_name, list_ML = self.list_ML, ext_val=self.ext_val)
		return requirements

	def run(self):
		setupLog(self.__class__.__name__)

		rows =[]
		for model in self.list_ML:
			row = {'Model Name': model}
			with open(self.input()[model]["results"].path, 'rb') as f:
				results_dict=pickle.load(f)
			for metric in WF_info[self.wf_name]['metrics']:
				m = metrics_list[metric]()
				row[m.name+'_rawvalue']=m.optimization_sign*results_dict[f"avg_{m.name}"]
				row[m.name] = f'{results_dict[f"avg_{m.name}"]:1.4f} ({results_dict[f"{m.name}_95ci_low"]:1.4f}-{results_dict[f"{m.name}_95ci_high"]:1.4f})'
			rows.append(row)
		df = pd.DataFrame(rows)
		df = df.sort_values(metrics_list[WF_info[self.wf_name]['metrics'][0]]().name+'_rawvalue', ascending=False).reset_index(drop=True)
		best_ml = df.loc[0, 'Model Name']
		df.loc[:,['Model Name']+[metrics_list[metric]().name for metric in WF_info[self.wf_name]['metrics']]].to_excel(self.output()['xlsx'].path, index=False)


		# First we open the results dictionary for every ML model in the workflow wf_name to determine
		# the best ML model
		# metric_ml = {}
		# m = metrics_list[WF_info[self.wf_name]['metrics'][0]]()
		# m_name = m.name
		# m_sign = m.optimization_sign
		#
		# for i in self.list_ML:
		# 	with open(self.input()[i]["results"].path, 'rb') as f:
		# 		results_dict=pickle.load(f)
		# 		metric_ml[i]=m_sign*results_dict[f"avg_{m_name}"]
		#
		# best_ml = max(metric_ml.keys(), key=(lambda k: metric_ml[k]))

		with open(self.input()[best_ml]["results"].path, 'rb') as f:
			best_ml_results_dict=pickle.load(f)

		with open(self.output()["txt"].path,'w') as f:
			f.write(f"Model name: {best_ml}\n")
			for m in WF_info[self.wf_name]['metrics']:
				m_name = metrics_list[m]().name
				f.write(f"{m_name}: {best_ml_results_dict[f'avg_{m_name}']}\n")
				f.write(f"{m_name} stderr: {best_ml_results_dict[f'avg_{m_name}_stderr']}\n")
				f.write(f"{m_name} Confidence Interval (95%): {best_ml_results_dict[f'{m_name}_95ci_low']}-{best_ml_results_dict[f'{m_name}_95ci_high']}\n")
			f.write("\n")
			if WF_info[self.wf_name]['type'] == 'classification':
				with open(self.input()[best_ml+'_threshold'].path, 'r') as f2:
					for line in f2.readlines():
						f.write(line)

	def output(self):
		try:
			os.makedirs(os.path.join(report_path+f'-{self.datestring}',self.wf_name))
		except:
			pass
		if self.ext_val == 'No':
			return {"txt": luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}',self.wf_name,f"BestML_Model_report_{self.wf_name}.txt")),
					"xlsx": luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}',self.wf_name,f"Model_Summary_{self.wf_name}.xlsx"))}
		elif self.ext_val == 'Yes':
			return {"txt": luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}',self.wf_name,f"BestML_Model_report_{self.wf_name}_EXT.txt")),
					"xlsx": luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}',self.wf_name,f"Model_Summary_{self.wf_name}_EXT.xlsx"))}


# class AllThresholds(luigi.Task):
# 	clf_or_score=luigi.Parameter()
# 	wf_name = luigi.Parameter()
# 	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
# 	ext_val = luigi.Parameter(default='No')
#
# 	def requires(self):
# 		return Evaluate_ML(wf_name=self.wf_name, clf_name=self.clf_or_score, ext_val = self.ext_val)
#
#
# 	def run(self):
# 		setupLog(self.__class__.__name__)
# 		with open(self.input()["pred_prob"].path, 'rb') as f:
# 			pred_prob=pickle.load(f)
# 		with open(self.input()["true_label"].path, 'rb') as f:
# 			true_label=pickle.load(f)
#
# 		list_thresholds = all_thresholds(pred_prob, true_label)
#
# 		with open(self.output()['txt'].path,'w') as f:
# 			rows = []
# 			for i in list_thresholds:
# 				(threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = i
# 				f.write(f'Threshold: {threshold}\n')
# 				f.write(f'TP:{tprate*100:.1f} FP:{fprate*100:.1f} TN:{tnrate*100:.1f} FN:{fnrate*100:.1f}\n')
# 				f.write(f'Sensitivity:{sens*100:.1f} Specificity:{spec*100:.1f} Precision:{prec*100:.1f} NPRv:{nprv*100:.1f}\n')
# 				rows.append([threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv])
# 		df_thr = pd.DataFrame(rows, columns=['Threshold','TP','FP','TN','FN', 'sensitivity','specificity','precision','nprv'])
# 		with open(self.output()['df'].path,'w') as f:
# 			df_thr.to_csv(f)
# 	def output(self):
# 		try:
# 			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
# 		except:
# 			pass
# 		if self.ext_val == 'No':
# 			return {'txt': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}.txt")),
# 					'df': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}.csv"))}
# 		elif self.ext_val == 'Yes':
# 			return {'txt': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}_EXT.txt")),
# 					'df': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}_EXT.csv"))}

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

					# masker = shap.maskers.Independent(data =  df_train.loc[:,feature_list].astype(float).values)
					masker = shap.maskers.Independent(data =  df_train.loc[:,feature_list].astype(float))
					if WF_info[self.wf_name]['type'] == 'classification':
						if hasattr(model, 'predict_proba'):
							explainer = shap.PermutationExplainer(lambda x: model.predict_proba(x)[:,1], masker, link = shap.links.logit)
						else:
							explainer = shap.PermutationExplainer(lambda x: model.decision_function(x), masker, link = shap.links.identity)
					else:
						explainer = shap.PermutationExplainer(lambda x: model.predict(x), masker, link = shap.links.identity)
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

			# try:
			# 	if hasattr(model, '__getitem__'):
			# 		data = df_train.loc[:,feature_list]
			# 		for i in range(len(model)-1):
			# 			data = model[i].transform(data)
			# 		masker = shap.maskers.Independent(data = data)
			# 		explainer = shap.Explainer(model, masker)
			# 		data_test = df_test
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
			masker = shap.maskers.Independent(data =  df_train.loc[:,feature_list].astype(float).values)
			if WF_info[self.wf_name]['type'] == 'classification':
				if hasattr(model, 'predict_proba'):
					explainer = shap.PermutationExplainer(lambda x: model.predict_proba(x)[:,1], masker, link = shap.links.logit)
				else:
					explainer = shap.PermutationExplainer(lambda x: model.decision_function(x), masker, link = shap.links.identity)
			else:
				explainer = shap.PermutationExplainer(lambda x: model.predict(x), masker, link = shap.links.identity)
			shap_values = explainer(df_test.astype(float).values).values

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
	metrics = luigi.ListParameter()

	def requires(self):
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

		rows = []
		if self.ext_val == 'No':
			df_folded = pd.read_pickle(self.input()['CreateFolds']['pickle'].path)
			for rep in range(WF_info[self.wf_name]["cv_repetitions"]):
				for fold in range(WF_info[self.wf_name]["cv_folds"]):
					df_test = df_folded.loc[df_folded[f"Repetition_{rep}_folds"]==fold]
					df_train = df_folded.loc[df_folded[f"Repetition_{rep}_folds"]!=fold]
					df_results = pd.read_pickle(self.input()[f'CalculateKFold_{rep}'][f"Test_{fold}"].path)
					df_test.set_index("CustomIndex")
					df_results.set_index("CustomIndex")
					df_test["Prediction"] = df_results["Prediction"]
					label = WF_info[self.wf_name]["label_name"]
					if WF_info[self.wf_name]["type"] == 'survival':
						time = WF_info[self.wf_name]["label_time"]
					df_test["True Label"] = df_test[label]
					df_train["True Label"] = df_train[label]
					if WF_info[self.wf_name]["type"] == 'classification':
						df_test["True Label"] = df_test["True Label"].astype(int)
						df_train["True Label"] = df_train["True Label"].astype(int)
					if WF_info[self.wf_name]["type"] == 'survival':
						df_test["Time"] = df_test[time].astype(int)
						df_train["Time"] = df_train[time].astype(int)
					with open(self.input()[f'CalculateKFold_{rep}'][f"Model_{fold}"].path, "rb") as f:
						model = pickle.load(f)

					for metric in self.metrics:
						m = metrics_list[metric]()
						metric_original = m(df_test, df_train)

						for feat in feature_list:
							df_shuffled = df_test.copy()

							for i in range(self.n_iterations):
								df_shuffled[feat] = np.random.permutation(df_test[feat].values)

								if WF_info[self.wf_name]["type"] == 'classification':
									try:
										df_shuffled['Prediction'] = model.predict_proba(df_shuffled.loc[:, feature_list])[:,1]
									except:
										df_shuffled['Prediction'] = model.decision_function(df_shuffled.loc[:, feature_list])
								elif WF_info[self.wf_name]["type"] == 'regression':
									df_shuffled['Prediction'] = model.predict(df_shuffled.loc[:, feature_list])
								elif WF_info[self.wf_name]["type"] == 'survival':
									df_shuffled['Prediction'] = model.predict(df_shuffled.loc[:, feature_list])

								metric_shuffled = m(df_shuffled, df_train)

								row = {'feat':feat,
									   'rep':rep,
									   'fold':fold,
									   'iter':i,
									   'metric_name': m.name,
									   'metric_original':metric_original,
									   'metric_shuffled':metric_shuffled,
									   'mda':metric_original-metric_shuffled}
								rows.append(row)

		elif self.ext_val == 'Yes':

			label = WF_info[self.wf_name]["label_name"]
			time = WF_info[self.wf_name]["label_time"]
			filter_function = WF_info[self.wf_name]["filter_function"]
			filter_function_external = WF_info[self.wf_name]["filter_external_validation"]

			df_input_train = pd.read_pickle(self.input()["train_data"]["pickle"].path)
			df_train = filter_function(df_train)

			df_input_test = pd.read_pickle(self.input()["test_data"]["pickle"].path)
			df_test = filter_function_external(df_input_test)

			df_train['True Label'] = df_train[label].values
			df_test['True Label'] = df_test[label].values
			if WF_info[self.wf_name]["type"] == 'classification':
				df_test["True Label"] = df_test["True Label"].astype(int)
				df_train["True Label"] = df_train["True Label"].astype(int)
			if WF_info[self.wf_name]["type"] == 'survival':
				df_train['Time'] = df_train[time].values
				df_test['Time'] = df_test[time].values

			with open(self.input()["model"].path, "rb") as f:
				model = pickle.load(f)

			if WF_info[self.wf_name]["type"] == 'classification':
				try:
					df_test['Prediction'] = model.predict_proba(df_test.loc[:, feature_list])[:,1]
				except:
					df_test['Prediction'] = model.decision_function(df_test.loc[:, feature_list])
			elif WF_info[self.wf_name]["type"] == 'regression':
				df_test['Prediction'] = model.predict(df_test.loc[:, feature_list])
			elif WF_info[self.wf_name]["type"] == 'survival':
				df_test['Prediction'] = model.predict(df_test.loc[:, feature_list])

			for metric in self.metrics:
				m = metrics_list[metric]()
				metric_original = m(df_test, df_train)

				for feat in feature_list:
					df_shuffled = df_test.copy()

					for i in range(self.n_iterations):
						df_shuffled[feat] = np.random.permutation(df_test[feat].values)

						if WF_info[self.wf_name]["type"] == 'classification':
							try:
								df_shuffled['Prediction'] = clf.predict_proba(df_shuffled.loc[:, feature_list])[:,1]
							except:
								df_shuffled['Prediction'] = clf.decision_function(df_shuffled.loc[:, feature_list])
						elif WF_info[self.wf_name]["type"] == 'regression':
							df_shuffled['Prediction'] = clf.predict(df_shuffled.loc[:, feature_list])
						elif WF_info[self.wf_name]["type"] == 'survival':
							df_shuffled['Prediction'] = clf.predict(df_shuffled.loc[:, feature_list])

						metric_shuffled = m(df_shuffled, df_train)

						row = {'feat':feat,
							   'rep':0,
							   'fold':0,
							   'iter':i,
							   'metric_name': m.name,
							   'metric_original':metric_original,
							   'metric_shuffled':metric_shuffled,
							   'mda':metric_original-metric_shuffled}
						rows.append(row)


		df_mda = pd.DataFrame(rows)
		df_mda.to_pickle(self.output()['iter'].path)
		# sorted_feats = sorted(feature_list, key= lambda x: mda[feat]/(np.sqrt(mda2[feat]-mda[feat]**2)+1e-14))

		for metric in self.metrics:
			m = metrics_list[metric]()
			df_m = df_mda.loc[(df_mda['metric_name']==m.name)]
			sorted_feats = sorted(feature_list, key= lambda x:  m.optimization_sign * df_m.loc[df_m['feat']==x,'mda'].mean(), reverse=True)
			rows = []
			for feat in sorted_feats:
				row = {"Feature":feat,
					   "MDA_norm": df_m.loc[df_m['feat']==feat,'mda'].mean()/df_m.loc[df_m['feat']==sorted_feats[0],'mda'].mean(),
					   "MDA": df_m.loc[df_m['feat']==feat,'mda'].mean(),
					   "stdev": df_m.loc[df_m['feat']==feat,'mda'].std(),
					   "z-score": df_m.loc[df_m['feat']==feat,'mda'].mean()/(df_m.loc[df_m['feat']==feat,'mda'].std()+1e-14)}
				rows.append(row)
			pd.DataFrame(rows).to_csv(self.output()[f'{metric}_csv'].path)

			with open(self.output()[f'{metric}_txt'].path,'w') as f:
				print(f"{'Feature':30.30} {'MDA_norm':10.10} {'MDA':10.10} {'Variation':10.10} {'z-score':10.10}", file=f)
				for row in rows:
					print(f"{row['Feature']:30.30} {row['MDA_norm']:0.4e} {row['MDA']:0.4e} {row['stdev']:0.4e} {row['z-score']:0.4e}", file=f)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__,self.wf_name))
		except:
			pass

		outputs = {}
		outputs['iter'] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"df_iter_{self.wf_name}_{self.clf_name}{'_EXT' if self.ext_val == 'Yes' else ''}.pickle"))
		for metric in self.metrics:
			outputs[f'{metric}_csv'] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"MDA_{metric}_Log_{self.wf_name}_{self.clf_name}{'_EXT' if self.ext_val == 'Yes' else ''}.csv"))
			outputs[f'{metric}_txt'] = luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"MDA_{metric}_Log_{self.wf_name}_{self.clf_name}{'_EXT' if self.ext_val == 'Yes' else ''}.txt"))
		return outputs

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

class FinalModelTrainResults(luigi.Task):
	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		requirements = {}
		requirements["data"] = ProcessDatabase(wf_name = self.wf_name)
		requirements["clf"] = FinalModelAndHyperparameterResults(wf_name = self.wf_name, clf_name = self.clf_name)
		return requirements

	def run(self):
		setupLog(self.__class__.__name__)

		clf = pd.read_pickle(self.input()["clf"].path)
		df = pd.read_pickle(self.input()["data"]["pickle"].path)
		filter_function = WF_info[self.wf_name]["filter_function"]
		features = WF_info[self.wf_name]["feature_list"]
		label = WF_info[self.wf_name]["label_name"]
		group_label = WF_info[self.wf_name]["group_label"]
		if WF_info[self.wf_name]['type'] == 'survival':
			time = WF_info[self.wf_name]["label_time"]

		df = filter_function(df).reset_index(drop=True)
		X = df.loc[:,features]

		if WF_info[self.wf_name]['type'] == 'classification':
			try:
				df['Prediction'] = clf.predict_proba(X)[:,1]
			except:
				df['Prediction'] = clf.decision_function(X)
		else:
			df['Prediction']= clf.predict(X)
		df['True Label'] = df[label]
		if WF_info[self.wf_name]['type'] == 'survival':
			df['Time'] = df[time]

		critical_pvalue=0.05
		results_dict = {}

		for metric in WF_info[self.wf_name]['metrics']:
			m = metrics_list[metric]()
			score = m(df, df)
			results_dict[f'avg_{m.name}'] = score

			if m.variance is not None:
				c = sc_st.t.ppf(1-critical_pvalue/2, df=0)
				results_dict[f'avg_{m.name}_stderr'] = np.sqrt(m.variance)
				results_dict[f'{m.name}_95ci_low'] = score - c*np.sqrt(m.variance)
				results_dict[f'{m.name}_95ci_high'] = score + c*np.sqrt(m.variance)
			else:
				results_dict[f'avg_{m.name}_stderr'] = np.nan
				results_dict[f'{m.name}_95ci_low'] = np.nan
				results_dict[f'{m.name}_95ci_high'] = np.nan
			results_dict[f'pool_{m.name}'] = score

		df.to_excel(self.output()["xls"].path)
		df.to_pickle(self.output()["pickle"].path)

		with open(self.output()[f"results"].path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)

		with open(self.output()["results_txt"].path, 'w') as f:
			print(results_dict, file=f)

		# Shapley values, internal value
		# masker = shap.maskers.Independent(data =  X.astype(float).values)
		masker = shap.maskers.Independent(data =  X.astype(float))
		if WF_info[self.wf_name]['type'] == 'classification':
			if hasattr(model, 'predict_proba'):
				explainer = shap.PermutationExplainer(lambda x: clf.predict_proba(x)[:,1], masker, link = shap.links.logit)
			else:
				explainer = shap.PermutationExplainer(lambda x: clf.decision_function(x), masker, link = shap.links.identity)
		else:
			explainer = shap.PermutationExplainer(lambda x: clf.predict(x), masker, link = shap.links.identity)
		shap_values = explainer(X.astype(float).values).values

		with open(self.output()[f"shapvalues"].path, 'wb') as f:
			pickle.dump(shap_values, f, pickle.HIGHEST_PROTOCOL)

		shap.summary_plot(shap_values, X, max_display = 100, show=False)
		plt.savefig(self.output()['shap'].path, bbox_inches='tight', dpi=300)
		plt.close()


	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__,self.wf_name))
		except:
			pass

		return {"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"FinalModelTrainPredictions_df_{self.clf_name}.xlsx")),
				"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"FinalModelTrainPredictions_df_{self.clf_name}.pickle")),
				"results": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"FinalModelTrainResults_{self.clf_name}.pickle")),
				"results_txt": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"FinalModelTrainResults_{self.clf_name}.txt")),
				"shap": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"FinalModelTrainShapValues_{self.clf_name}.png")),
				"shapvalues": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,f"FinalModelTrainShapValues_{self.clf_name}.pickle"))}

class FairnessReport(luigi.Task):
	list_ML = luigi.ListParameter()
	wf_name = luigi.Parameter()
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))
	ext_val = luigi.Parameter(default = 'No')

	def requires(self):
		requirements = {}
		requirements['Descriptive'] = DescriptiveFairness(wf_name = self.wf_name, ext_val = self.ext_val)
		for model in self.list_ML:
			requirements[model] = Evaluate_Fairness(wf_name = self.wf_name, clf_name = model, ext_val = self.ext_val)
			requirements['shap_'+model] = InterpretationShapFairness(wf_name = self.wf_name, clf_name = model, ext_val = self.ext_val)
		return requirements

	def run(self):
		setupLog(self.__class__.__name__)
		for model in self.list_ML:
			shutil.copy(self.input()[model]['xls'].path, self.output()['fairness_'+model].path)
			shutil.copy(self.input()['shap_'+model].path, self.output()['shap_'+model].path)
		shutil.copy(self.input()['Descriptive'].path, self.output()['xls'].path)



	def output(self):
		try:
			os.makedirs(os.path.join(report_path+f'-{self.datestring}', self.wf_name, "Fairness Evaluation"))
		except:
			pass
		outputs = {}
		for model in self.list_ML:
			outputs['fairness_'+model] = luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}', self.wf_name, "Fairness Evaluation", f"EvaluationResults_FairnessSubgroups_{model}.xlsx"))
			outputs['shap_'+model] = luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}', self.wf_name, "Fairness Evaluation", f"ShapFairness_{model}.png"))
		outputs['xls'] = luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}', self.wf_name, "Fairness Evaluation", "Fairness_Group_Comparison.xlsx"))
		return outputs

class TrainingReport(luigi.Task):
	list_ML = luigi.ListParameter()
	wf_name = luigi.Parameter()
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	def requires(self):
		requirements = {}
		for model in self.list_ML:
			requirements[model] = FinalModelTrainResults(wf_name = self.wf_name, clf_name = model)
		return requirements

	def run(self):
		setupLog(self.__class__.__name__)
		for model in self.list_ML:
			shutil.copy(self.input()[model]['shap'].path, self.output()['shap_'+model].path)

		for metric in WF_info[self.wf_name]['metrics']:
			m = metrics_list[metric]()
			fig, ax = m.figure(len(self.list_ML))
			fig.suptitle(WF_info[self.wf_name]["formal_title"], fontsize = 14)
			for model in self.list_ML:
				with open(self.input()[model]["pickle"].path, 'rb') as f:
					df=pickle.load(f)
				with open(self.input()[model]["results"].path, 'rb') as f:
					results_dict=pickle.load(f)
				m.plot(df, results_dict, ax, ML_info[model]["formal_name"])
			m.save_figure(fig, ax, self.output()['metric_'+metric].path)

	def output(self):
		try:
			os.makedirs(os.path.join(report_path+f'-{self.datestring}', self.wf_name, "Training Report"))
		except:
			pass
		outputs = {}
		for model in self.list_ML:
			outputs['shap_'+model] = luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}', self.wf_name, "Training Report", f"TrainResults_Shap_Values_{model}.png"))
		for metric in WF_info[self.wf_name]['metrics']:
			outputs['metric_'+metric] = luigi.LocalTarget(os.path.join(report_path+f'-{self.datestring}', self.wf_name, "Training Report", f"TrainResults_All_Models_{metric}.png"))
		return outputs

class AllFairnessReports(luigi.Task):
	list_WF = luigi.ListParameter(default=list(WF_info.keys()))
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	def requires(self):
		for it_wf_name in self.list_WF:
			if 'fairness_label' in WF_info[it_wf_name].keys():
				yield FairnessReport(wf_name = it_wf_name, list_ML=WF_info[it_wf_name]['models'], datestring=self.datestring)

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write("prueba\n")

	def output(self):
		return luigi.LocalTarget(os.path.join(log_path, f"AllFairnessReports_Log-{self.datestring}.txt"))

class AllTrainingReports(luigi.Task):
	list_WF = luigi.ListParameter(default=list(WF_info.keys()))
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	def requires(self):
		for it_wf_name in self.list_WF:
			yield TrainingReport(wf_name = it_wf_name, list_ML=WF_info[it_wf_name]['models'], datestring=self.datestring)
			if(WF_info[it_wf_name]['external_validation'] == 'Yes'):
				yield TrainingReport(wf_name = it_wf_name, list_ML=WF_info[it_wf_name]['models'], ext_val = 'Yes', datestring=self.datestring)

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write("prueba\n")

	def output(self):
		return luigi.LocalTarget(os.path.join(log_path, f"AllTrainingReports_Log-{self.datestring}.txt"))

class InterpretationReport(luigi.Task):
	list_ML = luigi.ListParameter()
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
				requirements[i+'_mda'] = MDAFeatureImportances(clf_name = i, wf_name = self.wf_name, ext_val=self.ext_val, metrics = WF_info[self.wf_name]['metrics'][0:1])
			if self.all_shap == 'Yes':
				requirements[i+'_shap'] = ShapleyValues(clf_name = i, wf_name = self.wf_name, ext_val=self.ext_val)
		return requirements


	def run(self):
		setupLog(self.__class__.__name__)

		if (self.best_MDA == 'Yes') or (self.best_shap == 'Yes'):
			score_ml = {}
			m = metrics_list[WF_info[self.wf_name]['metrics'][0]]()
			for i in self.list_ML:
				with open(self.input()[i]["results"].path, 'rb') as f:
					results_dict=pickle.load(f)
					score_ml[i]=m.optimization_sign*results_dict[f"avg_{m.name}"]
			best_ml = max(score_ml.keys(), key=(lambda k: score_ml[k]))
			if self.best_MDA == 'Yes':
				prerequisite = MDAFeatureImportances(clf_name = best_ml, wf_name = self.wf_name, ext_val = self.ext_val, metrics = WF_info[self.wf_name]['metrics'][0:1])
				luigi.build([prerequisite], local_scheduler = False)
				shutil.copy(prerequisite.output()[f'{WF_info[self.wf_name]["metrics"][0]}_txt'].path, self.output()['best_mda'].path)
			if self.best_shap == 'Yes':
				prerequisite = ShapleyValues(clf_name = best_ml, wf_name = self.wf_name, ext_val = self.ext_val)
				luigi.build([prerequisite], local_scheduler = False)
				shutil.copy(prerequisite.output()["png"].path, self.output()['best_shap'].path)
		if self.all_MDA == 'Yes':
			for i in self.list_ML:
				shutil.copy(self.input()[i+'_mda'][f'{WF_info[self.wf_name]["metrics"][0]}_txt'].path, self.output()[i+'_mda'].path)
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
	list_WF = luigi.ListParameter(default=list(WF_info.keys()))
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	best_MDA = luigi.Parameter(default = 'Yes')
	all_MDA = luigi.Parameter(default = 'Yes')
	best_shap = luigi.Parameter(default = 'No')
	all_shap = luigi.Parameter(default = 'No')

	def requires(self):
		for it_wf_name in self.list_WF:
			yield InterpretationReport(wf_name = it_wf_name, list_ML=WF_info[it_wf_name]['models'], datestring=self.datestring,
										best_MDA = self.best_MDA, best_shap = self.best_shap, all_MDA = self.all_MDA, all_shap = self.all_shap)
			if(WF_info[it_wf_name]['external_validation'] == 'Yes'):
				yield InterpretationReport(wf_name = it_wf_name, list_ML=WF_info[it_wf_name]['models'], ext_val = 'Yes', datestring=self.datestring,
											best_MDA = self.best_MDA, best_shap = self.best_shap, all_MDA = self.all_MDA, all_shap = self.all_shap)

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write("prueba\n")

	def output(self):
		return luigi.LocalTarget(os.path.join(log_path, f"AllInterpretationReports_Log-{self.datestring}.txt"))

class AllPerformanceReports(luigi.Task):

	list_WF = luigi.ListParameter(default=list(WF_info.keys()))
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	def requires(self):
		for it_wf_name in self.list_WF:
			yield BestMLModelReport(wf_name = it_wf_name, list_ML=WF_info[it_wf_name]['models'], datestring=self.datestring)
			if(WF_info[it_wf_name]['external_validation'] == 'Yes'):
				yield BestMLModelReport(wf_name = it_wf_name, list_ML=WF_info[it_wf_name]['models'], datestring=self.datestring, ext_val = 'Yes')


	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write("prueba\n")

	def output(self):
		return luigi.LocalTarget(os.path.join(log_path, f"AllPerformanceReports_Log-{self.datestring}.txt"))

class AllModels(luigi.Task):

	list_WF = luigi.ListParameter(default=list(WF_info.keys()))
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	def requires(self):
		for it_wf_name in self.list_WF:
			for it_clf_name in WF_info[it_wf_name]['models']:
				yield FinalModelAndHyperparameterResults(wf_name = it_wf_name, clf_name = it_clf_name)

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
			yield HistogramsPDF(wf_name = it_wf_name, label_name = WF_info[it_wf_name]['label_name'])
			if(WF_info[it_wf_name]['external_validation'] == 'Yes'):
				yield HistogramsPDF(wf_name = it_wf_name, ext_val = 'Yes', label_name = WF_info[it_wf_name]['label_name'])

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
	list_WF = luigi.ListParameter(default=list(WF_info.keys()))
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))

	def requires(self):
		for it_wf_name in self.list_WF:
			for metric in WF_info[it_wf_name]['metrics']:
				yield GraphsWF(wf_name = it_wf_name, list_ML=WF_info[it_wf_name]['models'],  datestring=self.datestring, metric=metric)
				if(WF_info[it_wf_name]['external_validation'] == 'Yes'):
					yield GraphsWF(wf_name = it_wf_name, list_ML=WF_info[it_wf_name]['models'], ext_val = 'Yes', datestring=self.datestring, metric = metric)

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write("prueba\n")

	def output(self):
		return luigi.LocalTarget(os.path.join(log_path, f"AllGraphs_Log-{self.datestring}.txt"))

class AllTasks(luigi.Task):

	list_WF = luigi.ListParameter(default=list(WF_info.keys()))
	datestring = luigi.Parameter(default=dt.datetime.now().strftime("%y%m%d-%H%M%S"))
	best_MDA = luigi.Parameter(default = 'Yes')
	all_MDA = luigi.Parameter(default = 'Yes')
	best_shap = luigi.Parameter(default = 'No')
	all_shap = luigi.Parameter(default = 'No')


	def __init__(self, *args, **kwargs):
		super(AllTasks, self).__init__(*args, **kwargs)

	def requires(self):

		return [AllGraphs(list_WF = self.list_WF, datestring=self.datestring),
				AllDescriptiveReports(list_WF = self.list_WF, datestring=self.datestring),
				AllHistograms(list_WF = self.list_WF, datestring=self.datestring),
				AllModels(list_WF = self.list_WF, datestring=self.datestring),
				AllPerformanceReports(list_WF = self.list_WF, datestring=self.datestring),
				AllTrainingReports(list_WF = self.list_WF, datestring=self.datestring),
				AllFairnessReports(list_WF = self.list_WF, datestring=self.datestring),
				AllInterpretationReports(list_WF = self.list_WF, datestring=self.datestring,
										best_MDA = self.best_MDA, best_shap = self.best_shap, all_MDA = self.all_MDA, all_shap = self.all_shap),
										]

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write("prueba\n")

	def output(self):
		return luigi.LocalTarget(os.path.join(log_path, f"AllTask_Log-{self.datestring}.txt"))
