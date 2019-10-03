import os
import luigi
import numpy as np
import pandas as pd
import pickle
import contextlib

from utils.crossvalidation import predict_kfold_ML, predict_kfold_RS
from utils.analysis import AUC_stderr_classic,AUC_stderr_hanley, group_files_analyze, mdaeli5_analysis, plot_all_aucs, paired_ttest, cutoff_threshold_maxfbeta, cutoff_threshold_single, cutoff_threshold_double, cutoff_threshold_triple,cutoff_threshold_accuracy, all_thresholds, create_descriptive_xls

from user_data_utils import load_database, clean_database, process_database, fillna_database
from user_MLmodels_info import ML_info
from user_RiskScores_info import RS_info
from user_Workflow_info import WF_info

# Global variables for path folders
log_path = os.path.abspath("log")
tmp_path = os.path.abspath("intermediate")
model_path = os.path.abspath("models")
report_path = os.path.abspath("report")

#Luigi Tasks
class CleanDatabase(luigi.Task):
	def run(self):
		df_input = load_database()
		df_output = clean_database(df_input)
		df_output.to_pickle(self.output()["pickle"].path)
		df_output.to_excel(self.output()["xls"].path, index=False)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_clean.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_clean.xlsx"))}

class ProcessDatabase(luigi.Task):

	def requires(self):
		return CleanDatabase()
	def run(self):
		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_output = process_database(df_input)
		df_output.to_pickle(self.output()["pickle"].path)
		df_output.to_excel(self.output()["xls"].path, index=False)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_processed.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_processed.xls"))}

class FillnaDatabase(luigi.Task):
	def requires(self):
		return ProcessDatabase()

	def run(self):
		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_output = fillna_database(df_input)
		df_output.to_pickle(self.output()["pickle"].path)
		df_output.to_excel(self.output()["xls"].path, index=False)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_fillna.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_fillna.xls"))}


class CalculateKFold(luigi.Task):

	seed = luigi.IntParameter()
	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()
	folds = luigi.IntParameter(default=10)

	def requires(self):
		return FillnaDatabase()

	def run(self):
		try:
			os.makedirs(os.path.join(log_path,self.__class__.__name__))
		except:
			pass

		with open(os.path.join(log_path,self.__class__.__name__,f"Log_{self.wf_name}_{self.clf_name}_{self.seed}.txt"),'w') as f:
			with contextlib.redirect_stdout(f):
				df_input = pd.read_pickle(self.input()["pickle"].path)
				df_filtered = WF_info[self.wf_name]["filter_function"](df_input)
				features = WF_info[self.wf_name]["feature_list"]
				label = WF_info[self.wf_name]["label_name"]
				group_label = WF_info[self.wf_name]["group_label"]
				clf = ML_info[self.clf_name]["clf"]

				if group_label is None:
					tl_pp_dict = predict_kfold_ML(df_filtered, label, features, clf, self.clf_name, self.seed, self.folds)
				else:
					 tl_pp_dict = predict_groupkfold_ML(df_filtered, label, features, group_label, clf, self.clf_name, self.seed, self.folds)

		with open(self.output().path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(tl_pp_dict, f, pickle.HIGHEST_PROTOCOL)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"TrueLabel_PredProb_{self.wf_name}_{self.clf_name}_{self.seed}.dict"))

class RiskScore_KFold(luigi.Task):

	seed = luigi.IntParameter()
	score_name = luigi.Parameter()
	wf_name = luigi.Parameter()
	folds = luigi.IntParameter(default=10)

	def requires(self):
		return FillnaDatabase()

	def run(self):
		try:
			os.makedirs(os.path.join(log_path,self.__class__.__name__))
		except:
			pass
		with open(os.path.join(log_path,self.__class__.__name__,f"Log_{self.wf_name}_{self.score_name}_{self.seed}.txt"),'w') as f:
			with contextlib.redirect_stdout(f):
				df_input = pd.read_pickle(self.input()["pickle"].path)
				df_filtered = WF_info[self.wf_name]["filter_function"](df_input)
				label = WF_info[self.wf_name]["label_name"]
				features = WF_info[self.wf_name]["feature_list"]
				group_label = WF_info[self.wf_name]["group_label"]
				sign = RS_info[self.score_name]["sign"]
				RS_name = RS_info[self.score_name]["label_name"]

				if group_label is None:
					tl_pp_dict = predict_kfold_RS(df_filtered, label, features, sign, RS_name, self.seed, self.folds)
				else:
					tl_pp_dict = predict_groupkfold_RS(df_filtered, label, features, group_label, sign, RS_name, self.seed, self.folds)

		with open(self.output().path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(tl_pp_dict, f, pickle.HIGHEST_PROTOCOL)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"TrueLabel_PredProb_{self.wf_name}_{self.score_name}_{self.seed}.dict"))


class Evaluate_ML(luigi.Task):

	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()
	repetitions = luigi.IntParameter(default=10)
	folds = luigi.IntParameter(default=10)

	def requires(self):
		for i in range(1,self.repetitions+1):
			yield CalculateKFold(wf_name=self.wf_name, seed=i,clf_name=self.clf_name, folds=self.folds)

	def run(self):
		try:
			os.makedirs(os.path.join(log_path,self.__class__.__name__))
		except:
			pass
		with open(os.path.join(log_path,self.__class__.__name__,f"EvaluateML_Log_{self.wf_name}_{self.clf_name}.txt"),'w') as f:
			with contextlib.redirect_stdout(f):
				(unfolded_pred_prob,unfolded_true_label, results_dict) = group_files_analyze(self.input(), self.clf_name)
				with open(self.output()["pred_prob"].path, 'wb') as f:
					# Pickle the 'data' dictionary using the highest protocol available.
					pickle.dump(unfolded_pred_prob, f, pickle.HIGHEST_PROTOCOL)
				with open(self.output()["true_label"].path, 'wb') as f:
					# Pickle the 'data' dictionary using the highest protocol available.
					pickle.dump(unfolded_true_label, f, pickle.HIGHEST_PROTOCOL)
				with open(self.output()["auc_results"].path, 'wb') as f:
					# Pickle the 'data' dictionary using the highest protocol available.
					pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)


	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pred_prob": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"Unfolded_Pred_Prob_{self.wf_name}_{self.clf_name}.pickle")),
				"true_label": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"Unfolded_True_Label_{self.wf_name}_{self.clf_name}.pickle")),
				"auc_results": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"AUC_results_{self.wf_name}_{self.clf_name}.pickle"))}

class EvaluateRiskScore(luigi.Task):
	wf_name = luigi.Parameter()
	score_name = luigi.Parameter()
	repetitions = luigi.IntParameter(default=10)
	folds = luigi.IntParameter(default=10)

	def requires(self):
		for i in range(1,self.repetitions+1):
			yield RiskScore_KFold(wf_name=self.wf_name, seed=i,score_name=self.score_name, folds=self.folds)

	def run(self):
		try:
			os.makedirs(os.path.join(log_path,self.__class__.__name__))
		except:
			pass
		with open(os.path.join(log_path,self.__class__.__name__,f"EvaluateRS_Log_{self.wf_name}_{self.score_name}.txt"),'w') as f:
			with contextlib.redirect_stdout(f):
				(unfolded_pred_prob,unfolded_true_label,results_dict) = group_files_analyze(self.input(), self.score_name)
				with open(self.output()["pred_prob"].path, 'wb') as f:
					# Pickle the 'data' dictionary using the highest protocol available.
					pickle.dump(unfolded_pred_prob, f, pickle.HIGHEST_PROTOCOL)
				with open(self.output()["true_label"].path, 'wb') as f:
					# Pickle the 'data' dictionary using the highest protocol available.
					pickle.dump(unfolded_true_label, f, pickle.HIGHEST_PROTOCOL)
				with open(self.output()["auc_results"].path, 'wb') as f:
					# Pickle the 'data' dictionary using the highest protocol available.
					pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path, self.__class__.__name__))
		except:
			pass
		return {"pred_prob": luigi.LocalTarget(os.path.join(tmp_path, self.__class__.__name__,f"Unfolded_Pred_Prob_{self.wf_name}_{self.score_name}.pickle")),
				"true_label": luigi.LocalTarget(os.path.join(tmp_path, self.__class__.__name__,f"Unfolded_True_Label_{self.wf_name}_{self.score_name}.pickle")),
				"auc_results": luigi.LocalTarget(os.path.join(tmp_path, self.__class__.__name__,f"AUC_results_{self.wf_name}_{self.score_name}.pickle"))}

class ConfidenceIntervalHanleyRS(luigi.Task):
	wf_name = luigi.Parameter()
	score_name = luigi.Parameter()

	def requires(self):
		return FillnaDatabase()

	def run(self):
		df_input = pd.read_pickle(self.input()["pickle"].path)
		(auc, stderr) = AUC_stderr_classic(df_input, label_name=WF_info[self.wf_name]["label_name"], score_name=RS_info[self.score_name]["label_name"])
		ci95_low= auc-1.96*stderr
		ci95_high= auc+1.96*stderr


		with open(self.output().path,'w') as f:
			f.write(f"Confidence Interval Upper Bound Classic(95%): {auc} ({ci95_low}-{ci95_high})")

			(auc, stderr) = AUC_stderr_hanley(df_input , label_name=WF_info[self.wf_name]["label_name"], score_name=RS_info[self.score_name]["label_name"])
			ci95_low= auc-1.96*stderr
			ci95_high= auc+1.96*stderr
			f.write(f"Confidence Interval Hanley(95%): {ci95_low}-{ci95_high}")
	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path, self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(tmp_path, self.__class__.__name__, f"CI_{self.wf_name}_{self.score_name}.txt"))


class DescriptiveXLS(luigi.Task):
	wf_name = luigi.Parameter()

	def requires(self):
		return ProcessDatabase()

	def run(self):
		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_filtered = WF_info[self.wf_name]["filter_function"](df_input)
		label = WF_info[self.wf_name]["label_name"]

		df_output=create_descriptive_xls(df_filtered, self.wf_name, label)
		df_output.to_excel(self.output().path)

	def output(self):
		try:
			os.makedirs(os.path.join(report_path, self.wf_name))
		except:
			pass
		return luigi.LocalTarget(os.path.join(report_path, self.wf_name, f"{self.wf_name}_descriptivo.xlsx"))

class FinalModelAndHyperparameterResults(luigi.Task):
	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return FillnaDatabase()

	def run(self):
		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_filtered = WF_info[self.wf_name]["filter_function"](df_input)
		label = WF_info[self.wf_name]["label_name"]
		features = WF_info[self.wf_name]["feature_list"]

		self.clf=ML_info[self.clf_name]["clf"]

		X = df_filtered.loc[:,features]
		Y = df_filtered.loc[:,[label]]

		self.clf.fit(X,Y)

		try:
			self.final_clf = self.clf.best_estimator_
			with open(self.output().path,'wb') as f:
				pickle.dump(self.final_clf, f, pickle.HIGHEST_PROTOCOL)
			pd.DataFrame(self.clf.cv_results_).to_excel(os.path.join(model_path,self.wf_name,f"HyperparameterResults_{self.wf_name}_{self.clf_name}.xlsx"))
		except:
			with open(self.output().path,'wb') as f:
				pickle.dump(self.clf, f, pickle.HIGHEST_PROTOCOL)

	def output(self):
		try:
			os.makedirs(os.path.join(model_path,self.wf_name))
		except:
			pass
		return luigi.LocalTarget(os.path.join(model_path,self.wf_name,f"ML_model_{self.wf_name}_{self.clf_name}.pickle"))


class AllModels_PairedTTest(luigi.Task):
	wf_name = luigi.Parameter()
	repetitions = luigi.Parameter(default=10)
	folds = luigi.Parameter(default=10)
	list_ML = luigi.ListParameter()
	list_RS = luigi.ListParameter()

	def requires(self):
		requirements={}
		for clf_or_score1 in self.list_RS:
			requirements[clf_or_score1] = EvaluateRiskScore(wf_name=self.wf_name, score_name = clf_or_score1, repetitions=self.repetitions, folds=self.folds)
		for clf_or_score2 in self.list_ML:
			requirements[clf_or_score2] = Evaluate_ML(wf_name=self.wf_name, clf_name=clf_or_score2, repetitions=self.repetitions, folds=self.folds)
		return requirements

	def run(self):
		with open(self.output().path,'w') as f:
			list_comparisons = []
			for clf_or_score1 in self.list_ML+self.list_RS:
				for clf_or_score2 in self.list_ML+self.list_RS:
					if (clf_or_score1 != clf_or_score2):
						(averaging_diff, pvalue) = paired_ttest(self.requires()[clf_or_score1],self.requires()[clf_or_score2], self.wf_name, tmp_path, self.repetitions)
						if clf_or_score1 in self.list_ML:
							formal_name1 = ML_info[clf_or_score1]["formal_name"]
						else:
							formal_name1 = RS_info[clf_or_score1]["formal_name"]
						if clf_or_score2 in self.list_ML:
							formal_name2 = ML_info[clf_or_score2]["formal_name"]
						else:
							formal_name2 = RS_info[clf_or_score2]["formal_name"]
						wf_formal_title = WF_info[self.wf_name]["formal_title"]
						f.write(f"{wf_formal_title}: {formal_name1}-{formal_name2}, Avg Diff: {averaging_diff}, p-value: {pvalue}")

	def output(self):
		try:
			os.makedirs(os.path.join(report_path, self.wf_name))
		except:
			pass
		return luigi.LocalTarget(os.path.join(report_path, self.wf_name,f"AllModelsPairedTTest_{self.wf_name}.txt"))

class GraphsWF(luigi.Task):

	wf_name = luigi.Parameter()
	repetitions = luigi.IntParameter(default=10)
	folds = luigi.IntParameter(default=10)
	list_ML = luigi.ListParameter()
	list_RS = luigi.ListParameter()
	n_best_ML = luigi.IntParameter(default=1)
	n_best_RS = luigi.IntParameter(default=2)

	def requires(self):
		requirements = {}
		for i in self.list_ML:
			requirements[i] = Evaluate_ML(clf_name = i, wf_name = self.wf_name, repetitions = self.repetitions, folds=self.folds)
		for i in self.list_RS:
			requirements[i] = EvaluateRiskScore(score_name = i, wf_name = self.wf_name, repetitions = self.repetitions, folds=self.folds)
		return requirements

	def run(self):

		# First we plot every ML model and risk score
		plot_all_aucs(task_requires=self.input(), fig_path= self.output()["plot_all"].path,title=WF_info[self.wf_name]["formal_title"])

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
					auc_ml[i]=results_dict["avg_auc"]

			sorted_ml = sorted(auc_ml.keys(), key=lambda x: auc_ml[x])

			auc_rs = {}
			for i in self.list_RS:
				with open(self.input()[i]["auc_results"].path, 'rb') as f:
						results_dict=pickle.load(f)
						auc_rs[i]=results_dict["avg_auc"]

			sorted_rs = sorted(auc_rs.keys(), key=lambda x: auc_rs[x])

			# We plot the ROC of the best ML model and the best risk score
			tasks={}
			for ml in sorted_ml[0:self.n_best_ML]:
				tasks[ml]=self.input()[ml]
			for rs in sorted_rs[0:self.n_best_RS]:
				tasks[rs]=self.input()[rs]

			plot_all_aucs(task_requires=tasks , fig_path= self.output()["plot_best"].path,title=WF_info[self.wf_name]["formal_title"])

	def output(self):
		try:
			os.makedirs(os.path.join(report_path,self.wf_name))
		except:
			pass
		if((len(self.list_RS) > 0) & (len(self.list_ML) > 0)):
			return {"plot_all": luigi.LocalTarget(os.path.join(report_path,self.wf_name, f"AllModelsPlot_{self.wf_name}.png")),
					"plot_best": luigi.LocalTarget(os.path.join(report_path,self.wf_name, f"BestModelsPlot_{self.wf_name}.png"))}
		else:
			return {"plot_all": luigi.LocalTarget(os.path.join(report_path,self.wf_name, f"AllModelsPlot_{self.wf_name}.png"))}



class ThresholdPoints(luigi.Task):
	clf_or_score=luigi.Parameter()
	wf_name = luigi.Parameter()
	repetitions = luigi.IntParameter(default=10)
	folds = luigi.IntParameter(default=10)
	list_RS = luigi.ListParameter(default=[])
	list_ML = luigi.ListParameter(default=[])


	def requires(self):
		if (self.clf_or_score in self.list_RS):
			return EvaluateRiskScore(wf_name=self.wf_name, score_name = self.clf_or_score, repetitions=self.repetitions,folds = self.folds)
		elif (self.clf_or_score in self.list_ML):
			return Evaluate_ML(wf_name=self.wf_name, clf_name=self.clf_or_score, repetitions=self.repetitions,folds = self.folds)
		else:
			raise Exception(f"{self.clf_score} not in list_ristkscores or list_MLmodels")

	def run(self):
		with open(self.input()["pred_prob"].path, 'rb') as f:
			pred_prob=pickle.load(f)
		with open(self.input()["true_label"].path, 'rb') as f:
			true_label=pickle.load(f)

		with open(self.output().path,'w') as f:
			(best_threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = cutoff_threshold_accuracy(pred_prob, true_label)
			f.write(f'Threshold: {best_threshold} Optimum for accuracy')
			f.write(f'TP:{tprate*100:.1f} FP:{fprate*100:.1f} TN:{tnrate*100:.1f} FN:{fnrate*100:.1f}')
			f.write(f'Sensitivity:{sens*100:.1f} Specifity:{spec*100:.1f} Precision:{prec*100:.1f} NPRv:{nprv*100:.1f}')
			f.write("")

			(best_threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = cutoff_threshold_single(pred_prob, true_label)
			f.write(f'Threshold: {best_threshold} Optimum for single point AUC')
			f.write(f'TP:{tprate*100:.1f} FP:{fprate*100:.1f} TN:{tnrate*100:.1f} FN:{fnrate*100:.1f}')
			f.write(f'Sensitivity:{sens*100:.1f} Specifity:{spec*100:.1f} Precision:{prec*100:.1f} NPRv:{nprv*100:.1f}')
			f.write("")

			threshold_dict = cutoff_threshold_double(pred_prob, true_label)
			(best_threshold1, tprate1, fprate1, tnrate1, fnrate1, sens1, spec1, prec1, nprv1) = threshold_dict["threshold1"]
			(best_threshold2, tprate2, fprate2, tnrate2, fnrate2, sens2, spec2, prec2, nprv2) = threshold_dict["threshold2"]

			f.write('Optimum for double point AUC')
			f.write(f'Threshold: {best_threshold1}')
			f.write(f'TP:{tprate1*100:.1f} FP:{fprate1*100:.1f} TN:{tnrate1*100:.1f} FN:{fnrate1*100:.1f}')
			f.write(f'Sensitivity:{sens1*100:.1f} Specifity:{spec1*100:.1f} Precision:{prec1*100:.1f} NPRv:{nprv1*100:.1f}')
			f.write(f'Threshold: {best_threshold2}')
			f.write(f'TP:{tprate2*100:.1f} FP:{fprate2*100:.1f} TN:{tnrate2*100:.1f} FN:{fnrate2*100:.1f}')
			f.write(f'Sensitivity:{sens2*100:.1f} Specifity:{spec2*100:.1f} Precision:{prec2*100:.1f} NPRv:{nprv2*100:.1f}')
			f.write("")

			threshold_dict = cutoff_threshold_triple(pred_prob, true_label)
			(best_threshold1, tprate1, fprate1, tnrate1, fnrate1, sens1, spec1, prec1, nprv1) = threshold_dict["threshold1"]
			(best_threshold2, tprate2, fprate2, tnrate2, fnrate2, sens2, spec2, prec2, nprv2) = threshold_dict["threshold2"]
			(best_threshold3, tprate3, fprate3, tnrate3, fnrate3, sens3, spec3, prec3, nprv3) = threshold_dict["threshold3"]

			f.write('Optimum for triple point AUC')
			f.write(f'Threshold: {best_threshold1}')
			f.write(f'TP:{tprate1*100:.1f} FP:{fprate1*100:.1f} TN:{tnrate1*100:.1f} FN:{fnrate1*100:.1f}')
			f.write(f'Sensitivity:{sens1*100:.1f} Specifity:{spec1*100:.1f} Precision:{prec1*100:.1f} NPRv:{nprv1*100:.1f}')
			f.write(f'Threshold: {best_threshold2}')
			f.write(f'TP:{tprate2*100:.1f} FP:{fprate2*100:.1f} TN:{tnrate2*100:.1f} FN:{fnrate2*100:.1f}')
			f.write(f'Sensitivity:{sens2*100:.1f} Specifity:{spec2*100:.1f} Precision:{prec2*100:.1f} NPRv:{nprv2*100:.1f}')
			f.write(f'Threshold: {best_threshold3}')
			f.write(f'TP:{tprate3*100:.1f} FP:{fprate3*100:.1f} TN:{tnrate3*100:.1f} FN:{fnrate3*100:.1f}')
			f.write(f'Sensitivity:{sens3*100:.1f} Specifity:{spec3*100:.1f} Precision:{prec3*100:.1f} NPRv:{nprv3*100:.1f}')
			f.write("")

			for beta in [0.5,1,2]:
				(max_f1_threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = cutoff_threshold_maxfbeta(pred_prob, true_label, beta)
				f.write(f'Threshold: {max_f1_threshold} Optimum for f{beta}')
				f.write(f'TP:{tprate*100:.1f} FP:{fprate*100:.1f} TN:{tnrate*100:.1f} FN:{fnrate*100:.1f}')
				f.write(f'Sensitivity:{sens*100:.1f} Specifity:{spec*100:.1f} Precision:{prec*100:.1f} NPRv:{nprv*100:.1f}')
				f.write("")

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}.txt"))

class BestMLModelReport(luigi.Task):
	wf_name = luigi.Parameter()
	repetitions = luigi.IntParameter(default=10)
	folds = luigi.IntParameter(default=10)
	list_ML = luigi.ListParameter()

	def requires(self):
		requirements = {}
		requirements['fillna_DB'] = FillnaDatabase()
		for i in self.list_ML:
			requirements[i] = Evaluate_ML(clf_name = i, wf_name = self.wf_name, repetitions = self.repetitions, folds = self.folds)
			requirements[i+'_threshold'] = ThresholdPoints(clf_or_score = i, wf_name = self.wf_name, repetitions = self.repetitions, folds = self.folds, list_ML = self.list_ML)
		return requirements

	def run(self):
		# First we open the results dictionary for every ML model in the workflow wf_name to determine
		# the best ML model
		auc_ml = {}
		for i in self.list_ML:
			with open(self.input()[i]["auc_results"].path, 'rb') as f:
				results_dict=pickle.load(f)
				auc_ml[i]=results_dict["avg_auc"]

		best_ml = max(auc_ml.keys(), key=(lambda k: auc_ml[k]))

		with open(self.input()[best_ml]["auc_results"].path, 'rb') as f:
			best_ml_results_dict=pickle.load(f)



		with open(self.output().path,'w') as f:
			f.write(f"Model name: {best_ml}")
			f.write(f"AUC: {best_ml_results_dict['avg_auc']}")
			f.write(f"AUC stderr: {best_ml_results_dict['avg_auc_stderr']}")
			f.write(f"Confidence Interval (95%): {best_ml_results_dict['95ci_low']}-{best_ml_results_dict['95ci_high']}")
			f.write("")
			with open(self.input()[best_ml+'_threshold'].path, 'rb') as f2:
				for line in f2.readlines():
					f.write(line)

			df_input = pd.read_pickle(self.input()["fillna_DB"]["pickle"].path)
			df_filtered = WF_info[self.wf_name]["filter_function"](df_input)
			label = WF_info[self.wf_name]["label_name"]
			features = WF_info[self.wf_name]["feature_list"]

			prerequisite = MDAFeatureImportances(clf_name = best_ml, wf_name = self.wf_name)
			luigi.build([prerequisite], local_scheduler = False)
			with open(prerequisite.output().path, 'rb') as f3:
				for line in f3.readlines():
					f.write(line)

	def output(self):
		try:
			os.makedirs(os.path.join(report_path,self.wf_name))
		except:
			pass
		return luigi.LocalTarget(os.path.join(report_path,self.wf_name,f"BestML_Model_report_{self.wf_name}.txt"))

class BestRSReport(luigi.Task):
	wf_name = luigi.Parameter()
	repetitions = luigi.IntParameter(default=10)
	folds = luigi.IntParameter(default=10)
	list_RS = luigi.ListParameter()

	def requires(self):
		requirements = {}
		for i in self.list_RS:
			requirements[i] = EvaluateRiskScore(score_name = i, wf_name = self.wf_name, repetitions = self.repetitions, folds=self.folds)
			requirements[i+'_threshold'] = AllThresholds(clf_or_score = i, wf_name = self.wf_name, repetitions = self.repetitions, folds=self.folds, list_RS=self.list_RS)
			requirements[i+'_hanley'] = ConfidenceIntervalHanleyRS(score_name = i, wf_name = self.wf_name)
		return requirements

	def run(self):
		# First we open the results dictionary for every ML model in the workflow wf_name to determine
		# the best risk score
		auc_rs = {}
		for i in self.list_RS:
			with open(self.input()[i]["auc_results"].path, 'rb') as f:
				results_dict=pickle.load(f)
				auc_rs[i]=results_dict["avg_auc"]

		best_rs = max(auc_rs.keys(), key=(lambda k: auc_rs[k]))

		with open(self.input()[best_rs]["auc_results"].path, 'rb') as f:
			best_rs_results_dict=pickle.load(f)

		with open(self.output().path,'w') as f:
			f.write(f"Score name: {RS_info[best_rs]['formal_name']}")
			f.write(f"AUC: {best_rs_results_dict['avg_auc']}")
			f.write(f"AUC stderr: {best_rs_results_dict['avg_auc_stderr']}")
			f.write(f"Confidence Interval Subsampling(95%): {best_rs_results_dict['95ci_low']}- {best_rs_results_dict['95ci_high']}")
			with open(self.input()[best_rs+'_hanley'].path, 'rb') as f2:
				for line in f2.readlines():
					f.write(line)
			f.write("")
			with open(self.input()[best_rs+'_threshold'].path, 'rb') as f3:
				for line in f3.readlines():
					f.write(line)
	def output(self):
		try:
			os.makedirs(os.path.join(report_path,self.wf_name))
		except:
			pass
		return luigi.LocalTarget(os.path.join(report_path,self.wf_name,f"BestRS_report_{self.wf_name}.txt"))

class MDAFeatureImportances(luigi.Task):
	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return FillnaDatabase()

	def run(self):
		with open(self.output().path,'w') as f:
			with contextlib.redirect_stdout(f):
				df_input = pd.read_pickle(self.input()["pickle"].path)
				df_filtered = WF_info[self.wf_name]["filter_function"](df_input)
				label = WF_info[self.wf_name]["label_name"]
				features = WF_info[self.wf_name]["feature_list"]

				(pi_cv, std_pi_cv) = mdaeli5_analysis(df_filtered, label, features, clf=ML_info[self.clf_name]["clf"],clf_name=self.clf_name)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"MDAeli5_Log_{self.wf_name}_{self.clf_name}.txt"))

class AllThresholds(luigi.Task):
	clf_or_score=luigi.Parameter()
	wf_name = luigi.Parameter()
	repetitions = luigi.IntParameter(default=10)
	folds = luigi.IntParameter(default=10)
	list_RS = luigi.ListParameter(default=[])
	list_ML = luigi.ListParameter(default=[])

	def requires(self):
		if (self.clf_or_score in self.list_RS):
			return EvaluateRiskScore(wf_name=self.wf_name, score_name = self.clf_or_score, repetitions=self.repetitions, folds=self.folds)
		elif (self.clf_or_score in self.list_ML):
			return Evaluate_ML(wf_name=self.wf_name, clf_name=self.clf_or_score, repetitions=self.repetitions, folds=self.folds)
		else:
			raise Exception(f"{self.clf_score} not in list_ristkscores or list_MLmodels")

	def run(self):
		with open(self.input()["pred_prob"].path, 'rb') as f:
			pred_prob=pickle.load(f)
		with open(self.input()["true_label"].path, 'rb') as f:
			true_label=pickle.load(f)

		list_thresholds = all_thresholds(pred_prob, true_label)

		with open(self.output().path,'w') as f:
			for i in list_thresholds:
				(threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = i
				f.write(f'Threshold: {threshold}')
				f.write(f'TP:{tprate*100:.1f} FP:{fprate*100:.1f} TN:{tnrate*100:.1f} FN:{fnrate*100:.1f}')
				f.write(f'Sensitivity:{sens*100:.1f} Specifity:{spec*100:.1f} Precision:{prec*100:.1f} NPRv:{nprv*100:.1f}')
	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}.txt"))

class AllTasks(luigi.Task):
	repetitions = luigi.IntParameter(default=10)
	folds = luigi.IntParameter(default=10)
	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	list_WF = luigi.ListParameter(default=list(WF_info.keys()))


	def __init__(self, *args, **kwargs):
		super(AllTasks, self).__init__(*args, **kwargs)

	def requires(self):

		for it_wf_name in self.list_WF:
			yield DescriptiveXLS(wf_name = it_wf_name)
			yield GraphsWF(wf_name = it_wf_name, list_ML=self.list_ML, list_RS=self.list_RS, repetitions = self.repetitions, folds = self.folds)
			yield BestMLModelReport(wf_name = it_wf_name, list_ML=self.list_ML, repetitions = self.repetitions, folds = self.folds)
			yield BestRSReport(wf_name = it_wf_name, list_RS=self.list_RS, repetitions = self.repetitions, folds = self.folds)
			yield AllModels_PairedTTest(wf_name = it_wf_name, list_ML=self.list_ML, list_RS=self.list_RS, repetitions = self.repetitions, folds = self.folds)
			for it_clf_name in self.list_ML:
				yield FinalModelAndHyperparameterResults(wf_name = it_wf_name, clf_name = it_clf_name)

	def run(self):
		with open(self.output().path,'w') as f:
			f.write("prueba")

	def output(self):
		return luigi.LocalTarget(os.path.join(log_path, "AllTask_Log.txt"))
