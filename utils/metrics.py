import numpy as np
import pandas as pd
import sklearn.metrics as sk_m
import sksurv.metrics as su_m
import matplotlib.pyplot as plt
from user_variables_info import dict_var


# class fairness_class_balance():
# 	def __init__(self):
# 		self.name = 'fairness_class_balance'
# 		self.optimization_sign = -1
# 		self.variance = None
#
# 	def __call__(self, df, df_train=None):
# 		for c in df.columns:
# 			if 'Fairness' in c:
# 				fairness_label = c
# 		fpr, tpr, thresholds = sk_m.roc_curve(df.loc[df['True Label'].notnull(), 'True Label'].astype(bool),
# 											  df.loc[df['True Label'].notnull(), 'Prediction'])
# 		id_max = (1-fpr+tpr).argmax()
# 		thr_opt = thresholds[id_max]
#
# 		fprs = []
# 		tprs = []
# 		for cat in df[fairness_label].unique():
# 			df_cat = df.loc[df[fairness_label] == cat]
# 			fpr = ((df_cat['Prediction'] >= thr_opt)&(df_cat['True Label']==0)).sum() / (df_cat['True Label']==0).sum()
# 			tpr = ((df_cat['Prediction'] >= thr_opt)&(df_cat['True Label']==1)).sum() / (df_cat['True Label']==1).sum()
# 			fprs.append(fpr)
# 			tprs.append(tpr)
# 		fprs = np.array(fprs)
# 		tprs = np.array(fprs)
#
# 		#Esto es muy arbitrario, pero para poner algo, es más importante el gráfico
# 		return tprs.max()+fprs.max()-tprs.min()-fprs.min()
#
#
# 	def plot(self, df, results, ax, score_name):
# 		nx = self.index % self.sidex
# 		ny = self.index // self.sidex
#
# 		for c in df.columns:
# 			if 'Fairness' in c:
# 				fairness_label = c
#
#
#
# 		fpr, tpr, thresholds = sk_m.roc_curve(df.loc[df['True Label'].notnull(), 'True Label'].astype(bool),
# 											  df.loc[df['True Label'].notnull(), 'Prediction'])
# 		id_max = (1-fpr+tpr).argmax()
# 		thr_opt = thresholds[id_max]
#
# 		index = -2
# 		for cat in df[fairness_label].unique():
# 			index+=2
# 			df_cat = df.loc[df[fairness_label] == cat]
# 			fpr_opt = ((df_cat['Prediction'] >= thr_opt)&(df_cat['True Label']==0)).sum() / (df_cat['True Label']==0).sum()
# 			tpr_opt = ((df_cat['Prediction'] >= thr_opt)&(df_cat['True Label']==1)).sum() / (df_cat['True Label']==1).sum()
# 			fpr, tpr, thresholds = sk_m.roc_curve(df_cat.loc[df_cat['True Label'].notnull(), 'True Label'].astype(bool),
# 												  df_cat.loc[df_cat['True Label'].notnull(), 'Prediction'])
# 			aucroc = sk_m.auc(fpr, tpr)
# 			ax[ny, nx].plot(fpr, tpr, lw=2, alpha=1, color=self.cmap(index) , label = f'{cat}: AUC ={aucroc:1.2f}')
# 			ax[ny, nx].scatter(fpr_opt, tpr_opt, color=self.cmap(index+1), edgecolors=self.cmap(index) , label = f'{cat}: Sens={tpr_opt:1.2f} Spec={1-fpr_opt:1.2f}')
# 		ax[ny, nx].legend(loc="lower right", fontsize = 10, title=fairness_label)
# 		ax[ny, nx].set_title(score_name)
# 		self.index+=1
#
# 	def figure(self, n, cmap = "tab20"):
# 		self.sidex = np.ceil(np.sqrt(n)).astype(int)
# 		self.sidey = np.ceil(n/self.sidex).astype(int)
# 		fig, ax = plt.subplots(self.sidey, self.sidex, figsize=(5*self.sidex,5*self.sidey), squeeze = False)
#
# 		for nx in range(self.sidex):
# 			for ny in range(self.sidey):
# 				ax[ny, nx].set_xlabel('1-Specificity', fontsize = 10)
# 				ax[ny, nx].set_ylabel('Sensitivity', fontsize = 10)
#
# 		self.index = 0
# 		self.cmap=plt.get_cmap(cmap)
#
# 		return fig, ax
#
# 	def save_figure(self, fig, ax, name):
# 		fig.savefig(name)
# 		return

class fairness_aucroc():
	def __init__(self):
		self.name = 'fairness aucroc'
		self.optimization_sign = -1
		self.variance = None

	def __call__(self, df, df_train=None):
		for c in df.columns:
			if 'Fairness' in c:
				fairness_label = c
		fpr, tpr, thresholds = sk_m.roc_curve(df.loc[df['True Label'].notnull(), 'True Label'].astype(bool),
											  df.loc[df['True Label'].notnull(), 'Prediction'])
		id_max = (1-fpr+tpr).argmax()
		thr_opt = thresholds[id_max]

		fprs = []
		tprs = []
		for cat in df[fairness_label].unique():
			df_cat = df.loc[df[fairness_label] == cat]
			fpr = ((df_cat['Prediction'] >= thr_opt)&(df_cat['True Label']==0)).sum() / (df_cat['True Label']==0).sum()
			tpr = ((df_cat['Prediction'] >= thr_opt)&(df_cat['True Label']==1)).sum() / (df_cat['True Label']==1).sum()
			fprs.append(fpr)
			tprs.append(tpr)
		fprs = np.array(fprs)
		tprs = np.array(fprs)

		#Esto es muy arbitrario, pero para poner algo, es más importante el gráfico
		return tprs.max()+fprs.max()-tprs.min()-fprs.min()


	def plot(self, df, results, ax, score_name):
		nx = self.index % self.sidex
		ny = self.index // self.sidex

		for c in df.columns:
			if 'Fairness' in c:
				fairness_label = c
		if 'Repetition' not in df.columns:
			df['Repetition']=1
		if 'Fold' not in df.columns:
			df['Fold']=1
		fpr, tpr, thresholds = sk_m.roc_curve(df.loc[df['True Label'].notnull(), 'True Label'].astype(bool),
											  df.loc[df['True Label'].notnull(), 'Prediction'])
		id_max = (1-fpr+tpr).argmax()
		thr_opt = thresholds[id_max]

		color_index = -2
		# for cat in df[fairness_label].unique():
		# 	color_index+=2
		# 	df_cat = df.loc[df[fairness_label] == cat]
		# 	fpr_opt = ((df_cat['Prediction'] >= thr_opt)&(df_cat['True Label']==0)).sum() / (df_cat['True Label']==0).sum()
		# 	tpr_opt = ((df_cat['Prediction'] >= thr_opt)&(df_cat['True Label']==1)).sum() / (df_cat['True Label']==1).sum()
		# 	fpr, tpr, thresholds = sk_m.roc_curve(df_cat.loc[df_cat['True Label'].notnull(), 'True Label'].astype(bool),
		# 										  df_cat.loc[df_cat['True Label'].notnull(), 'Prediction'])
		# 	aucroc = sk_m.auc(fpr, tpr)
		# 	ax[ny, nx].plot(fpr, tpr, lw=2, alpha=1, color=self.cmap(color_index) , label = f'{cat}: AUC ={aucroc:1.2f}')
		# 	ax[ny, nx].scatter(fpr_opt, tpr_opt, color=self.cmap(color_index+1), edgecolors=self.cmap(color_index) , label = f'{cat}: Sens={tpr_opt:1.2f} Spec={1-fpr_opt:1.2f}')
		for cat in df[fairness_label].unique():
			color_index+=2
			df_cat = df.loc[df[fairness_label] == cat]

			roc_divisions = 1001
			fpr_va = np.linspace(0,1,roc_divisions)
			tpr_va = np.zeros(roc_divisions)
			aucroc_list = []
			# fpr_opt_list = []
			# tpr_opt_list = []
			for rep in df_cat['Repetition'].unique():
				for fold in df_cat['Fold'].unique():
					true_label = df_cat.loc[df_cat['True Label'].notnull()&(df_cat['Repetition']==rep)&(df_cat['Fold']==fold), 'True Label'].astype(bool).values
					pred_prob = df_cat.loc[df_cat['True Label'].notnull()&(df_cat['Repetition']==rep)&(df_cat['Fold']==fold), 'Prediction'].values
					if len(np.unique(true_label))==2:
						fpr, tpr, thresholds = sk_m.roc_curve(true_label,pred_prob)
						tpr_va += np.interp(fpr_va, fpr, tpr)
						aucroc_list.append(sk_m.auc(fpr, tpr))
			tpr_va = tpr_va / (len(aucroc_list))

			aucroc_list = np.array(aucroc_list)
			if len(aucroc_list)>1:
				n_repfolds = len(df_cat['Repetition'].unique())*len(df_cat['Fold'].unique())
				n_folds = len(df_cat['Fold'].unique())
				aucroc_stderr = aucroc_list.std(ddof = 1)*np.sqrt(1/n_repfolds+1/(n_folds-1))
			else:
				m = (df_cat['True Label']==0).sum()
				n = (df_cat['True Label']==1).sum()
				auc = aucroc_list[0]
				pxxy = auc/(2-auc)
				pxyy = 2*auc**2/(1+auc)
				aucroc_stderr = np.sqrt((auc*(1-auc)+(m-1)*(pxxy-auc**2)+(n-1)*(pxyy-auc**2))/(m*n))
			c=1.96
			aucroc_95ci_low = aucroc_list.mean() - c*aucroc_stderr
			aucroc_95ci_high = aucroc_list.mean() + c*aucroc_stderr


			fpr_opt = ((df_cat['Prediction'] >= thr_opt)&(df_cat['True Label']==0)).sum() / (df_cat['True Label']==0).sum()
			tpr_opt = ((df_cat['Prediction'] >= thr_opt)&(df_cat['True Label']==1)).sum() / (df_cat['True Label']==1).sum()
			# fpr, tpr, thresholds = sk_m.roc_curve(df_cat.loc[df_cat['True Label'].notnull(), 'True Label'].astype(bool),
			# 									  df_cat.loc[df_cat['True Label'].notnull(), 'Prediction'])
			# aucroc = sk_m.auc(fpr, tpr)
			fpr, tpr, aucroc = fpr_va, tpr_va, aucroc_list.mean()
			ax[ny, nx].plot(fpr, tpr, lw=2, alpha=1, color=self.cmap(color_index) , label = f'{cat}: AUC ={aucroc:1.2f} ({aucroc_95ci_low:1.2f} - {aucroc_95ci_high:1.2f})')
			ax[ny, nx].scatter(fpr_opt, tpr_opt, color=self.cmap(color_index+1), edgecolors=self.cmap(color_index) , label = f'{cat}: Sens={tpr_opt:1.2f} Spec={1-fpr_opt:1.2f}')
		ax[ny, nx].legend(loc="lower right", fontsize = 10, title=fairness_label)
		ax[ny, nx].set_title(score_name)
		self.index+=1

	def figure(self, n, cmap = "tab20"):
		self.sidex = np.ceil(np.sqrt(n)).astype(int)
		self.sidey = np.ceil(n/self.sidex).astype(int)
		fig, ax = plt.subplots(self.sidey, self.sidex, figsize=(5*self.sidex,5*self.sidey), squeeze = False)

		for nx in range(self.sidex):
			for ny in range(self.sidey):
				ax[ny, nx].set_xlabel('1-Specificity', fontsize = 10)
				ax[ny, nx].set_ylabel('Sensitivity', fontsize = 10)
				ax[ny, nx].set_xlim([-0.05, 1.05])
				ax[ny, nx].set_ylim([-0.05, 1.05])

		self.index = 0
		self.cmap=plt.get_cmap(cmap)

		return fig, ax

	def save_figure(self, fig, ax, name):
		fig.savefig(name)
		return

class fairness_aucpr():
	def __init__(self):
		self.name = 'fairness aucpr'
		self.optimization_sign = -1
		self.variance = None

	def __call__(self, df, df_train=None):
		for c in df.columns:
			if 'Fairness' in c:
				fairness_label = c
		prec, recall, thresholds = sk_m.precision_recall_curve(df.loc[df['True Label'].notnull(), 'True Label'].astype(bool),
															   df.loc[df['True Label'].notnull(), 'Prediction'])
		thresholds = np.array([min(thresholds.min()-1e-20, 0.)] + list(thresholds))
		id_max = (2/(1/prec+1/recall)).argmax()
		thr_opt = thresholds[id_max]

		precs = []
		recalls = []
		for cat in df[fairness_label].unique():
			df_cat = df.loc[df[fairness_label] == cat]
			prec = ((df_cat['Prediction'] >= thr_opt)&(df_cat['True Label']==1)).sum() / (df_cat['Prediction'] >= thr_opt).sum()
			recall = ((df_cat['Prediction'] >= thr_opt)&(df_cat['True Label']==1)).sum() / (df_cat['True Label']==1).sum()
			precs.append(prec)
			recalls.append(recall)
		precs = np.array(precs)
		recalls = np.array(recalls)

		#Esto es muy arbitrario, pero para poner algo, es más importante el gráfico
		return recalls.max()+precs.max()-recalls.min()-precs.min()


	def plot(self, df, results, ax, score_name):
		nx = self.index % self.sidex
		ny = self.index // self.sidex

		for c in df.columns:
			if 'Fairness' in c:
				fairness_label = c
		if 'Repetition' not in df.columns:
			df['Repetition']=1
		if 'Fold' not in df.columns:
			df['Fold']=1
		prec, recall, thresholds = sk_m.precision_recall_curve(df.loc[df['True Label'].notnull(), 'True Label'].astype(bool),
															   df.loc[df['True Label'].notnull(), 'Prediction'])
		thresholds = np.array([min(thresholds.min()-1e-20, 0.)] + list(thresholds))
		id_max = (2/(1/prec+1/recall)).argmax()
		thr_opt = thresholds[id_max]

		color_index = -2

		for cat in df[fairness_label].unique():
			color_index+=2
			df_cat = df.loc[df[fairness_label] == cat]

			recall_divisions = 1001
			recall_va = np.linspace(0,1,recall_divisions)
			prec_va = np.zeros(recall_divisions)
			aucpr_list = []
			if 'Repetition' not in df.columns:
				df_cat['Repetition']=1
			if 'Fold' not in df.columns:
				df_cat['Fold']=1
			for rep in df_cat['Repetition'].unique():
				for fold in df_cat['Fold'].unique():
					true_label = df_cat.loc[df_cat['True Label'].notnull()&(df_cat['Repetition']==rep)&(df_cat['Fold']==fold), 'True Label'].astype(bool).values
					pred_prob = df_cat.loc[df_cat['True Label'].notnull()&(df_cat['Repetition']==rep)&(df_cat['Fold']==fold), 'Prediction'].values
					if len(np.unique(true_label))==2:
						prec, recall, thresholds = sk_m.precision_recall_curve(true_label,pred_prob)
						prec_va += np.interp(recall_va, recall[::-1], prec[::-1])
						aucpr_list.append(sk_m.auc(recall, prec))
			prec_va = prec_va / (len(aucpr_list))

			aucpr_list = np.array(aucpr_list)
			if len(aucpr_list)>1:
				n_repfolds = len(df_cat['Repetition'].unique())*len(df_cat['Fold'].unique())
				n_folds = len(df_cat['Fold'].unique())
				aucpr_stderr = aucpr_list.std(ddof = 1)*np.sqrt(1/n_repfolds+1/(n_folds-1))
			else:
				m = (df_cat['True Label']==0).sum()
				n = (df_cat['True Label']==1).sum()
				auc = aucpr_list[0]
				pxxy = auc/(2-auc)
				pxyy = 2*auc**2/(1+auc)
				aucpr_stderr = np.sqrt((auc*(1-auc)+(m-1)*(pxxy-auc**2)+(n-1)*(pxyy-auc**2))/(m*n))
			c=1.96
			aucpr_95ci_low = aucpr_list.mean() - c*aucpr_stderr
			aucpr_95ci_high = aucpr_list.mean() + c*aucpr_stderr

			prec_opt = ((df_cat['Prediction'] >= thr_opt)&(df_cat['True Label']==1)).sum() / (df_cat['Prediction'] >= thr_opt).sum()
			recall_opt = ((df_cat['Prediction'] >= thr_opt)&(df_cat['True Label']==1)).sum() / (df_cat['True Label']==1).sum()

			prec, recall, aucpr = prec_va, recall_va, aucpr_list.mean()
			ax[ny, nx].plot(recall, prec, lw=2, alpha=1, color=self.cmap(color_index) , label = f'{cat}: AUC ={aucpr:1.2f} ({aucpr_95ci_low:1.2f} - {aucpr_95ci_high:1.2f})')
			ax[ny, nx].scatter(recall_opt, prec_opt, color=self.cmap(color_index+1), edgecolors=self.cmap(color_index) , label = f'{cat}: Sens={recall_opt:1.2f} Prec={prec_opt:1.2f}')

		ax[ny, nx].legend(loc="upper right", fontsize = 10, title=fairness_label)
		ax[ny, nx].set_title(score_name)
		self.index+=1

	def figure(self, n, cmap = "tab20"):
		self.sidex = np.ceil(np.sqrt(n)).astype(int)
		self.sidey = np.ceil(n/self.sidex).astype(int)
		fig, ax = plt.subplots(self.sidey, self.sidex, figsize=(5*self.sidex,5*self.sidey), squeeze = False)

		for nx in range(self.sidex):
			for ny in range(self.sidey):
				ax[ny, nx].set_xlabel('Recall (Sensitivity)', fontsize = 10)
				ax[ny, nx].set_ylabel('Precision', fontsize = 10)
				ax[ny, nx].set_xlim([-0.05, 1.05])
				ax[ny, nx].set_ylim([-0.05, 1.05])

		self.index = 0
		self.cmap=plt.get_cmap(cmap)

		return fig, ax

	def save_figure(self, fig, ax, name):
		fig.savefig(name)
		return

class aucroc():
	def __init__(self):
		self.name = 'aucroc'
		self.optimization_sign = 1
		self.variance = None


	def __call__(self, df, df_train=None):
		m = (df['True Label']==0).sum()
		n = (df['True Label']==1).sum()
		auc = sk_m.roc_auc_score(df.loc[df['True Label'].notnull(), 'True Label'].astype(bool),df.loc[df['True Label'].notnull(), 'Prediction'])
		pxxy = auc/(2-auc)
		pxyy = 2*auc**2/(1+auc)
		self.variance = (auc*(1-auc)+(m-1)*(pxxy-auc**2)+(n-1)*(pxyy-auc**2))/(m*n)
		self.std_error = np.sqrt(self.variance)
		return auc

	def plot(self, df, results, ax, score_name):
		roc_divisions = 1001
		fpr_va = np.linspace(0,1,roc_divisions)
		tpr_va = np.zeros(roc_divisions)
		if 'Repetition' not in df.columns:
			df['Repetition']=1
		if 'Fold' not in df.columns:
			df['Fold']=1
		for rep in df['Repetition'].unique():
			for fold in df['Fold'].unique():
				true_label = df.loc[df['True Label'].notnull()&(df['Repetition']==rep)&(df['Fold']==fold), 'True Label'].astype(bool).values
				pred_prob = df.loc[df['True Label'].notnull()&(df['Repetition']==rep)&(df['Fold']==fold), 'Prediction'].values
				fpr, tpr, thresholds = sk_m.roc_curve(true_label,pred_prob)
				tpr_va += np.interp(fpr_va, fpr, tpr)
		tpr_va = tpr_va / (len(df['Repetition'].unique())*len(df['Fold'].unique()))


		plt.plot(fpr_va, tpr_va, lw=2, alpha=1, color=self.cmap(self.color_index) , label = f'{score_name}: AUC ={results["avg_aucroc"]:1.2f} ({results["aucroc_95ci_low"]:1.2f}-{results["aucroc_95ci_high"]:1.2f})' )
		self.color_index+=1

	def figure(self, n, cmap = "tab10"):
		fig, ax = plt.subplots(figsize=(10,10))
		ax.set_xlim([-0.05, 1.05])
		ax.set_ylim([-0.05, 1.05])

		ax.set_xlabel('1-specificity', fontsize = 15)
		ax.set_ylabel('sensitivity', fontsize = 15)

		self.color_index = 0
		self.cmap=plt.get_cmap(cmap)

		return fig, ax

	def save_figure(self, fig, ax, name):
		ax.legend(loc="lower right", fontsize = 10)
		fig.savefig(name)
		return
class aucroc_pooled():
	def __init__(self):
		self.name = 'aucroc_pooled'
		self.optimization_sign = 1
		self.variance = None


	def __call__(self, df, df_train=None):
		m = (df['True Label']==0).sum()
		n = (df['True Label']==1).sum()
		auc = sk_m.roc_auc_score(df.loc[df['True Label'].notnull(), 'True Label'].astype(bool),df.loc[df['True Label'].notnull(), 'Prediction'])
		pxxy = auc/(2-auc)
		pxyy = 2*auc**2/(1+auc)
		self.variance = (auc*(1-auc)+(m-1)*(pxxy-auc**2)+(n-1)*(pxyy-auc**2))/(m*n)
		self.std_error = np.sqrt(self.variance)
		return auc

	def plot(self, df, results, ax, score_name):
		true_label = df.loc[df['True Label'].notnull(), 'True Label'].astype(bool).values
		pred_prob = df.loc[df['True Label'].notnull(), 'Prediction'].values
		fpr, tpr, thresholds = sk_m.roc_curve(true_label,pred_prob)

		plt.plot(fpr, tpr, lw=2, alpha=1, color=self.cmap(self.color_index) , label = f'{score_name}: AUC ={results["avg_aucroc"]:1.2f} ({results["aucroc_95ci_low"]:1.2f}-{results["aucroc_95ci_high"]:1.2f})' )
		self.color_index+=1

	def figure(self, n, cmap = "tab10"):
		fig, ax = plt.subplots(figsize=(10,10))
		ax.set_xlim([-0.05, 1.05])
		ax.set_ylim([-0.05, 1.05])

		ax.set_xlabel('1-specificity', fontsize = 15)
		ax.set_ylabel('sensitivity', fontsize = 15)

		self.color_index = 0
		self.cmap=plt.get_cmap(cmap)

		return fig, ax

	def save_figure(self, fig, ax, name):
		ax.legend(loc="lower right", fontsize = 10)
		fig.savefig(name)
		return

class aucpr():
	def __init__(self):
		self.name = 'aucpr'
		self.optimization_sign = 1
		self.variance = None

	def __call__(self, df, df_train=None):
		m = (df['True Label']==0).sum()
		n = (df['True Label']==1).sum()
		auc = sk_m.average_precision_score(df.loc[df['True Label'].notnull(), 'True Label'].astype(bool),df.loc[df['True Label'].notnull(), 'Prediction'])
		pxxy = auc/(2-auc)
		pxyy = 2*auc**2/(1+auc)
		self.variance = (auc*(1-auc)+(m-1)*(pxxy-auc**2)+(n-1)*(pxyy-auc**2))/(m*n)
		self.std_error = np.sqrt(self.variance)
		return auc

	def plot(self, df, results, ax, score_name):
		# pr_divisions = 1001
		# prec_va = np.linspace(0,1,pr_divisions)
		# recall_va = np.zeros(pr_divisions)
		# if 'Repetition' not in df.columns:
		# 	df['Repetition']=1
		# if 'Fold' not in df.columns:
		# 	df['Fold']=1
		# for rep in df['Repetition'].unique():
		# 	for fold in df['Fold'].unique():
		# 		true_label = df.loc[df['True Label'].notnull()&(df['Repetition']==rep)&(df['Fold']==fold), 'True Label'].astype(bool).values
		# 		pred_prob = df.loc[df['True Label'].notnull()&(df['Repetition']==rep)&(df['Fold']==fold), 'Prediction'].values
		# 		prec, recall, thresholds = sk_m.precision_recall_curve(true_label,pred_prob)
		#
		# 		recall_va += np.interp(prec_va, prec, recall)
		# recall_va = recall_va / (len(df['Repetition'].unique())*len(df['Fold'].unique()))

		recall_divisions = 1001
		recall_va = np.linspace(0,1,recall_divisions)
		prec_va = np.zeros(recall_divisions)
		if 'Repetition' not in df.columns:
			df['Repetition']=1
		if 'Fold' not in df.columns:
			df['Fold']=1
		for rep in df['Repetition'].unique():
			for fold in df['Fold'].unique():
				true_label = df.loc[df['True Label'].notnull()&(df['Repetition']==rep)&(df['Fold']==fold), 'True Label'].astype(bool).values
				pred_prob = df.loc[df['True Label'].notnull()&(df['Repetition']==rep)&(df['Fold']==fold), 'Prediction'].values
				prec, recall, thresholds = sk_m.precision_recall_curve(true_label,pred_prob)

				prec_va += np.interp(recall_va, recall[::-1], prec[::-1])
		prec_va = prec_va / (len(df['Repetition'].unique())*len(df['Fold'].unique()))

		plt.plot(recall_va, prec_va, lw=2, alpha=1, color=self.cmap(self.color_index) , label = f'{score_name}: AUC ={results["avg_aucpr"]:1.2f} ({results["aucpr_95ci_low"]:1.2f}-{results["aucpr_95ci_high"]:1.2f})' )
		self.color_index+=1

	def figure(self, n, cmap = "tab10"):
		fig, ax = plt.subplots(figsize=(10,10))
		ax.set_xlim([-0.05, 1.05])
		ax.set_ylim([-0.05, 1.05])

		ax.set_xlabel('sensitivity (recall)', fontsize = 15)
		ax.set_ylabel('precision', fontsize = 15)

		self.color_index = 0
		self.cmap=plt.get_cmap(cmap)

		return fig, ax

	def save_figure(self, fig, ax, name):
		ax.legend(loc="upper right", fontsize = 10)
		fig.savefig(name)
		return

class rmse():
	def __init__(self):
		self.name = 'rmse'
		self.optimization_sign = -1
		self.variance = None
		self.max_x = None
		self.min_x = None
		self.max_y = None
		self.min_y = None
		self.index = 0

	def __call__(self, df, df_train=None):
		diff = df.loc[df['True Label'].notnull(), 'True Label'] - df.loc[df['True Label'].notnull(), 'Prediction']
		rmse = np.sqrt((diff**2).mean())

		self.variance = 0
		self.std_error = np.sqrt(self.variance)
		return rmse

	def plot(self, df, results, ax, score_name):
		nx = self.index % self.sidex
		ny = self.index // self.sidex
		ax[ny, nx].scatter(df.loc[df['True Label'].notnull(), 'True Label'], df.loc[df['True Label'].notnull(), 'Prediction'],
							color=self.cmap(self.index%10), label = f'{score_name}: RMSE ={results["avg_rmse"]:1.2f} ({results["rmse_95ci_low"]:1.2f}-{results["rmse_95ci_high"]:1.2f})')
		ax[ny, nx].plot([0, df.loc[df['True Label'].notnull(), 'True Label'].max()], [0, df.loc[df['True Label'].notnull(), 'True Label'].max()], c='black', linestyle=':')
		ax[ny, nx].legend(loc="lower right", fontsize = 10)
		self.index+=1

	def figure(self, n, cmap = "tab10"):
		self.sidex = np.ceil(np.sqrt(n)).astype(int)
		self.sidey = np.ceil(n/self.sidex).astype(int)
		fig, ax = plt.subplots(self.sidey, self.sidex, figsize=(5*self.sidex,5*self.sidey), squeeze = False)

		for nx in range(self.sidex):
			for ny in range(self.sidey):
				ax[ny, nx].set_xlabel('True Value', fontsize = 15)
				ax[ny, nx].set_ylabel('Predicted Value', fontsize = 15)

		self.index = 0
		self.cmap=plt.get_cmap(cmap)

		return fig, ax

	def save_figure(self, fig, ax, name):
		fig.savefig(name)
		return


class r2():
# 	def __init__(self):
# 		self.name = 'r2'
# 		self.optimization_sign = 1
# 		self.variance = None
#
# 	def __call__(self, df, df_train=None):
# 		diff = df.loc[df['True Label'].notnull(), 'True Label'] - df.loc[df['True Label'].notnull(), 'Prediction']
# 		r2 = 1-(diff**2).mean()/df.loc[df['True Label'].notnull(), 'True Label'].var()
#
# 		self.variance = 0
# 		self.std_error = np.sqrt(self.variance)
# 		return r2
#
# 	def plot(self, df, results, ax, score_name):
#
# 		ax.barh(f'{score_name}', results["avg_r2"], color=self.cmap(self.color_index),
# 			   label = f'{score_name}: R2 ={results["avg_r2"]:1.2f} ({results["r2_95ci_low"]:1.2f}-{results["r2_95ci_high"]:1.2f})')
# 		self.color_index+=1
#
# 	def figure(self, n, cmap = "tab10"):
# 		fig, ax = plt.subplots(figsize=(10,10))
#
#
# 		self.color_index = 0
# 		self.cmap=plt.get_cmap(cmap)
#
# 		return fig, ax
#
# 	def save_figure(self, fig, ax, name):
# 		ax.legend(loc="lower right", fontsize = 10)
# 		ax.set_xlim([-0.05, 1.05])
#
# 		fig.savefig(name)
# 		return
	def __init__(self):
		self.name = 'mse_r2'
		self.optimization_sign = -1
		self.variance = None

	def __call__(self, df, df_train=None):
		diff = df.loc[df['True Label'].notnull(), 'True Label'] - df.loc[df['True Label'].notnull(), 'Prediction']

		self.variance = 0
		self.std_error = np.sqrt(self.variance)
		return (diff**2).mean()

	def plot(self, df, results, ax, score_name):
		var_tl = df['True Label'].var()

		avg_r2 = 1- results["avg_mse_r2"]/var_tl
		r2_95ci_low = 1- results["mse_r2_95ci_low"]/var_tl
		r2_95ci_high = 1- results["mse_r2_95ci_high"]/var_tl

		ax.barh(f'{score_name}', avg_r2, color=self.cmap(self.color_index),
			   label = f'{score_name}: R2 ={avg_r2:.3g} ({r2_95ci_low:.3g}-{r2_95ci_high:.3g})')
		self.color_index+=1

	def figure(self, n, cmap = "tab10"):
		fig, ax = plt.subplots(figsize=(10,10))
		self.color_index = 0
		self.cmap=plt.get_cmap(cmap)

		return fig, ax

	def save_figure(self, fig, ax, name):
		ax.legend(loc="lower right", fontsize = 10)
		ax.set_xlim([-0.05, 1.05])

		fig.savefig(name)
		return


class cindex_censored():
	def __init__(self):
		self.name = 'concordance_index_censored'
		self.optimization_sign = 1
		self.variance = None

	def __call__(self, df, df_train=None):
		cindex = su_m.concordance_index_censored(df.loc[df['True Label'].notnull(), 'True Label'].astype(bool),
												df.loc[df['True Label'].notnull(), 'Time'],
												df.loc[df['True Label'].notnull(), 'Prediction'])[0]
		self.variance = 0
		self.std_error = np.sqrt(self.variance)
		return cindex

	def plot(self, df, results, ax, score_name):

		ax.barh(f'{score_name}', results["avg_concordance_index_censored"], color=self.cmap(self.color_index),
			   label = f'{score_name}: C-index ={results["avg_concordance_index_censored"]:1.2f} ({results["concordance_index_censored_95ci_low"]:1.2f}-{results["concordance_index_censored_95ci_high"]:1.2f})')
		self.color_index+=1

	def figure(self, n, cmap = "tab10"):
		fig, ax = plt.subplots(figsize=(10,10))


		self.color_index = 0
		self.cmap=plt.get_cmap(cmap)

		return fig, ax

	def save_figure(self, fig, ax, name):
		ax.legend(loc="lower right", fontsize = 10)
		ax.set_xlim([-0.05, 1.05])

		fig.savefig(name)
		return
#
# class cindex_ipcw():
# 	def __init__(self):
# 		self.name = 'concordance_index_ipcw'
# 		self.optimization_sign = 1
# 		self.variance = None
#
# 	def __call__(self, df, df_train):
# 		cindex = su_m.concordance_index_ipcw(df.loc[df['True Label'].notnull(), 'True Label'],
# 												df.loc[df['True Label'].notnull(), 'Time Label'],
# 												df.loc[df['True Label'].notnull(), 'Prediction'])
# 		self.variance = 0
# 		self.std_error = np.sqrt(self.variance)
# 		return cindex
#
# 	def plot(self, df, results, ax, score_name):
#
# 		ax.barh(f'{score_name}', results["avg_concordance_index_ipcw"], color=self.cmap(self.color_index),
# 			   label = f'{score_name}: C-index ={results["avg_concordance_index_ipcw"]:1.2f} ({results["concordance_index_ipcw_95ci_low"]:1.2f}-{results["concordance_index_ipcw_95ci_high"]:1.2f})')
# 		self.color_index+=1
#
# 	def figure(self, n, cmap = "tab10"):
# 		fig, ax = plt.subplots(figsize=(10,10))
#
#
# 		self.color_index = 0
# 		self.cmap=plt.get_cmap(cmap)
#
# 		return fig, ax
#
# 	def save_figure(self, fig, ax, name):
# 		ax.legend(loc="lower right", fontsize = 10)
# 		ax.set_xlim([-0.05, 1.05])
#
# 		fig.savefig(name)
# 		return
#
#
class cumulative_auc():
	def __init__(self):
		self.name = 'cd auc'
		self.optimization_sign = 1
		self.variance = None

	def __call__(self, df, df_train = None):
		if df_train is None:
			gt = df.loc[df['True Label'].notnull(),['True Label', 'Time']].to_records(index = False).astype([('Events',bool), ('Times', float)])
			pp = df.loc[df['True Label'].notnull(), 'Prediction']
			ti = np.sort(df.loc[df['True Label'].notnull(), 'Time'].unique())
			min_time = df.loc[df['True Label'], 'Time'].min()
			ti = ti[ti>min_time][:-1]
			cd_auc = su_m.cumulative_dynamic_auc(gt, gt, pp, ti)[1]
		else:
			# All the data is used to compute the censoring survival function
			total = pd.concat([df.loc[df['True Label'].notnull(),['True Label', 'Time']],
							   df_train.loc[df_train['True Label'].notnull(),['True Label', 'Time']]]).to_records(index = False).astype([('Events',bool), ('Times', float)])
			gt = df.loc[df['True Label'].notnull(),['True Label', 'Time']].to_records(index = False)
			pp = df.loc[df['True Label'].notnull(), 'Prediction']
			ti = np.sort(df.loc[df['True Label'].notnull(), 'Time'].unique())
			min_time = df.loc[df['True Label'].astype(bool), 'Time'].min()
			ti = ti[ti>min_time][:-1]
			cd_auc = su_m.cumulative_dynamic_auc(total, gt, pp, ti)[1]

		self.variance = 0
		self.std_error = np.sqrt(self.variance)
		return cd_auc

	def plot(self, df, results, ax, score_name):

		ev = df.loc[df['True Label'].notnull(), 'True Label'].astype(bool)
		pp = df.loc[df['True Label'].notnull(), 'Prediction']
		fu = df.loc[df['True Label'].notnull(), 'Time']
		ti = np.sort(fu.unique())
		min_time = df.loc[df['True Label'].astype(bool), 'Time'].min()
		ti = ti[ti>min_time][:-1]

		auc_list = []
		ti_list = []
		for t in ti:
			ev_t = ev & (fu < t)
			pp_t = pp[ev_t | (fu >= t)]
			ev_t = ev_t[ev_t | (fu >= t)]
			ti_list.append(t)
			auc_list.append(sk_m.roc_auc_score(ev_t,pp_t))

		ax.plot(ti_list, auc_list, c=self.cmap(self.color_index), label = f'{score_name}: C/D AUC ={results["avg_cd auc"]:1.2f} ({results["cd auc_95ci_low"]:1.2f}-{results["cd auc_95ci_high"]:1.2f})')
		self.color_index+=1

	def figure(self, n, cmap = "tab10"):
		fig, ax = plt.subplots(figsize=(12,6))

		self.color_index = 0
		self.cmap=plt.get_cmap(cmap)

		return fig, ax

	def save_figure(self, fig, ax, name):
		ax.legend(loc="lower right", fontsize = 10)
		ax.set_ylim([-0.05, 1.05])

		fig.savefig(name)
		return


metrics_list = {'aucroc': aucroc,
				'aucroc_pooled': aucroc_pooled,
				'aucpr':aucpr,
				'rmse':rmse,
				'r2':r2,
				'cindex':cindex_censored,
				'cd auc':cumulative_auc,
				'fairness_aucroc': fairness_aucroc,
				'fairness_aucpr': fairness_aucpr}
