import numpy as np
import pandas as pd
import sklearn.metrics as sk_m
import sksurv.metrics as su_m
import matplotlib.pyplot as plt


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
		pr_divisions = 1001
		prec_va = np.linspace(0,1,pr_divisions)
		recall_va = np.zeros(pr_divisions)
		if 'Repetition' not in df.columns:
			df['Repetition']=1
		if 'Fold' not in df.columns:
			df['Fold']=1
		for rep in df['Repetition'].unique():
			for fold in df['Fold'].unique():
				true_label = df.loc[df['True Label'].notnull()&(df['Repetition']==rep)&(df['Fold']==fold), 'True Label'].astype(bool).values
				pred_prob = df.loc[df['True Label'].notnull()&(df['Repetition']==rep)&(df['Fold']==fold), 'Prediction'].values
				prec, recall, thresholds = sk_m.precision_recall_curve(true_label,pred_prob)

				recall_va += np.interp(prec_va, prec, recall)
		recall_va = recall_va / (len(df['Repetition'].unique())*len(df['Fold'].unique()))

		plt.plot(recall_va, prec_va, lw=2, alpha=1, color=self.cmap(self.color_index) , label = f'{score_name}: AUC ={results["avg_aucpr"]:1.2f} ({results["aucpr_95ci_low"]:1.2f}-{results["aucpr_95ci_high"]:1.2f})' )
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
	def __init__(self):
		self.name = 'r2'
		self.optimization_sign = 1
		self.variance = None

	def __call__(self, df, df_train=None):
		diff = df.loc[df['True Label'].notnull(), 'True Label'] - df.loc[df['True Label'].notnull(), 'Prediction']
		r2 = 1-(diff**2).mean()/df.loc[df['True Label'].notnull(), 'True Label'].var()

		self.variance = 0
		self.std_error = np.sqrt(self.variance)
		return r2

	def plot(self, df, results, ax, score_name):

		ax.barh(f'{score_name}', results["avg_r2"], color=self.cmap(self.color_index),
			   label = f'{score_name}: R2 ={results["avg_r2"]:1.2f} ({results["r2_95ci_low"]:1.2f}-{results["r2_95ci_high"]:1.2f})')
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
			min_time = df.loc[df['True Label'], 'Time'].min()
			ti = ti[ti>min_time][:-1]
			cd_auc = su_m.cumulative_dynamic_auc(total, gt, pp, ti)[1]

		self.variance = 0
		self.std_error = np.sqrt(self.variance)
		return cd_auc

	def plot(self, df, results, ax, score_name):

		ev = df.loc[df['True Label'].notnull(), 'True Label']
		pp = df.loc[df['True Label'].notnull(), 'Prediction']
		fu = df.loc[df['True Label'].notnull(), 'Time']
		ti = np.sort(fu.unique())
		min_time = df.loc[df['True Label'], 'Time'].min()
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
				'aucpr':aucpr,
				'rmse':rmse,
				'r2':r2,
				'cindex':cindex_censored,
				'cd auc':cumulative_auc}
