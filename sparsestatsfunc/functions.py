#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import t, f
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.cross_decomposition import PLSRegression

from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

# suppress console because of weird permission around r
import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)

from sparsestatsfunc.cynumstats import cy_lin_lstsqr_mat_residual, cy_lin_lstsqr_mat, fast_se_of_slope, tval_fast

stats = importr('stats')
base = importr('base')
spls = importr('spls')
utils = importr('utils')

numpy2ri.activate()

class permute_model_parallel():
	"""
	Calculates PLS metrics and significance using sklearn PLSRegression and joblib locky parallelization.
	"""
	def __init__(self, n_jobs, n_permutations = 10000):
		self.n_jobs = n_jobs
		self.n_permutations = n_permutations
	def zscaler_XY(self, X, y, axis = 0, ddof = 1, w_mean = True, scale_x = True, scale_y = True):
		"""
		Applies scaling to X and y, return means and std regardless
		"""
		X_ = np.zeros_like(X)
		X_[:] = np.copy(X)
		X_mean_ = np.nanmean(X_, axis)
		X_std_ = np.nanstd(X_, axis = axis, ddof = ddof)
		Y_ = np.zeros_like(y)
		Y_[:] = np.copy(y)
		Y_mean_ = np.nanmean(Y_, axis)
		Y_std_ = np.nanstd(Y_, axis = axis, ddof = ddof)
		if w_mean:
			X_ -= X_mean_
			Y_ -= Y_mean_
		if scale_x:
			X_ /= X_std_
		if scale_y:
			Y_ /= Y_std_
		return(X_, Y_, X_mean_, Y_mean_, X_std_, Y_std_)
	def index_perm(self, unique_arr, arr, variable, within_group = True):
		"""
		Shuffles an array within group (within_group = True) or the groups (within_group = False)
		"""
		if within_group:
			perm_u = unique_arr
		else:
			perm_u = np.random.permutation(unique_arr)
		out = []
		for unique in perm_u:
			if within_group:
				out.append(np.random.permutation(variable[unique == arr]))
			else:
				out.append(variable[unique == arr])
		return np.concatenate(out)
	def check_variables(self, X_Train, Y_Train, X_Test, Y_Test):
		X_Train, Y_Train, X_Train_mean, Y_Train_mean, X_Train_std, Y_Train_std = self.zscaler_XY(X = X_Train, y = Y_Train)
		X_Test, Y_Test, X_Test_mean, Y_Test_mean, X_Test_std, Y_Test_std = self.zscaler_XY(X = X_Test, y = Y_Test)
		has_issue = False
		if np.sum((X_Train_std == 0)*1) != 0:
			print("Warning: zero standard deviation predictors detected in Training data. Printing index array")
			print((X_Train_std != 0)*1)
			has_issue = True
		if np.sum((X_Test_std == 0)*1) != 0:
			print("Warning: zero standard deviation predictors detect in Testing data. Printing index array")
			print((X_Test_std != 0)*1)
			has_issue = True
		outindex = (X_Test_std != 0) * (X_Train_std != 0)
		return(outindex)
	def fit_model(self, X_Train, Y_Train, X_Test, Y_Test, n_components, group_train):
		"""
		Calcules R2_train, R2_train_components, Q2_train, Q2_train_components, R2_test, R2_test_components for overal model and targets
		"""
		X_Train, Y_Train, X_Train_mean, Y_Train_mean, X_Train_std, Y_Train_std = self.zscaler_XY(X = X_Train, y = Y_Train)
		X_Test, Y_Test, X_Test_mean, Y_Test_mean, X_Test_std, Y_Test_std = self.zscaler_XY(X = X_Test, y = Y_Test)
		ugroup_train = np.unique(group_train)

		# Calculate Q2 squared
		CV_Q2 = np.zeros((len(ugroup_train)))
		CV_Q2_roi = np.zeros((len(ugroup_train), Y_Train.shape[1]))
		CV_Q2_components = np.zeros((len(ugroup_train), n_components))
		CV_Q2_components_roi = np.zeros((len(ugroup_train), n_components, Y_Train.shape[1]))
		for g, group in enumerate(ugroup_train):
			X_gtrain = X_Train[group_train != group]
			Y_gtrain = Y_Train[group_train != group]
			X_gtest = X_Train[group_train == group]
			Y_gtest = Y_Train[group_train == group]
			pls2 = PLSRegression(n_components = n_components).fit(X_gtrain, Y_gtrain)
			CV_Q2[g] = explained_variance_score(Y_gtest, pls2.predict(X_gtest))
			CV_Q2_roi[g] = explained_variance_score(Y_gtest, pls2.predict(X_gtest), multioutput = 'raw_values')
			for c in range(n_components):
				yhat_c = np.dot(pls2.x_scores_[:,c].reshape(-1,1), pls2.y_loadings_[:,c].reshape(-1,1).T) * Y_gtrain.std(axis=0, ddof=1) + Y_gtrain.mean(axis=0)
				CV_Q2_components[g,c] = explained_variance_score(Y_gtrain, yhat_c)
				CV_Q2_components_roi[g,c,:] = explained_variance_score(Y_gtrain, yhat_c, multioutput = 'raw_values')
		self.Q2_train_ = CV_Q2.mean(0)
		self.Q2_train_std_ = CV_Q2.std(0)
		self.Q2_train_targets_ = CV_Q2_roi.mean(0)
		self.Q2_train_components_ = CV_Q2_components.mean(0)
		self.Q2_train_components_std_ = CV_Q2_components.std(0)
		self.Q2_train_components_targets_ = CV_Q2_components_roi.mean(0)

		# Calculate R2 squared for training data
		pls2 = PLSRegression(n_components = n_components).fit(X_Train, Y_Train)
		components_variance_explained_train = []
		components_variance_explained_train_roi = []
		for c in range(n_components):
			yhat_c = np.dot(pls2.x_scores_[:,c].reshape(-1,1), pls2.y_loadings_[:,c].reshape(-1,1).T) * Y_Train_std + Y_Train_mean
			components_variance_explained_train.append(explained_variance_score(Y_Train, yhat_c))
			components_variance_explained_train_roi.append(explained_variance_score(Y_Train, yhat_c, multioutput = 'raw_values'))
		self.R2_train_ = explained_variance_score(Y_Train, pls2.predict(X_Train))
		self.R2_train_targets_ = explained_variance_score(Y_Train, pls2.predict(X_Train), multioutput = 'raw_values')
		self.R2_train_components_ = np.array(components_variance_explained_train)
		self.R2_train_components_targets_ = np.array(components_variance_explained_train_roi)

		# Calculate R2P squared for test data
		components_variance_explained_test = []
		components_variance_explained_test_roi = []
		x_scores_test, y_scores_test = pls2.transform(X_Test, Y_Test)
		for c in range(n_components):
			yhat_c = np.dot(x_scores_test[:,c].reshape(-1,1), pls2.y_loadings_[:,c].reshape(-1,1).T) * Y_Test.std(axis=0, ddof=1) + Y_Test.mean(axis=0)
			components_variance_explained_test.append(explained_variance_score(Y_Test, yhat_c))
			components_variance_explained_test_roi.append(explained_variance_score(Y_Test, yhat_c, multioutput = 'raw_values'))
		self.R2_test_ = explained_variance_score(Y_Test, pls2.predict(X_Test))
		self.R2_test_targets_ = explained_variance_score(Y_Test, pls2.predict(X_Test), multioutput = 'raw_values')
		self.R2_test_components_ = np.array(components_variance_explained_test)
		self.R2_test_components_targets_ = np.array(components_variance_explained_test_roi)
		self.n_components_ = n_components
		self.group_train_ = group_train
		self.ugroup_train_ = ugroup_train
		self.X_Train_ = X_Train
		self.Y_Train_ = Y_Train
		self.X_Train_mean_ = X_Train_mean
		self.Y_Train_mean_ = Y_Train_mean
		self.X_Train_std_ = X_Train_std
		self.Y_Train_std_ = Y_Train_std
		self.X_Test_ = X_Test
		self.Y_Test_ = Y_Test
		self.X_Test_mean_ = X_Test_mean
		self.Y_Test_mean_ = Y_Test_mean
		self.X_Test_std_ = X_Test_std
		self.Y_Test_std_ = Y_Test_std
		self.model_obj_ = pls2
	def fwer_corrected_p(self, permuted_arr, target, right_tail_probability = True, apply_fwer_correction = True):
		"""
		Calculates the FWER corrected p-value
		
		Parameters
		----------
		permuted_arr : array
			Array of permutations [N_permutations, N_factors]
		target : array or float
			statistic(s) to check against null array
		right_tail_probability : bool
			Use right tail distribution (default: True)
		apply_fwer_correction : bool
			If True, output the family-wise error rate across all factors, else output permuted p-value for each factors' distribution (default: True)
		Returns
		---------
		pval_corrected : array
			Family-wise error rate corrected p-values or permuted p-values
		"""
		if permuted_arr.ndim == 1:
			permuted_arr = permuted_arr.reshape(-1,1)
		if isinstance(target, float):
			target = np.array([target])
		assert target.ndim == 1, "Error: target array must be 1D array or float"
		n_perm, n_factors = permuted_arr.shape
		if apply_fwer_correction: 
			permuted_arr = permuted_arr.max(1)
			pval_corrected = np.divide(np.searchsorted(np.sort(permuted_arr), target), n_perm)
		else:
			if n_factors == 1:
				pval_corrected = np.divide(np.searchsorted(np.sort(permuted_arr), target), n_perm)
			else:
				assert n_factors == target.shape[0], "Error: n_factors must equal length of target for elementwise comparison"
				pval_corrected = np.zeros_like(target)
				for i in range(n_factors):
					pval_corrected[i] = np.divide(np.searchsorted(np.sort(permuted_arr[:,i]), target[i]), n_perm)
		if right_tail_probability:
			pval_corrected = 1 - pval_corrected
		return(pval_corrected)
	def permute_function_pls(self, p, compute_targets = False):
		assert hasattr(self,'model_obj_'), "Error: run fit_model"
		if p % 200 == 0:
			print(p)
		X_perm = self.index_perm(unique_arr = self.ugroup_train_,
										arr = self.group_train_,
										variable = self.X_Train_)
		perm_pls2 = PLSRegression(n_components = self.n_components_).fit(X_perm, self.Y_Train_)
		x_scores_test = perm_pls2.transform(self.X_Test_)
		components_ve = np.zeros((self.n_components_))
		if compute_targets:
			components_ve_roi = np.zeros((self.n_components_, self.Y_Train_.shape[1]))
		for c in range(self.n_components_):
			yhat_c = np.dot(x_scores_test[:,c].reshape(-1,1), perm_pls2.y_loadings_[:,c].reshape(-1,1).T) * self.Y_Test_std_ + self.Y_Test_mean_
			components_ve[c] = explained_variance_score(self.Y_Test_, yhat_c)
			if compute_targets:
				components_ve_roi[c, :] = explained_variance_score(self.Y_Test_, yhat_c, multioutput = 'raw_values')
		ve = explained_variance_score(self.Y_Test_, perm_pls2.predict(self.X_Test_))
		if compute_targets:
			ve_roi = explained_variance_score(self.Y_Test_, perm_pls2.predict(self.X_Test_), multioutput = 'raw_values')
		abs_max_coef = np.max(np.abs(perm_pls2.coef_),1)
		if compute_targets:
			return(ve, ve_roi, components_ve, components_ve_roi, abs_max_coef)
		else:
			return(ve, components_ve, abs_max_coef)
	def run_permute_pls(self, compute_targets = True, calulate_pvalues = True):
		assert hasattr(self,'model_obj_'), "Error: run fit_model"
		output = Parallel(n_jobs = self.n_jobs)(delayed(self.permute_function_pls)(p, compute_targets = compute_targets) for p in range(self.n_permutations))
		if compute_targets:
			perm_ve, perm_ve_roi, perm_components_ve, perm_components_ve_roi, perm_abs_max_coef = zip(*output)
			self.perm_R2_test_targets_ = np.array(perm_ve_roi)
			self.perm_R2_test_components_targets_ = np.array(perm_components_ve_roi)
		else:
			perm_ve, perm_components_ve, perm_abs_max_coef = zip(*output)
		self.perm_R2_test_ = np.array(perm_ve)
		self.perm_R2_test_components_ = np.array(perm_components_ve)
		self.perm_coef_fwer_ = np.array(perm_abs_max_coef)
		if calulate_pvalues:
			self.compute_permuted_pvalues()
	def compute_permuted_pvalues(self):
		"""
		Calculates p-values (and FWER p-values) using the permuted null distribution.
		"""
		assert hasattr(self,'perm_R2_test_'), "Error: no permuted variables. Run run_permute_pls first."
		if hasattr(self,'perm_R2_test_targets_'):
			self.pFWER_test_targets_ = self.fwer_corrected_p(self.perm_R2_test_targets_, self.R2_test_targets_)
			CompFWER = []
			for c in range(self.n_components_):
				CompFWER.append(self.fwer_corrected_p(self.perm_R2_test_components_targets_[:,0,:],self.R2_test_components_targets_[c]))
			self.pFWER_test_components_targets_ = np.array(CompFWER)
		self.pvalue_R2_test_ = self.fwer_corrected_p(self.perm_R2_test_, self.R2_test_)[0]
		self.pvalue_R2_test_components_ = self.fwer_corrected_p(self.perm_R2_test_components_, self.R2_test_components_, apply_fwer_correction = False)
		coef_p = []
		for co in range(len(self.model_obj_.coef_)):
			coef_p.append(self.fwer_corrected_p(self.perm_coef_fwer_[:,co], np.abs(self.model_obj_.coef_[co])))
		self.pFWER_coef_ = np.array(coef_p).T 
	def plot_model(self, n_jitters = 1000, png_basename = None, add_Q2_from_train = False):
		assert hasattr(self,'pvalue_R2_test_'), "Error: Run compute_permuted_pvalues"
		if n_jitters > self.n_permutations:
			n_jitters = self.n_permutations
		n_plots = self.n_components_ + 1
		p_num = 1
		plt.subplots(figsize=(int(2*n_plots), 6), dpi=100, tight_layout = True, sharey='row')
		plt.subplot(1, n_plots, p_num)
		jitter = np.random.normal(0, scale = 0.1, size=n_jitters)
		rand_dots = self.perm_R2_test_[:n_jitters]
		plt.scatter(jitter, rand_dots, marker = '.', alpha = 0.3)
		plt.xlim(-.5, .5)
		plt.ylabel("R2 predicted vs actual (Test Data)")
		plt.title("Model")
		plt.scatter(0, self.R2_test_, marker = 'o', alpha = 1.0, c = 'k')
		if add_Q2_from_train:
			plt.errorbar(0.1, self.Q2_train_, self.Q2_train_std_, linestyle='None', marker='.', c = 'r', alpha = 0.5)
		plt.xticks(color='w')
		p_num += 1
		x1,x2,y1,y2 = plt.axis()
		y1 = round(y1,3) - 0.01
		y2 = round(y2,3) + 0.01
		plt.ylim(y1, y2)
		if self.pvalue_R2_test_ == 0:
			plt.xlabel("R2=%1.3f, P<%1.1e" % (self.R2_test_, (1 / self.n_permutations)), fontsize=10)
		else:
			plt.xlabel("R2=%1.3f, P=%1.1e" % (self.R2_test_, self.pvalue_R2_test_), fontsize=10)
		for c in range(self.n_components_):
			plt.subplot(1, n_plots, p_num)
			jitter = np.random.normal(0, scale = 0.1, size=n_jitters)
			rand_dots = self.perm_R2_test_components_[:n_jitters, c]
			plt.scatter(jitter, rand_dots, marker = '.', alpha = 0.3)
			plt.xlim(-.5, .5)
			plt.title("Component %d" % (c+1))
			plt.scatter(0, self.R2_test_components_[c], marker = 'o', alpha = 1.0, c = 'k')
			if add_Q2_from_train:
				plt.errorbar(0.1, self.Q2_train_components_[c], self.Q2_train_components_std_[c], linestyle='None', marker='.', c = 'r', alpha = 0.5)
			plt.xticks(color='w')
			plt.ylim(y1, y2)
			if self.pvalue_R2_test_components_[c] == 0:
				plt.xlabel("R2=%1.3f, P<%1.1e" % (self.R2_test_components_[c], (1 / self.n_permutations)), fontsize=10)
			else:
				plt.xlabel("R2=%1.3f, P=%1.1e" % (self.R2_test_components_[c], self.pvalue_R2_test_components_[c]), fontsize=10)
			p_num += 1
		if png_basename is not None:
			plt.savefig("%s_model_fit_to_test_with_null.png" % png_basename)
			plt.close()
		else:
			plt.show()
	def plot_rmsep_components(self, component_range = np.arange(1,11,1), png_basename = None):
		full_model_rmse = []
		full_model_ve = []
		for i in component_range:
			pls2 = PLSRegression(n_components=i)
			pls2.fit(self.X_Train_, self.Y_Train_)
			Y_proj = pls2.predict(self.X_Train_)
			score = explained_variance_score(self.Y_Train_, Y_proj)
			full_model_ve.append(score)
			full_model_rmse.append(mean_squared_error(self.Y_Train_, Y_proj, squared = False))
		rmse_cv = []
		ve_cv = []
		for i in component_range:
			rmse_cv_by_subject = []
			CV_temp_rmse = []
			CV_temp_ve = []
			for group in self.ugroup_train_:
				Y_gtrain = self.Y_Train_[self.group_train_ != group]
				Y_gtest = self.Y_Train_[self.group_train_ == group]
				X_gtrain = self.X_Train_[self.group_train_ != group]
				X_gtest = self.X_Train_[self.group_train_ == group]
				pls2 = PLSRegression(n_components=i)
				pls2.fit(X_gtrain, Y_gtrain)
				Y_proj = pls2.predict(X_gtest)
				CV_temp_ve.append(explained_variance_score(Y_gtest, Y_proj))
				CV_temp_rmse.append(mean_squared_error(Y_gtest, Y_proj, squared = False))
			rmse_cv.append(np.mean(CV_temp_rmse))
			ve_cv.append(np.mean(CV_temp_ve))
		plt.plot(component_range, np.array(full_model_rmse), c = 'b', label = "Model RMSEP")
		plt.plot(component_range, np.array(rmse_cv), c = 'r', label = "CV RMSEP")
		plt.legend()
		if png_basename is not None:
			plt.savefig("%s_rmse_versus_component_number.png" % png_basename)
			plt.close()
		else:
			plt.show()
		plt.plot(component_range, np.array(full_model_ve), c = 'k', label = "Model R2")
		plt.plot(component_range, np.array(ve_cv), c = 'r', label = "CV Q2")
		plt.legend()
		if png_basename is not None:
			plt.savefig("%s_R2_versus_component_number.png" % png_basename)
			plt.close()
		else:
			plt.show()

class bootstraper_parallel():
	def __init__(self, n_jobs, n_boot = 1000, split = 0.5):
		self.n_jobs = n_jobs
		self.n_boot = n_boot
		self.split = split
	def nfoldsplit_group(self, group, n_fold = 5, holdout = 0, train_index = None, verbose = False, debug_verbose = False):
		"""
		Creates indexed array(s) for k-fold cross validation with holdout option for test data. The ratio of the groups are maintained. To reshuffle the training, if can be passed back through via index_train.
		The indices are always based on the original grouping variable. i.e., the orignal data.
		
		Parameters
		----------
		group : array
			List array with length of number of subjects. 
		n_fold : int
			The number of folds
		holdout : float
			The amount of data to holdout ranging from 0 to <1. A reasonable holdout is around 0.3 or 30 percent. If holdout = None, then returns test_index = None. (default = 0)
		train_index : array
			Indexed array of training data. Holdout must be zero (holdout = 0). It is useful for re-shuffling the fold indices or changing the number of folds.
		verbose : bool
			Prints out the splits and some basic information
		debug_verbose: bool
			Prints out the indices by group
		Returns
		---------
		train_index : array
			index array of training data
		fold_indices : object
			the index array for each fold (n_folds, training_fold_size)
		test_index : array or None
			index array of test data
		"""
		test_index = None
		original_group = group[:]
		ugroup = np.unique(group)
		lengroup = len(group)
		indices = np.arange(0,lengroup,1)
		if holdout != 0:
			assert holdout < 1., "Error: Holdout ratio must be >0 and <1.0. Try .3"
			assert train_index is None, "Error: train index already exists."
			indx_0 = []
			indx_1 = []
			for g in ugroup:
				pg = np.random.permutation(indices[group==g])
				indx_0.append(pg[:int(len(pg)*holdout)])
				indx_1.append(pg[int(len(pg)*holdout):])
			train_index = np.concatenate(indx_1)
			test_index = np.concatenate(indx_0)
			group = group[train_index]
			if verbose:
				print("Train data size = %s, Test data size = %s [holdout = %1.2f]" %(len(train_index), len(test_index), holdout))
		else:
			if train_index is None:
				train_index = indices[:]
			else:
				group = group[train_index]
		# reshuffle for good luck
		gsize = []
		shuffle_train = []
		for g in ugroup:
			pg = np.random.permutation(train_index[group==g])
			gsize.append(len(pg))
			shuffle_train.append(pg)
		train_index = np.concatenate(shuffle_train)
		group = original_group[train_index]
		split_sizes = np.divide(gsize, n_fold).astype(int)
		if verbose:
			for s in range(len(ugroup)):
				print("Training group [%s]: size n=%d, split size = %d, remainder = %d" % (ugroup[s], gsize[s], split_sizes[s], int(gsize[s] % split_sizes[s])))
			if test_index is not None:
				for s in range(len(ugroup)):
					original_group[test_index] == ugroup[s]
					test_size = np.sum((original_group[test_index] == ugroup[s])*1)
					print("Test group [%s]: size n=%d, holdout percentage = %1.2f" % (ugroup[s], test_size, np.divide(test_size * 100, test_size+gsize[s])))
		fold_indices = []
		for n in range(n_fold):
			temp_index = []
			for i, g in enumerate(ugroup):
				temp = train_index[group==g]
				if n == n_fold-1:
					temp_index.append(temp[n*split_sizes[i]:])
				else:
					temp_index.append(temp[n*split_sizes[i]:((n+1)*split_sizes[i])])
				if debug_verbose:
					print(n)
					print(g)
					print(original_group[temp_index[-1]])
					print(temp_index[-1])
			fold_indices.append(np.concatenate(temp_index))
		train_index = np.sort(train_index)
		fold_indices = np.array(fold_indices, dtype = object)
		if holdout != 0:
			test_index = np.sort(test_index)
		if verbose:
			for i in range(n_fold):
				print("\nFOLD %d:" % (i+1))
				print(np.sort(original_group[fold_indices[i]]))
			if test_index is not None:
				print("\nTEST:" )
				print(np.sort(original_group[test_index]))
		return(fold_indices, train_index, test_index)
	def bootstrap_by_group(self, group, split = 0.5):
		ugroup = np.unique(group)
		lengroup = len(group)
		indices = np.arange(0,lengroup,1)
		indx_0 = []
		indx_1 = []
		for g in ugroup:
			pg = np.random.permutation(indices[group==g])
			indx_0.append(pg[:int(len(pg)*split)])
			indx_1.append(pg[int(len(pg)*split):])
		return(np.concatenate(indx_0), np.concatenate(indx_1))
	def create_nfold(self, X, y, group, n_fold = 5, holdout = 0.3, verbose = True):
		"""
		Imports the data and runs nfoldsplit_group.
		"""
		fold_indices, train_index, test_index  = self.nfoldsplit_group(group = group,
																							n_fold = n_fold,
																							holdout = holdout,
																							train_index = None,
																							verbose = verbose,
																							debug_verbose = False)
		X_train = X[train_index]
		y_train = y[train_index]
		if test_index is not None:
			X_test= X[test_index]
			y_test= y[test_index]
		self.train_index_ = train_index
		self.fold_indices_ = fold_indices
		self.test_index_ = test_index
		self.X_ = X
		self.y_ = y
		self.group_ = group
		self.n_fold_ = n_fold
		self.X_train_ = X_train
		self.y_train_ = y_train
		self.X_test_ = X_test
		self.y_test_ = y_test
	def nfold_params_search(self, c, X, y, group, train_index, fold_indices, eta_range = np.arange(.1,1.,.1), n_reshuffle = 1):
		"""
		"""
		n_fold = len(fold_indices)
		fold_index = np.arange(0,n_fold,1)
		nspls_runs = n_fold*n_reshuffle
		K = c + 1
		cQ2_SEARCH = np.zeros((len(eta_range)))
		cQ2_SEARCH_SD = np.zeros((len(eta_range)))
		cRMSEP_CV_SEARCH = np.zeros((len(eta_range)))
		cRMSEP_CV_SEARCH_SD = np.zeros((len(eta_range)))
		for e, eta in enumerate(eta_range):
			temp_Q2 = np.zeros((nspls_runs))
			temp_rmse = np.zeros((nspls_runs))
			if ((K+1) < X[train_index].shape[1]) and (y[train_index].shape[1] > (K+1)):
				p = 0
				for i in range(n_reshuffle):
					if n_reshuffle > 1:
						fold_indices, _, _ = self.nfoldsplit_group(group = group,
																			n_fold = n_fold,
																			holdout = 0,
																			train_index = train_index,
																			verbose = False,
																			debug_verbose = False)
					for n in range(n_fold):
						sel_train = fold_indices[n]
						sel_test = np.concatenate(fold_indices[fold_index != n])
						tmpX_train = X[sel_train]
						tmpY_train = y[sel_train]
						tmpX_test = X[sel_test]
						tmpY_test = y[sel_test]
						# kick out effective zero predictors
						tmpX_test = tmpX_test[:,tmpX_train.std(0) > 0.0001]
						tmpX_train = tmpX_train[:,tmpX_train.std(0) > 0.0001]
						spls2 = spls_rwrapper(n_components = K, eta = eta)
						spls2.fit(tmpX_train, tmpY_train)
						Y_proj = spls2.predict(tmpX_test)
						temp_Q2[p] = explained_variance_score(tmpY_test, Y_proj)
						temp_rmse[p] = mean_squared_error(tmpY_test, Y_proj, squared = False)
						p += 1
				cQ2_SEARCH[e] = np.mean(temp_Q2)
				cQ2_SEARCH_SD[e] = np.std(temp_Q2)
				cRMSEP_CV_SEARCH[e] = np.mean(temp_rmse)
				cRMSEP_CV_SEARCH_SD[e] = np.std(temp_rmse)
			else:
				cQ2_SEARCH[e] = 0
				cQ2_SEARCH_SD[e] = 0
				cRMSEP_CV_SEARCH[e] = 1
				cRMSEP_CV_SEARCH_SD[e] = 1
		print("Component %d finished" % K)
		return(c, cQ2_SEARCH, cQ2_SEARCH_SD, cRMSEP_CV_SEARCH, cRMSEP_CV_SEARCH_SD)
	def nfold_cv_params_search_spls(self, eta_range = np.arange(.1,1.,.1), n_reshuffle = 1, max_n_comp = 10):
		assert hasattr(self,'fold_indices_'), "Error: No fold indices. Run create_nfold first"
		assert isinstance(n_reshuffle, (int, np.integer)), "Error: n_reshuffle must be an interger"
		# parallel by max_n_comp
		output = Parallel(n_jobs=min(self.n_jobs,max_n_comp), backend='multiprocessing')(delayed(self.nfold_params_search)(c, X = self.X_, y = self.y_, group = self.group_, train_index = self.train_index_, fold_indices = self.fold_indices_, eta_range = eta_range, n_reshuffle = n_reshuffle) for c in range(max_n_comp))
		ord_k, Q2_SEARCH, Q2_SEARCH_SD, RMSEP_CV_SEARCH, RMSEP_CV_SEARCH_SD = zip(*output)
		ord_k = np.array(ord_k)
		Q2_SEARCH = np.row_stack(Q2_SEARCH)[ord_k]
		Q2_SEARCH_SD = np.row_stack(Q2_SEARCH_SD)[ord_k]
		RMSEP_CV_SEARCH = np.row_stack(RMSEP_CV_SEARCH)[ord_k]
		RMSEP_CV_SEARCH_SD = np.row_stack(RMSEP_CV_SEARCH_SD)[ord_k]
		# re-ordering stuff Comp [low to high], Eta [low to high]
		self.Q2_SEARCH_ = Q2_SEARCH.T
		self.Q2_SEARCH_SD_ = Q2_SEARCH_SD.T
		self.RMSEP_CV_SEARCH_ = RMSEP_CV_SEARCH.T
		self.RMSEP_CV_SEARCH_SD_ = RMSEP_CV_SEARCH_SD.T
		self.search_eta_range_ = eta_range[::-1]
		self.max_n_comp_ = max_n_comp
		xy = (self.RMSEP_CV_SEARCH_ == np.nanmin(self.RMSEP_CV_SEARCH_))*1
		print_optimal_values = True
		try:
			self.best_K_ = int(np.arange(1,self.max_n_comp_+1,1)[xy.mean(0) > 0])
		except:
			print_optimal_values = False
			print("Warning: multiple components have the best value.")
			print(np.arange(1,self.max_n_comp_+1,1)[xy.mean(0) > 0])
			self.best_K_ = np.arange(1,self.max_n_comp_+1,1)[xy.mean(0) > 0]
		try:
			self.best_eta_ = float(self.search_eta_range_[xy.mean(1) > 0])
		except:
			print_optimal_values = False
			print("Warning: multiple sparsity thresholds have the best value.")
			print(self.search_eta_range_[xy.mean(1) > 0])
			self.best_eta_ = self.search_eta_range_[xy.mean(1) > 0]
		if print_optimal_values:
			print("Best N-components = %d, Best eta = %1.2f" % (self.best_K_, self.best_eta_))
	def group_params_searcher(self, c, X, y, group, eta_range = np.arange(.1,1.,.1)):
		K = c + 1
		ugroup = np.unique(group)
		cQ2_SEARCH = np.zeros((len(eta_range)))
		cQ2_SEARCH_SD = np.zeros((len(eta_range)))
		cRMSEP_CV_SEARCH = np.zeros((len(eta_range)))
		cRMSEP_CV_SEARCH_SD = np.zeros((len(eta_range)))
		for e, eta in enumerate(eta_range):
			temp_Q2 = []
			temp_rmse = []
			if ((K+1) < X.shape[1]) and (y.shape[1] > (K+1)):
				for g in ugroup:
					Y_train = y[group != g]
					X_train = X[group != g]
					X_test = X[group == g]
					Y_test = y[group == g]
					# kick out effective zero predictors
					X_test = X_test[:,X_train.std(0) > 0.0001]
					X_train = X_train[:,X_train.std(0) > 0.0001]
					spls2 = spls_rwrapper(n_components = K, eta = eta)
					spls2.fit(X_train, Y_train)
					Y_proj = spls2.predict(X_test)
					temp_Q2.append(explained_variance_score(Y_test, Y_proj))
					temp_rmse.append(mean_squared_error(Y_test, Y_proj, squared = False))
				cQ2_SEARCH[e] = np.mean(temp_Q2)
				cQ2_SEARCH_SD[e] = np.std(temp_Q2)
				cRMSEP_CV_SEARCH[e] = np.mean(temp_rmse)
				cRMSEP_CV_SEARCH_SD[e] = np.std(temp_rmse)
			else:
				cQ2_SEARCH[e] = 0
				cQ2_SEARCH_SD[e] = 0
				cRMSEP_CV_SEARCH[e] = 1
				cRMSEP_CV_SEARCH_SD[e] = 1
		print("Component %d finished" % K)
		return(c, cQ2_SEARCH, cQ2_SEARCH_SD, cRMSEP_CV_SEARCH, cRMSEP_CV_SEARCH_SD)
	def group_cv_params_search_spls(self, X, y, group, eta_range = np.arange(.1,1.,.1), max_n_comp = 10):
		# parallel by max_n_comp
		output = Parallel(n_jobs=min(self.n_jobs,max_n_comp))(delayed(self.group_params_searcher)(c, X = X, y = y, group = group, eta_range = eta_range) for c in range(max_n_comp))
		ord_k, Q2_SEARCH, Q2_SEARCH_SD, RMSEP_CV_SEARCH, RMSEP_CV_SEARCH_SD = zip(*output)
		ord_k = np.array(ord_k)
		Q2_SEARCH = np.row_stack(Q2_SEARCH)[ord_k]
		Q2_SEARCH_SD = np.row_stack(Q2_SEARCH_SD)[ord_k]
		RMSEP_CV_SEARCH = np.row_stack(RMSEP_CV_SEARCH)[ord_k]
		RMSEP_CV_SEARCH_SD = np.row_stack(RMSEP_CV_SEARCH_SD)[ord_k]
		# re-ordering stuff Comp [low to high], Eta [low to high]
		self.Q2_SEARCH_ = Q2_SEARCH.T
		self.Q2_SEARCH_SD_ = Q2_SEARCH_SD.T
		self.RMSEP_CV_SEARCH_ = RMSEP_CV_SEARCH.T
		self.RMSEP_CV_SEARCH_SD_ = RMSEP_CV_SEARCH_SD.T
		self.search_eta_range_ = eta_range[::-1]
		self.max_n_comp_ = max_n_comp
		xy = (self.RMSEP_CV_SEARCH_ == np.nanmin(self.RMSEP_CV_SEARCH_))*1
		print_optimal_values = True
		try:
			self.best_K_ = int(np.arange(1,self.max_n_comp_+1,1)[xy.mean(0) > 0])
		except:
			print_optimal_values = False
			print("Warning: multiple components have the best value.")
			print(np.arange(1,self.max_n_comp_+1,1)[xy.mean(0) > 0])
			self.best_K_ = np.arange(1,self.max_n_comp_+1,1)[xy.mean(0) > 0]
		try:
			self.best_eta_ = float(self.search_eta_range_[xy.mean(1) > 0])
		except:
			print_optimal_values = False
			print("Warning: multiple sparsity thresholds have the best value.")
			print(self.search_eta_range_[xy.mean(1) > 0])
			self.best_eta_ = self.search_eta_range_[xy.mean(1) > 0]
		if print_optimal_values:
			print("Best N-components = %d, Best eta = %1.2f" % (self.best_K_, self.best_eta_))

	def plot_cv_params_search_spls(self, nan_unstable = False, png_basename = None):
		assert hasattr(self,'best_eta_'), "Error: run cv_params_search_spls"
		
		#Q2
		Q2_SEARCH = self.Q2_SEARCH_
		if nan_unstable:
			Q2_SEARCH[Q2_SEARCH < 0] = np.nan
		else:
			Q2_SEARCH[Q2_SEARCH < 0] = 0
		plt.imshow(Q2_SEARCH, interpolation = None, cmap='jet')
		plt.yticks(range(len(self.search_eta_range_)),[s[:3] for s in self.search_eta_range_.astype(str)])
		plt.ylabel('eta (sparsity)')
		plt.xticks(range(self.max_n_comp_),np.arange(1,self.max_n_comp_+1,1))
		plt.xlabel('Components')
		plt.colorbar()
		plt.title("Q-Squared [CV]")
		if png_basename is not None:
			plt.savefig("%s_feature_selection_Q2.png" % png_basename)
			plt.close()
		else:
			plt.show()

		Q2_SEARCH = self.Q2_SEARCH_
		Q2_SEARCH_SD = self.Q2_SEARCH_SD_
		if nan_unstable:
			Q2_SEARCH_SD[Q2_SEARCH < 0] = np.nan
		plt.imshow(Q2_SEARCH, interpolation = None, cmap='jet')
		plt.imshow(Q2_SEARCH_SD, interpolation = None, cmap='jet_r')
		plt.yticks(range(len(self.search_eta_range_)),[s[:3] for s in self.search_eta_range_.astype(str)])
		plt.ylabel('eta (sparsity)')
		plt.xticks(range(self.max_n_comp_),np.arange(1,self.max_n_comp_+1,1))
		plt.xlabel('Components')
		plt.colorbar()
		plt.title("Q-Squared [CV] St. Dev.")
		if png_basename is not None:
			plt.savefig("%s_feature_selection_Q2_SD.png" % png_basename)
			plt.close()
		else:
			plt.show()

		Q2_SEARCH_RATIO = np.divide(self.Q2_SEARCH_, self.Q2_SEARCH_SD_)
		if nan_unstable:
			Q2_SEARCH_RATIO[Q2_SEARCH < 0] = np.nan
		else:
			Q2_SEARCH_RATIO[Q2_SEARCH < 0] = 0
		plt.imshow(Q2_SEARCH_RATIO, interpolation = None, cmap='jet')
		plt.yticks(range(len(self.search_eta_range_)),[s[:3] for s in self.search_eta_range_.astype(str)])
		plt.ylabel('eta (sparsity)')
		plt.xticks(range(self.max_n_comp_),np.arange(1,self.max_n_comp_+1,1))
		plt.xlabel('Components')
		plt.colorbar()
		plt.title("Q-Squared [CV] / St. Dev.")
		if png_basename is not None:
			plt.savefig("%s_feature_selection_Q2divSD.png" % png_basename)
			plt.close()
		else:
			plt.show()

		RMSEP_CV_SEARCH = self.RMSEP_CV_SEARCH_
		if nan_unstable:
			RMSEP_CV_SEARCH[RMSEP_CV_SEARCH > 1] = np.nan
		else:
			RMSEP_CV_SEARCH[RMSEP_CV_SEARCH > 1] = 1.
		plt.imshow(self.RMSEP_CV_SEARCH_, interpolation = None, cmap='jet_r')
		plt.yticks(range(len(self.search_eta_range_)),[s[:3] for s in self.search_eta_range_.astype(str)])
		plt.ylabel('eta (sparsity)')
		plt.xticks(range(self.max_n_comp_),np.arange(1,self.max_n_comp_+1,1))
		plt.xlabel('Components')
		plt.colorbar()
		plt.title("RMSEP [CV]")
		if png_basename is not None:
			plt.savefig("%s_feature_selection_RMSEP.png" % png_basename)
			plt.close()
		else:
			plt.show()

	def bootstrap_spls(self, i, X, y, n_comp, group, split, eta):
		if i % 100 == 0: 
			print("Bootstrap : %d" % (i))
		train_idx, _ = self.bootstrap_by_group(group = group, split = split)
		X = X[train_idx]
		y = y[train_idx]
		boot_spls2 = spls_rwrapper(n_components = n_comp, eta = eta)
		boot_spls2.fit(X, y)
		selector = np.zeros((X.shape[1]))
		selector[boot_spls2.selectedvariablesindex_] = 1
		return(selector)
	def run_bootstrap_spls(self, X, y, n_comp, group, eta, split = 0.5):
		selected_vars = Parallel(n_jobs=self.n_jobs, backend='multiprocessing')(delayed(self.bootstrap_spls)(i, X = X, y = y, n_comp = n_comp, group = group, split = self.split, eta = eta) for i in range(self.n_boot))
		self.selected_vars_ = np.array(selected_vars)
		self.selected_vars_mean_ = np.mean(selected_vars, 0)
		self.X = X  # I need to fix this
		self.y = y
		self.n_comp = n_comp
		self.eta = eta
		self.split = split
		self.group = group
		self.ugroup = np.unique(group)
	def nfold_cv_search_array(self, search_array = np.arange(.2,1.,.05)):
		assert hasattr(self,'selected_vars_mean_'), "Error: bootstrap parallel is missing"
		ve_cv = []
		rmse_cv = []
		ve_cv_std = []
		rmse_cv_std = []
		full_model_ve = []
		full_model_rmse = []
		n_fold = len(self.fold_indices_)
		fold_index = np.arange(0,n_fold,1)
		for s in search_array:
			print("\nSelection Threshold = %1.2f" % np.round(s,3))
			selection_mask = self.selected_vars_mean_ > s
			CV_temp_rmse = []
			CV_temp_ve = []
			X_SEL = self.X_[:,selection_mask]
			print("\nNumber of Selected Variables: %d from %s " % (X_SEL.shape[1], self.X_.shape[1]))
			if ((self.n_comp+1) < X_SEL.shape[1]) and (self.y.shape[1] > (self.n_comp+1)):
				for n in range(self.n_fold_):
					sel_train = self.fold_indices_[n]
					sel_test = np.concatenate(self.fold_indices_[fold_index != n])
					X_train =X_SEL[sel_train]
					Y_train = self.y_[sel_train]
					X_test = X_SEL[sel_test]
					Y_test = self.y_[sel_test]
					pls2 = PLSRegression(n_components=self.n_comp).fit(X_train, Y_train)
					Y_proj = pls2.predict(X_test)
					score = explained_variance_score(Y_test, Y_proj)
					print("FOLD %d : %1.3f" % ((n+1), score))
					CV_temp_ve.append(score)
					CV_temp_rmse.append(mean_squared_error(Y_test, Y_proj, squared = False))
				print("CV MODEL : %1.3f +/- %1.3f" % (np.mean(CV_temp_ve), np.std(CV_temp_ve)))
				rmse_cv.append(np.mean(CV_temp_rmse))
				ve_cv.append(np.mean(CV_temp_ve))
				rmse_cv_std.append(np.std(CV_temp_rmse))
				ve_cv_std.append(np.std(CV_temp_ve))
				# full model
				pls2 = PLSRegression(n_components=self.n_comp).fit(X_SEL, self.y_)
				Y_proj = pls2.predict(X_SEL)
				score = explained_variance_score(self.y_, Y_proj)
				print("MODEL : %1.3f" % (score))
				full_model_ve.append(score)
				full_model_rmse.append(mean_squared_error(self.y_, Y_proj, squared = False))
			else:
				rmse_cv.append(0.)
				ve_cv.append(0.)
				rmse_cv_std.append(0.)
				ve_cv_std.append(0.)
				full_model_ve.append(0.)
				full_model_rmse.append(0.)
		self.RMSEP_CV_ = np.array(rmse_cv)
		self.Q2_ = np.array(ve_cv)
		self.RMSEP_CV_SD_ = np.array(rmse_cv_std)
		self.Q2_SD_ = np.array(ve_cv_std)
		self.RMSEP_LEARN_ = np.array(full_model_rmse)
		self.R2_LEARN_ = np.array(full_model_ve)
		self.search_thresholds_ = search_array
	def cv_search_array(self, search_array = np.arange(.2,1.,.05)):
		assert hasattr(self,'selected_vars_mean_'), "Error: bootstrap parallel is missing"
		ve_cv = []
		rmse_cv = []
		ve_cv_std = []
		rmse_cv_std = []
		full_model_ve = []
		full_model_rmse = []
		for s in search_array:
			print(np.round(s,3))
			selection_mask = self.selected_vars_mean_ > s
			CV_temp_rmse = []
			CV_temp_ve = []
			X_SEL = self.X[:,selection_mask]
			if ((self.n_comp+1) < X_SEL.shape[1]) and (self.y.shape[1] > (self.n_comp+1)):
				for g in self.ugroup:
					Y_train = self.y[self.group != g]
					X_train = X_SEL[self.group != g]
					X_test = X_SEL[self.group == g]
					Y_test = self.y[self.group == g]
					pls2 = PLSRegression(n_components=self.n_comp).fit(X_train, Y_train)
					Y_proj = pls2.predict(X_test)
					score = explained_variance_score(Y_test, Y_proj)
					print("%s : %1.3f" % (g, score))
					CV_temp_ve.append(score)
					CV_temp_rmse.append(mean_squared_error(Y_test, Y_proj, squared = False))
				print("CV MODEL : %1.3f +/- %1.3f" % (np.mean(CV_temp_ve), np.std(CV_temp_ve)))
				rmse_cv.append(np.mean(CV_temp_rmse))
				ve_cv.append(np.mean(CV_temp_ve))
				rmse_cv_std.append(np.std(CV_temp_rmse))
				ve_cv_std.append(np.std(CV_temp_ve))
				# full model
				pls2 = PLSRegression(n_components=self.n_comp).fit(X_SEL, self.y)
				Y_proj = pls2.predict(X_SEL)
				score = explained_variance_score(self.y, Y_proj)
				print("MODEL : %1.3f" % (score))
				full_model_ve.append(score)
				full_model_rmse.append(mean_squared_error(self.y, Y_proj, squared = False))
			else:
				rmse_cv.append(0.)
				ve_cv.append(0.)
				rmse_cv_std.append(0.)
				ve_cv_std.append(0.)
				full_model_ve.append(0.)
				full_model_rmse.append(0.)
		self.RMSEP_CV_ = np.array(rmse_cv)
		self.Q2_ = np.array(ve_cv)
		self.RMSEP_CV_SD_ = np.array(rmse_cv_std)
		self.Q2_SD_ = np.array(ve_cv_std)
		self.RMSEP_LEARN_ = np.array(full_model_rmse)
		self.R2_LEARN_ = np.array(full_model_ve)
		self.search_thresholds_ = search_array
	def plot_cv_search_array(self, png_basename = None):
		assert hasattr(self,'search_thresholds_'), "Error: run cv_search_array"
		isvalid_sd = np.zeros((len(self.search_thresholds_)), dtype = bool)
		for i, s in enumerate(self.search_thresholds_):
			selection_mask = self.selected_vars_mean_ > s
			CV_temp_rmse = []
			CV_temp_ve = []
			X_SEL = self.X[:,selection_mask]
			if ((self.n_comp+1) < X_SEL.shape[1]) and (self.y.shape[1] > (self.n_comp+1)):
				isvalid_sd[i] = True
		isvalid_sd[self.Q2_SD_ > 1] = False
		Xthresholds = self.search_thresholds_
		Q2_values = self.Q2_
		Q2_values[Q2_values < 0] = 0 
		plt.plot(Xthresholds, Q2_values, color='blue')
		plt.fill_between(Xthresholds[isvalid_sd], self.Q2_[isvalid_sd]-self.Q2_SD_[isvalid_sd], self.Q2_[isvalid_sd]+self.Q2_SD_[isvalid_sd], alpha=0.5, edgecolor='blue', facecolor='lightsteelblue', linestyle=":")
		plt.ylabel('Q-Squared')
		plt.xlabel('Selection Threshold')
		plt.xticks(Xthresholds, [s[:4] for s in np.round(Xthresholds,3).astype(str)])
		plt.title("sPLS Model: Components = %d, eta = %1.2f, n_boot = %d" % (self.n_comp, self.eta, self.n_boot))
		if png_basename is not None:
			plt.savefig("%s_feature_selection_thresholds.png" % png_basename)
			plt.close()
		else:
			plt.show()
class spls_rwrapper:
	"""
	Wrapper that uses the spls r package, and rpy2
	https://cran.r-project.org/web/packages/spls/
	Based on: Chun and Keles (2010), doi:10.1111/j.1467-9868.2009.00723.x
	"""
	def __init__(self, n_components, eta, kappa = 0.5, max_iter = 100, algorithm_selection = "pls2", algorithm_fit = "simpls", scale_x = True, scale_y = True, effective_zero = 0.0001):
		"""
		Setting for spls
		
		Parameters
		----------
		n_components : array
			The number of components to fit
		eta : float
			The regularization coefficient ranging from 0 to <1. At Eta = 0, sPLS is equivalent to PLS
		kappa : float
			Parameter to control the effect of the concavity of the objective function and the closeness of original and surrogate direction vector. kappa should be between 0 and 0.5 (default = 0.5).
		max_iter : int
			Maximum number of interactions for fitting direction vector (default = 100).
		algorithm_selection : str
			PLS algorithm for variable selection (default = "pls2"). Choices: {"pls2", "simpls"}
		algorithm_selection : str
			PLS algorithm for model fitting (default = "simpls"). Choices: {"kernelpls", "widekernelpls", "simpls","oscorespls"}
		scale_x : bool
			z-scale X
		scale_y : bool
			z-scale y
		effective_zero : float
			The threshold for effect zero (default = 0.0001)
		Returns
		---------
		The sPLS function
		"""
		self.n_components = n_components
		self.eta = eta
		self.kappa = kappa
		self.max_iter = max_iter
		self.scale_x = scale_x
		self.scale_y = scale_y
		self.penalty = "l1"
		self.algorithm_selection = algorithm_selection
		self.algorithm_fit = algorithm_fit
		self.effective_zero = effective_zero
		self.max_iter = max_iter
	def fit(self, X, y, transform = True):
		"""
		Fit for spls model
		
		Parameters
		----------
		X : array
			Array of predictors [N_subjects, N_predictors]
		y : float
			Array of responses [N_subjects, N_responses]
		Returns
		---------
		self.betacomponents_ : array
			beta component vectors [N_components, N_predictors, N_responses]
		self.selectedvariablescomponents_ : object
			selected variables for each component [N_components, N_selected]
		self.selectedvariablesindex_ : arr1
			Selected variables index. Useful for subsetting
		self.coef_ : array
			coefficient array [N_predictors, N_responses]
		"""
		X, y, X_mean, y_mean, X_std, y_std = self.zscaler_XY(X, y, scale_x = self.scale_x, scale_y = self.scale_y)
		X = np.array(X)
		y = np.array(y)
		model = spls.spls(X, y,
							K = self.n_components,
							eta = self.eta,
							select = self.algorithm_selection,
							fit = self.algorithm_fit,
							eps = self.effective_zero,
							maxstep = self.max_iter)
		components = np.zeros((self.n_components, X.shape[1], y.shape[1]))
		sel_vars = []
		for i in range(self.n_components):
			components[i] = model.rx2("betamat")[i]
			sel_vars.append(model.rx2("new2As")[i])
		self.betacomponents_ = np.array(components)
		self.selectedvariablescomponents_ = np.array(sel_vars, dtype=object) - 1
		self.selectedvariablesindex_ = np.sort(np.concatenate(sel_vars)) - 1
		self.coef_ = stats.coef(model)
		self.X_ = X
		self.X_mean_ = X_mean
		self.X_std_ = X_std
		self.y_ = y
		self.y_mean_ = y_mean
		self.y_std = y_std
	def zscaler_XY(self, X, y, axis=0, w_mean=True, scale_x = True, scale_y = True):
		"""
		Applies scaling to X and y, return means and std regardless
		"""
		X_ = np.zeros_like(X)
		X_[:] = np.copy(X)
		X_mean_ = np.nanmean(X_, axis)
		X_std_ = np.nanstd(X_, axis = axis, ddof=1)
		Y_ = np.zeros_like(y)
		Y_[:] = np.copy(y)
		Y_mean_ = np.nanmean(Y_, axis)
		Y_std_ = np.nanstd(Y_, axis = axis, ddof=1)
		if w_mean:
			X_ -= X_mean_
			Y_ -= Y_mean_
		if scale_x:
			X_ /= X_std_
		if scale_y:
			Y_ /= Y_std_
		return(X_, Y_, X_mean_, Y_mean_, X_std_, Y_std_)
	def transform(self, X, y = None):
		"""
		Calculate the component scores for predictors.
		"""
		# spls::spls uses pls::plsr. I'll use sklearn.cross_decomposition.PLSRegression
		pls2 = PLSRegression(n_components=self.n_components).fit(self.X_[:, self.selectedvariablesindex_], self.y_)
		if y is not None:
			return(pls2.transform(X[:, self.selectedvariablesindex_], y))
		else:
			return(pls2.transform(X[:, self.selectedvariablesindex_], y))
	def predict(self, X):
		"""
		Predict y from X using the spls model
		"""
		return(np.dot(X,self.coef_))

class linear_regression:
	def __init__(self):
		self.coef = None
	def fit(self, X, Y):
		self.coef = cy_lin_lstsqr_mat(X,Y)
	def predict(self, X):
		return(np.dot(X,self.coef))

class gradient_linear_regression:
	def __init__(self, X, y, learning_rate = 0.05, intercept = True):
		if intercept:
			X = self.stack_ones(X)
		self.X = X
		self.y = y
		self.coef = np.random.uniform(size=(X.shape[1], y.shape[1]))
		self.learning_rate = learning_rate
	def stack_ones(self, arr):
		"""
		Add a column of ones to an array
		
		Parameters
		----------
		arr : array

		Returns
		---------
		arr : array
			array with a column of ones
		"""
		return np.column_stack([np.ones(len(arr)),arr])
	def predict(self, X):
		return np.dot(X, self.coef)
	def gradient(self):
		y_hat = self.predict(self.X)
		y_res = self.y - y_hat
		return(np.dot((-2.*self.X.T), y_res/self.X.shape[0]))
	def update(self):
		self.coef -= self.learning_rate*self.gradient()
	def fit(self, iterations = 1000):
		for _ in range(iterations):
			self.update()

class proximal_gradient_lasso_regression:
	def __init__(self, alpha = 1.0, max_iter = 1000):
		self.alpha = alpha
		self.intercept = 0.
		self.beta = None
		self.penalty = "l1"
		self.max_iter = max_iter
		# for sklearn
	def zscaler(self, X, axis=0, w_mean=True, w_std=True):
		data = np.zeros_like(X)
		data[:] = np.copy(X)
		if w_mean:
			data -= np.mean(data, axis)
		if w_std:
			data /= np.std(data, axis)
		return data
	def soft_thresholding_operator(self, z, theta):
		return(np.sign(z)*np.maximum((np.abs(z) - np.full_like(z, theta)), 0.0))
	def fit(self, X, y):
		self.l = self.alpha*(2*X.shape[0])
		if y.ndim == 1:
			print("Reshaping y = y.reshape(-1,1)")
			y = y.reshape(-1,1)
		self.intercept = np.mean(y,0)
		y_c = y - self.intercept
		X_ = self.zscaler(X)
		# set the learning rate (gamma) to be the maximum eigenvalue of X'X
		gamma = np.max(np.linalg.eigh(np.dot(X_.T, X_))[0])**-1
		beta = np.zeros((X.shape[1], y.shape[1]))
		for _ in range (self.max_iter):
			nabla = -np.dot(X_.T,(y_c - np.dot(X_, beta)))
			z = beta - 2*gamma*nabla
			beta = self.soft_thresholding_operator(z, self.l*gamma)
		self.beta = beta
		# for sklearn
		self.coef_ = self.beta.T
	def predict(self, X):
		X_ = self.zscaler(X)
		return(np.dot(X_, self.beta) + self.intercept)

class ridge_regression:
	def __init__(self, l = 1.):
		self.coef = None
		self.l = l
	def stack_ones(self, arr):
		"""
		Add a column of ones to an array
		
		Parameters
		----------
		arr : array

		Returns
		---------
		arr : array
			array with a column of ones
		
		"""
		return np.column_stack([np.ones(len(arr)),arr])
	def stack_zeros(self, arr):
		"""
		Add a column of ones to an array
		
		Parameters
		----------
		arr : array

		Returns
		---------
		arr : array
			array with a column of ones
		
		"""
		return np.column_stack([np.zeros(len(arr)),arr])
	def zscaler(self, X, axis=0, w_mean=True, w_std=True):
		data = np.zeros_like(X)
		data[:] = np.copy(X)
		if w_mean:
			data -= np.mean(data, axis)
		if w_std:
			data /= np.std(data, axis)
		return data
	def fit(self, X, y):
		X_ = self.zscaler(X)
		X_lambda = np.vstack((self.stack_ones(X_), self.stack_zeros(np.diag([np.sqrt(self.l)]*X.shape[1]))))
		Y_lambda = np.vstack((y, np.zeros((X.shape[1],y.shape[1]))))
		self.coef = cy_lin_lstsqr_mat(X_lambda,Y_lambda)
	def predict(self, X):
		X_ = self.zscaler(X)
		X_ = self.stack_ones(X_)
		return(np.dot(X_,self.coef))


class tm_glm:
	"""
	To-do: help for tm_glm class
	"""
	def __init__(self, endog, exog, covars = None):
		self.endog = np.array(endog)
		self.exog = exog
		self.covars = covars
		self.pval_Fmodel = None
		self.pval_Fvar = None
		self.pval_T = None
		self.reduced_endog = None
		self.root_mse = None

	def init_covariates(self, exog):
		"""
		Returns the residuals for two-step regression
		
		Parameters
		----------
		exog : array
			Independent variables to regress out.
		"""
		# copy the original data
		self.endog_raw = self.endog[:]
		# calculate the residuals
		exog_model = self.stack_ones(np.column_stack(exog.dmy_arr))
		a = cy_lin_lstsqr_mat(exog_model, self.endog)
		self.endog = self.endog - np.dot(exog_model,a)

	def column_product(self, arr1, arr2):
		"""
		Multiply two dummy codes arrays
		
		Parameters
		----------
		arr1 : array
			2D array variable dummy coded array (nlength, nvars)

		arr2 : array
			2D array variable dummy coded array (nlength, nvars)

		Returns
		---------
		prod_arr : array
			dummy coded array [nlength, nvars(arr1)*nvars(arr2)]
		
		"""
		l1 = len(arr1)
		l2 = len(arr2)
		if l1 == l2:
			arr1 = np.array(arr1)
			arr2 = np.array(arr2)
			prod_arr = []
			if arr1.ndim == 1:
				prod_arr = (arr1*arr2.T).T
			elif arr2.ndim == 1:
				prod_arr = (arr2*arr1.T).T
			else:
				for i in range(arr1.shape[1]):
					prod_arr.append((arr1[:,i]*arr2.T).T)
				prod_arr = np.array(prod_arr)
				if prod_arr.ndim == 3:
					prod_arr = np.concatenate(prod_arr, axis=1)
			prod_arr[prod_arr==0]=0
			return prod_arr
		else:
			print("Error: arrays must be of same length")
			quit()

	def stack_ones(self, arr):
		"""
		Add a column of ones to an array
		
		Parameters
		----------
		arr : array

		Returns
		---------
		arr : array
			array with a column of ones
		
		"""
		return np.column_stack([np.ones(len(arr)),arr])

	def calculate_pvalues_T(self, two_sided = True):

		try:
			self.pval_T = t.sf(np.abs(self.Tvalues), self.DF_Total)
		except:
			print("T-values are missing. Run glm_T().")
		if two_sided:
			self.pval_T = self.pval_T*2

	def calculate_pvalues_F(self):
		try:
			self.pval_Fmodel = f.sf(self.Fmodel, self.DF_Between, self.DF_Within)
		except:
			print("T-values are missing. Run glm_F().")
		if self.Fvar is not None:
			pvalues = []
			for i, dfn in enumerate(self.exog.k):
				pvalues.append(f.sf(self.Fvar[i], dfn, self.DF_Within))
			self.pval_Fvar = np.array(pvalues)


	def nonparametric_fwer_correction(self, perm_arr, stat_type = 'T'):
		assert stat_type in ['T', 'F'], "Please specifiy statistic type: T or F."
		perm_arr = np.array(perm_arr)
		if stat_type == 'T':
			try:
				self.Tvalues
			except:
				print("Run self.glm_T() first.")
				return
			n_perm, n_factors = perm_arr.shape
			pval_corrected = []
			for i in range(n_factors):
				temp = perm_arr[:,i]
				temp.sort()
				pval_corrected.append(1 - np.divide(np.searchsorted(temp, np.abs(self.Tvalues[i,:])), n_perm))
			self.pval_T_corrected = np.array(pval_corrected)
		if stat_type == 'F':
			try:
				self.Fmodel
			except:
				print("Run self.glm_F() first.")
				return
			n_factors = len(self.exog.name)
			n_perm = perm_arr.shape[0]
			perm_arr = np.array(perm_arr)
			if perm_arr.ndim == 1:
				temp = perm_arr[:]
			else:
				temp = perm_arr[:,0]
			temp.sort()
			Fvalues = self.Fmodel
			self.pval_Fmodel_corrected = 1 - np.divide(np.searchsorted(temp, np.abs(Fvalues)), n_perm)
			# get the permutated f-values the factor separately for readabiltiy
			if n_factors == 1:
				self.pval_Fvar_corrected = self.pval_Fmodel_corrected
			else:
				pval_corrected = []
				# strip the model F permutations
				temp = perm_arr[:,1:]
				for i in range(n_factors):
					print("Factor %d" % i)
					temp_corr = (1 - np.divide(np.searchsorted(temp[:,i], np.abs(self.Fvar[i])), n_perm))
					pval_corrected.append(temp_corr)
#					pval_corrected.append(1 - np.divide(np.searchsorted(temp[:,i], np.abs(self.Fvar[i])), n_perm))
				self.pval_Fvar_corrected = np.array(pval_corrected)

	def multiple_comparison_correction(self, correction = 'FDR', alpha = 0.05, n_perm = 10000):
		"""
		"""
		assert correction in ['FDR', 'Bonferroni', 'Randomisation'], "Correction method [%s] is not understood. Only FDR and Bonferroni are currently available" % correction
		cm = 'fdr_bh'
		if correction == 'Bonferroni':
			cm = 'bonferroni'

		if self.pval_Fmodel is not None:
			if correction == 'Randomisation':
				f_perm = np.max([self.glm_F(iterator = i, randomise = True) for i in range(n_perm)],1)
				self.nonparametric_fwer_correction(f_perm, stat_type = 'F')
			else:
				self.pval_Fmodel_corrected = multipletests(self.pval_Fmodel, alpha = alpha, method = cm)[1]

		if self.pval_Fvar is not None:
			num_variable = len(self.exog.name)
			if num_variable == 1:
				self.pval_Fvar_corrected = self.pval_Fmodel_corrected
			else:
				pval_corrected = []
				if correction == 'Randomisation':
					pass
				else:
					for i in range(num_variable):
						pval_corrected.append(multipletests(self.pval_Fvar[i], alpha = alpha, method=cm, is_sorted=False, returnsorted=False)[1])
					self.pval_Fvar_corrected = np.array(pval_corrected)
		if self.pval_T is not None:
			num_variable = self.Tvalues.shape[0]
			pval_corrected = []
			if correction == 'Randomisation':
				t_perm = np.max([self.glm_T(iterator = i, randomise = True) for i in range(10000)],2)
				self.nonparametric_fwer_correction(t_perm, stat_type = 'T')
			else:
				for i in range(num_variable):
					pval_corrected.append(multipletests(self.pval_T[i], alpha = alpha, method=cm, is_sorted=False, returnsorted=False)[1])
				self.pval_T_corrected = np.array(pval_corrected)

	def glm_T(self, iterator = 0, blocking = None, randomise = False, output_statistics = False):

		endog = self.endog
		exog = self.exog
		covars = self.covars

		assert len(exog.dmy_arr) == len(exog.name), "[Error]: Exogenous dummy array doesn't match names. This likely means the exogeneous object is not defined properly."
		n = len(exog.dmy_arr[0])

		# Check that endog has two dimensions
		if endog.ndim == 1:
			endog = endog.reshape(len(endog),1)

		exog_model = self.stack_ones(np.column_stack(exog.dmy_arr))

		k = exog_model.shape[1]
		self.DF_Total = n - 1

		if covars is not None:
			covars_model = np.column_stack(covars.dmy_arr)
			exog_model = np.column_stack((exog_model, covars_model))

		if randomise:
			if self.reduced_endog is None:
				self.reduced_endog = cy_lin_lstsqr_mat_residual(exog_model, self.endog)[1]
			rand_array = np.random.permutation(list(range(n)))
			endog = self.reduced_endog[rand_array]

		a = cy_lin_lstsqr_mat_residual(exog_model, endog)[0]
		sigma2 = np.sum((endog - np.dot(exog_model, a))**2,axis=0) / (n - k)
		invXX = np.linalg.inv(np.dot(exog_model.T, exog_model))

		if endog.ndim == 1:
			se = np.sqrt(np.diag(sigma2 * invXX))
		else:
			num_depv = endog.shape[1]
			se = fast_se_of_slope(invXX, sigma2)
		# return permuted T-values
		if randomise:
			return(np.divide(a, se))
		else:
			self.Tvalues = np.divide(a, se)
			if output_statistics:
				return(self.Tvalues)

	def glm_F(self, iterator = 0, blocking = None, randomise = False, output_statistics = False):

		endog = self.endog
		exog = self.exog
		covars = self.covars

		assert len(exog.dmy_arr) == len(exog.name), "[Error]: Exogenous dummy array doesn't match names. This likely means the exogeneous object is not defined properly."
		n = len(exog.dmy_arr[0])

		# Check that endog has two dimensions
		if endog.ndim == 1:
			endog = endog.reshape(len(endog),1)

		exog_model = self.stack_ones(np.column_stack(exog.dmy_arr))
		if covars is not None:
			covars_model = np.column_stack(covars.dmy_arr)
			exog_model = np.column_stack((exog_model, covars_model))

		# create the reduced endogenous variable for permutation testing.
		if randomise:
			if self.reduced_endog is None:
				a = cy_lin_lstsqr_mat(exog_model, self.endog)
				self.reduced_endog = self.endog - np.dot(exog_model,a)
			rand_array = np.random.permutation(list(range(n)))
			endog = self.reduced_endog[rand_array]

		k =  np.sum(exog.k) + 1

		self.DF_Total = n - 1
		self.DF_Between = k - 1 # aka df model
		self.DF_Within = n - k # aka df residuals

		SS_Total = np.sum((endog - np.mean(endog,0))**2,0)
		SS_Residuals = cy_lin_lstsqr_mat_residual(exog_model, endog)[1]

		self.R2 = 1 - np.divide(SS_Residuals, SS_Total)
		self.R2_adj = 1 - np.divide(((1-self.R2)*self.DF_Total), self.DF_Within)

		SS_Between = SS_Total - SS_Residuals
		MS_Residuals = np.divide(SS_Residuals, self.DF_Within)
		self.root_mse = np.sqrt(MS_Residuals)
		Fmodel = np.divide(np.divide(SS_Between, self.DF_Between), MS_Residuals)

		Fvar = []
		PartialEtaSqr = []
		EtaSqr = []
		if len(exog.name) == 1: # do nothing if there is only one factor
			self.Fvar = Fvar = None
		else:
			for i in range(len(exog.name)):
				temp = exog.dmy_arr[:]
				_ = temp.pop(i)
				temp_exog = self.stack_ones(np.column_stack(temp))
				if covars is not None:
					temp_exog = np.column_stack((temp_exog, covars_model))
				SS_model = np.array(SS_Total - cy_lin_lstsqr_mat_residual(temp_exog, endog)[1])
				SS_var = (SS_Between - SS_model)
				Ftemp = np.divide(SS_var, (MS_Residuals*2))
				EtaSqrtemp = SS_var / SS_Total
				PartialEtaSqrtemp = SS_var / (SS_Residuals + SS_var)
				Fvar.append(Ftemp)
				PartialEtaSqr.append(PartialEtaSqrtemp)
				EtaSqr.append(EtaSqrtemp)
			Fvar = np.array(Fvar)
			PartialEtaSqr = np.array(PartialEtaSqr)
			EtaSqr = np.array(EtaSqr)

		# return permuted F-values or save the model
		if randomise:
			if Fvar is not None:
				return(np.column_stack((Fmodel, Fvar.T)))
			else:
				return(Fmodel)
		else:
			self.Fmodel = Fmodel
			self.Fvar = Fvar
			self.PartialEtaSqr = PartialEtaSqr
			self.EtaSqr = EtaSqr
			if output_statistics:
				return(self.Fmodel, self.Fvar)

	def calculate_cosinor_metrics(self, period_arr, exogenoues_variable, two_step_regression = False, calculate_cosinor_stats = False):
		"""
		https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3991883/
		https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3663600/
		"""
		endog = self.endog
		exog = self.exog
		covars = self.covars

		# re-order the variables.
		init_cosinor_flag = True
		init_other_flag = True
		for i, var_type in enumerate(self.exog.exog_type):
			if var_type == 'cosine':
				if init_cosinor_flag:
					reordered_exog = exogenoues_variable(name = self.exog.name[i],
								dmy_arr = self.exog.dmy_arr[i],
								exog_type = 'cosine')
					init_cosinor_flag = False
				else:
					reordered_exog.add_var(name = self.exog.name[i],
								dmy_arr = self.exog.dmy_arr[i],
								exog_type = 'cosine')
			else: 
				if init_other_flag:
					other_exog = exogenoues_variable(name = self.exog.name[i],
								dmy_arr = self.exog.dmy_arr[i],
								exog_type = var_type)
					init_other_flag = False
				else:
					other_exog.add_var(name = self.exog.name[i],
								dmy_arr = self.exog.dmy_arr[i],
								exog_type = var_type)

		if init_other_flag == False:
			covars_model = self.stack_ones(np.column_stack(other_exog.dmy_arr))
			a = cy_lin_lstsqr_mat(covars_model, self.endog)
			endog_reduced = self.endog - np.dot(covars_model,a)
		else:
			endog_reduced = self.endog[:]
			exog_model_cosinor = self.stack_ones(np.column_stack(reordered_exog.dmy_arr))
		if two_step_regression:
			endog = endog_reduced
			exog_model = exog_model_cosinor
		else:
			endog = self.endog[:]
			exog_model = self.stack_ones(np.column_stack(reordered_exog.dmy_arr))
			if init_other_flag == False:
				exog_model = np.column_stack((exog_model, np.column_stack(other_exog.dmy_arr)))

		if calculate_cosinor_stats:
			n = len(self.exog.dmy_arr[0])
			k = np.sum(reordered_exog.k) + 1

			DF_Total = n - 1
			DF_Between = k - 1 # aka df model
			DF_Within = n - k # aka df residuals

			SS_Total = np.sum((endog_reduced - np.mean(endog_reduced,0))**2,0)
			SS_Residuals = cy_lin_lstsqr_mat_residual(exog_model_cosinor, endog_reduced)[1]

			self.R2_cosinor = 1 - np.divide(SS_Residuals, SS_Total)
			self.R2_cosinor_adj = 1 - np.divide(((1-self.R2_cosinor)*DF_Total), DF_Within)

			SS_Between = SS_Total - SS_Residuals
			MS_Residuals = np.divide(SS_Residuals, DF_Within)
			self.Fcosinor = np.divide(np.divide(SS_Between, DF_Between), MS_Residuals)
			self.pval_Fcosinor = f.sf(self.Fcosinor, DF_Between, DF_Within)


		a, SS_Residuals = cy_lin_lstsqr_mat_residual(exog_model,endog)
		AMPLITUDE = []
		ACROPHASE = []
		MESOR = a[0]
		for i in range(len(period_arr)):
			# beta, gamma
			AMPLITUDE.append(np.sqrt((a[1+(i*2),:]**2) + (a[2+(i*2),:]**2)))
			# Acrophase calculation
			if i == 0: # awful hack
				ACROPHASE = np.arctan(np.abs(np.divide(-a[2+(i*2),:], a[1+(i*2),:])))
				ACROPHASE = ACROPHASE[np.newaxis,:]
			else:
				temp_acro = np.arctan(np.abs(np.divide(-a[2+(i*2),:], a[1+(i*2),:])))
				temp_acro = temp_acro[np.newaxis,:]
				ACROPHASE = np.append(ACROPHASE,temp_acro, axis=0)
			ACROPHASE = np.array(ACROPHASE)
			ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] >= 0)] = -ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] >= 0)]
			ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] < 0)] = (-1*np.pi) + ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] < 0)]
			ACROPHASE[i, (a[2+(i*2),:] < 0) & (a[1+(i*2),:] <= 0)] = (-1*np.pi) - ACROPHASE[i, (a[2+(i*2),:] < 0) & (a[1+(i*2),:] <= 0)]
			ACROPHASE[i, (a[2+(i*2),:] <= 0) & (a[1+(i*2),:] > 0)] = (-2*np.pi) + ACROPHASE[i, (a[2+(i*2),:] <= 0) & (a[1+(i*2),:] > 0)]

		# Make human readable (24H) acrophase
		ACROPHASE_24 = np.zeros_like(ACROPHASE)
		for j, per in enumerate(period_arr):
			acrotemp = np.abs(ACROPHASE[j]/(2*np.pi)) * per
			acrotemp[acrotemp>per] -= per
			ACROPHASE_24[j] = acrotemp

		self.period = period_arr
		self.MESOR = MESOR
		self.AMPLITUDE = np.array(AMPLITUDE)
		self.ACROPHASE = np.array(ACROPHASE)
		self.ACROPHASE_24 = np.array(ACROPHASE_24)


#class spls_mixOmics_rwrapper:
#	"""
#	Wrapper that uses the mixomics.spls r package, and rpy2
#	https://cran.r-project.org/web/packages/spls/

#	Package Reference:
#	Rohart F, Gautier B, Singh A, and Le Cao K-A (2017) mixOmics: An R
#	package for 'omics feature selection and multiple data integration.
#	PLoS computational biology 13(11):e1005752

#	Function refereces:
#	LE Cao, K.-A., Martin, P.G.P., Robert-Granie, C. and Besse, P. (2009). Sparse canonical methods
#	for biological data integration: application to a cross-platform study. BMC Bioinformatics 10:34.
#	LE Cao, K.-A., Rossouw, D., Robert-Granie, C. and Besse, P. (2008). A sparse PLS for variable
#	selection when integrating Omics data. Statistical Applications in Genetics and Molecular Biology
#	7, article 35.
#	
#	Sparse SVD: Shen, H. and Huang, J. Z. (2008). Sparse principal component analysis via regularized
#	low rank matrix approximation. Journal of Multivariate Analysis 99, 1015-1034.
#	PLS methods: Tenenhaus, M. (1998). La regression PLS: theorie et pratique. Paris: Editions
#	Technic. Chapters 9 and 11.
#	
#	Abdi H (2010). Partial least squares regression and projection on latent structure regression (PLS
#	Regression). Wiley Interdisciplinary Reviews: Computational Statistics, 2(1), 97-106.
#	Wold H. (1966). Estimation of principal components and related models by iterative least squares.
#	In: Krishnaiah, P. R. (editors), Multivariate Analysis. Academic Press, N.Y., 391-420.
#	"""

#X,
#Y,
#ncomp = 2,
#mode = c("regression", "canonical", "invariant", "classic"),
#keepX,
#keepY,
#scale = TRUE,
#tol = 1e-06,
#max.iter = 100,
#near.zero.var = FALSE,
#logratio = "none",
#multilevel = NULL,
#all.outputs = TRUE

#	def __init__(self, n_components, spls_mode = "regression", keepX = None, keepY = None, tol = 1e-6, max_iter = 100, all_output = True):
#		"""
#		Setting for spls
#		
#		Parameters
#		----------
#		n_components : array
#			The number of components to fit
#		eta : float
#			The regularization coefficient ranging from 0 to <1. At Eta = 0, sPLS is equivalent to PLS
#		kappa : float
#			Parameter to control the effect of the concavity of the objective function and the closeness of original and surrogate direction vector. kappa should be between 0 and 0.5 (default = 0.5).
#		max_iter : int
#			Maximum number of interactions for fitting direction vector (default = 100).
#		algorithm_selection : str
#			PLS algorithm for variable selection (default = "pls2"). Choices: {"pls2", "simpls"}
#		algorithm_selection : str
#			PLS algorithm for model fitting (default = "simpls"). Choices: {"kernelpls", "widekernelpls", "simpls","oscorespls"}
#		scale_x : bool
#			z-scale X
#		scale_y : bool
#			z-scale y
#		effective_zero : float
#			The threshold for effect zero (default = 0.0001)
#		Returns
#		---------
#		The sPLS function
#		"""
#		spls_modes = ["regression", "canonical", "invariant", "classic"]
#		assert spls_mode in spls_modes, "Error: sPLS model [%s] not recognized" % spls_mode

#		self.n_components = n_components
#		self.spls_mode = spls_mode
#		self.keepX = keepX
#		self.keepY = keepY
#		self.max_iter = max_iter
#		self.tol = tol
#		self.all_output = all_output
#		self.penalty = "l1"



