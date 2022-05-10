#!/usr/bin/env python

import itertools
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import t, f, norm, chi2
from scipy.linalg import pinv
from statsmodels.stats.multitest import multipletests, fdrcorrection
from joblib import Parallel, delayed

from sklearn.preprocessing import scale
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.cross_decomposition import PLSRegression, CCA, PLSCanonical

from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

# suppress console because of weird permission around r
import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)
import warnings
warnings.filterwarnings('ignore') 

from sparsestatsfunc.cynumstats import cy_lin_lstsqr_mat_residual, cy_lin_lstsqr_mat, fast_se_of_slope, tval_fast

stats = importr('stats')
base = importr('base')
utils = importr('utils')

# autoinstalls the r packages... not the smartest thing to do.
# create an install R + packages script later
try:
	from sparsecca._cca_pmd import cca as sparsecca
	have_sparsecca = True
except:
	have_sparsecca = False
try:
	pma = importr('PMA')
except:
	utils.install_packages('PMA')
	pma = importr('PMA')
try:
	spls = importr('spls')
except:
	utils.install_packages('spls')
	pma = importr('spls')

def zscaler_XY(X, y, axis=0, w_mean=True, scale_x = True, scale_y = True):
	"""
	Applies scaling to X and y, return means and std in all cases.
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

def generate_seeds(n_seeds, maxint = int(2**32 - 1)):
	return([np.random.randint(0, maxint) for i in range(n_seeds)])


class parallel_scca():

	def __init__(self, n_jobs = 8, n_permutations = 10000):
		"""
		Main SCCA function
		"""
		self.n_jobs = n_jobs
		self.n_permutations = n_permutations
	def _check(self):
		print("2022_03_01")
	def index_perm(self, unique_arr, arr, variable, within_group = True):
		"""
		Shuffles an array within group (within_group = True) or the groups (within_group = False)
		"""
		rng = np.random.default_rng()
		if within_group:
			perm_u = unique_arr
		else:
			perm_u = rng.permutation(unique_arr)
		out = []
		for unique in perm_u:
			if within_group:
				out.append(rng.permutation(variable[unique == arr]))
			else:
				out.append(variable[unique == arr])
		return np.concatenate(out)
	def nfoldsplit_group(self, group, n_fold = 10, holdout = 0, train_index = None, verbose = False, debug_verbose = False, seed = None):
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
		
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)
		
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
	def create_nfold(self, X, y, group, n_fold = 10, holdout = 0.3, verbose = True):
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


	def _scca_params_cvgridsearch(self, X, y, n_components, l1x_pen, l1y_pen, group, train_index, fold_indices, n_reshuffle = 1, max_iter = 20, verbose = True, optimize_x = False, optimize_y = False, optimize_primary_component = False, optimize_global_redundancy_index = False, optimize_selected_variables = True, seed = None):
		"""
		return CV 
		"""
		if n_reshuffle > 1:
			if seed is None:
				np.random.seed(np.random.randint(4294967295))
			else:
				np.random.seed(seed)
		p = 0
		n_fold = len(fold_indices)
		fold_index = np.arange(0,self.n_fold_,1)
		temp_Q2 = np.zeros((n_fold*n_reshuffle))
		for r in range(n_reshuffle):
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
				n_samples = tmpX_train.shape[0]
				n_features = tmpX_train.shape[1]
				n_targets = tmpY_train.shape[1]
				cvscca = scca_rwrapper(n_components = n_components,
												X_L1_penalty = l1x_pen,
												y_L1_penalty = l1y_pen,
												max_iter = max_iter).fit(tmpX_train, tmpY_train, calculate_loadings = optimize_global_redundancy_index)
				selx = np.ones((self.X_train_.shape[1]), dtype = bool)
				sely = np.ones((self.y_train_.shape[1]), dtype = bool)
				if optimize_selected_variables:
					selx[cvscca.x_selectedvariablesindex_ == 0] = False
					sely[cvscca.y_selectedvariablesindex_ == 0] = False
				if optimize_primary_component:
					temp_cors = cvscca.canonicalcorr(tmpX_test, tmpY_test)[0]
					temp_Q2[p] = temp_cors**2
				elif optimize_global_redundancy_index:
					temp_Q2[p] = cvscca.x_redundacy_variance_explained_global_
				else: 
					tmpX_test_predicted, tmpY_test_predicted = cvscca.predict(X = tmpX_test,
																								y = tmpY_test,
																								toself = True)
					pscorex = cvscca._rscore(tmpX_test[:,selx], tmpX_test_predicted[:,selx])
					pscorey = cvscca._rscore(tmpY_test[:,sely], tmpY_test_predicted[:,sely])
					if (optimize_x*optimize_y)==1:
						temp_Q2[p] = np.divide((pscorex + pscorey), 2)
					elif optimize_x:
						temp_Q2[p] = pscorex
					elif optimize_y:
						temp_Q2[p] = pscorey
					else:
						temp_Q2[p] = np.divide((pscorex + pscorey), 2)
				p+1
		Q2_mean = np.mean(temp_Q2)
		Q2_std = np.std(temp_Q2)
		if verbose:
			if optimize_primary_component:
				print("FINISHED: Comp %d, l1x = %1.3f, l1y = %1.3f, CC1 = %1.3f +/- %1.3f" % (n_components, l1x_pen, l1y_pen, Q2_mean, Q2_std))
			else:
				print("FINISHED: Comp %d, l1x = %1.3f, l1y = %1.3f, mean global correlation = %1.3f +/- %1.3f" % (n_components, l1x_pen, l1y_pen, Q2_mean, Q2_std))
		return(Q2_mean, Q2_std)

	def nfold_cv_params_search_scca(self, l1x_range = np.arange(0.1,1.1,.1), l1y_range = np.arange(0.1,1.1,.1), n_reshuffle = 1, max_iter = 20, optimize_x = False, optimize_y = False, optimize_primary_component = False, optimize_global_redundancy_index = False, optimize_selected_variables = True, max_n_comp = None, debug = False):
		if optimize_primary_component:
			max_n_comp = 1
		if max_n_comp is None:
			# auto select max number of components of smallest feature
			n_samples = self.X_test_.shape[0]
			n_features = self.X_test_.shape[1]
			n_targets = self.y_test_.shape[1]
			max_n_comp = int(min(n_samples, n_features, n_targets))
			print("Search for up to %d components (Sqrt of min(n_samples, n_features, n_targets)" % (max_n_comp))
		component_range = np.arange(1,(max_n_comp+1),1)
		search_i_size = len(component_range)
		search_j_size = len(l1x_range)
		search_k_size = len(l1y_range)
		Q2_GRIDSEARCH = np.zeros((search_i_size, search_j_size, search_k_size))
		Q2_GRIDSEARCH_SD = np.zeros((search_i_size, search_j_size, search_k_size))
		# because of indexing, everything is passed to the gridsearch. The function (above) only optimizes training data! np.sort(np.concatenate(self.fold_indices_)) == self.train_index_
		
		# generate independent seeds for the grid search. This is doesn't do anything unless n_reshuffle > 1.
		n_seeds  = len(list(itertools.product(range(search_i_size), range(search_j_size), range(search_k_size))))
		seed_grid = np.array(generate_seeds(n_seeds)).reshape(search_i_size, search_j_size, search_k_size)
		output = Parallel(n_jobs=self.n_jobs, backend='multiprocessing')(delayed(self._scca_params_cvgridsearch)(X = self.X_, 
																																	y = self.y_,
																																	n_components = component_range[i],
																																	l1x_pen = l1x_range[j],
																																	l1y_pen = l1y_range[k],
																																	group = self.group_,
																																	train_index = self.train_index_,
																																	fold_indices = self.fold_indices_,
																																	n_reshuffle = n_reshuffle,
																																	optimize_x = optimize_x,
																																	optimize_y = optimize_y,
																																	optimize_primary_component = optimize_primary_component,
																																	optimize_global_redundancy_index = optimize_global_redundancy_index,
																																	optimize_selected_variables = optimize_selected_variables,
																																	max_iter = max_iter,
																																	seed = seed_grid[i, j, k]) for i, j, k in list(itertools.product(range(search_i_size), range(search_j_size), range(search_k_size))))
		output_mean, output_sd = zip(*output)
		count = 0
		best_component = 0
		best_l1_x = 0
		best_l1_y = 0
		highest = 0
		for i, j, k in list(itertools.product(range(search_i_size), range(search_j_size), range(search_k_size))):
			Q2_GRIDSEARCH[i,j,k] = output_mean[count]
			Q2_GRIDSEARCH_SD[i,j,k] = output_sd[count]
			if output_mean[count] > highest:
				highest = output_mean[count]
				best_component = component_range[i]
				best_l1_x = l1x_range[j]
				best_l1_y = l1y_range[k]
				if optimize_global_redundancy_index:
					print("Current best Global Redundacy Index (X) = %1.3f [Components = %d, l1[x] penalty = %1.2f, and l1[y] penalty = %1.2f]" % (highest, best_component, best_l1_x, best_l1_y))
				else:
					print("Current best prediction scores = %1.3f [Components = %d, l1[x] penalty = %1.2f, and l1[y] penalty = %1.2f]" % (highest, best_component, best_l1_x, best_l1_y))
			count+=1
		if highest == 0:
			print("Q-squared was never above zero")
		if debug:
			self.output = output
		self.STAT_GRIDSEARCH_ = np.array(Q2_GRIDSEARCH)
		self.STAT_GRIDSEARCH_SD_ = np.array(Q2_GRIDSEARCH_SD)
		self.GRIDSEARCH_BEST_COMPONENT_ = best_component
		self.GRIDSEARCH_L1X_PENALTY_ = best_l1_x
		self.GRIDSEARCH_L1Y_PENALTY_ = best_l1_y
		self.l1x_gridsearch_range_ = l1x_range
		self.l1y_gridsearch_range_ = l1y_range
		self.gridsearch_maxncomp_ = max_n_comp

	def _scca_canonical_corr_cvgridsearch(self, l1x_pen, l1y_pen, n_components, max_iter = 20, verbose = True, maxcc = False):
		n_fold = self.n_fold_
		fold_index = np.arange(0,self.n_fold_,1)
		fold_indices = self.fold_indices_
		X = self.X_
		y = self.y_
		tempz = np.zeros((n_fold))
		for n in range(n_fold):
			sel_train = fold_indices[n]
			sel_test = np.concatenate(fold_indices[fold_index != n])
			tmpX_train = X[sel_train]
			tmpY_train = y[sel_train]
			tmpX_test = X[sel_test]
			tmpY_test = y[sel_test]
			cvscca = scca_rwrapper(n_components = n_components,
											X_L1_penalty = l1x_pen,
											y_L1_penalty = l1y_pen,
											max_iter = max_iter).fit(tmpX_train, tmpY_train, calculate_loadings = False)
			# fisher z-transformation of correlations
			fisherz_cancor = np.arctanh(cvscca.canonicalcorr(tmpX_test, tmpY_test))
			# calculate p-values for test data
			pval = self._pearsonr_to_t(abs(fisherz_cancor), len(tmpX_test))[1]
			# fisher's method to calculate chi2
			chi2_stat = -2*np.sum(np.log10(pval))
			# convert to z-values
			tempz[n] = norm.ppf(chi2.sf(chi2_stat,n_components*2))*-1
		if maxcc:
			outstat = np.max(tempz)
			outstd = np.std(tempz)
			if verbose:
				print("FINISHED: Comp %d, l1x = %1.3f, l1y = %1.3f, Best Z = %1.3f +/- %1.3f" % (n_components, l1x_pen, l1y_pen, outstat, outstd))
		else:
			outstat = np.mean(tempz)
			outstd = np.std(tempz)
			if verbose:
				print("FINISHED: Comp %d, l1x = %1.3f, l1y = %1.3f, Average Z = %1.3f +/- %1.3f" % (n_components, l1x_pen, l1y_pen, outstat, outstd))
		return(outstat, outstd)

	def nfold_cv_canonical_corr_gridsearch(self, l1x_range = np.arange(0.1,1.1,.1), l1y_range = np.arange(0.1,1.1,.1), max_iter = 20, max_n_comp = None, debug = False, maxcc = False):
		if max_n_comp is None:
			# auto select max number of components of smallest feature
			n_samples = self.X_test_.shape[0]
			n_features = self.X_test_.shape[1]
			n_targets = self.y_test_.shape[1]
			max_n_comp = int(min(n_samples, n_features, n_targets))
			print("Search for up to %d components (Sqrt of min(n_samples, n_features, n_targets)" % (max_n_comp))
		component_range = np.arange(1,(max_n_comp+1),1)
		search_i_size = len(component_range)
		search_j_size = len(l1x_range)
		search_k_size = len(l1y_range)
		meanZ_GRIDSEARCH = np.zeros((search_i_size, search_j_size, search_k_size))
		meanZ_GRIDSEARCH_SD = np.zeros((search_i_size, search_j_size, search_k_size))
		n_seeds  = len(list(itertools.product(range(search_i_size), range(search_j_size), range(search_k_size))))
		output = Parallel(n_jobs=self.n_jobs, backend='multiprocessing')(delayed(self._scca_canonical_corr_cvgridsearch)(l1x_pen = l1x_range[j],
																																	l1y_pen = l1y_range[k],
																																	n_components = component_range[i],
																																	max_iter = max_iter,
																																	maxcc = maxcc) for i, j, k in list(itertools.product(range(search_i_size), range(search_j_size), range(search_k_size))))
		output_mean, output_sd = zip(*output)
		count = 0
		best_component = 0
		best_l1_x = 0
		best_l1_y = 0
		highest = 0
		for i, j, k in list(itertools.product(range(search_i_size), range(search_j_size), range(search_k_size))):
			meanZ_GRIDSEARCH[i,j,k] = output_mean[count]
			meanZ_GRIDSEARCH_SD[i,j,k] = output_sd[count]
			if output_mean[count] > highest:
				highest = output_mean[count]
				best_component = component_range[i]
				best_l1_x = l1x_range[j]
				best_l1_y = l1y_range[k]
				print("Current best prediction stat = %1.3f [Components = %d, l1[x] penalty = %1.2f, and l1[y] penalty = %1.2f]" % (highest, best_component, best_l1_x, best_l1_y))
			count+=1
		if highest == 0:
			print("Z-score was never above zero (should not occur...)")
		self.STAT_GRIDSEARCH_ = np.array(meanZ_GRIDSEARCH)
		self.STAT_GRIDSEARCH_SD_ = np.array(meanZ_GRIDSEARCH_SD)
		self.l1x_gridsearch_range_ = l1x_range
		self.l1y_gridsearch_range_ = l1y_range
		self.gridsearch_maxncomp_ = max_n_comp

	def plot_gridsearch(self, png_basename = None, component = None, nan_unstable = False, cmap = 'jet'):
		l1x_range = self.l1x_gridsearch_range_
		l1y_range = self.l1y_gridsearch_range_
		if component is not None:
			vmax = np.max(self.STAT_GRIDSEARCH_)
			vmin = 0
			Q2_SEARCH = np.array(self.STAT_GRIDSEARCH_[component-1]).T
			if nan_unstable:
				Q2_SEARCH[Q2_SEARCH < 0] = np.nan
			else:
				Q2_SEARCH[Q2_SEARCH < 0] = 0
			plt.imshow(Q2_SEARCH, interpolation = None, cmap = cmap, vmin = vmin, vmax = vmax)
			plt.xticks(range(len(l1x_range)),[s[:3] for s in l1x_range.astype(str)])
			plt.xlabel('L1(X) Penalty')
			plt.yticks(range(len(l1y_range)),[s[:3] for s in l1y_range.astype(str)])
			plt.ylabel('L1(Y) Penalty')
			plt.colorbar()
			plt.title("Mean CV Statistic: Component %d" % component)
			plt.gca().invert_yaxis()
			plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)
			if png_basename is not None:
				plt.savefig("%s_gridsearch_stat_component%d.png" % (png_basename, component))
				plt.close()
			else:
				plt.show()
		else:
			for c in range(self.gridsearch_maxncomp_):
				component = int(c+1)
				vmax = np.max(self.STAT_GRIDSEARCH_)
				vmin = 0
				Q2_SEARCH = np.array(self.STAT_GRIDSEARCH_[c]).T
				if nan_unstable:
					Q2_SEARCH[Q2_SEARCH < 0] = np.nan
				else:
					Q2_SEARCH[Q2_SEARCH < 0] = 0
				plt.imshow(Q2_SEARCH, interpolation = None, cmap = cmap, vmin = vmin, vmax = vmax)
				plt.xticks(range(len(l1x_range)),[s[:3] for s in l1x_range.astype(str)])
				plt.xlabel('L1(X) Penalty')
				plt.yticks(range(len(l1y_range)),[s[:3] for s in l1y_range.astype(str)])
				plt.ylabel('L1(Y) Penalty')
				plt.colorbar()
				plt.title("Q-Squared [CV] Component %d" % component)
				plt.gca().invert_yaxis()
				plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)
				if png_basename is not None:
					plt.savefig("%s_gridsearch_stat_component%d.png" % (png_basename, component))
					plt.close()
				else:
					plt.show()


	def _pearsonr_to_t(self, r, N):
		tvalues = r / np.sqrt(np.divide((1-(r*r)),(N-2)))
		pvalues = t.sf(np.abs(tvalues), N-1)*2
		return(tvalues, pvalues)

	def _bootstrap_loadings(self, i, seed = None):
		"""
		Base function to calculates boostrapped loading.
		"""
		assert hasattr(self,'model_obj_'), "Error: run fit_model"
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)
		if i % 200 == 0:
			print(i)
		subindex = np.array(self.train_index_)
		bindex = np.random.choice(subindex, replace = True, size = len(subindex))
		bxscore, byscore = self.model_obj_.transform(self.X_[bindex], self.y_[bindex])
		bxloading = self.model_obj_._calculate_loadings(bxscore, self.X_[bindex])
		byloading = self.model_obj_._calculate_loadings(byscore, self.y_[bindex])
		return(bxloading, byloading)

	def run_model_bootstrap_loadings(self, n_bootstrap = 10000):
		"""
		Bootstrap values for the training data's loadings with replacement. It's probably better to just use the permutation testing for signficance testing.
		(1) Bootstrapped training index created with replacement (allows duplicates) with same length as training data
		(2) Loadings are estimated for each bootstrap
		(3) Bootrapped distribution can be used to create normative confidence intervals
		"""
		assert hasattr(self,'model_obj_'), "Error: run fit_model"
		seeds = generate_seeds(n_bootstrap)
		output = Parallel(n_jobs = self.n_jobs, backend='multiprocessing')(delayed(self._bootstrap_loadings)(i, seed = seeds[i]) for i in range(n_bootstrap))
		bxloading, byloading = zip(*output)
		self.model_boostrapping_loadings_X_train_ = np.array(bxloading)
		self.model_boostrapping_loadings_Y_train = np.array(byloading)

	def _permute_loadings(self, i, permute_test_data = False, seed = None):
		"""
		Base function to calculates permuted loading.
		"""
		assert hasattr(self,'model_obj_'), "Error: run fit_model"
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)
		if i % 200 == 0:
			print(i)
		if permute_test_data:
			data_index = self.test_index_
		else:
			data_index = self.train_index_
		pxscore, pyscore = self.model_obj_.transform(np.random.permutation(self.X_[data_index]), np.random.permutation(self.y_[data_index]))
		pxloading = self.model_obj_._calculate_loadings(pxscore, self.X_[data_index])
		pyloading = self.model_obj_._calculate_loadings(pyscore, self.y_[data_index])
		return(pxloading, pyloading)

	def run_model_permute_loadings(self):
		"""
		Created permuted distributions for the data views from the training model. This can be used to calculate family-wise error rate corrections.
		(1) Scores are estimated using the training model using permuted training data.
		(2) Loadings estimated by correlating the permuted score and the training data.
		(3) Null distribution per component can be used to calculate permuted p-values and family-wise error rate corrected p-values. 
		e.g., pcrit = np.sort(np.max(pxloading[:,componenent,:],1))[int(Nperm*0.95)]
		Note, the correlations are already z-transformed (fischer transformation). i.e., np.arctanh(loadings) = loadings
		"""
		seeds = generate_seeds(self.n_permutations)
		output = Parallel(n_jobs = self.n_jobs, backend='multiprocessing')(delayed(self._permute_loadings)(i, seed = seeds[i]) for i in range(self.n_permutations))
		pxloading, pyloading = zip(*output)
		self.model_permutations_loadings_X_train_ = np.array(pxloading)
		self.model_permutations_loadings_X_train_ = np.array(pyloading)

	def fit_model(self, n_components, X_L1_penalty, y_L1_penalty, max_iter = 20, toself = False):
		"""
		Calculates r_train, r_train_components, q_train, q_train_components, r_test, r_test_components for overal model and targets
		"""
		assert hasattr(self,'X_train_'), "Error: run create_nfold"
		
		X_Train = self.X_train_
		Y_Train = self.y_train_
		X_Test = self.X_test_
		Y_Test = self.y_test_

		grouping_var = np.array(self.group_)
		grouping_var[self.test_index_] = "TEST"
		for i in range(len(self.fold_indices_)):
			grouping_var[self.fold_indices_[i]] = "FOLD%d" % (i+1)
		group_train = grouping_var[self.train_index_]
		ugroup_train = np.unique(group_train)
		self.cvgroups_ = grouping_var

		# Calculate Q2 squared
		X_CV_Q2 = np.zeros((len(ugroup_train)))
		Y_CV_Q2 = np.zeros((len(ugroup_train)))
		X_CV_Q2_roi = np.zeros((len(ugroup_train), X_Train.shape[1]))
		Y_CV_Q2_roi = np.zeros((len(ugroup_train), Y_Train.shape[1]))
		X_CV_redundacy = np.zeros((len(ugroup_train), n_components))
		Y_CV_redundacy = np.zeros((len(ugroup_train), n_components))
		CV_canonicalcorrelation = np.zeros((len(ugroup_train), n_components))
		for g, group in enumerate(ugroup_train):
			X_gtrain = X_Train[group_train != group]
			Y_gtrain = Y_Train[group_train != group]
			X_gtest = X_Train[group_train == group]
			Y_gtest = Y_Train[group_train == group]
			cvscca = scca_rwrapper(n_components = n_components,
											X_L1_penalty = X_L1_penalty, y_L1_penalty =  y_L1_penalty,
											max_iter = max_iter).fit(X_gtrain, Y_gtrain, calculate_loadings = True)
			X_gtest_hat, Y_gtest_hat = cvscca.predict(X = X_gtest, y = Y_gtest, toself = toself)
			X_CV_Q2[g] = cvscca._rscore(X_gtest, X_gtest_hat)
			Y_CV_Q2[g] = cvscca._rscore(Y_gtest, Y_gtest_hat)
			X_CV_Q2_roi[g] = cvscca._rscore(X_gtest, X_gtest_hat, mean_score = False)
			Y_CV_Q2_roi[g] = cvscca._rscore(Y_gtest, Y_gtest_hat, mean_score = False)
			X_CV_redundacy[g] = cvscca.x_redundacy_variance_explained_components_
			Y_CV_redundacy[g] = cvscca.y_redundacy_variance_explained_components_
			CV_canonicalcorrelation[g] = cvscca.canonicalcorr(X_gtest, Y_gtest)
		self.Q2_X_train_ = X_CV_Q2.mean(0)
		self.Q2_Y_train_ = Y_CV_Q2.mean(0)
		self.Q2_X_train_std_ = X_CV_Q2.std(0)
		self.Q2_Y_train_std_ = Y_CV_Q2.std(0)
		self.Q2_X_train_targets_ = X_CV_Q2_roi.mean(0)
		self.Q2_Y_train_targets_ = Y_CV_Q2_roi.mean(0)
		self.Q2_X_train_targets_std_ = X_CV_Q2_roi.std(0)
		self.Q2_Y_train_targets_std_ = Y_CV_Q2_roi.std(0)
		self.CVRDI_X_component_ = X_CV_redundacy.mean(0)
		self.CVRDI_Y_component_ = Y_CV_redundacy.mean(0)
		self.CVRDI_X_component_std_ = X_CV_redundacy.std(0)
		self.CVRDI_Y_component_std_ = Y_CV_redundacy.std(0)
		self.CV_canonicalcorrelation_ = CV_canonicalcorrelation.mean(0)
		self.CV_canonicalcorrelation_std_ = CV_canonicalcorrelation.std(0)

		# Calculate R2 squared for training data
		scca = scca_rwrapper(n_components = n_components,
										X_L1_penalty = X_L1_penalty, y_L1_penalty =  y_L1_penalty,
										max_iter = max_iter).fit(X_Train, Y_Train, calculate_loadings = True)
		self.R2_X_train_ = scca.x_variance_explained_
		self.R2_Y_train_ = scca.y_variance_explained_
		X_Train_hat, Y_Train_hat = scca.predict(X = X_Train, y = Y_Train)
		self.R2_X_train_targets_ = scca._rscore(X_Train, X_Train_hat, mean_score = False)
		self.R2_Y_train_targets_ = scca._rscore(Y_Train, Y_Train_hat, mean_score = False)
		self.RDI_X_train_components_ = scca.x_redundacy_variance_explained_components_
		self.RDI_Y_train_components_ = scca.y_redundacy_variance_explained_components_
		self.RDI_global_ = np.divide(np.sum(scca.x_redundacy_variance_explained_components_) + np.sum(scca.y_redundacy_variance_explained_components_), 2)
		self.canonicalcorrelation_train_ = scca.cors
		self.pvalue_canonicalcorrelation_train_ = self._pearsonr_to_t(scca.cors, len(scca.X_))[1]
		self.X_loadings = scca.x_loadings_
		self.Y_loadings = scca.y_loadings_

		self.pvalue_X_train_ = self._pearsonr_to_t(self.R2_X_train_, len(X_Train))[1]
		t,p = self._pearsonr_to_t(self.R2_X_train_targets_, len(X_Train))
		self.pvalue_X_train_targets_ = p
		self.tvalue_X_train_targets_ = t
		self.qvalue_X_train_targets_ = fdrcorrection(p)[1]

		self.pvalue_Y_train_ = self._pearsonr_to_t(self.R2_Y_train_, len(Y_Test))[1]
		t,p = self._pearsonr_to_t(self.R2_Y_train_targets_, len(Y_Train))
		self.pvalue_Y_train_targets_ = p
		self.tvalue_Y_train_targets_ = t
		self.qvalue_Y_train_targets_ = fdrcorrection(p)[1]

		t,p = self._pearsonr_to_t(self.X_loadings, len(self.X_train_))
		self.pvalue_X_loadings_ = p
		self.tvalue_X_loadings_targets_ = t
		self.qvalue_X_loadings_ = np.zeros_like(self.X_loadings)
		for c in range(n_components):
			self.qvalue_X_loadings_[:,c] = fdrcorrection(p[:,c])[1]

		t,p = self._pearsonr_to_t(self.Y_loadings, len(self.y_train_))
		self.pvalue_Y_loadings_ = p
		self.tvalue_Y_loadings_targets_ = t
		self.qvalue_Y_loadings_ = np.zeros_like(self.Y_loadings)
		for c in range(n_components):
			self.qvalue_Y_loadings_[:,c] = fdrcorrection(p[:,c])[1]

		# Calculate R2 squared for test data
		X_Test_hat, Y_Test_hat = scca.predict(X_Test, Y_Test, toself = toself)
		self.R2_X_test_ = scca._rscore(X_Test, X_Test_hat)
		self.R2_Y_test_ = scca._rscore(Y_Test, Y_Test_hat)
		self.R2_X_test_targets_ = scca._rscore(X_Test, X_Test_hat, mean_score = False)
		self.R2_Y_test_targets_ = scca._rscore(Y_Test, Y_Test_hat, mean_score = False)
		self.canonicalcorrelation_test_ = scca.canonicalcorr(self.X_test_, self.y_test_)
		self.pvalue_canonicalcorrelation_test_ = self._pearsonr_to_t(self.canonicalcorrelation_test_, len(self.X_test_))[1]

		self.pvalue_X_test_ = self._pearsonr_to_t(self.R2_X_test_, len(X_Test))[1]
		t,p = self._pearsonr_to_t(self.R2_X_test_targets_, len(X_Test))
		self.pvalue_X_test_targets_ = p
		self.tvalue_X_test_targets_ = t
		self.qvalue_X_test_targets_ = fdrcorrection(p)[1]
		self.pvalue_Y_test_ = self._pearsonr_to_t(self.R2_Y_test_, len(Y_Test))[1]
		t,p = self._pearsonr_to_t(self.R2_Y_test_targets_, len(Y_Test))
		self.pvalue_Y_test_targets_ = p
		self.tvalue_Y_test_targets_ = t
		self.qvalue_Y_test_targets_ = fdrcorrection(p)[1]

		self.n_components_ = n_components
		self.X_L1_penalty_ = X_L1_penalty
		self.y_L1_penalty_ = y_L1_penalty
		self.max_iter_ = max_iter
		self.group_train_ = group_train
		self.ugroup_train_ = ugroup_train
		self.model_obj_ = scca
		self.toself_ = toself

	def _permute_function_scca(self, p, compute_targets = True, permute_loadings = False, seed = None):
		assert hasattr(self,'model_obj_'), "Error: run fit_model"
		
		if seed is None:
			np.random.seed(np.random.randint(4294967295))
		else:
			np.random.seed(seed)
		if p % 200 == 0:
			print(p)
		
		X_perm = self.index_perm(unique_arr = self.ugroup_train_,
										arr = self.group_train_,
										variable = self.X_train_)
		Y_perm = self.index_perm(unique_arr = self.ugroup_train_,
										arr = self.group_train_,
										variable = self.y_train_)
		perm_ssca = scca_rwrapper(n_components = self.n_components_,
											X_L1_penalty = self.X_L1_penalty_,
											y_L1_penalty =  self.y_L1_penalty_,
											max_iter = self.max_iter_).fit(X_perm, Y_perm, calculate_loadings = True)
		perm_X_Test_hat, perm_Y_Test_hat = perm_ssca.predict(self.X_test_, self.y_test_, toself = self.toself_)
		X_VE = perm_ssca._rscore(self.X_test_, perm_X_Test_hat)
		Y_VE = perm_ssca._rscore(self.y_test_, perm_Y_Test_hat)
		if compute_targets:
			X_VE_ROI = perm_ssca._rscore(self.X_test_, perm_X_Test_hat, mean_score = False)
			Y_VE_ROI = perm_ssca._rscore(self.y_test_, perm_Y_Test_hat, mean_score = False)
		else:
			X_VE_ROI = None
			Y_VE_ROI = None
		if permute_loadings:
			perm_X_sqr_rcs_ = np.square(perm_ssca.x_loadings_)
			perm_Y_sqr_rcs_ = np.square(perm_ssca.y_loadings_)
		else:
			perm_X_sqr_rcs_ = None
			perm_Y_sqr_rcs_ = None
		X_RDI = perm_ssca.x_redundacy_variance_explained_components_
		Y_RDI = perm_ssca.y_redundacy_variance_explained_components_
		CANCORS = perm_ssca.canonicalcorr(self.X_test_, self.y_test_)
		return(X_VE, Y_VE, X_RDI, Y_RDI, CANCORS, X_VE_ROI, Y_VE_ROI, perm_X_sqr_rcs_, perm_Y_sqr_rcs_)
	def run_permute_scca(self, compute_targets = True, calulate_pvalues = True, permute_loadings = False):
		assert hasattr(self,'model_obj_'), "Error: run fit_model"
		seeds = generate_seeds(self.n_permutations)
		output = Parallel(n_jobs = self.n_jobs, backend='multiprocessing')(delayed(self._permute_function_scca)(p, compute_targets = compute_targets, permute_loadings = permute_loadings, seed = seeds[p]) for p in range(self.n_permutations))
		perm_X_VE, perm_Y_VE, perm_X_RDI, perm_Y_RDI, perm_CANCORS, perm_X_VE_ROI, perm_Y_VE_ROI, perm_X_sqr_rcs_, perm_Y_sqr_rcs_ = zip(*output)
		self.perm_R2_X_test_ = np.array(perm_X_VE)
		self.perm_R2_Y_test_ = np.array(perm_Y_VE)
		self.perm_RDI_X_ = np.array(perm_X_RDI)
		self.perm_RDI_Y_ = np.array(perm_Y_RDI)
		self.perm_canonicalcorrelation_ = np.array(perm_CANCORS)
		if compute_targets:
			self.perm_R2_X_test_targets_ = np.array(perm_X_VE_ROI)
			self.perm_R2_Y_test_targets_ = np.array(perm_Y_VE_ROI)
		if permute_loadings:
			self.perm_X_sqr_rcs_ = np.array(perm_X_sqr_rcs_)
			self.perm_Y_sqr_rcs_ = np.array(perm_Y_sqr_rcs_)
		if calulate_pvalues:
			self.compute_permuted_pvalues()
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
	def compute_permuted_pvalues(self):
		"""
		Calculates p-values (and FWER p-values) using the permuted null distribution.
		"""
		assert hasattr(self,'perm_R2_X_test_'), "Error: no permuted variables. Run run_permute_scca first."
		if hasattr(self,'perm_R2_X_test_targets_'):
			self.perm_pvalue_test_X_targets_ = self.fwer_corrected_p(self.perm_R2_X_test_targets_, self.R2_X_test_targets_, apply_fwer_correction = False)
			self.perm_pvalue_test_Y_targets_ = self.fwer_corrected_p(self.perm_R2_Y_test_targets_, self.R2_Y_test_targets_, apply_fwer_correction = False)
			self.pFWER_test_X_targets_ = self.fwer_corrected_p(self.perm_R2_X_test_targets_, self.R2_X_test_targets_)
			self.pFWER_test_Y_targets_ = self.fwer_corrected_p(self.perm_R2_Y_test_targets_, self.R2_Y_test_targets_)
		if hasattr(self,'perm_X_sqr_rcs_'):
			self.perm_pvalue_X_loadings_ = np.zeros_like(self.X_loadings)
			self.perm_pvalue_Y_loadings_ = np.zeros_like(self.Y_loadings)
			for c in range(self.n_components_):
				self.perm_pvalue_X_loadings_[c,:] = self.fwer_corrected_p(self.perm_X_sqr_rcs_[:,c,:], np.square(self.X_loadings)[c,:], apply_fwer_correction = False)
				self.perm_pvalue_Y_loadings_[c,:] = self.fwer_corrected_p(self.perm_Y_sqr_rcs_[:,c,:], np.square(self.Y_loadings)[c,:], apply_fwer_correction = False)
		self.perm_pvalue_R2_X_test_ = self.fwer_corrected_p(self.perm_R2_X_test_, self.R2_X_test_)[0]
		self.perm_pvalue_R2_Y_test_ = self.fwer_corrected_p(self.perm_R2_Y_test_, self.R2_Y_test_)[0]
		self.perm_pvalue_R2_test_ = self.fwer_corrected_p(np.sort((self.perm_R2_X_test_ + self.perm_R2_Y_test_)/2), ((self.R2_X_test_ + self.R2_Y_test_) /2))[0]
		self.perm_pvalue_RDI_X_train_components_ = self.fwer_corrected_p(self.perm_RDI_X_, self.RDI_X_train_components_, apply_fwer_correction = False)
		self.perm_pvalue_RDI_Y_train_components_ = self.fwer_corrected_p(self.perm_RDI_Y_, self.RDI_Y_train_components_, apply_fwer_correction = False)
		self.perm_pvalue_RDI_X_train_ = self.fwer_corrected_p(np.sum(self.perm_RDI_X_, 1), np.sum(self.RDI_X_train_components_))[0]
		self.perm_pvalue_RDI_Y_train_ = self.fwer_corrected_p(np.sum(self.perm_RDI_Y_, 1), np.sum(self.RDI_Y_train_components_))[0]
		self.perm_pvalue_RDI_model_ = self.fwer_corrected_p(np.divide(np.sum(self.perm_RDI_X_, 1) + np.sum(self.perm_RDI_Y_, 1),2), np.divide(np.sum(self.RDI_X_train_components_) + np.sum(self.RDI_Y_train_components_),2))[0]
		self.perm_pvalue_canonicalcorrelation_ = self.fwer_corrected_p(self.perm_canonicalcorrelation_, np.abs(self.canonicalcorrelation_test_), apply_fwer_correction = False)

	def plot_permuted_canonical_correlations(self, png_basename = None, n_jitters = 1000, add_Q2_from_train = False, plot_model = False):
		assert hasattr(self,'perm_pvalue_R2_test_'), "Error: Run compute_permuted_pvalues"
		if n_jitters > self.n_permutations:
			n_jitters = self.n_permutations
		p_num = 1
		n_plots = self.n_components_ 
		plt.subplots(figsize=(int(2*n_plots), 6), dpi=100, tight_layout = True, sharey='row')
		if plot_model:
			n_plots += 2
			plt.subplot(1, n_plots, p_num)
			jitter = np.random.normal(0, scale = 0.1, size=n_jitters)
			rand_dots = self.perm_R2_X_test_[:n_jitters]
			plt.scatter(jitter, rand_dots, marker = '.', alpha = 0.3)
			plt.xlim(-.5, .5)
			plt.ylabel("R2_X predicted vs actual (Test Data)")
			plt.title("Model(X)")
			plt.scatter(0, self.R2_X_test_, marker = 'o', alpha = 1.0, c = 'k')
			if add_Q2_from_train:
				plt.errorbar(0.1, self.Q2_X_train_, self.Q2_X_train_std_, linestyle='None', marker='.', c = 'r', alpha = 0.5)
			plt.xticks(color='w')
			p_num += 1
			x1,x2,y1,_ = plt.axis()
			y1 = round(y1,3) - 0.01
			y2 = np.max(self.R2_X_test_)
			if np.max(self.R2_X_test_) > y2:
				y2 =np.max(self.R2_X_test_)
			if np.max(np.squeeze(self.perm_R2_X_test_)) > y2:
				y2 = np.max(np.squeeze(self.perm_R2_X_test_))
			if np.max(np.squeeze(self.perm_R2_Y_test_)) > y2:
				y2 = np.max(np.squeeze(self.perm_R2_Y_test_))
			y2 = round(y2,3) + 0.01
			if self.perm_pvalue_R2_test_ == 0:
				plt.xlabel("R2=%1.3f, P<%1.1e" % (self.R2_X_test_, (1 / self.n_permutations)), fontsize=10)
			else:
				plt.xlabel("R2=%1.3f, P=%1.1e" % (self.R2_X_test_, self.pvalue_R2_X_test_), fontsize=10)
			plt.ylim(y1, y2)
			plt.subplot(1, n_plots, p_num)
			jitter = np.random.normal(0, scale = 0.1, size=n_jitters)
			rand_dots = self.perm_R2_Y_test_[:n_jitters]
			plt.scatter(jitter, rand_dots, marker = '.', alpha = 0.3)
			plt.xlim(-.5, .5)
			plt.ylabel("R2_Y predicted vs actual (Test Data)")
			plt.title("Model(Y)")
			plt.scatter(0, self.R2_Y_test_, marker = 'o', alpha = 1.0, c = 'k')
			if add_Q2_from_train:
				plt.errorbar(0.1, self.Q2_Y_train_, self.Q2_Y_train_std_, linestyle='None', marker='.', c = 'r', alpha = 0.5)
			plt.xticks(color='w')
			p_num += 1
			plt.ylim(y1, y2)
			if self.perm_pvalue_R2_Y_test_ == 0:
				plt.xlabel("R2=%1.3f, P<%1.1e" % (self.R2_Y_test_, (1 / self.n_permutations)), fontsize=10)
			else:
				plt.xlabel("R2=%1.3f, P=%1.1e" % (self.R2_Y_test_, self.perm_pvalue_R2_Y_test_), fontsize=10)
		y1 =  round(np.min(np.concatenate((self.canonicalcorrelation_test_, (self.perm_canonicalcorrelation_).flatten()))),3) - 0.01
		y2 =  round(np.max(np.concatenate((self.canonicalcorrelation_test_, (self.perm_canonicalcorrelation_).flatten()))),3) + 0.01
		for c in range(self.n_components_):
			plt.subplot(1, n_plots, p_num)
			jitter = np.random.normal(0, scale = 0.1, size=n_jitters)
			rand_dots = self.perm_canonicalcorrelation_[:n_jitters, c]
			plt.scatter(jitter, rand_dots, marker = '.', alpha = 0.3)
			plt.xlim(-.5, .5)
			plt.title("Component %d" % (c+1))
			plt.scatter(0, self.canonicalcorrelation_test_[c], marker = 'o', alpha = 1.0, c = 'k')
			plt.xticks(color='w')
			plt.ylim(y1, y2)
			if self.perm_pvalue_canonicalcorrelation_[c] == 0:
				plt.xlabel("r=%1.3f, P<%1.2e" % (self.canonicalcorrelation_test_[c], (1 / self.n_permutations)), fontsize=10)
			elif self.perm_pvalue_canonicalcorrelation_[c] > 0.001:
				plt.xlabel("r=%1.3f, P=%1.3f" % (self.canonicalcorrelation_test_[c], self.perm_pvalue_canonicalcorrelation_[c]), fontsize=10)
			else:
				plt.xlabel("r=%1.3f, P=%1.2e" % (self.canonicalcorrelation_test_[c], self.perm_pvalue_canonicalcorrelation_[c]), fontsize=10)
			p_num += 1
		if png_basename is not None:
			plt.savefig("%s_model_fit_to_test_with_null.png" % png_basename)
			plt.close()
		else:
			plt.show()

	def plot_component_range(self, lamdba_x, lamdba_y, component_range = [1, 16], plotx = True, ploty = True, selected_subset = False, png_basename = None):
		fold_indices = self.fold_indices_
		n_fold = len(fold_indices)
		X = np.array(self.X_)
		y = np.array(self.y_)
		fold_index = np.arange(0,n_fold,1)
		comp_range = np.arange(int(component_range[0]), int(component_range[1]+1), 1)
		n_comps = len(comp_range)
		cv_ve = np.zeros((n_comps))
		cv_vex = np.zeros((n_comps))
		cv_vey = np.zeros((n_comps))
		cv_ve_err = np.zeros((n_comps))
		cv_vex_err = np.zeros((n_comps))
		cv_vey_err = np.zeros((n_comps))
		for i, c in enumerate(comp_range):
			cv_corr = []
			cv_corr_x = []
			cv_corr_y = []
			for n in range(n_fold):
				selx = np.ones((self.X_train_.shape[1]), dtype = bool)
				sely = np.ones((self.y_train_.shape[1]), dtype = bool)
				sel_train = fold_indices[n]
				sel_test = np.concatenate(fold_indices[fold_index != n])
				tmpX_train = X[sel_train]
				tmpY_train = y[sel_train]
				tmpX_test = X[sel_test]
				tmpY_test = y[sel_test]
				cvscca = scca_rwrapper(n_components = c, X_L1_penalty = lamdba_x, y_L1_penalty = lamdba_y, max_iter = 100).fit(tmpX_train, tmpY_train)
				if selected_subset:
					selx[cvscca.x_selectedvariablesindex_ == 0] = False
					sely[cvscca.y_selectedvariablesindex_ == 0] = False
				tmpX_test_predicted, tmpY_test_predicted = cvscca.predict(X = tmpX_test, y = tmpY_test, toself = True)
				corr_x = cvscca._rscore(tmpX_test[:,selx], tmpX_test_predicted[:,selx])
				corr_y = cvscca._rscore(tmpY_test[:,sely], tmpY_test_predicted[:,sely])
				cv_corr_x.append(corr_x)
				cv_corr_y.append(corr_y)
				cv_corr.append((corr_x + corr_y)/2)
			cv_ve[i] = np.mean(cv_corr)
			cv_vex[i] = np.mean(cv_corr_x)
			cv_vey[i] = np.mean(cv_corr_y)
			cv_ve_err[i] = np.std(cv_corr)
			cv_vex_err[i] = np.std(cv_corr_x)
			cv_vey_err[i] = np.std(cv_corr_y)
		plt.plot(comp_range, cv_ve)
		plt.fill_between(comp_range, cv_ve-cv_ve_err, cv_ve+cv_ve_err, alpha = 0.5)
		plt.ylabel('CV prediction score')
		plt.xticks(comp_range,comp_range)
		plt.xlabel('Number of Latent Variables')
		if png_basename is not None:
			plt.savefig("%s_cv_test_prediction_component_range.png" % (png_basename))
			plt.close()
		else:
			plt.show()
		if plotx:
			plt.plot(comp_range, cv_vex)
			plt.fill_between(comp_range, cv_vex-cv_vex_err, cv_vex+cv_vex_err, alpha = 0.5)
			plt.ylabel('CV prediction score [X variates]')
			plt.xticks(comp_range,comp_range)
			plt.xlabel('Number of Latent Variables')
			if png_basename is not None:
				plt.savefig("%s_cv_test_xprediction_component_range.png" % (png_basename))
				plt.close()
			else:
				plt.show()
		if ploty:
			plt.plot(comp_range, cv_vey)
			plt.fill_between(comp_range, cv_vey-cv_vey_err, cv_vey+cv_vey_err, alpha = 0.5)
			plt.ylabel('CV prediction score [Y variates]')
			plt.xticks(comp_range,comp_range)
			plt.xlabel('Number of Latent Variables')
			if png_basename is not None:
				plt.savefig("%s_cv_test_yprediction_component_range.png" % (png_basename))
				plt.close()
			else:
				plt.show()

	def plot_canonical_correlations(self, png_basename = None, component = None, swapXY = False, Xlabel = None, Ylabel = None):
		assert hasattr(self,'model_obj_'), "Error: run fit_model"
		score_x_train, score_y_train = self.model_obj_.transform(self.X_train_, self.y_train_)
		score_x_test, score_y_test = self.model_obj_.transform(self.X_test_, self.y_test_)
		if component is not None:
			c = component -1
			if swapXY:
				plt.scatter(score_y_train[:,c], score_x_train[:,c])
				b, m = np.polynomial.polynomial.polyfit(score_y_train[:,c], score_x_train[:,c],1)
				plt.plot(score_y_train[:,c], b + m * score_y_train[:,c], '-')
			else:
				plt.scatter(score_x_train[:,c], score_y_train[:,c])
				b, m = np.polynomial.polynomial.polyfit(score_x_train[:,c], score_y_train[:,c],1)
				plt.plot(score_x_train[:,c], b + m * score_x_train[:,c], '-')
			if Xlabel is not None:
				plt.xlabel(Xlabel)
			if Ylabel is not None:
				plt.ylabel(Xlabel)
			plt.title("Canonical variate %d" % (c+1))
			if png_basename is not None:
				plt.savefig("%s_canonical_corr_train_component%d.png" % (png_basename, component))
				plt.close()
			else:
				plt.show()
			if swapXY:
				plt.scatter(score_y_test[:,c], score_x_test[:,c])
				b, m = np.polynomial.polynomial.polyfit(score_y_test[:,c], score_x_test[:,c],1)
				plt.plot(score_y_test[:,c], b + m * score_y_test[:,c], '-')
			else:
				plt.scatter(score_x_test[:,c], score_y_test[:,c])
				b, m = np.polynomial.polynomial.polyfit(score_x_test[:,c], score_y_test[:,c],1)
				plt.plot(score_x_test[:,c], b + m * score_x_test[:,c], '-')
			if Xlabel is not None:
				plt.xlabel(Xlabel)
			if Ylabel is not None:
				plt.ylabel(Xlabel)
			plt.title("Canonical variate %d" % (c+1))
			if png_basename is not None:
				plt.savefig("%s_canonical_corr_test_component%d.png" % (png_basename, component))
				plt.close()
			else:
				plt.show()
		else:
			for c in range(self.n_components_):
				component = int(c+1)
				if swapXY:
					plt.scatter(score_y_train[:,c], score_x_train[:,c])
					b, m = np.polynomial.polynomial.polyfit(score_y_train[:,c], score_x_train[:,c],1)
					plt.plot(score_y_train[:,c], b + m * score_y_train[:,c], '-')
				else:
					plt.scatter(score_x_train[:,c], score_y_train[:,c])
					b, m = np.polynomial.polynomial.polyfit(score_x_train[:,c], score_y_train[:,c],1)
					plt.plot(score_x_train[:,c], b + m * score_x_train[:,c], '-')
				if Xlabel is not None:
					plt.xlabel(Xlabel)
				if Ylabel is not None:
					plt.ylabel(Xlabel)
				plt.title("Canonical Component %d [Train]" % (c+1))
				if png_basename is not None:
					plt.savefig("%s_canonical_corr_train_component%d.png" % (png_basename, component))
					plt.close()
				else:
					plt.show()
				if swapXY:
					plt.scatter(score_y_test[:,c], score_x_test[:,c])
					b, m = np.polynomial.polynomial.polyfit(score_y_test[:,c], score_x_test[:,c],1)
					plt.plot(score_y_test[:,c], b + m * score_y_test[:,c], '-')
				else:
					plt.scatter(score_x_test[:,c], score_y_test[:,c])
					b, m = np.polynomial.polynomial.polyfit(score_x_test[:,c], score_y_test[:,c],1)
					plt.plot(score_x_test[:,c], b + m * score_x_test[:,c], '-')
				if Xlabel is not None:
					plt.xlabel(Xlabel)
				if Ylabel is not None:
					plt.ylabel(Xlabel)
				plt.title("Canonical Component %d [Test]" % (c+1))
				if png_basename is not None:
					plt.savefig("%s_canonical_corr_test_component%d.png" % (png_basename, component))
					plt.close()
				else:
					plt.show()


class scca_rwrapper:
	"""
	
	By Default:
	R Wrapper that uses the PMA (PMA-package: Penalized Multivariate Analysis) r package, and rpy2
	https://rdrr.io/cran/PMA/man/CCA.html
	Added calculation of loadings, prediction, redundacy metrics
	
	Set force_pma to false to use sparsecca package (cca_pmd) which provides identical results to R PMA library with option to the R package
	https://github.com/Teekuningas/sparsecca
	
	default sets force_pma = True because the R implementation is slighlty faster (16 Thread Ryzen 7: sparsecca is 380ms and PMA is 300ms) => optimize the cython/c ? 
	
	References:

	Witten, D. M., Tibshirani, R., & Hastie, T. (2009). A penalized matrix decomposition, with applications to sparse principal components and
	canonical correlation analysis. Biostatistics, 10(3), 515–534. http://doi.org/10.1093/biostatistics/kxp008
	
	Witten, D. M., & Tibshirani, R. J. (2009). Extensions of Sparse Canonical Correlation Analysis with Applications to Genomic Data.
	Statistical Applications in Genetics and Molecular Biology, 8(1), 29. http://doi.org/10.2202/1544-6115.1470
	"""
	def __init__(self, n_components, X_L1_penalty = 0.3, y_L1_penalty = 0.3, max_iter = 100, scale_x = True, scale_y = True, force_pma = True, effective_zero = 1e-15):
		self.n_components = n_components
		assert (X_L1_penalty > 0) and (X_L1_penalty <= 1), "Error: X_L1_penalty must be between 0 and 1"
		assert (y_L1_penalty > 0) and (y_L1_penalty <= 1), "Error: y_L1_penalty must be between 0 and 1"
		self.X_L1_penalty = X_L1_penalty
		self.y_L1_penalty = y_L1_penalty
		self.max_iter = max_iter
		self.scale_x = scale_x
		self.scale_y = scale_y
		self.penalty = "l1"
		self.force_pma = force_pma
		self.effective_zero = effective_zero
		
	def fit(self, X, y, calculate_loadings = True):
		"""
		Fit for scca model. The functions saves outputs using sklearn's naming convention.
		https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/cross_decomposition/_pls.py
		
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
		X, y, X_mean, y_mean, X_std, y_std = zscaler_XY(X, y, scale_x = self.scale_x, scale_y = self.scale_y)
		Xk = np.array(X)
		yk = np.array(y)
		
		# select the algorithm
		if self.force_pma:
			numpy2ri.activate()
			model = pma.CCA(x = X, z = y,
								K = self.n_components,
								penaltyx = self.X_L1_penalty,
								penaltyz = self.y_L1_penalty,
								niter = self.max_iter,
								trace  = False)
			u = model.rx2("u")
			v = model.rx2("v")
			d = model.rx2("d")
			numpy2ri.deactivate()
		else:
			if have_sparsecca:
				u, v, d = sparsecca(X, y,
									K = self.n_components,
									penaltyx = self.X_L1_penalty,
									penaltyz = self.y_L1_penalty,
									niter = self.max_iter)
			else:
				numpy2ri.activate()
				model = pma.CCA(x = X, z = y,
									K = self.n_components,
									penaltyx = self.X_L1_penalty,
									penaltyz = self.y_L1_penalty,
									niter = self.max_iter,
									trace  = False)
				u = model.rx2("u")
				v = model.rx2("v")
				d = model.rx2("d")
				numpy2ri.deactivate()

		self.X_ = X
		self.X_mean_ = X_mean
		self.X_std_ = X_std
		self.y_ = y
		self.y_mean_ = y_mean
		self.y_std_ = y_std

		self.x_selectedvariablescomponents_ = (u != 0)*1
		self.y_selectedvariablescomponents_ = (v != 0)*1
		self.x_selectedvariablesindex_ = (np.mean((u != 0)*1,1) > 0)*1
		self.y_selectedvariablesindex_ = (np.mean((v != 0)*1,1) > 0)*1
		self.x_weights_ = u
		self.y_weights_ = v
		self.d_ = d
		self.x_scores_ = np.dot(X, u)
		self.y_scores_ = np.dot(y, v)
		self.cors = self.canonicalcorr()
		if calculate_loadings:
#			https://pure.uvt.nl/ws/portalfiles/portal/596531/useofcaa_ab5.pdf and https://scholarscompass.vcu.edu/cgi/viewcontent.cgi?article=1001&context=socialwork_pubs
			# the loadings are equivalent to the structure coefficient. Square of the loading (squared structure coefficient) is the proportion of variance each variable shares with the canonical variate/component.
			self.x_loadings_ = self._calculate_loadings(self.x_scores_, X)
			self.y_loadings_ = self._calculate_loadings(self.y_scores_, y)
			# Another measure of effect size. The redundacy component is the amount of variance in the X account by Y through each canonical variate/component (X->Y and Y->X are not equivlent). 
			self.x_redundacy_variance_explained_components_ = np.mean(self.x_loadings_**2)*(self.cors**2)
			self.x_redundacy_variance_explained_global_ = np.sum(self.x_redundacy_variance_explained_components_)
			self.y_redundacy_variance_explained_components_ = np.mean(self.y_loadings_**2)*(self.cors**2)
			self.y_redundacy_variance_explained_global_ = np.sum(self.y_redundacy_variance_explained_components_)
			X_hat, Y_hat = self.predict(X = X, y = y)
			self.x_variance_explained_ = self._rscore(self.X_, X_hat)
			self.y_variance_explained_ = self._rscore(self.y_, Y_hat)
			self.x_variance_explained_selected_ = self._rscore(self.X_[:,self.x_selectedvariablesindex_==1], X_hat[:,self.x_selectedvariablesindex_==1])
			self.y_variance_explained_selected_ = self._rscore(self.y_[:,self.y_selectedvariablesindex_==1], Y_hat[:,self.y_selectedvariablesindex_==1])
		return(self)

	def _calculate_loadings(self, data_scores, data):
		"""
		calculates loadings using cython optimizated least square regression
		returns loading with the shape n_components, n_targets
		"""
		
		n_components = data_scores.shape[1]
		n_targets = data.shape[1]
		loadings = np.zeros((n_components, n_targets))
		for c in range(n_components):
			loadings[c,:] = cy_lin_lstsqr_mat(data_scores[:,c].reshape(-1,1), data)[0]
		return(loadings)

	def transform(self, X = None, y = None):
		"""
		Calculate the component scores for selected variables.
		"""
		if X is not None:
			X = scale(X)
			x_scores = np.dot(X, self.x_weights_)
		else:
			x_scores = None
		if y is not None:
			y = scale(y)
			y_scores = np.dot(y, self.y_weights_)
		else:
			y_scores = None
		return(x_scores, y_scores)
	def canonicalcorr(self, X = None, y = None):
		"""
		Returns the canonical correlation
		"""
		if X is not None:
			assert y is not None, "Error: for test data, by X and y must be specified"
			X, y, _, _, _, _ = zscaler_XY(X, y)
		else:
			X = self.X_
			y = self.y_
		x_scores = np.dot(X, self.x_weights_)
		y_scores = np.dot(y, self.y_weights_)
		cancors = np.corrcoef(x_scores.T, y_scores.T).diagonal(self.n_components)
		return(cancors)

	def _prediction_cor(self, true, predicted):
		"""
		Calculates the correlation between the true and predicted values.
		
		The same method used is this citation:
		Bilenko NY, Gallant JL. Pyrcca: Regularized Kernel Canonical Correlation Analysis in Python and Its Applications to Neuroimaging. Front Neuroinform. 2016 Nov 22;10:49. doi: 10.3389/fninf.2016.00049.
		"""
		n_targets = true.shape[1]
		return(np.array([cy_lin_lstsqr_mat(true[:,target].reshape(-1,1), predicted[:,target])[0] for target in range(n_targets)]))

	def _rscore(self, true, predicted, mean_score = True):
		score = self._prediction_cor(true, predicted)
		if mean_score:
			return(np.mean(score))
		else:
			return(score)

	def predict(self, X = None, y = None, toself = False):
		"""
		Predict X or y by calculating canonical scores from the model, and then dotting be the inverse of the model weights by the canonical scores.
		"""
		if X is not None:
			x_scores = np.dot(X, self.x_weights_)
			if toself: # toself may be useful for optimization...
				x_predicted = scale(np.dot(pinv(self.x_weights_, rcond=self.effective_zero).T, x_scores.T).T)
			else:
				y_predicted = scale(np.dot(pinv(self.y_weights_, rcond=self.effective_zero).T, x_scores.T).T)
		else:
			if toself:
				x_predicted = None
			else:
				y_predicted = None
		if y is not None:
			y_scores = np.dot(y, self.y_weights_)
			if toself:
				y_predicted = scale(np.dot(pinv(self.y_weights_, rcond=self.effective_zero).T, y_scores.T).T)
			else:
				x_predicted = scale(np.dot(pinv(self.x_weights_, rcond=self.effective_zero).T, y_scores.T).T)
		else:
			if toself:
				y_predicted = None
			else:
				x_predicted = None
		return(x_predicted, y_predicted)

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
		X, y, X_mean, y_mean, X_std, y_std = zscaler_XY(X, y, scale_x = self.scale_x, scale_y = self.scale_y)
		X = np.array(X)
		y = np.array(y)
		numpy2ri.activate()
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
		numpy2ri.deactivate()
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
		return(self)
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
		X -= self.X_mean_
		X /= self.X_std_
		return(np.dot(X,self.coef_) + self.y_mean_)


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
		# this should really be written in c or cython
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
		self.coef = cy_lin_lstsqr_mat(X_lambda, Y_lambda)
	def predict(self, X):
		X_ = self.zscaler(X)
		X_ = self.stack_ones(X_)
		return(np.dot(X_,self.coef))


class exogenoues_variable:
	def __init__(self, name, dmy_arr, exog_type):
		try:
			k = np.array(dmy_arr).shape[1]
		except:
			k = 1
		self.name = [str(name)]
		self.dmy_arr = [dmy_arr]
		self.k = [k]
		self.exog_type = [exog_type]
	def add_var(self, name, dmy_arr, exog_type):
		try:
			k = np.array(dmy_arr).shape[1]
		except:
			k = 1
		self.name.append(str(name))
		self.k.append(k)
		self.dmy_arr.append(dmy_arr)
		self.exog_type.append(exog_type)
	def print_variables(self):
		for i in range(len(self.name)):
			print("{" + str(self.name[i]) + " : " + str(self.exog_type[i]) + " : " + str(self.dmy_arr[i].shape) + "}")

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

	def calculate_cosinor_metrics(self, period_arr, two_step_regression = False, calculate_cosinor_stats = False):
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

# older PLS code.

class permute_model_parallel():
	"""
	Calculates PLS metrics and significance using sklearn PLSRegression and joblib locky parallelization.
	"""
	def __init__(self, n_jobs, n_permutations = 10000):
		self.n_jobs = n_jobs
		self.n_permutations = n_permutations
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
		X_Train, Y_Train, X_Train_mean, Y_Train_mean, X_Train_std, Y_Train_std = zscaler_XY(X = X_Train, y = Y_Train)
		X_Test, Y_Test, X_Test_mean, Y_Test_mean, X_Test_std, Y_Test_std = zscaler_XY(X = X_Test, y = Y_Test)
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
		X_Train, Y_Train, X_Train_mean, Y_Train_mean, X_Train_std, Y_Train_std = zscaler_XY(X = X_Train, y = Y_Train)
		X_Test, Y_Test, X_Test_mean, Y_Test_mean, X_Test_std, Y_Test_std = zscaler_XY(X = X_Test, y = Y_Test)
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
	def create_nfold(self, X, y, group, n_fold = 10, holdout = 0.3, verbose = True):
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
				components_variance_explained_test = np.zeros((self.n_fold_, self.n_comp))
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
					x_scores_test, y_scores_test = pls2.transform(X_test, Y_test)
					for c in range(self.n_comp):
						yhat_c = np.dot(x_scores_test[:,c].reshape(-1,1), pls2.y_loadings_[:,c].reshape(-1,1).T) * Y_test.std(axis=0, ddof=1) + Y_test.mean(axis=0)
						components_variance_explained_test[n, c] = explained_variance_score(Y_test, yhat_c)
				print("CV MODEL : %1.3f +/- %1.3f" % (np.mean(CV_temp_ve), np.std(CV_temp_ve)))
				for c in range(self.n_comp):
					print("  CV COMP%d : %1.3f +/- %1.3f" % ((c+1), np.mean(components_variance_explained_test, 0)[c], np.std(components_variance_explained_test, 0)[c]))
				rmse_cv.append(np.mean(CV_temp_rmse))
				ve_cv.append(np.mean(CV_temp_ve))
				rmse_cv_std.append(np.std(CV_temp_rmse))
				ve_cv_std.append(np.std(CV_temp_ve))
				# full model
				X_SEL = self.X_train_[:,selection_mask]
				Y_actual = self.y_train_
				pls2 = PLSRegression(n_components=self.n_comp).fit(X_SEL, Y_actual)
				Y_proj = pls2.predict(X_SEL)
				score = explained_variance_score(Y_actual, Y_proj)
				print("TRAIN FULL MODEL : %1.3f" % (score))
				x_scores_test, y_scores_test = pls2.transform(X_SEL, Y_actual)
				for c in range(self.n_comp):
					yhat_c = np.dot(x_scores_test[:,c].reshape(-1,1), pls2.y_loadings_[:,c].reshape(-1,1).T) * Y_actual.std(axis=0, ddof=1) + Y_actual.mean(axis=0)
					print("  TRAIN COMP%d : %1.3f" % ((c+1),explained_variance_score(Y_actual, yhat_c)))
				full_model_ve.append(score)
				full_model_rmse.append(mean_squared_error(Y_actual, Y_proj, squared = False))
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


#	def scca_params_search(self, l1x_range = np.arange(0.1,1.1,.1), l1y_range = np.arange(0.1,1.1,.1), png_basename = None):
#		search_x_size = len(l1x_range)
#		search_y_size = len(l1y_range)
#		cancor = np.zeros((search_x_size, search_y_size))
#		highest = 0
#		for i,j in np.array(list(itertools.product(range(search_x_size), range(search_y_size)))):
#			scca = scca_rwrapper(n_components = 1, X_L1_penalty = l1x_range[i], y_L1_penalty = l1y_range[j], max_iter = 100, scale_x = True, scale_y = True).fit(X = self.X_train_, y = self.y_train_)
#			cancor[i,j] = scca.cors[0]
#			if scca.cors[0] > highest:
#				highest = scca.cors[0]
#				best_l1_x = l1x_range[i]
#				best_l1_y = l1y_range[j]
#				print("Current best penalties: l1[x] = %1.2f and l1[y] = %1.2f, Correlation = %1.3f" % (best_l1_x, best_l1_y, highest))
#		plt.imshow(cancor, interpolation = None, cmap='jet')
#		plt.yticks(range(search_y_size),[s[:3] for s in l1y_range.astype(str)])
#		plt.ylabel('L1 sparsity y')
#		plt.xticks(range(search_x_size),[s[:3] for s in l1x_range.astype(str)])
#		plt.xlabel('L1 sparsity X')
#		plt.colorbar()
#		if png_basename is not None:
#			plt.savefig("%s_sparsity_params_search.png" % png_basename)
#			plt.close()
#		else:
#			plt.show()
#	def scca_bootstrap(self, i, n_components, X, y, l1x, l1y, group, split):
#		if i % 100 == 0: 
#			print("Bootstrap : %d" % (i))
#		train_idx, _ = self.bootstrap_by_group(group = group, split = split)
#		X = X[train_idx]
#		y = y[train_idx]
#		scca = scca_rwrapper(n_components = n_components, X_L1_penalty = l1x, y_L1_penalty = l1y, max_iter = 100).fit(X = X, y = y)
#		return(scca.x_selectedvariablesindex_, scca.y_selectedvariablesindex_)

#	def run_scca_bootstrap(self, l1x, l1y, thresholds = np.arange(0.1,1.0,0.1), max_ncomp = 8):
#		grouping_var = np.array(self.group_)
#		grouping_var[self.test_index_] = "TEST"
#		for i in range(len(self.fold_indices_)):
#			grouping_var[self.fold_indices_[i]] = "FOLD%d" % (i+1)
#		self.nfold_groups = grouping_var
#		fold_index = np.arange(0,self.n_fold_,1)
#		Q2_grid_arr = np.zeros((max_ncomp, len(thresholds)))
#		RMSE_grid_arr = np.zeros((max_ncomp, len(thresholds)))
#		Q2_grid_arr_sd = np.zeros((max_ncomp, len(thresholds)))
#		RMSE_grid_arr_sd = np.zeros((max_ncomp, len(thresholds)))
#		X_mean_selected = []
#		y_mean_selected = []
#		for c in range(max_ncomp):
#			K = c + 1
#			output = Parallel(n_jobs=self.n_jobs, backend='multiprocessing')(delayed(self.scca_bootstrap)(i, n_components = K, X = self.X_train_, y = self.y_train_, l1x = l1x, l1y = l1y, group = self.nfold_groups[self.nfold_groups != "TEST"], split = self.split) for i in range(self.n_boot))
#			X_selected, y_selected = zip(*output)
#			X_selected = np.array(X_selected)
#			X_selected_mean = np.mean(X_selected, 0)
#			y_selected = np.array(y_selected)
#			y_selected_mean = np.mean(y_selected, 0)
#			X_mean_selected.append(X_selected_mean)
#			y_mean_selected.append(y_selected_mean)
#			for s, threshold in enumerate(thresholds):
#				X_selection_mask = X_selected_mean > threshold
#				y_selection_mask = y_selected_mean > threshold
#				X_SEL = self.X_[:,X_selection_mask]
#				Y_SEL = self.y_[:,y_selection_mask]
#				temp_Q2 = np.zeros((self.n_fold_))
#				temp_rmse = np.zeros((self.n_fold_))
#				n_samples = X_SEL.shape[0]
#				n_features = X_SEL.shape[1]
#				n_targets = Y_SEL.shape[1]
#				if K <= min(n_samples, n_features, n_targets):
#					for n in range(self.n_fold_):
#						sel_train = self.fold_indices_[n]
#						sel_test = np.concatenate(self.fold_indices_[fold_index != n])
#						tmpX_train = X_SEL[sel_train]
#						tmpY_train = Y_SEL[sel_train]
#						tmpX_test = X_SEL[sel_test]
#						tmpY_test = Y_SEL[sel_test]
#						# kick out effective zero predictors
#						tmpX_test = tmpX_test[:,tmpX_train.std(0) > 0.0001]
#						tmpX_train = tmpX_train[:,tmpX_train.std(0) > 0.0001]
#						tmpY_test = tmpY_test[:,tmpY_train.std(0) > 0.0001]
#						tmpY_train = tmpY_train[:,tmpY_train.std(0) > 0.0001]
#						# no sparsity
#						bsscca = PLSCanonical(n_components = K, max_iter=100).fit(tmpX_train, tmpY_train)
#						Y_proj = bsscca.predict(tmpX_test)
#						temp_Q2[n] = explained_variance_score(tmpY_test, Y_proj)
#						temp_rmse[n] = mean_squared_error(tmpY_test, Y_proj, squared = False)
#					Q2_grid_arr[c,s] = np.mean(temp_Q2)
#					Q2_grid_arr_sd[c,s] = np.std(temp_Q2)
#					RMSE_grid_arr[c,s] = np.mean(temp_rmse)
#					RMSE_grid_arr_sd[c,s] = np.std(temp_rmse)
#				else:
#					Q2_grid_arr[c,s] = 0.
#					Q2_grid_arr_sd[c,s] = 1.
#					RMSE_grid_arr[c,s] =  0.
#					RMSE_grid_arr_sd[c,s] = 1.
#		self.STAT_GRIDSEARCH_ = Q2_grid_arr
#		self.STAT_GRIDSEARCH_SD_ = Q2_grid_arr_sd
#		self.RMSEP_CV_GRIDSEARCH_ = RMSE_grid_arr
#		self.RMSEP_CV_GRIDSEARCH_SD_ = RMSE_grid_arr_sd
#		self.mean_selected_X = np.array(X_mean_selected)
#		self.mean_selected_y = np.array(y_mean_selected)


