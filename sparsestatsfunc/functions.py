#!/usr/bin/env python

import numpy as np

from scipy.stats import t, f
from statsmodels.stats.multitest import multipletests

from sparsestatsfunc.cynumstats import cy_lin_lstsqr_mat_residual, cy_lin_lstsqr_mat, fast_se_of_slope, tval_fast

import rpy2.robjects as robjects
from rpy2.robjects import FloatVector, numpy2ri
from rpy2.robjects.packages import importr
stats = importr('stats')
base = importr('base')
spls = importr('spls')
numpy2ri.activate()

# spls( x, y, K, eta, kappa=0.5, select="pls2", fit="simpls", scale.x=TRUE, scale.y=FALSE, eps=1e-4, maxstep=100, trace=FALSE)

class spls_rwrapper:
	"""
	Wrapper that uses the spls r package, and rpy2
	https://cran.r-project.org/web/packages/spls/
	Based on: Chun and Keles (2010), doi:10.1111/j.1467-9868.2009.00723.x
	"""
	def __init__(self, n_components, eta, kappa = 0.5, max_iter = 100, algorithm_selection = "pls2", algorithm_fit = "simpls", scale_x = True, scale_y = False, effective_zero = 0.0001):
		"""
		Setting for spls
		
		Parameters
		----------
		n_components : array
			The number of components to fit
		eta : float
			The sparsity metric ranging from 0 to 1. At Eta = 0 sPLS is equivalent to PLS
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
		self.scale_x = scale_x # implement later
		self.scale_y = scale_y # implement later
		self.penalty = "l1"
		self.algorithm_selection = algorithm_selection
		self.algorithm_fit = algorithm_fit
		self.effective_zero = effective_zero
		self.max_iter = max_iter
	def fit(self, X, y):
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
		self.components_ : array
			beta component vectors [N_components, N_predictors, N_responses]
		self.selectedvariablescomponents_ : object
			selected variables for each component [N_components, N_selected]
		self.selectedvariablesindex_ : arr1
			Selected variables index. Useful for subsetting
		self.coef_ : array
			coefficient array
		"""
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
		self.components_ = np.array(components)
		self.selectedvariablescomponents_ = np.array(sel_vars, dtype=object) - 1
		self.selectedvariablesindex_ = np.sort(np.concatenate(sel_vars)) - 1
		self.coef_ = stats.coef(model)
	def predict(self, X):
		"""
		Predict y from X using the spls model
		"""
		return(np.dot(X,self.coef_))


def dummy_code_cosine(time_variable, period = [24.0]):
	exog_vars = np.ones((len(time_variable)))
	for i in range(len(period)):
		exog_vars = np.column_stack((exog_vars, np.cos(np.divide(2.0 * np.pi * time_variable, period[i]))))
		exog_vars = np.column_stack((exog_vars, np.sin(np.divide(2.0 * np.pi * time_variable, period[i]))))
	return(np.array(exog_vars))

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
