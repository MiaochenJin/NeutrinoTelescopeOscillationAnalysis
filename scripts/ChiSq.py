import numpy as np
import math
from scipy.special import gamma
from Systematics import *

# compute chi squared without penalty or jacobian or analytic prior for jacobian 
def ChiSq_only_no_prior(analysis, syst, N_dat, N_mod_hypo=None):
	sim = analysis.sim
	syst_ls = analysis.systNames
	syst_reg = analysis.systRegistry
	if N_mod_hypo is None:
	N_mod = analysis.sim.ReturnBFBinned()
	else:
		N_mod = N_mod_hypo
	assert(N_dat.shape == N_mod.shape)
	syst_shift = 0
	for i, sname in enumerate(syst_ls):
		apply_fn, diff_fn = syst_reg[sname]
		syst_shift += apply_fn(syst[i], sim)
	N_mod_syst = N_mod * (1 + syst_shift)
	X2 = 2 * (N_mod_syst - N_dat + N_dat * np.log(N_dat / N_mod_syst))
	return np.sum(X2)

# this chi squared returns X2 and jacobian without priors
def ChiSq_Jac_no_prior(analysis, syst, N_dat, N_mod_hypo=None):
	JX2 = [0] * len(syst)
	sim = analysis.sim
	syst_ls = analysis.systNames
	syst_reg = analysis.systRegistry
	if N_mod_hypo is None:
		N_mod = analysis.sim.ReturnBFBinned()
	else:
		N_mod = N_mod_hypo
	assert(N_dat.shape == N_mod.shape)
	syst_shift = 0
	dNdx = [0] * len(syst)

	for i, sname in enumerate(syst_ls):
		print(sname)
		apply_fn, diff_fn = syst_reg[sname]
		syst_shift += apply_fn(syst[i], sim)
		dNdx[i] += N_mod * diff_fn(syst[i], sim)

	N_mod_syst = N_mod * (1 + syst_shift)

	X2 = np.sum(2 * (N_mod_syst - N_dat + N_dat * np.log(N_dat / N_mod_syst)))
	for i in range(len(JX2)):
		JX2[i] += 2 * np.sum((1 - N_dat / N_mod_syst) * dNdx[i])
	return X2, np.array(JX2)

# this chi squared returns X2 and jacobian without priors
def ChiSq_Jac_with_penalty(analysis, syst, N_dat, N_mod_hypo=None):
	JX2 = [0] * len(syst)
	sim = analysis.sim
	syst_ls = analysis.systNames
	syst_reg = analysis.systRegistry
	if N_mod_hypo is None:
	N_mod = analysis.sim.ReturnBFBinned()
	else:
		N_mod = N_mod_hypo
	assert(N_dat.shape == N_mod.shape)
	syst_shift = 0
	dNdx = [0] * len(syst)
	for i, sname in enumerate(syst_ls):
		apply_fn, diff_fn = syst_reg[sname]
		syst_shift += apply_fn(syst[i], sim)
		dNdx[i] += N_mod * diff_fn(syst[i], sim)

	N_mod_syst = N_mod * (1 + syst_shift)
	X2 = np.sum(2 * (N_mod_syst - N_dat + N_dat * np.log(N_dat / N_mod_syst)))
	for i in range(len(JX2)):
		JX2[i] += 2 * np.sum((1 - N_dat / N_mod_syst) * dNdx[i])
	# apply penalty
	for i,(x,mu,sig) in enumerate(zip(syst,analysis.systNominal,analysis.systSigma)):
		X2 += ((x-mu) / sig)**2
		JX2[i] += 2 * (x-mu) / sig**2

	return X2, np.array(JX2)

# this method does not yet have jacobian implemented
def ChiSq_with_penalty_with_error(analysis, syst, N_dat, N_mod_hypo=None, N_mod_hypo_err=None):
	sim = analysis.sim
	# prepare systematics
	syst_ls = analysis.systNames
	syst_reg = analysis.systRegistry
	# prepare different things that go into the ChiSq calculation
	if N_mod_hypo is None:
	N_mod = analysis.sim.ReturnBFBinned()
	else:
		N_mod = N_mod_hypo

	if N_mod_hypo_err is None:
	err_mod = sim.ReturnBFErrorBinned()
	else:
		err_mod = N_mod_hypo_err
	# make sure shapes are correct
	assert(N_dat.shape == N_mod.shape)
	assert(N_mod.shape == err_mod.shape)
	# start calculation of systematics effect
	syst_shift = 0
	for i, sname in enumerate(syst_ls):
		apply_fn, diff_fn = syst_reg[sname]
		syst_shift += apply_fn(syst[i], sim)
	# compute N_mod(theta, syst)
	N_mod_syst = N_mod * (1 + syst_shift)
	# compute BB error based on N_mod_syst
	def compute_beta():
		beta = np.zeros_like(err_mod)
		model_uncert = 1 - N_mod_syst * err_mod
		beta = .5 * (model_uncert + np.sqrt(model_uncert ** 2 + 4 * N_dat * err_mod))
		return beta
	beta = compute_beta()
	BB_error = np.sum(np.nan_to_num((beta - 1) ** 2 / err_mod, nan = 0))
	beta_N_mod = beta * N_mod_syst
	# sum up to get X2 with analytically minimized beta
	X2 = np.sum(2 * (beta_N_mod - N_dat + N_dat * np.log(N_dat / beta_N_mod)))
	X2 += BB_error
	# apply BB penalty and syst penalty
	for i,(x,mu,sig) in enumerate(zip(syst,analysis.systNominal,analysis.systSigma)):
		X2 += ((x-mu) / sig)**2
	return X2

# compute the penalty and jacobian due to analytic prior bounds
def syst_penalty_prior(analysis, syst, N_dat, N_mod_hypo=None, N_mod_hypo_err=None):
	sim = analysis.sim
	n_syst = len(syst)
	lin_terms = np.zeros(n_syst)
	quad_terms = np.zeros(n_syst)
	syst_ls = analysis.systNames
	syst_reg = analysis.systRegistry
	if N_mod_hypo is None:
	N_mod = analysis.sim.ReturnBFBinned()
	else:
		N_mod = N_mod_hypo
	
	# The Hessian (quad_terms) depends on the variance of the bin contents
	if N_mod_hypo_err is None:
		# Original case: variance is just data statistics, approximated by N_dat
		variance = N_dat
	else:
		# With MC error: variance is sum of data and model variance
		variance = N_dat + N_mod_hypo_err
	
	# Avoid division by zero for bins with zero variance
	safe_variance = np.where(variance > 0, variance, 1)

	for i, sname in enumerate(syst_ls):
		apply_fn, diff_fn = syst_reg[sname]
		# compute derivative at nominal pull
		mu_i = analysis.systNominal[i]
		dFdx = diff_fn(mu_i, sim) # This is df/ds
		
		# Gradient term: d(Chi2)/ds
		lin_terms[i] += np.sum((N_mod - N_dat) * dFdx)

		# Hessian term: d^2(Chi2)/ds^2, which is sum_bins[ (dL/ds)^2 / variance ]
		dLds = N_mod * dFdx
		quad_terms[i] += np.sum(dLds**2 / safe_variance)

	priors = []
	bounds = []
	for i, (mu_i, sigma_i) in enumerate(zip(analysis.systNominal, analysis.systSigma)):
		# prior = mu - gradient / (Hessian + prior_variance)
		pr = mu_i - lin_terms[i] / (quad_terms[i] + 1.0/sigma_i**2)
		delta = min(abs(pr - mu_i), sigma_i)
		center = 0.5 * (mu_i + pr)
		priors.append(center)
		if delta > 0:
			bounds.append((center - delta, center + delta))
		else:
			bounds.append((mu_i - sigma_i, mu_i + sigma_i))
	return np.array(priors), tuple(bounds)