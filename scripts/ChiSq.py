import numpy as np
import math
from scipy.special import gamma
from Systematics import *

# compute chi squared without penalty or jacobian or analytic prior for jacobian 
def ChiSq_only_no_prior(analysis, syst, N_dat):
	sim = analysis.sim
	syst_ls = analysis.systNames
	syst_reg = analysis.systRegistry
	N_mod = sim._BF_rates_weighted_binned
	assert(N_dat.shape == N_mod.shape)
	syst_shift = 0
	print(N_mod.shape)
	for i, sname in enumerate(syst_ls):
		apply_fn, diff_fn = syst_reg[sname]
		syst_shift += apply_fn(syst[i], sim)
		# print(sname)
		print(syst_shift.shape)
	N_mod_syst = N_mod * (1 + syst_shift)

	X2 = 2 * (N_mod_syst - N_dat + N_dat * np.log(N_dat / N_mod_syst))
	return np.sum(X2)

# this chi squared returns X2 and jacobian without priors
def ChiSq_Jac_no_prior(analysis, syst, N_dat):
	JX2 = [0] * len(syst)
	sim = analysis.sim
	syst_ls = analysis.systNames
	syst_reg = analysis.systRegistry
	N_mod = sim._BF_rates_weighted_binned
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
def ChiSq_Jac_with_penalty(analysis, syst, N_dat):
	JX2 = [0] * len(syst)
	sim = analysis.sim
	syst_ls = analysis.systNames
	syst_reg = analysis.systRegistry
	N_mod = sim._BF_rates_weighted_binned
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

# compute the penalty and jacobian due to analytic prior bounds
def syst_penalty_prior(analysis, syst, N_dat):
	sim = analysis.sim
	n_syst = len(syst)
	lin_terms = np.zeros(n_syst)
	quad_terms = np.zeros(n_syst)
	syst_ls = analysis.systNames
	syst_reg = analysis.systRegistry
	N_mod = sim._BF_rates_weighted_binned
	N_diff = N_dat - N_mod
	for i, sname in enumerate(syst_ls):
		apply_fn, diff_fn = syst_reg[sname]
		# compute derivative at nominal pull
		mu_i = analysis.systNominal[i]
		dFdx = diff_fn(mu_i, sim)
		lin_terms[i] += np.sum(N_diff * dFdx)
		quad_terms[i] += np.sum(N_dat * dFdx ** 2)
	priors = []
	bounds = []
	for i, (mu_i, sigma_i) in enumerate(zip(analysis.systNominal, analysis.systSigma)):
		pr = mu_i + lin_terms[i] / (quad_terms[i] + 1.0/sigma_i**2)
		delta = min(abs(pr - mu_i), sigma_i)
		center = 0.5 * (mu_i + pr)
		priors.append(center)
		if delta > 0:
			bounds.append((center - delta, center + delta))
		else:
			bounds.append((mu_i - sigma_i, mu_i + sigma_i))
	return np.array(priors), tuple(bounds)