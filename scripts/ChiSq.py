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
		N_mod_total = analysis.sim.ReturnBFBinned()
	else:
		N_mod_total = N_mod_hypo
	assert(N_dat.shape == N_mod_total.shape)

	# Separate neutrino and muon components if necessary
	if sim._experiment == 'ORCA' and sim._muon_bkg_binned is not None:
		N_mod_muon = sim._muon_bkg_binned[sim._cut_bins]
		N_mod_nu = N_mod_total - N_mod_muon
	else:
		N_mod_nu = N_mod_total
		N_mod_muon = 0

	syst_shift = 0
	dNdx = [0] * len(syst)
	for i, sname in enumerate(syst_ls):
		apply_fn, diff_fn = syst_reg[sname]
		# Systematics apply only to the neutrino component
		syst_shift += apply_fn(syst[i], sim)
		dNdx[i] += N_mod_nu * diff_fn(syst[i], sim)

	# Apply systematics to neutrino part, then add back muon background
	N_mod_syst = N_mod_nu * (1 + syst_shift) + N_mod_muon
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
	# Separate neutrino and muon components if necessary
	if sim._experiment == 'ORCA' and sim._muon_bkg_binned is not None:
		N_mod_muon = sim._muon_bkg_binned[sim._cut_bins]
		N_mod_nu = N_mod - N_mod_muon
	else:
		N_mod_nu = N_mod
		N_mod_muon = 0

	# start calculation of systematics effect
	syst_shift = 0
	for i, sname in enumerate(syst_ls):
		apply_fn, diff_fn = syst_reg[sname]
		syst_shift += apply_fn(syst[i], sim)

	# Apply systematics to neutrino part, then add back muon background
	N_mod_syst = N_mod_nu * (1 + syst_shift) + N_mod_muon

	# --- Barlow-Beeston implementation ---
	# The method accounts for MC statistical error (err_mod = sigma_MC^2)
	# by finding an optimal scaling factor `beta` for the model in each bin.

	# 1. Calculate squared relative MC error, tau = sigma_MC^2 / N_mod^2
	# Use np.divide for safe division. Where N_mod_syst is 0, tau will be 0.
	tau = np.divide(err_mod, N_mod_syst**2,
					out=np.zeros_like(err_mod), where=N_mod_syst!=0)

	# 2. Solve the quadratic equation for beta in each bin:
	# beta^2 + (N_mod * tau - 1)*beta - N_dat*tau = 0
	a = 1.0
	b = N_mod_syst * tau - 1.0
	c = -N_dat * tau
	
	# The positive solution for beta is (-b + sqrt(b^2 - 4ac)) / 2a
	sqrt_discriminant = np.sqrt(np.maximum(0, b**2 - 4*a*c)) # Ensure non-negative
	beta = 0.5 * (-b + sqrt_discriminant)

	# 3. Calculate the two components of the chi-squared
	
	# The penalty term for the MC statistics: sum_i (beta_i - 1)^2 / tau_i
	bb_penalty = np.sum(np.divide((beta - 1)**2, tau,
								  out=np.zeros_like(tau), where=tau!=0))

	# The scaled model prediction
	beta_N_mod = beta * N_mod_syst

	# --- Start Debugging Block ---
	if np.any(beta_N_mod <= 0):
		print("\n--- DEBUG: Problem detected in ChiSq calculation ---")
		problem_indices = np.where(beta_N_mod <= 0)[0]
		print(f"Problem in {len(problem_indices)} bins. Indices: {problem_indices}")
		
		# To avoid spamming, print details for the first problematic bin only
		i = problem_indices[0]
		print(f"\n--- Details for First Problematic Bin (Index: {i}) ---")
		print(f"  Data (N_dat)          : {N_dat[i]}")
		print(f"  Model Total (N_mod)   : {N_mod[i]}")
		if sim._experiment == 'ORCA':
			# Recalculate for printing, ensuring correct context
			N_mod_muon_val = sim._muon_bkg_binned[sim._cut_bins][i] if sim._muon_bkg_binned is not None else 0
			N_mod_nu_val = N_mod[i] - N_mod_muon_val
			print(f"  Model Neutrino (N_mod_nu) : {N_mod_nu_val}")
			print(f"  Model Muon (N_mod_muon) : {N_mod_muon_val}")
		print(f"  Model Error^2 (err_mod) : {err_mod[i]}")
		print(f"  Syst Shift (syst_shift) : {syst_shift[i]}")
		print(f"  Syst Model (N_mod_syst) : {N_mod_syst[i]}")
		print(f"  BB Factor (beta)        : {beta[i]}")
		print(f"  Final Model (beta_N_mod): {beta_N_mod[i]}")

		print("\n--- Systematics Pulls (syst) ---")
		for s_name, s_val in zip(analysis.systNames, syst):
			print(f"  {s_name:<30}: {s_val}")
		print("--- END DEBUG ---\n")
		# Clip values to prevent crash during this debug run
		beta_N_mod = np.maximum(beta_N_mod, 1e-9)
	# --- End Debugging Block ---
	
	# The Poisson log-likelihood term comparing scaled model to data
	# Use safe division for the log term
	log_term = np.log(np.divide(N_dat, beta_N_mod,
								out=np.ones_like(N_dat), where=beta_N_mod!=0))
	
	# Set terms to zero where data is zero to avoid log(0) = -inf issues.
	log_term[N_dat == 0] = 0
	
	poisson_chi2 = np.sum(2 * (beta_N_mod - N_dat + N_dat * log_term))

	# 4. Sum the components for the total chi-squared
	X2 = poisson_chi2 + bb_penalty
	
	# 5. Add the standard penalty for systematic nuisance parameters
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