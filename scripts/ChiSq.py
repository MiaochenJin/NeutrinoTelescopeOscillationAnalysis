import numpy as np
import math
from scipy.special import gamma
from Systematics import *

# compute chi squared without penalty or jacobian or analytic prior for jacobian 
def ChiSq_only_no_prior(analysis, syst, N_dat, N_mod_hypo=None):
	sim = analysis.sim
	if N_mod_hypo is None:
		N_mod = analysis.sim.ReturnBFBinned()
	else:
		N_mod = N_mod_hypo
	assert(N_dat.shape == N_mod.shape)

	# Use the generic systematics application logic
	syst_factor = np.ones_like(N_mod)
	for i, syst_obj in enumerate(analysis.syst_objects):
		pull = syst[i]
		# In this simple function, we assume all systematics affect the total model.
		# This is sufficient for its purpose of scaling the tolerance.
		syst_factor *= (1 + syst_obj.apply(pull, sim))

	N_mod_syst = N_mod * syst_factor
	X2 = 2 * (N_mod_syst - N_dat + N_dat * np.log(N_dat / N_mod_syst))
	return np.sum(X2)

# this chi squared returns X2 and jacobian without priors
def ChiSq_Jac_no_prior(analysis, syst, N_dat, N_mod_hypo=None):
    sim = analysis.sim
    if N_mod_hypo is None:
        N_mod = analysis.sim.ReturnBFBinned()
    else:
        N_mod = N_mod_hypo
    assert(N_dat.shape == N_mod.shape)

    # Initialize for combined loop
    syst_factor = np.ones_like(N_mod)
    dNdx = [np.zeros_like(N_mod) for _ in syst] # Initialize dNdx for each systematic

    for i, syst_obj in enumerate(analysis.syst_objects):
        pull = syst[i]
        apply_val = syst_obj.apply(pull, sim)
        
        # Update syst_factor
        syst_factor *= (1 + apply_val)

        # Calculate dNdx for the current systematic
        denominator = (1 + apply_val)
        if np.any(denominator == 0):
            dNdx[i] = np.where(denominator != 0, N_mod / denominator * syst_obj.diff(pull, sim), 0)
        else:
            dNdx[i] = N_mod / denominator * syst_obj.diff(pull, sim)
            
    N_mod_syst = N_mod * syst_factor
    N_mod_syst = np.maximum(N_mod_syst, 1e-9) # Safeguard against division by zero or log(0)

    JX2 = [0] * len(syst)
    X2 = np.sum(2 * (N_mod_syst - N_dat + N_dat * np.log(N_dat / N_mod_syst)))
    
    for i in range(len(JX2)):
        JX2[i] = 2 * np.sum((1 - N_dat / N_mod_syst) * dNdx[i])
        
    return X2, np.array(JX2)

# this chi squared returns X2 and jacobian without priors
def ChiSq_Jac_with_penalty(analysis, syst, N_dat, N_mod_hypo=None):
    sim = analysis.sim
    if N_mod_hypo is None:
        N_mod_total = analysis.sim.ReturnBFBinned()
    else:
        N_mod_total = N_mod_hypo
    assert(N_dat.shape == N_mod_total.shape)

    if sim._experiment == 'ORCA' and sim._muon_bkg_binned is not None:
        N_mod_muon = sim._muon_bkg_binned[sim._cut_bins]
        N_mod_nu = N_mod_total - N_mod_muon
    else:
        N_mod_nu = N_mod_total
        N_mod_muon = 0

    syst_factor_nu = np.ones_like(N_mod_nu)
    syst_factor_mu = 1.0
    dNdx = [np.zeros_like(N_mod_nu) for _ in syst] # Initialize dNdx for each systematic

    for i, syst_obj in enumerate(analysis.syst_objects):
        pull = syst[i]
        
        if getattr(syst_obj, 'affects', 'neutrino') == 'muon':
            syst_factor_mu *= pull
            dNdx[i] = N_mod_muon # Derivative of N_mod_muon * pull with respect to pull is N_mod_muon
        else:
            apply_val = syst_obj.apply(pull, sim)
            syst_factor_nu *= (1 + apply_val)
            
            denominator = (1 + apply_val)
            if np.any(denominator == 0):
                dNdx[i] = np.where(denominator != 0, N_mod_nu * syst_factor_nu / denominator * syst_obj.diff(pull, sim), 0)
            else:
                dNdx[i] = N_mod_nu * syst_factor_nu / denominator * syst_obj.diff(pull, sim)
    
    N_mod_syst = N_mod_nu * syst_factor_nu + N_mod_muon * syst_factor_mu
    N_mod_syst = np.maximum(N_mod_syst, 1e-9) # Safeguard

    JX2 = [0] * len(syst)
    X2 = np.sum(2 * (N_mod_syst - N_dat + N_dat * np.log(N_dat / N_mod_syst)))
    
    for i in range(len(JX2)):
        JX2[i] = 2 * np.sum((1 - N_dat / N_mod_syst) * dNdx[i])
        
        x = syst[i]
        mu = analysis.systNominal[i]
        sig = analysis.systSigma[i]
        X2 += ((x - mu) / sig)**2
        JX2[i] += 2 * (x - mu) / sig**2

    return X2, np.array(JX2)

# this method does not yet have jacobian implemented
def ChiSq_with_penalty_with_error(analysis, syst, N_dat, N_mod_hypo=None, N_mod_hypo_err=None):
	sim = analysis.sim
	# prepare systematics
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

	# --- Generic Systematics Application ---
	syst_factor_nu = np.ones_like(N_mod_nu)
	syst_factor_mu = 1.0

	for i, syst_obj in enumerate(analysis.syst_objects):
		pull = syst[i]
		if getattr(syst_obj, 'affects', 'neutrino') == 'muon':
			syst_factor_mu *= pull
		else:
			syst_factor_nu *= (1 + syst_obj.apply(pull, sim))

	N_mod_syst = N_mod_nu * syst_factor_nu + N_mod_muon * syst_factor_mu
	N_mod_syst = np.maximum(N_mod_syst, 1e-9)

	# --- Barlow-Beeston implementation ---
	tau = np.divide(err_mod, N_mod_syst**2, out=np.zeros_like(err_mod), where=N_mod_syst!=0)
	b = N_mod_syst * tau - 1.0
	c = -N_dat * tau
	sqrt_discriminant = np.sqrt(np.maximum(0, b**2 - 4*c))
	beta = 0.5 * (-b + sqrt_discriminant)
	bb_penalty = np.sum(np.divide((beta - 1)**2, tau, out=np.zeros_like(tau), where=tau!=0))
	beta_N_mod = np.maximum(beta * N_mod_syst, 1e-9)
	
	log_term = np.log(np.divide(N_dat, beta_N_mod, out=np.ones_like(N_dat), where=beta_N_mod!=0))
	
	# Set terms to zero where data is zero to avoid log(0) = -inf issues.
	log_term[N_dat == 0] = 0
	
	poisson_chi2 = np.sum(2 * (beta_N_mod - N_dat + N_dat * log_term))

	# 4. Sum the components for the total chi-squared
	X2 = poisson_chi2 + bb_penalty
	
	# 5. Add the standard penalty for systematic nuisance parameters
	for i,(x,mu,sig) in enumerate(zip(syst,analysis.systNominal,analysis.systSigma)):
		X2 += ((x-mu) / sig)**2
		
	return X2

# this chi squared returns X2 and jacobian with penalty and MC error (Barlow-Beeston)
def ChiSq_Jac_with_penalty_with_error(analysis, syst, N_dat, N_mod_hypo=None, N_mod_hypo_err=None):
    sim = analysis.sim
    if N_mod_hypo is None:
        N_mod = analysis.sim.ReturnBFBinned()
    else:
        N_mod = N_mod_hypo

    if N_mod_hypo_err is None:
        err_mod = sim.ReturnBFErrorBinned()
    else:
        err_mod = N_mod_hypo_err
    assert(N_dat.shape == N_mod.shape)
    assert(N_mod.shape == err_mod.shape)

    if sim._experiment == 'ORCA' and sim._muon_bkg_binned is not None:
        N_mod_muon = sim._muon_bkg_binned[sim._cut_bins]
        N_mod_nu = N_mod - N_mod_muon
    else:
        N_mod_nu = N_mod
        N_mod_muon = 0

    syst_factor_nu = np.ones_like(N_mod_nu)
    syst_factor_mu = 1.0
    dNdx = [np.zeros_like(N_mod_nu) for _ in syst] # Initialize dNdx for each systematic

    for i, syst_obj in enumerate(analysis.syst_objects):
        pull = syst[i]
        
        if getattr(syst_obj, 'affects', 'neutrino') == 'muon':
            syst_factor_mu *= pull
            dNdx[i] = N_mod_muon # Derivative of N_mod_muon * pull with respect to pull is N_mod_muon
        else:
            apply_val = syst_obj.apply(pull, sim)
            syst_factor_nu *= (1 + apply_val)
            
            denominator = (1 + apply_val)
            if np.any(denominator == 0):
                dNdx[i] = np.where(denominator != 0, N_mod_nu * syst_factor_nu / denominator * syst_obj.diff(pull, sim), 0)
            else:
                dNdx[i] = N_mod_nu * syst_factor_nu / denominator * syst_obj.diff(pull, sim)

    N_mod_syst = N_mod_nu * syst_factor_nu + N_mod_muon * syst_factor_mu
    N_mod_syst = np.maximum(N_mod_syst, 1e-9)

    tau = np.divide(err_mod, N_mod_syst**2, out=np.zeros_like(err_mod), where=N_mod_syst!=0)
    b = N_mod_syst * tau - 1.0
    c = -N_dat * tau
    sqrt_discriminant = np.sqrt(np.maximum(0, b**2 - 4*c))
    beta = 0.5 * (-b + sqrt_discriminant)
    bb_penalty = np.sum(np.divide((beta - 1)**2, tau, out=np.zeros_like(tau), where=tau!=0))
    beta_N_mod = np.maximum(beta * N_mod_syst, 1e-9)
    
    log_term = np.log(np.divide(N_dat, beta_N_mod, out=np.ones_like(N_dat), where=beta_N_mod!=0))
    log_term[N_dat == 0] = 0
    
    poisson_chi2 = np.sum(2 * (beta_N_mod - N_dat + N_dat * log_term))

    X2 = poisson_chi2 + bb_penalty
    
    JX2 = [0] * len(syst)
    for i in range(len(JX2)):
        JX2[i] += 2 * np.sum((1 - N_dat / beta_N_mod) * beta * dNdx[i])

        x = syst[i]
        mu = analysis.systNominal[i]
        sig = analysis.systSigma[i]
        X2 += ((x - mu) / sig)**2
        JX2[i] += 2 * (x - mu) / sig**2

    return X2, np.array(JX2)

# compute the penalty and jacobian due to analytic prior bounds
def syst_penalty_prior(analysis, syst, N_dat, N_mod_hypo=None, N_mod_hypo_err=None):
	sim = analysis.sim
	n_syst = len(syst)
	lin_terms = np.zeros(n_syst)
	quad_terms = np.zeros(n_syst)
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

	for i, syst_obj in enumerate(analysis.syst_objects):
		# compute derivative at nominal pull
		mu_i = analysis.systNominal[i]
		dFdx = syst_obj.diff(mu_i, sim) # This is df/ds
		
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