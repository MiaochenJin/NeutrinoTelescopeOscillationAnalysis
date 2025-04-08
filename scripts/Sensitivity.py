import numpy as np
from scipy.optimize import minimize
from Systematics import Systematics


def compute_chi2(pred_binned, asimov_binned):
    """
    Compute Poisson chi-squared.
    """
    pred = pred_binned.flatten()
    obs = asimov_binned.flatten()
    mask = (obs > 0) & (pred > 0)
    pred = pred[mask]
    obs = obs[mask]
    chi2 = 2 * np.sum(pred - obs + obs * np.log(obs / pred))
    return chi2


def chi2_wrapper(syst_params, syst_template, binned_nominal, E_true_bins, E_true_centers, cth_centers, asimov_binned):
    """
    Wrapper that updates a Systematics object from a flat parameter list, applies systematics,
    and returns chi2 between prediction and Asimov dataset.
    """
    syst = Systematics(*syst_params)
    binned_syst = syst.apply_systematics(binned_nominal, E_true_bins, E_true_centers, cth_centers)
    return compute_chi2(binned_syst, asimov_binned)


def get_minimized_chi2(binned_nominal, E_true_bins, E_true_centers, cth_centers, asimov_binned, syst0):
    """
    Minimize chi2 over systematic parameters.
    """
    x0 = [
        syst0.f_all, syst0.f_HPT, syst0.f_S, syst0.f_HE, syst0.f_mu,
        syst0.f_tauCC, syst0.f_NC, syst0.s_mu_mubar, syst0.s_e_ebar,
        syst0.s_e_mu, syst0.delta_gamma, syst0.delta_theta
    ]

    bounds = [
        (0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (0.5, 1.5),
        (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1), (-0.1, 0.1),
        (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)
    ]

    res = minimize(
        chi2_wrapper, x0, args=(syst0, binned_nominal, E_true_bins, E_true_centers, cth_centers, asimov_binned),
        bounds=bounds, method='L-BFGS-B'
    )

    return res.fun, res.x