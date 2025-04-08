import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SimReader import Reader
from utils import *

class Systematics:
    def __init__(self, f_all, f_HPT, f_S, f_HE, f_mu, f_tauCC, f_NC, s_mu_mubar, s_e_ebar, s_e_mu, delta_gamma, delta_theta):
        self.f_all = f_all
        self.f_HPT = f_HPT
        self.f_S = f_S
        self.f_HE = f_HE
        self.f_mu = f_mu
        self.f_tauCC = f_tauCC
        self.f_NC = f_NC
        self.s_mu_mubar = s_mu_mubar
        self.s_e_ebar = s_e_ebar
        self.s_e_mu = s_e_mu
        self.delta_gamma = delta_gamma
        self.delta_theta = delta_theta

    def apply_systematics(self, binned_events, E_true_bins, E_true_centers, cth_true_centers):
        # # Normalizations
        binned_events = self.apply_overall_normalization(binned_events)
        binned_events = self.apply_high_energy_normalization(binned_events, E_true_bins)
        binned_events = self.apply_pid_normalization(binned_events, 0)  # Showers
        binned_events = self.apply_pid_normalization(binned_events, 1)  # High Purity Tracks
        binned_events = self.apply_tau_CC_normalization(binned_events)
        binned_events = self.apply_NC_normalization(binned_events)

        # Flux normalizations
        binned_events = self.apply_flux_spectral_index(binned_events, E_true_centers)
        binned_events = self.apply_flux_v_to_h_ratio(binned_events, cth_true_centers)

        binned_events = self.apply_electron_flux(binned_events)
        binned_events = self.apply_muon_flux(binned_events)
        binned_events = self.apply_electron_bar_flux(binned_events)
        binned_events = self.apply_muon_bar_flux(binned_events)
        return binned_events

    # this file defines the application of eyetematics, including the post-bin (ORCA) and pre-bin (IC) approaches
    def apply_overall_normalization(self, binned_events):
        return binned_events * (1 + self.f_all)

    def apply_high_energy_normalization(self, binned_events, true_energy_bins):
        # Compute masks for CC and NC based on true energy bins
        cc_energy_mask = true_energy_bins[:-1] > 500  # CC bins (True Energy > 500 GeV)
        nc_energy_mask = true_energy_bins[:-1] > 100  # NC bins (True Energy > 100 GeV)
        # set up the interaction masks
        cc_interaction_mask = np.array([False, True])
        nc_interaction_mask = np.array([True, False])
        # Expand dimensions to match the shape of binned_events
        cc_energy_mask = cc_energy_mask[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        nc_energy_mask = nc_energy_mask[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        cc_interaction_mask = cc_interaction_mask[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        nc_interaction_mask = nc_interaction_mask[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        # Apply the correct mask for CC and NC bins
        affected_bins = (cc_energy_mask & cc_interaction_mask) | (nc_energy_mask & nc_interaction_mask)
        # Apply the systematic shift only to the affected bins
        return binned_events + self.f_HPT * affected_bins * binned_events

    def apply_pid_normalization(self, binned_events, pid_index):
        affected_bins = np.zeros_like(binned_events)
        affected_bins[:, :, :, :, :, :, :, pid_index] = 1  # Select only the specified pid
        if pid_index == 0:
            return binned_events + self.f_S * affected_bins * binned_events
        elif pid_index == 1:
            return binned_events + self.f_HPT * affected_bins * binned_events


    def apply_NC_normalization(self, binned_events):
        affected_bins = np.zeros_like(binned_events)
        affected_bins[:, :, :, :, :, :, 0, :] = 1  # Select only the specified pid
        return binned_events + self.f_NC * affected_bins * binned_events


    def apply_tau_CC_normalization(self, binned_events):
        affected_bins = np.zeros_like(binned_events)
        affected_bins[:, :, :, :, :, 2, 1, :] = 1  # Select only the specified pid
        return binned_events + self.f_tauCC * affected_bins * binned_events

    def apply_muon_normalization(self, binned_muons):
        return binned_muons * (1 + self.f_mu)

    def apply_flux_spectral_index(self, binned_events, energy_bin_centers, E0=1):
        # Compute spectral weight modification
        spectral_shift = (energy_bin_centers / E0) ** self.delta_gamma

        # Expand dimensions to match binned_events shape
        spectral_shift = spectral_shift[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

        # Apply the spectral modification to all bins
        return binned_events * spectral_shift

    def apply_flux_v_to_h_ratio(self, binned_events, cth_bin_centers):
        ratio_shift = self.delta_theta * np.abs(cth_bin_centers)
        ratio_shift = ratio_shift[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        return binned_events + ratio_shift * binned_events

    def apply_electron_flux(self, binned_events):
        affected_bins = np.zeros_like(binned_events)
        affected_bins[:, :, :, :, 0, 0, :, :]
        return binned_events + (self.s_e_mu + self.s_e_ebar + self.s_e_mu * self.s_e_ebar) * affected_bins * binned_events

    def apply_muon_flux(self, binned_events):
        affected_bins = np.zeros_like(binned_events)
        affected_bins[:, :, :, :, 0, 1, :, :]
        I_e = np.sum(binned_events[:, :, :, :, 0, 0, :, :])
        I_ebar = np.sum(binned_events[:, :, :, :, 1, 0, :, :])
        I_mu = np.sum(binned_events[:, :, :, :, 0, 1, :, :])
        I_mubar = np.sum(binned_events[:, :, :, :, 1, 1, :, :])
        I_ratio = (I_e + I_ebar) / (I_mu + I_mubar)
        return binned_events + (self.s_mu_mubar - self.s_e_mu * self.s_mu_mubar * I_ratio - self.s_e_mu * I_ratio) * affected_bins * binned_events

    def apply_electron_bar_flux(self, binned_events):
        affected_bins = np.zeros_like(binned_events)
        affected_bins[:, :, :, :, 1, 0, :, :]
        I_e = np.sum(binned_events[:, :, :, :, 0, 0, :, :])
        I_ebar = np.sum(binned_events[:, :, :, :, 1, 0, :, :])
        I_ratio = I_e / I_ebar
        return binned_events + (self.s_e_mu - self.s_e_ebar * I_ratio - self.s_e_mu * self.s_e_ebar * I_ratio) * affected_bins * binned_events

    def apply_muon_bar_flux(self, binned_events):
        affected_bins = np.zeros_like(binned_events)
        affected_bins[:, :, :, :, 1, 1, :, :]
        I_e = np.sum(binned_events[:, :, :, :, 0, 0, :, :])
        I_ebar = np.sum(binned_events[:, :, :, :, 1, 0, :, :])
        I_mu = np.sum(binned_events[:, :, :, :, 0, 1, :, :])
        I_mubar = np.sum(binned_events[:, :, :, :, 1, 1, :, :])
        I_ratio = (I_e + I_ebar) / (I_mu + I_mubar)
        I_mu_ratio = I_mu / I_mubar
        return binned_events + \
                (- self.s_mu_mubar * I_mu_ratio - self.s_e_mu * I_ratio + self.s_mu_mubar * self.s_e_mu * I_ratio * I_mu_ratio) \
                * affected_bins * binned_events
