import numpy as np
import pandas as pd
import h5py
import nuflux
import math
import nuSQuIDS as nsq
import nuSQUIDSTools
from itertools import repeat
from math import asin, sqrt
from itertools import repeat
from utils import *
from scipy.interpolate import griddata
class Simulation:
	def __init__(self, experiment = 'ORCA', livetime = 1.39, filename = '../datafiles/ORCA/ORCA_MC.parquet', mode = 'Sterile'):
		# some global variables
		self._experiment = experiment
		self._filename = filename
		if mode == "Sterile":
			self.flavors = 4
		else: self.flavors = 3
		if experiment == 'ORCA':
			mc = pd.read_parquet(filename)
			mu_mc = mc[(mc["MC_type"] == -1)]
			nu_mc = mc[(mc["MC_type"] != -1)]
			self._nu_mc = nu_mc
			self._mu_mc = mu_mc
			self._mc_et_bin = (np.array(nu_mc["true_energy_bin_num"])).astype(int)
			self._mc_er_bin = (np.array(nu_mc["reco_energy_bin_num"])).astype(int)
			self._mc_ct_bin = (np.array(nu_mc["true_cos_zenith_bin_num"])).astype(int)
			self._mc_cr_bin = (np.array(nu_mc["reco_cos_zenith_bin_num"])).astype(int)
			self._num_morphology = 3
			self._E_true_bins = np.load("../datafiles/ORCA/_E_true_bins.npy")
			self._E_reco_bins = np.load("../datafiles/ORCA/_E_reco_bins.npy")
			self._cosT_true_bins = np.load("../datafiles/ORCA/_cosT_true_bins.npy")
			self._cosT_reco_bins = np.linspace(-1, 0, 11)
		elif experiment == 'IC':
			mc = pd.read_csv(filename)
			self._nu_mc = mc
			self._mu_mc = None
			self._num_morphology = 2
			self._E_true_bins = np.logspace(np.log10(1), np.log10(1e4), 40+1, endpoint = True)
			self._E_reco_bins = self._E_true_bins
			self._cosT_true_bins = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
			self._cosT_reco_bins = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
		# MC event information
		condition = (self._nu_mc["true_energy"] > 1) & (self._nu_mc["true_energy"] < 1e3) & (self._nu_mc["reco_energy"] > 1)
		self._num_entries = len(self._nu_mc[condition])
		self._mc_etrue = np.array(self._nu_mc["true_energy"][condition])
		self._mc_cthtrue = np.array(np.cos(self._nu_mc["true_zenith"])[condition])
		self._mc_ereco = np.array(self._nu_mc["reco_energy"][condition])
		self._mc_cthreco = np.array(np.cos(self._nu_mc["reco_zenith"])[condition])
		self._mc_nutype = np.array(self._nu_mc["pdg"].apply(lambda pdg: nuflux_dict[int(pdg)][0])[condition])
		self._mc_neuflavor = np.array(self._nu_mc["pdg"].apply(lambda pdg: nuflux_dict[int(pdg)][1])[condition])
		self._mc_weights = np.array(self._nu_mc["weight"][condition])
		self._mc_current = np.array(self._nu_mc["current_type"][condition])
		self._mc_morphology = np.array(self._nu_mc["pid"][condition])
		self._mc_interaction = np.array(self._nu_mc["interaction_type"][condition])

		# experiment constants
		self._livetime = livetime
		self._unit_norm = 1e4
		self._unit = nsq.Const().GeV # energy unit definitions
		# flux related settings
		self._atm_initial_flux = None
		self._flux_emin = 1e-1 * self._unit
		self._flux_emax = 1e3 * self._unit
		self._flux_enodes = 100
		self._flux_cthmin = -1.0
		self._flux_cthmax = 1.0
		self._flux_cnodes = 40
		self._flux_energy_nodes = None
		self._flux_cth_nodes = None
		# energy and zenith binning
		self._E_true_centers = np.sqrt(self._E_true_bins[1:] * self._E_true_bins[:-1])
		self._cth_bin_centers = (self._cosT_true_bins[:-1] + self._cosT_true_bins[1:]) / 2
		# mask to cut the bins with too little entries
		self._cut_bins = []
		# storage room for the best fit unweighted rates, filled in once per analysis
		self._BF_rates = None
		self._BF_rates_weighted_binned = None

	# set up the atmospheric initial flux object
	def SetInitialFlux(self):
		self._flux_energy_nodes = nsq.logspace(self._flux_emin, self._flux_emax, self._flux_enodes)
		self._flux_cth_nodes = nsq.linspace(self._flux_cthmin, self._flux_cthmax, self._flux_cnodes)
		nsq_atm = nsq.nuSQUIDSAtm(self._flux_cth_nodes,self._flux_energy_nodes,self.flavors,nsq.NeutrinoType.both,interactions)
		AtmInitialFlux = np.zeros((len(self._flux_cth_nodes),len(self._flux_energy_nodes),2,self.flavors))
		for ic,cth in enumerate(nsq_atm.GetCosthRange()):
			for ie,E in enumerate(nsq_atm.GetERange()):
				nu_energy = E/units.GeV
				nu_cos_zenith = cth
				for f in ['nue', 'numu']:
					f_ = f_dict[f]
					for t in ['nu', 'nubar']:
						t_ = t_dict[t]
						AtmInitialFlux[ic][ie][t_][f_] = flux.getFlux(nuflux_label["{}_{}".format(f, t)],nu_energy,nu_cos_zenith)
		self._atm_initial_flux = AtmInitialFlux

	# obtain the mc event unweighted rate for all events given oscillation parameters (phi * prob)
	def GetOscillatedRate(self, t12, t13, t23, dm21, dm31, dcp, Ordering='normal'):
		nsq_atm = nsq.nuSQUIDSAtm(self._flux_cth_nodes,self._flux_energy_nodes,self.flavors,nsq.NeutrinoType.both,interactions)
		nsq_atm.Set_rel_error(1.0e-4)
		nsq_atm.Set_abs_error(1.0e-4)
		nsq_atm.Set_MixingAngle(0, 1, t12)
		nsq_atm.Set_MixingAngle(0, 2, t13)
		nsq_atm.Set_MixingAngle(1, 2, t23)
		nsq_atm.Set_SquareMassDifference(1, dm21)
		nsq_atm.Set_SquareMassDifference(2, dm31)
		if Ordering!='normal': # change mass difference for IO setting
			AtmOsc.Set_SquareMassDifference(2,dm21-dm31)
		nsq_atm.Set_CPPhase(0, 2, dcp)
		nsq_atm.Set_initial_state(self._atm_initial_flux,nsq.Basis.flavor)
		nsq_atm.EvolveState() # progress bar is hidden here
		rate = np.zeros_like(self._mc_weights)
		# for i in range(len(rate)):
		# 	rate[i] = nsq_atm.EvalFlavor(int(self._mc_neuflavor[i]), float(self._mc_cthtrue[i]), float(self._mc_etrue[i] * self._unit), int(self._mc_nutype[i]))
		rate = list(map(nsq_atm.EvalFlavor, (self._mc_neuflavor.astype(int).tolist()), (self._mc_cthtrue.astype(float).tolist()), (self._mc_etrue*self._unit).astype(float).tolist(), (self._mc_nutype.astype(int).tolist()), repeat(True)))
		return rate

	# obtain the mc event unweighted rate for all events given oscillation sterile parameters (phi * prob)
	def GetOscillatedSterileRate(self, t12, t13, t23, dm21, dm31, dcp, t14, t24, t34, dm41, d24, Ordering='normal'):
		nsq_atm = nsq.nuSQUIDSAtm(self._flux_cth_nodes,self._flux_energy_nodes,self.flavors,nsq.NeutrinoType.both,interactions)
		nsq_atm.Set_rel_error(1.0e-4)
		nsq_atm.Set_abs_error(1.0e-4)
		nsq_atm.Set_MixingAngle(0, 1, t12)
		nsq_atm.Set_MixingAngle(0, 2, t13)
		nsq_atm.Set_MixingAngle(1, 2, t23)
		nsq_atm.Set_MixingAngle(0, 3, t14)
		nsq_atm.Set_MixingAngle(1, 3, t24)
		nsq_atm.Set_MixingAngle(2, 3, t34)			
		nsq_atm.Set_SquareMassDifference(1, dm21)
		nsq_atm.Set_SquareMassDifference(2, dm31)
		nsq_atm.Set_SquareMassDifference(3, dm41)
		if Ordering!='normal': # change mass difference for IO setting
			AtmOsc.Set_SquareMassDifference(2,dm21-dm31)
		nsq_atm.Set_CPPhase(0, 2, dcp)
		nsq_atm.Set_CPPhase(1, 3, d24)
		nsq_atm.Set_initial_state(self._atm_initial_flux,nsq.Basis.flavor)
		nsq_atm.EvolveState()
		rate = np.zeros_like(self._mc_weights)
		# for i in range(len(rate)):
		# 	rate[i] = nsq_atm.EvalFlavor(int(self._mc_neuflavor[i]), float(self._mc_cthtrue[i]), float(self._mc_etrue[i] * units.GeV), int(self._mc_nutype[i]))
		rate = list(map(nsq_atm.EvalFlavor, (self._mc_neuflavor.astype(int).tolist()), (self._mc_cthtrue.astype(float).tolist()), (self._mc_etrue*self._unit).astype(float).tolist(), (self._mc_nutype.astype(int).tolist()), repeat(True)))
		
		return rate
		
	# given the unweighted rates, multiply by weights and bin them 
	def BinWeightedRate8D(self, unweighted_rate, E_shift = 1):
		assert(len(unweighted_rate) == len(self._mc_weights))
		weighted_rate = unweighted_rate * self._mc_weights * self._livetime * self._unit_norm
		# Create an empty 8D histogram for neutrinos
		shape = (
			len(self._E_true_bins) - 1, len(self._E_reco_bins) - 1,
			len(self._cosT_true_bins) - 1, len(self._cosT_reco_bins) - 1,
			2, 3, 2, self._num_morphology
		)
		binned_events = np.zeros(shape)
		# Bin neutrino events
		for i in range(len(self._nu_mc)):
			te_idx = self._mc_et_bin[i] - 1
			re_idx = self._mc_er_bin[i] - 1
			tz_idx = self._mc_ct_bin[i] - 1
			rz_idx = self._mc_cr_bin[i] - 1
			n_idx = self._mc_nutype[i]
			f_idx = self._mc_neuflavor[i]
			i_idx = self._mc_current[i]
			m_idx = self._mc_morphology[i]
			if (0 <= te_idx < shape[0] and 0 <= re_idx < shape[1] and 
				0 <= tz_idx < shape[2] and 0 <= rz_idx < shape[3]):
				binned_events[te_idx, re_idx, tz_idx, rz_idx, n_idx, f_idx, i_idx, m_idx] += weighted_rate[i]
			else:
				print("Invalid index found in populating bins")
				print(shape[0], shape[1], shape[2], shape[3])
				print(te_idx, re_idx, tz_idx, rz_idx)
		return binned_events

	# the simple binning to bin events into only 3 bins
	def BinWeightedRate3DFlatten(self, unweighted_rate, E_shift = 1):
		weighted_rate = unweighted_rate * self._mc_weights * self._livetime * self._unit_norm
		shifted_E = self._mc_ereco * E_shift
		binned_events = np.array([])
		for m in range(self._num_morphology):
			morph = (self._mc_morphology == m)
			wrate_m = weighted_rate[morph]
			bins = (self._E_reco_bins, self._cosT_reco_bins)
			binned, _, _ = np.histogram2d(shifted_E[morph], self._mc_cthreco[morph], bins = bins, weights = wrate_m)
			binned_events = np.append(binned_events, binned)
		binned_events.reshape(-1)
		if len(self._cut_bins) > 0 : binned_events = binned_events[self._cut_bins]

		return binned_events
	
	# get the reco energy distribution of events, returns num_morph * num_ebin
	def BinWeightedRateRecoEnergy(self, unweighted_rate, E_shift = 1):
		weighted_rate = unweighted_rate * self._mc_weights * self._livetime * self._unit_norm
		shifted_E = self._mc_ereco * E_shift
		binned_events = np.zeros((self._num_morphology, len(self._E_reco_bins) - 1))
		for m in range(self._num_morphology):
			morph = (self._mc_morphology == m)
			wrate_m = weighted_rate[morph]
			binned, _ = np.histogram(shifted_E[morph], bins = self._E_reco_bins, weights = wrate_m)
			binned_events[m] = binned
		return binned_events

	# computes the best fit 
	def ComputeBFRates(self, t12, t13, t23, dm21, dm31, dcp, Ordering='normal'):
		if self._BF_rates is not None:
			print("Best fit unweighted rates is being set multiple times")
			exit(1)
		self._BF_rates = self.GetOscillatedRate(t12, t13, t23, dm21, dm31, dcp, Ordering='normal')
		self._BF_rates_weighted_binned = self.BinWeightedRate3DFlatten(self._BF_rates) # no E shift needed
		self._cut_bins = self._BF_rates_weighted_binned > 4 # cut all bins with fewer than 4 events
		self._BF_rates_weighted_binned = self._BF_rates_weighted_binned[self._cut_bins]
	
	# computes the best fit 
	def ComputeBFRatesSterile(self, t12, t13, t23, t14, t24, t34, dm21, dm31, dm41, dcp, dcp24, Ordering='normal'):
		if self._BF_rates is not None:
			print("Best fit unweighted rates is being set multiple times")
			exit(1)
		self._BF_rates = self.GetOscillatedSterileRate(t12, t13, t23, dm21, dm31, dcp, t14, t24, t34, dm41, dcp24, Ordering=Ordering)
		self._BF_rates_weighted_binned = self.BinWeightedRate3DFlatten(self._BF_rates) # no E shift needed
		self._cut_bins = self._BF_rates_weighted_binned > 4 # cut all bins with fewer than 4 events
		self._BF_rates_weighted_binned = self._BF_rates_weighted_binned[self._cut_bins]

	# method to deal with IC systematics
	def ReadDetSystTables(self, syst):
		if self._experiment != 'IC':
			print("skipping: IC systematics incorrectly loading for experiment ", self._experiment)
			return
		
		Ebin = dict({0 : self._E_true_bins, 1 : self._E_true_bins})
		Zbin = dict({0 : self._cosT_true_bins, 1 : self._cosT_true_bins})
		cut = self._cut_bins
		hp_nue_cc = pd.read_csv("../datafiles/IC/hyperplanes_nue_cc.csv")
		hp_numu_cc = pd.read_csv("../datafiles/IC/hyperplanes_numu_cc.csv")
		hp_nutau_cc = pd.read_csv("../datafiles/IC/hyperplanes_nutau_cc.csv")
		hp_nu_nc = pd.read_csv("../datafiles/IC/hyperplanes_all_nc.csv")

		method = 'nearest'

		grid_cz = np.array(hp_nue_cc['reco_coszen'])
		grid_Er =  np.array(hp_nue_cc['reco_energy'])
		pid_nue =  np.array(hp_nue_cc['pid'])
		ICSyst = [ "offset", "ice_absorption", "ice_scattering", "opt_eff_headon", "opt_eff_lateral", "opt_eff_overall", "coin_fraction"]
		if syst not in ICSyst:
			print('Systematic source not known for IC.')
		
		# for syst in ICSyst:
		offset_nueCC = np.array(hp_nue_cc['offset'])
		offset_numuCC =  np.array(hp_numu_cc['offset'])
		offset_nutauCC =  np.array(hp_nutau_cc['offset'])
		offset_NC =  np.array(hp_nu_nc['offset'])

		values_nueCC =  np.array(hp_nue_cc[syst]) / offset_nueCC
		values_numuCC =  np.array(hp_numu_cc[syst]) / offset_numuCC
		values_nutauCC =  np.array(hp_nutau_cc[syst]) / offset_nutauCC
		values_NC =  np.array(hp_nu_nc[syst]) / offset_NC

		dic = {}
		nuecc = np.array([])
		numucc = np.array([])
		nutaucc = np.array([])
		nc = np.array([])

		for sample in Ebin:
			c = pid_nue==sample
			points = np.array([grid_cz[c],grid_Er[c]]).T

			nue = values_nueCC[c]
			numu = values_numuCC[c]
			nutau = values_nutauCC[c]
			nuNC = values_NC[c]

			cz = Zbin[sample]
			e = Ebin[sample]
			zbin = cz[:-1] + np.diff(cz)/2
			ebin = e[:-1] + np.diff(e)/2

			newCz, newEr = np.meshgrid(zbin,ebin)

			nuecc = np.append(nuecc,griddata(points, nue, (newCz, newEr), method=method))
			numucc = np.append(numucc,griddata(points, numu, (newCz, newEr), method=method))
			nutaucc = np.append(nutaucc,griddata(points, nutau, (newCz, newEr), method=method))
			nc = np.append(nc,griddata(points, nuNC, (newCz, newEr), method=method))

		if len(cut)>0:
			dic['nueCC'] = nuecc.reshape(-1)[cut]
			dic['numuCC'] = numucc.reshape(-1)[cut]
			dic['nutauCC'] = nutaucc.reshape(-1)[cut]
			dic['NC'] = nc.reshape(-1)[cut]
		else:
			dic['nueCC'] = nuecc.reshape(-1)
			dic['numuCC'] = numucc.reshape(-1)
			dic['nutauCC'] = nutaucc.reshape(-1)
			dic['NC'] = nc.reshape(-1)

		return dic

	def SetDetSyst(self):
		ev = np.zeros(self._num_entries)
		c = (np.abs(self._mc_neuflavor)==0) * (self._mc_current)
		ev[c] = 1
		self.ExpFracNuECC = self.BinWeightedRate3DFlatten(ev * self._BF_rates) / self._BF_rates_weighted_binned
		ev = np.zeros(self._num_entries)
		c = (np.abs(self._mc_neuflavor)==1) * (self._mc_current)
		ev[c] = 1
		self.ExpFracNuMuCC = self.BinWeightedRate3DFlatten(ev * self._BF_rates) / self._BF_rates_weighted_binned
		ev = np.zeros(self._num_entries)
		c = (np.abs(self._mc_neuflavor)==2) * (self._mc_current)
		ev[c] = 1
		self.ExpFracNuTauCC = self.BinWeightedRate3DFlatten(ev * self._BF_rates) / self._BF_rates_weighted_binned
		ev = np.zeros(self._num_entries)
		c = np.logical_not(self._mc_current)
		ev[c] = 1
		self.ExpFracNC = self.BinWeightedRate3DFlatten(ev * self._BF_rates) / self._BF_rates_weighted_binned
		# Load systematics tables
		self.ice_absorption = self.ReadDetSystTables('ice_absorption')
		self.ice_scattering = self.ReadDetSystTables('ice_scattering')
		self.offset = self.ReadDetSystTables('offset')
		self.opt_eff_headon = self.ReadDetSystTables('opt_eff_headon')
		self.opt_eff_lateral = self.ReadDetSystTables('opt_eff_lateral')
		self.opt_eff_overall = self.ReadDetSystTables('opt_eff_overall')
		self.coin_fraction = self.ReadDetSystTables('coin_fraction')
