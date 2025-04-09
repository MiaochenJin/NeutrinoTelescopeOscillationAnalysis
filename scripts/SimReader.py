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
 
 class Reader:
 	def __init__(self, source = None, experiment = 'ORCA', exposure = 1.39, filename = '../../datafiles/ORCA_MC.parquet'):
 		# some global variables
 		self._experiment = experiment
 		self._exposure = exposure
 		self._filename = filename
 		mc = pd.read_parquet(filename)
		if experiment == 'ORCA':
			mu_mc = mc[(mc["MC_type"] == -1)]
			nu_mc = mc[(mc["MC_type"] != -1)]
			self._nu_mc = nu_mc
			self._mu_mc = mu_mc
			self._mc_et_bin = (np.array(nu_mc["true_energy_bin_num"])).astype(int)
 			self._mc_er_bin = (np.array(nu_mc["reco_energy_bin_num"])).astype(int)
 			self._mc_ct_bin = (np.array(nu_mc["true_cos_zenith_bin_num"])).astype(int)
 			self._mc_cr_bin = (np.array(nu_mc["reco_cos_zenith_bin_num"])).astype(int)
		elif experiment == 'IceCube':
			self._nu_mc = mc
			self._mu_mc = None
 		# MC event information
 		self._mc_etrue = nu_mc["true_energy"]
 		self._mc_cthtrue = np.cos(nu_mc["true_zenith"])
 		self._mc_nutype = nu_mc["pdg"].apply(lambda pdg: nuflux_dict[pdg][0])
 		self._mc_neuflavor = nu_mc["pdg"].apply(lambda pdg: nuflux_dict[pdg][1])
 		self._mc_weights = nu_mc["weight"]
 		self._mc_current = nu_mc["current_type"]
 		self._mc_morphology = nu_mc["pid"]
 		# experiment constants
 		self._livetime = 1.39
 		self._unit_norm = 1e4
 		# flux related settings
 		self._atm_initial_flux = None
 		self._flux_emin = 1 * units.GeV
 		self._flux_emax = 1e4 * units.GeV
 		self._flux_enodes = 100
 		self._flux_cthmin = -1.0
 		self._flux_cthmax = 1.0
 		self._flux_cnodes = 80
 		self._flux_energy_nodes = None
 		self._flux_cth_nodes = None
 		# energy and zenith binning
 		self._E_true_bins = np.load("../../datafiles/Analysis_Release_ORCA433kton-years/_E_true_bins.npy")
 		self._E_reco_bins = np.load("../../datafiles/Analysis_Release_ORCA433kton-years/_E_reco_bins.npy")
 		self._cosT_true_bins = np.load("../../datafiles/Analysis_Release_ORCA433kton-years/_cosT_true_bins.npy")
 		self._cosT_reco_bins = np.linspace(-1, 0, 11)
 		self._E_true_centers = (self._E_true_bins[1:] - self._E_true_bins[:-1]) / np.log(self._E_true_bins[1:] / self._E_true_bins[:-1])
 		self._cth_bin_centers = (self._cosT_true_bins[:-1] + self._cosT_true_bins[1:]) / 2
 		# flavor, nu type and other bining
 		# self._flavor_bins = np.array([12, 14, 16])  # νe, νμ, ντ
 		# self._nu_type_bins = np.array([-1, 1])  # Neutrino (1), Antineutrino (-1)
 		# self._interaction_bins = np.array([0, 1])  # CC (1), NC (0)
 		# self._morphology_bins = np.array([0, 1, 2])  # 3 Morphology categories
 		print(f"Finished setting up experiment {experiment}")
 
 	# set up the atmospheric initial flux object
 	def SetInitialFlux(self):
 		self._flux_energy_nodes = nsq.logspace(self._flux_emin, self._flux_emax, self._flux_enodes)
 		self._flux_cth_nodes = nsq.linspace(self._flux_cthmin, self._flux_cthmax, self._flux_cnodes)
 		nsq_atm = nsq.nuSQUIDSAtm(self._flux_cth_nodes,self._flux_energy_nodes,neutrino_flavors,nsq.NeutrinoType.both,interactions)
 		AtmInitialFlux = np.zeros((len(self._flux_cth_nodes),len(self._flux_energy_nodes),2,neutrino_flavors))
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
 		nsq_atm = nsq.nuSQUIDSAtm(self._flux_cth_nodes,self._flux_energy_nodes,neutrino_flavors,nsq.NeutrinoType.both,interactions)
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
 		for i in range(len(rate)):
 			rate[i] = nsq_atm.EvalFlavor(int(self._mc_neuflavor[i]), float(self._mc_cthtrue[i]), float(self._mc_etrue[i] * units.GeV), int(self._mc_nutype[i]))
 		# rate = list(map(nsq_atm.EvalFlavor, self._mc_neuflavor, self._mc_cthtrue, self._mc_etrue*units.GeV, self._mc_nutype, repeat(True)))
 		return rate
 	
 	# obtain the mc event unweighted rate for all events given oscillation sterile parameters (phi * prob)
 	def GetOscillatedSterileRate(self, neutrino_flavors, t12, t13, t23, dm21, dm31, dcp, t14, t24, t34, dm41, d24, Ordering='normal'):
 		nsq_atm = nsq.nuSQUIDSAtm(self._flux_cth_nodes,self._flux_energy_nodes,neutrino_flavors,nsq.NeutrinoType.both,interactions)
 		nsq_atm.Set_rel_error(1.0e-4)
 		nsq_atm.Set_abs_error(1.0e-4)
 		nsq_atm.Set_MixingAngle(0, 1, asin(sqrt(t12)))
 		nsq_atm.Set_MixingAngle(0, 2, asin(sqrt(t13)))
 		nsq_atm.Set_MixingAngle(1, 2, asin(sqrt(t23)))
 		nsq_atm.Set_MixingAngle(0, 3, asin(sqrt(t14)))
 		nsq_atm.Set_MixingAngle(1, 3, asin(sqrt(t24)))
 		nsq_atm.Set_MixingAngle(2, 3, asin(sqrt(t34)))			
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
 		for i in range(len(rate)):
 			rate[i] = nsq_atm.EvalFlavor(int(self._mc_neuflavor[i]), float(self._mc_cthtrue[i]), float(self._mc_etrue[i] * units.GeV), int(self._mc_nutype[i]))
 		# rate = list(map(nsq_atm.EvalFlavor, self._mc_neuflavor, self._mc_cthtrue, self._mc_etrue*units.GeV, self._mc_nutype, repeat(True)))
 		return rate
 		
 	# given the unweighted rates, multiply by weights and bin them 
 	def BinWeightedRate(self, unweighted_rate, E_shift = 1):
 		assert(len(unweighted_rate) == len(self._mc_weights))
 		weighted_rate = unweighted_rate * self._mc_weights * self._livetime * self._unit_norm
 		# Create an empty 8D histogram for neutrinos
 		shape = (
 			len(self._E_true_bins) - 1, len(self._E_reco_bins) - 1,
 			len(self._cosT_true_bins) - 1, len(self._cosT_reco_bins) - 1,
 			2, 3, 2, 3
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