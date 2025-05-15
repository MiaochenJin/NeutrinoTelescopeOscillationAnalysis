# registers systematics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Simulation import Simulation
from utils import *

class Systematics:
    name = None
    def apply(self, x, sim): return x
    def diff(self, x): return x

class FluxNormalization(Systematics):
    name = 'FluxNormalization'
    def apply(self, x, sim): 
        return x - 1
    def diff(self, x, sim): return 1

class FluxNormalization_Above1GeV(Systematics):
    name = "FluxNormalization_Above1GeV"
    def apply(self, x, sim): 
        nev = np.ones(sim._num_entries)
        nev[sim._mc_etrue>1] = x
        return sim.BinWeightedRate3DFlatten(nev * sim._BF_rates) / sim._BF_rates_weighted_binned - 1
    def diff(self, x, sim): 
        nev = np.ones(sim._num_entries)
        nev[sim._mc_etrue>1] = 1
        return sim.BinWeightedRate3DFlatten(nev * sim._BF_rates) / sim._BF_rates_weighted_binned

class FluxNormalization_Below1GeV(Systematics):
    name = "FluxNormalization_Below1GeV"
    def apply(self, x, sim): 
        nev = np.ones(sim._num_entries)
        nev[sim._mc_etrue<1] = x
        return sim.BinWeightedRate3DFlatten(nev * sim._BF_rates) / sim._BF_rates_weighted_binned - 1
    def diff(self, x, sim): 
        nev = np.ones(sim._num_entries)
        nev[sim._mc_etrue<1] = 1
        return sim.BinWeightedRate3DFlatten(nev * sim._BF_rates) / sim._BF_rates_weighted_binned

class FluxTilt(Systematics):
    name = "FluxTilt"
    def apply(self, x, sim): 
        E0Gam = 10 # GeV
        nev = (sim._mc_etrue / E0Gam) ** x
        return sim.BinWeightedRate3DFlatten(nev * sim._BF_rates) / sim._BF_rates_weighted_binned - 1
    def diff(self, x, sim): 
        E0Gam = 10 # GeV
        nev = (sim._mc_etrue / E0Gam)**x * np.log(sim._mc_etrue / E0Gam)
        return sim.BinWeightedRate3DFlatten(nev * sim._BF_rates) / sim._BF_rates_weighted_binned

class NuNubarRatio(Systematics):
    name = "NuNubarRatio"
    def apply(self, x, sim): 
        nnbar = np.ones(sim._num_entries)
        nnbar[sim._mc_nutype<0] = x
        return sim.BinWeightedRate3DFlatten(nnbar * sim._BF_rates) / sim._BF_rates_weighted_binned - 1
    def diff(self, x, sim): 
        nnbar = np.ones(sim._num_entries)
        nnbar[sim._mc_nutype<0] = 1
        return sim.BinWeightedRate3DFlatten(nnbar * sim._BF_rates) / sim._BF_rates_weighted_binned

class FlavorRatio(Systematics):
    name = "FlavorRatio"
    def apply(self, x, sim): 
        eovermu = np.ones(sim._num_entries)
        eovermu[np.abs(sim._mc_neuflavor)==12] = x
        return sim.BinWeightedRate3DFlatten(eovermu * sim._BF_rates) / sim._BF_rates_weighted_binned - 1
    def diff(self, x, sim): 
        eovermu = np.ones(sim._num_entries)
        eovermu[np.abs(sim._mc_neuflavor)==12] = 1
        return sim.BinWeightedRate3DFlatten(eovermu * sim._BF_rates) / sim._BF_rates_weighted_binned

class ZenithFluxUp(Systematics):
    name = "ZenithFluxUp"
    def apply(self, x, sim): 
        zenith = np.ones(sim._num_entries)
        zenith[sim._mc_cthtrue>=0] = zenith[sim._mc_cthtrue>=0] - x * np.tanh(sim._mc_cthtrue[sim._mc_cthtrue>=0])**2
        return sim.BinWeightedRate3DFlatten(zenith * sim._BF_rates) / sim._BF_rates_weighted_binned - 1
    def diff(self, x, sim): 
        zenith = np.zeros(sim._num_entries)
        zenith[sim._mc_cthtrue>=0] = - np.tanh(sim._mc_cthtrue[sim._mc_cthtrue>=0])**2
        return sim.BinWeightedRate3DFlatten(zenith * sim._BF_rates) / sim._BF_rates_weighted_binned
    
class ZenithFluxDown(Systematics):
    name = "ZenithFluxDown"
    def apply(self, x, sim): 
        zenith = np.ones(sim._num_entries)
        zenith[sim._mc_cthtrue<0] = zenith[sim._mc_cthtrue<0] - x * np.tanh(sim._mc_cthtrue[sim._mc_cthtrue<0])**2
        return sim.BinWeightedRate3DFlatten(zenith * sim._BF_rates) / sim._BF_rates_weighted_binned - 1
    def diff(self, x, sim): 
        zenith = np.zeros(sim._num_entries)
        zenith[sim._mc_cthtrue<0] = - np.tanh(sim._mc_cthtrue[sim._mc_cthtrue<0])**2
        return sim.BinWeightedRate3DFlatten(zenith * sim._BF_rates) / sim._BF_rates_weighted_binned

SYST_CLASSES = [FluxNormalization, FluxNormalization_Above1GeV, FluxNormalization_Below1GeV, FluxTilt, NuNubarRatio, \
                FlavorRatio, ZenithFluxUp, ZenithFluxDown]
SYST_NAMES = [cls.name for cls in SYST_CLASSES]
SYST_REGISTRY = {cls.name: (cls().apply, cls().diff)
                 for cls in SYST_CLASSES}