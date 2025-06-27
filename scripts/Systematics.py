# registers systematics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Simulation import Simulation
from utils import *

def _get_base_rates(sim):
    """
    Helper function to get the correct base event rates for systematics calculations.
    In data-fitting mode, it uses the rates from the current model hypothesis.
    In sensitivity mode, it falls back to the pre-calculated best-fit rates.
    """
    base_rates = getattr(sim, '_hypothesis_rates', sim._BF_rates)
    if base_rates is None:
        raise ValueError("Base rates for systematics calculation not found in simulation object. "
                         "Ensure that either BF rates are computed or hypothesis_rates are set.")
    return base_rates

class Systematics:
    name = None
    nominal = 0.
    sigma = 0.
    def apply(self, x, sim): return x
    def diff(self, x): return x

class FluxNormalization(Systematics):
    name = 'FluxNormalization'
    nominal = 1.
    sigma = .25
    def apply(self, x, sim): 
        base_rates = _get_base_rates(sim)
        return x - 1
    def diff(self, x, sim): 
        base_rates = _get_base_rates(sim)
        return 1

class FluxNormalization_Above1GeV(Systematics):
    name = "FluxNormalization_Above1GeV"
    nominal = 1.
    sigma = .15
    def apply(self, x, sim): 
        base_rates = _get_base_rates(sim)
        nev = np.ones(sim._num_entries)
        nev[sim._mc_etrue>1] = x
        return sim.BinEvents(nev * base_rates) / sim.BinEvents(base_rates) - 1
    def diff(self, x, sim): 
        base_rates = _get_base_rates(sim)
        nev = np.zeros(sim._num_entries)
        nev[sim._mc_etrue>1] = 1
        return sim.BinEvents(nev * base_rates) / sim.BinEvents(base_rates)

class FluxNormalization_Below1GeV(Systematics):
    name = "FluxNormalization_Below1GeV"
    nominal = 1.
    sigma = .25
    def apply(self, x, sim): 
        base_rates = _get_base_rates(sim)
        nev = np.ones(sim._num_entries)
        nev[sim._mc_etrue<1] = x
        return sim.BinEvents(nev * base_rates) / sim.BinEvents(base_rates) - 1
    def diff(self, x, sim): 
        base_rates = _get_base_rates(sim)
        nev = np.zeros(sim._num_entries)
        nev[sim._mc_etrue<1] = 1
        return sim.BinEvents(nev * base_rates) / sim.BinEvents(base_rates)

class FluxTilt(Systematics):
    name = "FluxTilt"
    nominal = 0.
    sigma = .2
    def apply(self, x, sim): 
        base_rates = _get_base_rates(sim)
        E0Gam = 10 # GeV
        nev = (sim._mc_etrue / E0Gam) ** x
        return sim.BinEvents(nev * base_rates) / sim.BinEvents(base_rates) - 1
    def diff(self, x, sim): 
        base_rates = _get_base_rates(sim)
        E0Gam = 10 # GeV
        nev = (sim._mc_etrue / E0Gam)**x * np.log(sim._mc_etrue / E0Gam)
        return sim.BinEvents(nev * base_rates) / sim.BinEvents(base_rates)

class NuNubarRatio(Systematics):
    name = "NuNubarRatio"
    nominal = 1.
    sigma = .02
    def apply(self, x, sim): 
        base_rates = _get_base_rates(sim)
        nnbar = np.ones(sim._num_entries)
        nnbar[sim._mc_nutype == 1] = x
        return sim.BinEvents(nnbar * base_rates) / sim.BinEvents(base_rates) - 1
    def diff(self, x, sim): 
        base_rates = _get_base_rates(sim)
        nnbar = np.zeros(sim._num_entries)
        nnbar[sim._mc_nutype == 1] = 1
        return sim.BinEvents(nnbar * base_rates) / sim.BinEvents(base_rates)

class FlavorRatio(Systematics):
    name = "FlavorRatio"
    nominal = 1.
    sigma = .05
    def apply(self, x, sim): 
        base_rates = _get_base_rates(sim)
        eovermu = np.ones(sim._num_entries)
        eovermu[np.abs(sim._mc_neuflavor)==0] = x
        return sim.BinEvents(eovermu * base_rates) / sim.BinEvents(base_rates) - 1
    def diff(self, x, sim): 
        base_rates = _get_base_rates(sim)
        eovermu = np.zeros(sim._num_entries)
        eovermu[np.abs(sim._mc_neuflavor)==0] = 1
        return sim.BinEvents(eovermu * base_rates) / sim.BinEvents(base_rates)

class ZenithFluxUp(Systematics):
    name = "ZenithFluxUp"
    nominal = 0.
    sigma = .2
    def apply(self, x, sim): 
        base_rates = _get_base_rates(sim)
        zenith = np.ones(sim._num_entries)
        zenith[sim._mc_cthtrue>=0] = zenith[sim._mc_cthtrue>=0] - x * np.tanh(sim._mc_cthtrue[sim._mc_cthtrue>=0])**2
        return sim.BinEvents(zenith * base_rates) / sim.BinEvents(base_rates) - 1
    def diff(self, x, sim): 
        base_rates = _get_base_rates(sim)
        zenith = np.zeros(sim._num_entries)
        zenith[sim._mc_cthtrue>=0] = - np.tanh(sim._mc_cthtrue[sim._mc_cthtrue>=0])**2
        return sim.BinEvents(zenith * base_rates) / sim.BinEvents(base_rates)
    
class ZenithFluxDown(Systematics):
    name = "ZenithFluxDown"
    nominal = 0.
    sigma = .2
    def apply(self, x, sim): 
        base_rates = _get_base_rates(sim)
        zenith = np.ones(sim._num_entries)
        zenith[sim._mc_cthtrue<0] = zenith[sim._mc_cthtrue<0] - x * np.tanh(sim._mc_cthtrue[sim._mc_cthtrue<0])**2
        return sim.BinEvents(zenith * base_rates) / sim.BinEvents(base_rates) - 1
    def diff(self, x, sim): 
        base_rates = _get_base_rates(sim)
        zenith = np.zeros(sim._num_entries)
        zenith[sim._mc_cthtrue<0] = - np.tanh(sim._mc_cthtrue[sim._mc_cthtrue<0])**2
        return sim.BinEvents(zenith * base_rates) / sim.BinEvents(base_rates)

# cross section systematics
class XSecNuTau(Systematics):
    name = "XSecNuTau"
    nominal = 1.
    sigma = .25
    def apply(self, x, sim):
        if x == 1: return 0 
        base_rates = _get_base_rates(sim)
        tau = np.ones(sim._num_entries)
        tau[np.abs(sim._mc_neuflavor)==2] = x
        return sim.BinWeightedRate3DFlatten(tau * base_rates) / sim.BinEvents(base_rates) - 1
    def diff(self, x, sim):
        base_rates = _get_base_rates(sim)
        tau = np.zeros(sim._num_entries)
        tau[np.abs(sim._mc_neuflavor)==2] = 1
        return sim.BinWeightedRate3DFlatten(tau * base_rates) / sim.BinEvents(base_rates)

class NCoverCC(Systematics):
    name = "NCoverCC"
    nominal = 1.
    sigma = .2
    def apply(self, x, sim):
        if x == 1: return 0 
        base_rates = _get_base_rates(sim)
        nc = np.ones(sim._num_entries)
        nc[sim._mc_current==0] = x 
        return sim.BinWeightedRate3DFlatten(nc * base_rates) / sim.BinEvents(base_rates) - 1
    def diff(self, x, sim):
        base_rates = _get_base_rates(sim)
        nc = np.zeros(sim._num_entries)
        nc[sim._mc_current==0] = 1 
        return sim.BinWeightedRate3DFlatten(nc * base_rates) / sim.BinEvents(base_rates)

class AxialMass(Systematics):
    name = "AxialMass"
    nominal = 1.
    sigma = .1
    def apply(self, x, sim):
        if x == 1: return 0 
        base_rates = _get_base_rates(sim)
        cc = np.ones(sim._num_entries)
        cc[sim._mc_current==1] = 1+0.042*(x-1)*1.05*np.log10(sim._mc_etrue[sim._mc_current==1]) 
        return sim.BinWeightedRate3DFlatten(cc * base_rates) / sim.BinEvents(base_rates) - 1
    def diff(self, x, sim):
        base_rates = _get_base_rates(sim)
        cc = np.zeros(sim._num_entries)
        cc[sim._mc_current==1] = 0.042*1.05*np.log10(sim._mc_etrue[sim._mc_current==1]) 
        return sim.BinWeightedRate3DFlatten(cc * base_rates) / sim.BinEvents(base_rates)

class NCHad(Systematics):
    name = "NCHad"
    nominal = 1.
    sigma = .1
    def apply(self, x, sim):
        if x == 1: return 0 
        base_rates = _get_base_rates(sim)
        nc = np.ones(sim._num_entries)
        nc[sim._mc_current==0] = x 
        return sim.BinWeightedRate3DFlatten(nc * base_rates) / sim.BinEvents(base_rates) - 1
    def diff(self, x, sim):
        base_rates = _get_base_rates(sim)
        nc = np.zeros(sim._num_entries)
        nc[sim._mc_current==0] = 1 
        return sim.BinWeightedRate3DFlatten(nc * base_rates) / sim.BinEvents(base_rates)

class DIS(Systematics):
    name = "DIS"
    nominal = 1.
    sigma = .05
    def apply(self, x, sim):
        if x == 1: return 0 
        base_rates = _get_base_rates(sim)
        dis = np.ones(sim._num_entries)
        dis[(sim._mc_interaction in [0, 3]) & (sim._mc_current == 1)] = x 
        return sim.BinWeightedRate3DFlatten(dis * base_rates) / sim.BinEvents(base_rates) - 1
    def diff(self, x, sim):
        base_rates = _get_base_rates(sim)
        dis = np.zeros(sim._num_entries)
        dis[((sim._mc_interaction == 0) | ( sim._mc_interaction == 3)) & (sim._mc_current == 1)] = 1
        return sim.BinWeightedRate3DFlatten(dis * base_rates) / sim.BinEvents(base_rates)

class CCQE(Systematics):
    name = "CCQE"
    nominal = 1.
    sigma = .1
    def apply(self, x, sim):
        if x == 1: return 0 
        base_rates = _get_base_rates(sim)
        ccqe = np.ones(sim._num_entries)
        cond = (sim._mc_interaction==1) & (sim._mc_nutype == 0)
        ccqe[cond] = x 
        return sim.BinWeightedRate3DFlatten(ccqe * base_rates) / sim.BinEvents(base_rates) - 1
    def diff(self, x, sim):
        base_rates = _get_base_rates(sim)
        ccqe = np.zeros(sim._num_entries)
        cond = (sim._mc_interaction==1) & (sim._mc_nutype == 0)
        ccqe[cond] = 1
        return sim.BinWeightedRate3DFlatten(ccqe * base_rates) / sim.BinEvents(base_rates)

class CCQENuBarNu(Systematics):
    name = "CCQENuBarNu"
    nominal = 1.
    sigma = .1
    def apply(self, x, sim):
        if x == 1: return 0 
        base_rates = _get_base_rates(sim)
        ccqe = np.ones(sim._num_entries)
        cond = (sim._mc_interaction==1) & (sim._mc_nutype == 1)
        ccqe[cond] = x 
        return sim.BinWeightedRate3DFlatten(ccqe * base_rates) / sim.BinEvents(base_rates) - 1
    def diff(self, x, sim):
        base_rates = _get_base_rates(sim)
        ccqe = np.zeros(sim._num_entries)
        cond = (sim._mc_interaction==1) & (sim._mc_nutype == 1)
        ccqe[cond] = 1
        return sim.BinWeightedRate3DFlatten(ccqe * base_rates) / sim.BinEvents(base_rates)

class CCQEMuE(Systematics):
    name = "CCQEMuE"
    nominal = 1.
    sigma = .1
    def apply(self, x, sim):
        if x == 1: return 0 
        base_rates = _get_base_rates(sim)
        ccqe = np.ones(sim._num_entries)
        cond = (sim._mc_interaction==1) & (sim._mc_neuflavor == 1)
        ccqe[cond] = x 
        return sim.BinWeightedRate3DFlatten(ccqe * base_rates) / sim.BinEvents(base_rates) - 1
    def diff(self, x, sim):
        base_rates = _get_base_rates(sim)
        ccqe = np.zeros(sim._num_entries)
        cond = (sim._mc_interaction==1) & (sim._mc_neuflavor == 1)
        ccqe[cond] = 1
        return sim.BinWeightedRate3DFlatten(ccqe * base_rates) / sim.BinEvents(base_rates)

# Detector systematics
class IceAbsorption(Systematics):
    name = "IceAbsorption"
    nominal = 1.
    sigma = .1
    def apply(self, x, sim):
        xx = x - 1
        base_rates = _get_base_rates(sim)
        d = sim.ExpFracNuECC * sim.ice_absorption['nueCC'] + \
        sim.ExpFracNuMuCC * sim.ice_absorption['numuCC'] + \
        sim.ExpFracNuTauCC * sim.ice_absorption['nutauCC'] + \
        sim.ExpFracNC * sim.ice_absorption['NC']
        return xx * d
    def diff(self, x, sim):
        base_rates = _get_base_rates(sim)
        d = sim.ExpFracNuECC * sim.ice_absorption['nueCC'] + \
        sim.ExpFracNuMuCC * sim.ice_absorption['numuCC'] + \
        sim.ExpFracNuTauCC * sim.ice_absorption['nutauCC'] + \
        sim.ExpFracNC * sim.ice_absorption['NC']
        return d

class IceScattering(Systematics):
    name = "IceScattering"
    nominal = 1.
    sigma = .1
    def apply(self, x, sim):
        xx = x - 1
        base_rates = _get_base_rates(sim)
        d = sim.ExpFracNuECC * sim.ice_scattering['nueCC'] + \
        sim.ExpFracNuMuCC * sim.ice_scattering['numuCC'] + \
        sim.ExpFracNuTauCC * sim.ice_scattering['nutauCC'] + \
        sim.ExpFracNC * sim.ice_scattering['NC']
        return xx * d
    def diff(self, x, sim):
        base_rates = _get_base_rates(sim)
        d = sim.ExpFracNuECC * sim.ice_scattering['nueCC'] + \
        sim.ExpFracNuMuCC * sim.ice_scattering['numuCC'] + \
        sim.ExpFracNuTauCC * sim.ice_scattering['nutauCC'] + \
        sim.ExpFracNC * sim.ice_scattering['NC']
        return d

class OffSet(Systematics):
    name = "OffSet"
    nominal = 0.
    sigma = 10.
    def apply(self, x, sim):
        xx = x 
        base_rates = _get_base_rates(sim)
        d = sim.ExpFracNuECC * sim.offset['nueCC'] + \
        sim.ExpFracNuMuCC * sim.offset['numuCC'] + \
        sim.ExpFracNuTauCC * sim.offset['nutauCC'] + \
        sim.ExpFracNC * sim.offset['NC']
        return xx * d
    def diff(self, x, sim):
        d = sim.ExpFracNuECC * sim.offset['nueCC'] + \
        sim.ExpFracNuMuCC * sim.offset['numuCC'] + \
        sim.ExpFracNuTauCC * sim.offset['nutauCC'] + \
        sim.ExpFracNC * sim.offset['NC']
        return d

class OptEffHeadon(Systematics):
    name = "OptEffHeadon"
    nominal = 0.
    sigma = 1000.
    def apply(self, x, sim):
        xx = x 
        d = sim.ExpFracNuECC * sim.opt_eff_headon['nueCC'] + \
        sim.ExpFracNuMuCC * sim.opt_eff_headon['numuCC'] + \
        sim.ExpFracNuTauCC * sim.opt_eff_headon['nutauCC'] + \
        sim.ExpFracNC * sim.opt_eff_headon['NC']
        return xx * d
    def diff(self, x, sim):
        d = sim.ExpFracNuECC * sim.opt_eff_headon['nueCC'] + \
        sim.ExpFracNuMuCC * sim.opt_eff_headon['numuCC'] + \
        sim.ExpFracNuTauCC * sim.opt_eff_headon['nutauCC'] + \
        sim.ExpFracNC * sim.opt_eff_headon['NC']
        return d

class OptEffLateral(Systematics):
    name = "OptEffLateral"
    nominal = 25
    sigma = 10.
    def apply(self, x, sim):
        xx = x - 25
        d = sim.ExpFracNuECC * sim.opt_eff_lateral['nueCC'] + \
        sim.ExpFracNuMuCC * sim.opt_eff_lateral['numuCC'] + \
        sim.ExpFracNuTauCC * sim.opt_eff_lateral['nutauCC'] + \
        sim.ExpFracNC * sim.opt_eff_lateral['NC']
        return xx * d
    def diff(self, x, sim):
        d = sim.ExpFracNuECC * sim.opt_eff_lateral['nueCC'] + \
        sim.ExpFracNuMuCC * sim.opt_eff_lateral['numuCC'] + \
        sim.ExpFracNuTauCC * sim.opt_eff_lateral['nutauCC'] + \
        sim.ExpFracNC * sim.opt_eff_lateral['NC']
        return d

class OptEffOverall(Systematics):
    name = "OptEffOverall"
    nominal = 1.
    sigma = .1
    def apply(self, x, sim):
        xx = x - 1
        d = sim.ExpFracNuECC * sim.opt_eff_overall['nueCC'] + \
        sim.ExpFracNuMuCC * sim.opt_eff_overall['numuCC'] + \
        sim.ExpFracNuTauCC * sim.opt_eff_overall['nutauCC'] + \
        sim.ExpFracNC * sim.opt_eff_overall['NC']
        return xx * d
    def diff(self, x, sim):
        d = sim.ExpFracNuECC * sim.opt_eff_overall['nueCC'] + \
        sim.ExpFracNuMuCC * sim.opt_eff_overall['numuCC'] + \
        sim.ExpFracNuTauCC * sim.opt_eff_overall['nutauCC'] + \
        sim.ExpFracNC * sim.opt_eff_overall['NC']
        return d

class CoinFraction(Systematics):
    name = "CoinFraction"
    nominal = 0.
    sigma = .1
    def apply(self, x, sim):
        xx = x
        d = sim.ExpFracNuECC * sim.coin_fraction['nueCC'] + \
        sim.ExpFracNuMuCC * sim.coin_fraction['numuCC'] + \
        sim.ExpFracNuTauCC * sim.coin_fraction['nutauCC'] + \
        sim.ExpFracNC * sim.coin_fraction['NC']
        return xx * d
    def diff(self, x, sim):
        d = sim.ExpFracNuECC * sim.coin_fraction['nueCC'] + \
        sim.ExpFracNuMuCC * sim.coin_fraction['numuCC'] + \
        sim.ExpFracNuTauCC * sim.coin_fraction['nutauCC'] + \
        sim.ExpFracNC * sim.coin_fraction['NC']
        return d