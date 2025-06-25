import numpy as np
import pandas as pd
from Simulation import Simulation
from utils import *
from Systematics import *
from ChiSq import *
import click, yaml
from scipy.optimize import minimize
import os

class Analysis:
    def __init__(self, experiment, livetime, filename, config = "../config/config.yaml"):
        # read config
        config_path = config
        full_config = yaml.safe_load(open(config_path, 'r'))
        config_dir = os.path.dirname(os.path.abspath(config_path))

        mode = full_config["Oscillation"]
        self.binning = full_config["Binning"]
        self.data_fitting_settings = full_config.get("DataFitting", {'do_data_fitting': False})
        self.do_data_fitting = self.data_fitting_settings['do_data_fitting']
        # set up experiment
        self.sim = Simulation(experiment, livetime, filename, mode = mode)
        self.sim._analysis_binning = self.binning
        syst_settings = {}
        syst_settings.update(full_config.get('FluxSyst', {}))
        syst_settings.update(full_config.get('DetSyst', {}))
        syst_settings.update(full_config.get('XsecSyst', {}))
        
        FLUX_SYST = [
            FluxNormalization_Below1GeV, FluxNormalization_Above1GeV,
            FluxTilt, NuNubarRatio, FlavorRatio,
            ZenithFluxUp, ZenithFluxDown
        ]
        DET_SYST = [
            IceAbsorption, IceScattering, OffSet,
            OptEffHeadon, OptEffLateral, OptEffOverall,
            CoinFraction
        ]
        XSEC_SYST = [XSecNuTau, NCoverCC, AxialMass, NCHad, 
            DIS, CCQE, CCQENuBarNu, CCQEMuE]

        all_systs = FLUX_SYST + DET_SYST + XSEC_SYST
        
        self.syst = [
            cls for cls in all_systs
            if syst_settings.get(cls.name, 1) == 1
        ]
        self.systNames = [cls.name for cls in self.syst]
        self.systRegistry = {cls.name: (cls().apply, cls().diff) for cls in self.syst}
        self.systNominal = [cls.nominal for cls in self.syst]
        self.systSigma = [cls.sigma for cls in self.syst]
        # set up best fit (model) values
        self.bf = full_config["BestFit"]
        # set up the simulation
        self.sim.SetInitialFlux()

        if self.do_data_fitting:
            # In data fitting mode, load N_dat from files and do not pre-compute a model.
            # The model (N_mod) will be calculated for each grid point in the run script.
            data_dir_relative = self.data_fitting_settings['data_dir']
            # Construct absolute path to data_dir relative to the config file's location
            data_dir_absolute = os.path.join(config_dir, data_dir_relative)
            self.N_dat_fixed = LoadBinnedData(data_dir_absolute, file_pattern="counts_*.csv")
        else:
            # In sensitivity mode, compute the best-fit rates as the model (N_mod).
            # N_dat will be calculated for each grid point in the run script.
            if mode == "Standard":
                self.sim.ComputeBFRates(np.arcsin(np.sqrt(self.bf['s2t12'])),\
                                        np.arcsin(np.sqrt(self.bf['s2t13'])),\
                                        np.arcsin(np.sqrt(self.bf['s2t23'])),\
                                        self.bf['m21'], self.bf['m31'], \
                                        self.bf['dCP'] * np.pi)
            elif mode == "Sterile":
                self.sim.ComputeBFRatesSterile(np.arcsin(np.sqrt(self.bf['s2t12'])),\
                                        np.arcsin(np.sqrt(self.bf['s2t13'])),\
                                        np.arcsin(np.sqrt(self.bf['s2t23'])),\
                                        np.arcsin(np.sqrt(self.bf['s2t14'])),\
                                        np.arcsin(np.sqrt(self.bf['s2t24'])),\
                                        np.arcsin(np.sqrt(self.bf['s2t34'])),\
                                        self.bf['m21'], self.bf['m31'],self.bf['m41'], \
                                        self.bf['dCP'] * np.pi, self.bf['dCP24'] * np.pi)
            if self.binning == "3DEZ": self.sim.BinBFRatesEnergyZenith()
            elif self.binning == "LoE": self.sim.BinBFRatesLoE()
            elif self.binning == "2DEZ": self.sim.BinBFRates2DEZ()

        if self.sim._experiment == "IC":
            self.sim.SetDetSyst()
    
    def Compute_N_dat(self, **osc):
        t12 = np.arcsin(np.sqrt(osc['sin2t12']))
        t13 = np.arcsin(np.sqrt(osc['sin2t13']))
        t23 = np.arcsin(np.sqrt(osc['sin2t23']))
        # 1) get unweighted model rates at this point
        unweighted = self.sim.GetOscillatedRate(t12, t13, t23,
                                                osc['dm21'], osc['dm31'],
                                                osc['dcp'])
        return self.sim.BinEvents(unweighted)

    def FitSystematics(
            self,
            nominal_syst: np.ndarray,
            N_dat: np.ndarray,
            tol: float,
            stat_chisq_fn,       # e.g. ChiSq_only_no_prior
            full_chisq_fn,       # e.g. ChiSq_Jac_with_penalty or any other χ²
            method: str = 'L-BFGS-B',
            chisq_kwargs: dict = None
        ):
            if chisq_kwargs is None:
                chisq_kwargs = {}

            # 1) compute stat-only χ² to scale tolerance
            # Only pass arguments that stat_chisq_fn expects.
            stat_chisq_args = chisq_kwargs.copy()
            if 'N_mod_hypo_err' in stat_chisq_args:
                del stat_chisq_args['N_mod_hypo_err']
            statOnly = stat_chisq_fn(self, nominal_syst, N_dat, **stat_chisq_args)
            
            startChisq = full_chisq_fn(self, nominal_syst, N_dat, **chisq_kwargs)
            # 2) wrap the full χ²
            def obj(syst):
                return full_chisq_fn(self, syst, N_dat, **chisq_kwargs)
            # 3) get analytic prior and bounds
            # The prior calculation only depends on the base model, not the systematics themselves
            prior_kwargs = {'N_mod_hypo': chisq_kwargs.get('N_mod_hypo')}
            self.SystPrior, bounds = syst_penalty_prior(self, nominal_syst, N_dat, **prior_kwargs)
            # 4) adjust tolerance
            tol_adj = max(tol, np.sqrt(statOnly) * tol)
            # 5) detect whether obj returns a gradient
            test_out = obj(self.SystPrior)
            use_jac = isinstance(test_out, tuple) and len(test_out) == 2
            # 6) run the minimizer
            res = minimize(
                obj,
                self.SystPrior,
                method=method,
                jac=use_jac,
                bounds=bounds,
                options={'ftol': tol_adj}
            )
            return res