import numpy as np
import pandas as pd
from Simulation import Simulation
from utils import *
from Systematics import *
from ChiSq import *
import click, yaml
from scipy.optimize import minimize
import os
import inspect

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
        self.use_analytic_priors = self.data_fitting_settings.get('use_analytic_priors', False)
        self.full_chisq_fn = None
        if full_config.get("FullChiSqFn") == "ChiSq_Jac_with_penalty":
            self.full_chisq_fn = ChiSq_Jac_with_penalty
        elif full_config.get("FullChiSqFn") == "ChiSq_with_penalty_with_error":
            self.full_chisq_fn = ChiSq_with_penalty_with_error
        else:
            raise ValueError(f"FullChiSqFn {full_config.get('FullChiSqFn')} not recognized")
        
        # Look up the chi-squared function by name from the ChiSq module
        try:
            self.stat_chisq_fn = globals()[full_config['StatChiSqFn']]
        except KeyError:
            raise ValueError(f"StatChiSqFn '{full_config.get('StatChiSqFn')}' not found in ChiSq module.")

        # set up experiment
        self.sim = Simulation(experiment, livetime, filename, mode = mode)
        self.sim._analysis_binning = self.binning
        if self.sim._experiment == "ORCA":
            self.sim.BinMuons()
        syst_settings = {}
        syst_settings.update(full_config.get('FluxSyst', {}))
        syst_settings.update(full_config.get('DetSyst', {}))
        syst_settings.update(full_config.get('XsecSyst', {}))
        
        FLUX_SYST = [
            FluxNormalization_Below1GeV, FluxNormalization_Above1GeV,
            FluxTilt, NuNubarRatio, FlavorRatio,
            ZenithFluxUp, ZenithFluxDown
        ]
        DET_SYST_IC = [
            IceAbsorption, IceScattering, OffSet,
            OptEffHeadon, OptEffLateral, OptEffOverall,
            CoinFraction
        ]
        DET_SYST_ORCA = [
            ORCA_f_all, ORCA_f_mu, ORCA_f_HPT, ORCA_f_Shower, ORCA_f_tauCC, ORCA_f_NC,
            ORCA_f_HE, ORCA_E_shift
        ]
        XSEC_SYST = [XSecNuTau, NCoverCC, AxialMass, NCHad,
            DIS, CCQE, CCQENuBarNu, CCQEMuE]
        DET_SYST = DET_SYST_IC + DET_SYST_ORCA
        all_systs = FLUX_SYST + DET_SYST + XSEC_SYST
        
        # Create instances of the selected systematics
        self.syst_objects = []
        for cls in all_systs:
            if syst_settings.get(cls.name, 0) == 1:
                self.syst_objects.append(cls())

        self.systNames = [s.name for s in self.syst_objects]
        self.systNominal = [s.nominal for s in self.syst_objects]
        self.systSigma = [s.sigma for s in self.syst_objects]
        # set up best fit (model) values
        self.bf = full_config["BestFit"]
        # set up the simulation
        self.sim.SetInitialFlux()

        if self.do_data_fitting:
            data_parquet_path_relative = self.data_fitting_settings.get('data_parquet_file')
            
            if data_parquet_path_relative:
                data_parquet_path_absolute = os.path.join(config_dir, data_parquet_path_relative)
                self.N_dat_fixed = self.sim.BinDataFromParquet(
                    data_parquet_path_absolute,
                    binning_type=self.binning
                )
            else:
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
    
    def Compute_N_dat_sterile(self, **osc):
        t12 = np.arcsin(np.sqrt(osc['sin2t12']))
        t13 = np.arcsin(np.sqrt(osc['sin2t13']))
        t23 = np.arcsin(np.sqrt(osc['sin2t23']))
        t14 = np.arcsin(np.sqrt(osc['sin2t14']))
        t24 = np.arcsin(np.sqrt(osc['sin2t24']))
        t34 = np.arcsin(np.sqrt(osc['sin2t34']))
        # 1) get unweighted model rates at this point
        unweighted = self.sim.GetOscillatedSterileRate(t12, t13, t23, t14, t24, t34,
                                                osc['dm21'], osc['dm31'], osc['dm41'],
                                                osc['dcp'], osc['dcp24'])
        return self.sim.BinEvents(unweighted)

    def FitSystematics(
            self,
            nominal_syst: np.ndarray,
            N_dat: np.ndarray,
            tol: float,
            method: str = 'L-BFGS-B',
            chisq_kwargs: dict = None
        ):
            if chisq_kwargs is None:
                chisq_kwargs = {}

            # Helper to filter kwargs for a specific function
            def get_valid_kwargs(func, all_kwargs):
                sig = inspect.signature(func)
                return {k: v for k, v in all_kwargs.items() if k in sig.parameters}

            # 1) compute stat-only χ² to scale tolerance
            stat_kwargs = get_valid_kwargs(self.stat_chisq_fn, chisq_kwargs)
            statOnly = self.stat_chisq_fn(self, nominal_syst, N_dat, **stat_kwargs)

            # Filter kwargs for the full chi-squared function
            full_chisq_kwargs = get_valid_kwargs(self.full_chisq_fn, chisq_kwargs)
            
            startChisq = self.full_chisq_fn(self, nominal_syst, N_dat, **full_chisq_kwargs)
            # 2) wrap the full χ²
            def obj(syst):
                return self.full_chisq_fn(self, syst, N_dat, **full_chisq_kwargs)
            # 3) get analytic prior and bounds
            # The prior calculation only depends on the base model, not the systematics themselves
            # 3) get analytic prior and bounds
            if self.use_analytic_priors:
                prior_kwargs = get_valid_kwargs(syst_penalty_prior, chisq_kwargs)
                self.SystPrior, bounds = syst_penalty_prior(self, nominal_syst, N_dat, **prior_kwargs)
            else:
                self.SystPrior = self.systNominal
                bounds = [(val - 5*sig, val + 5*sig) for val, sig in zip(self.systNominal, self.systSigma)]

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