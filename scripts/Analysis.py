import numpy as np
import pandas as pd
from Simulation import Simulation
from utils import *
from Systematics import *
import click, yaml

class Analysis:
    def __init__(self, experiment, livetime, filename, config = "../config/config.yaml"):
        # read config
        full_config = yaml.safe_load(open(config, 'r'))
        # set up experiment
        self.sim = Simulation(experiment, livetime, filename)
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
        self.sim.ComputeBFRates(np.arcsin(np.sqrt(self.bf['s2t12'])),\
                                np.arcsin(np.sqrt(self.bf['s2t13'])),\
                                np.arcsin(np.sqrt(self.bf['s2t23'])),\
                                self.bf['m21'], self.bf['m31'], \
                                self.bf['dCP'] * np.pi)
        self.sim.SetDetSyst()
    