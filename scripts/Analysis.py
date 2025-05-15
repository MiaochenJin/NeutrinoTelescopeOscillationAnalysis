import numpy as np
import pandas as pd
from Simulation import Simulation
from utils import *
from Systematics import *
import click, yaml

class Analysis:
    def __init__(self, experiment, livetime, filename, config = "../config/config.yaml"):
        # read config
        full_config = yaml.safe_load(open(config))
        # set up experiment
        self.sim = Simulation(experiment, livetime, filename)
        # set up systematics
        self.syst = SYST_CLASSES # this can be modified later to be input
        self.systNames = [cls.name for cls in self.syst]
        self.systRegistry = {cls.name: (cls().apply, cls().diff) for cls in self.syst}
        # set up best fit (model) values
        self.bf = full_config["BestFit"]
        # set up the simulation
        self.sim.SetInitialFlux()
        self.sim.ComputeBFRates(np.arcsin(np.sqrt(self.bf['s2t12'])),\
                                np.arcsin(np.sqrt(self.bf['s2t13'])),\
                                np.arcsin(np.sqrt(self.bf['s2t23'])),\
                                self.bf['m21'], self.bf['m31'], \
                                self.bf['dCP'] * np.pi)
    