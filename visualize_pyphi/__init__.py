from . import visualize_ces as viz
from . import simulate_ces as sim
from . import special_plots as plots
from . import network_generator as ng
from . import compute


import pyphi

pyphi.config.REPERTOIRE_DISTANCE = "AID"
pyphi.config.REPR_VERBOSITY = 1
pyphi.config.PARTITION_TYPE = 'TRI'
