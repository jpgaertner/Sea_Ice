
import numpy as np

from seaice_size import *
from seaice_params import *

from gendata import uWindField, vWindField, uVel, vVel, hIce_init

from seaice_dynsolver import dynsolver
from seaice_advdiff import advdiff
from seaice_growth import growth


### input
hIceMean = 1
hSnowMean = 1
Area = 1
