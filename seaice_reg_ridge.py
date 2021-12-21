
import numpy as np

from seaice_params import *
from seaice_size import *

# cleaning up after advection
# and driver for ice ridging

d_hIceMeanByNeg = np.zeros((sNx+2*OLx,sNy+2*OLy))
d_hSnowMeanByNeg = np.zeros((sNx+2*OLx,sNy+2*OLy))

# ifdef EXF_SEAICE_FRACTION
d_AreaByRLX = np.zeros((sNx+2*OLx,sNy+2*OLy))
d_hIceMeanByRLX = np.zeros((sNx+2*OLx,sNy+2*OLy))
# endif

# ifdef SEAICE_VARIABLE_SALINITY = False?


##### treat pathological cases #####

#ifdef EXF_SEAICE_FRACTION
