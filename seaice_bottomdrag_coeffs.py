from itertools import combinations_with_replacement
import numpy as np

from seaice_params import *
from seaice_size import *

R_low = 1 #in grid.h: base of fluid in r_unit (Depth(m) / Pressure(Pa) at top Atmos.)


### input:
# uIce: zonal ice velocity
# vIce: meridional ice velocity
# hIceMean: mean ice thickness
# Area: ice cover fraction

### output:
# cBot: non-linear drag coefficient for ice-bottom drag


def bottomdrag_coeffs(uIce, vIce, hIceMean, Area):

    u0sq = basalDragU0**2
    recip_k1 = 1 / basalDragK1

    fac = 10 #scales the soft maximum for more accuracy
    recip_fac = 1 / fac

    tmp = 0.25 * ((uIce[:-1,:-1] * maskInW[:-1,:-1] + uIce[:-1,1:] * maskInW[:-1,1:])**2 + (vIce[:-1,:-1] * maskInS[:-1,:-1] + vIce[1:,:-1] * maskInS[1:,:-1])**2)
    tmpFld = basalDragK2 / np.sqrt(tmp + u0sq)

    isArea = np.where(Area[:-1,:-1] > 0.01)

    hCrit = np.abs(R_low[:-1,:-1][isArea]) * Area[:-1,:-1][isArea] * recip_k1

    # soft maximum
    # isnt cBot set to 0 at the beginning?
    cBot = np.zeros((sNy+2*OLy,sNx+2*OLx))
    cBot[:-1,:-1][isArea] = cBot[:-1,:-1][isArea] + tmpFld[isArea] * np.log(np.exp(fac * (hIceMean[:-1,:-1][isArea] - hCrit)) + 1) * recip_fac * np.exp(-cBasalStar * (1 - Area[:-1,:-1][isArea]))  * hIceMeanMask[:-1,:-1][isArea]


    return cBot