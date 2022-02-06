import numpy as np

from seaice_params import *
from seaice_size import *

### input:
# uIce: zonal ice velocity
# vIce: meridional ice velocity
# hIceMean: mean ice thickness
# Area: ice cover fraction
# R_low: water depth

### output:
# cBot: non-linear drag coefficient for ice-bottom drag


def bottomdrag_coeffs(uIce, vIce, hIceMean, Area, R_low):

    u0sq = basalDragU0**2
    recip_k1 = 1 / basalDragK1

    fac = 10 #scales the soft maximum for more accuracy
    recip_fac = 1 / fac

    tmpFld = np.zeros((sNy+2*OLy,sNx+2*OLx))
    isArea = np.where(Area[:-1,:-1] > 0.01)

    tmp = 0.25 * ((uIce[:-1,:-1][isArea] * maskInW[:-1,:-1][isArea] + uIce[:-1,1:][isArea] * maskInW[:-1,1:][isArea])**2 + (vIce[:-1,:-1][isArea] * maskInS[:-1,:-1][isArea] + vIce[1:,:-1][isArea] * maskInS[1:,:-1][isArea])**2)
    tmpFld[:-1,:-1][isArea] = basalDragK2 / np.sqrt(tmp + u0sq)

    hCrit = np.abs(R_low[:-1,:-1][isArea]) * Area[:-1,:-1][isArea] * recip_k1

    # soft maximum
    # add an explanation
    cBot = np.zeros((sNy+2*OLy,sNx+2*OLx))
    cBot[:-1,:-1][isArea] = tmpFld[:-1,:-1][isArea] * np.log(np.exp(fac * (hIceMean[:-1,:-1][isArea] - hCrit)) + 1) * recip_fac * np.exp(-cBasalStar * (1 - Area[:-1,:-1][isArea]))  * iceMask[:-1,:-1][isArea]


    return cBot