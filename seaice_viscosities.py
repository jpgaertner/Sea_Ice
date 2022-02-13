"""Usage: zeta, eta, press =
          viscosities(e11,e22,e12,press0,iStep,myTime,myIter)
### input
# e11: 1,1 component of strain rate tensor
# e22: 2,2 component of strain rate tensor
# e12: 1,2 component of strain rate tensor
# press0: maximum compressive stres

### ouput
# zeta, eta : bulk and shear viscosity
# press : replacement pressure
"""
import numpy as np

from seaice_params import PlasDefCoeff, deltaMin, tensileStrFac, pressReplFac
from seaice_size import *

def viscosities(e11,e22,e12,press0,iStep,myTime,myIter):

    recip_PlasDefCoeffSq = 1. / PlasDefCoeff**2

    # use area weighted average of squares of e12 (more accurate)
    e12Csq = rAz * e12**2
    e12Csq =                     e12Csq + np.roll(e12Csq,-1,0)
    e12Csq = 0.25 * recip_rA * ( e12Csq + np.roll(e12Csq,-1,1) )

    deltaSq = (e11+e22)**2 + recip_PlasDefCoeffSq * (
        (e11-e22)**2 + 4. * e12Csq )
    deltaC = np.sqrt(deltaSq)

    # smooth regularization of delta for better differentiability
    deltaCreg = deltaC + deltaMin
    # deltaCreg = np.sqrt( deltaSq + deltaMin**2 )
    # deltaCreg = np.maximum(deltaC,deltaMin)

    zeta = 0.5 * (press0 * (1 + tensileStrFac)) / deltaCreg
    eta  = zeta * recip_PlasDefCoeffSq

    # recalculate pressure
    press = ( press0 * (1 - pressReplFac)
              + 2. * zeta * deltaC * pressReplFac / (1 + tensileStrFac)
             ) * (1 - tensileStrFac)

    return zeta, eta, press
