#Ice dynamics using LSR solver (see Zhang and Hibler,   JGR, 102, 8691-8702, 1997)

import numpy as np
from seaice_size import *
from seaice_params import *

from seaice_freedrift import seaIceFreeDrift
from seaice_get_dynforcing import get_dynforcing


### input:
# uIce: zonal ice velocity [m/s] at south-west B-grid (or c grid? ifdef cgrid) u point? (what does the grid look like) (>0 = from west to east)
# vIce: meridional ice velocty [m/s] at south-west B-grid v point (>0 = from south to north)
# hIceMean
# hSnowMean
# Area
# etaN: ocean surface elevation
# pLoad
# SeaIceLoad
# useRealFreshWaterFlux
# uVel: zonal ocean velocity [m/s]
# vVel: meridional ocean velocity [m/s]
# uWind: zonal wind velocity [m/s]
# vWind: meridional wind velocity [m/s]

### output:
# uIce
# vIce


def dynsolver(uIce, vIce, uVel, vVel, uWind, vWind, hIceMean, hSnowMean, Area, etaN, pLoad, SeaIceLoad, useRealFreshWaterFlux):

    # local variables:
    tauX = np.zeros((sNy+2*OLy,sNx+2*OLx)) # zonal wind stress over ice at u point
    tauY = np.zeros((sNy+2*OLy,sNx+2*OLx)) # meridional wind stress over ice at v point
    seaIceMassC = np.zeros((sNy+2*OLy,sNx+2*OLx))
    seaIceMassU = np.zeros((sNy+2*OLy,sNx+2*OLx))
    seaIceMassV = np.zeros((sNy+2*OLy,sNx+2*OLx))
    IceSurfStressX0 = np.zeros((sNy+2*OLy,sNx+2*OLx))
    IceSurfStressY0 = np.zeros((sNy+2*OLy,sNx+2*OLx))

    # set up mass per unit area
    seaIceMassC[1:,1:] = rhoIce * hIceMean[1:,1:]
    seaIceMassU[1:,1:] = rhoIce * 0.5 * (hIceMean[1:,1:] + hIceMean[1:,:-1])
    seaIceMassV[1:,1:] = rhoIce * 0.5 * (hIceMean[1:,1:] + hIceMean[:-1,1:])

    # if SEAICEaddSnowMass (true)
    seaIceMassC[1:,1:] = seaIceMassC[1:,1:] + rhoSnow * hSnowMean[1:,1:]
    seaIceMassU[1:,1:] = seaIceMassU[1:,1:] + rhoSnow * 0.5 * (hSnowMean[1:,1:] + hSnowMean[1:,:-1])
    seaIceMassV[1:,1:] = seaIceMassV[1:,1:] + rhoSnow * 0.5 * (hSnowMean[1:,1:] + hSnowMean[:-1,1:])

    # if SEAICE_maskRHS... (false)


    ##### set up forcing fields #####

    # compute surface stresses from wind, ocean and ice velocities
    tauX, tauY = get_dynforcing(uIce, vIce, uWind, vWind, uVel, vVel)

    # compute surface pressure at z = 0:
    # use actual sea surface height phiSurf for tilt computations
    phiSurf = gravity * etaN

    #if usingZCoords (true)
    if useRealFreshWaterFlux:
        phiSurf = phiSurf + (pLoad + SeaIceLoad * gravity * seaIceLoadFac) * recip_rhoConst
    else:
        phiSurf = phiSurf + pLoad * recip_rhoConst

    # forcing by wind
    #if SEAICEscaleSurfStress (true)
    IceSurfStressX0[1:,1:] = tauX[1:,1:] * 0.5 * (Area[1:,1:] + Area[1:,:-1]) #forcex0 in F
    IceSurfStressY0[1:,1:] = tauY[1:,1:] * 0.5 * (Area[1:,1:] + Area[:-1,1:]) #forcey0 in F

    # add in tilt
    #if SEAICEuseTILT (true)
    IceSurfStressX0[1:,1:] = IceSurfStressX0[1:,1:] - seaIceMassU[1:,1:] * recip_dxC[1:,1:] * (phiSurf[1:,1:] - phiSurf[1:,:-1])
    IceSurfStressY0[1:,1:] = IceSurfStressY0[1:,1:] - seaIceMassV[1:,1:] * recip_dyC[1:,1:] * (phiSurf[1:,1:] - phiSurf[:-1,1:])

    # calculate press0, where is it used?
    # call SEAICE_CALC_ICE_STRENGTH

    #if SEAICEuseDYNAMICS (true)
    uIce, vIce = seaIceFreeDrift(hIceMean, uVel, vVel, IceSurfStressX0, IceSurfStressY0)

    #ifdef ALLOW_OBCS
    #call OBCS_APPLY_UVICE
    #solver

    #ifdef SEAICE_ALLOW_EVP
    #call SEAICE_EVP
    #solver

    #if SEAICEuseLSR
    #call SEAICE_LSR
    #solver

    #if SEAICEuseKrylov
    #call SEAICE_KRYLOV
    #solver

    #if SEAICEuseJFNK
    #call SEAICE_JFNK
    #solver

    # update ocean surface stress
    #if SEAICEupdateOceanStress
    #call SEAICE_OCEAN_STRESS

    # cap the ice velicity at 0.4 m/s to avoid CFL violations in open water areas (drift of zero thickness ice)
    #if SEAICE_clipVelocities
    uIce.clip(-0.4, 0.4)
    vIce.clip(-0.4, 0.4)


    return uIce, vIce