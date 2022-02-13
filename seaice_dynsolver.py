import numpy as np

from seaice_size import *
from seaice_params import *

from seaice_freedrift import seaIceFreeDrift
from seaice_evp import evp
from seaice_implicit_solver import picard_solver
from seaice_lsr import lsr_solver
from seaice_get_dynforcing import get_dynforcing
from seaice_ocean_stress import ocean_stress


### input:
# uIce: zonal ice velocity at south-west B-grid (or c grid? ifdef cgrid) u point? (what does the grid look like) (>0 = from west to east)
# vIce: meridional ice velocty at south-west B-grid v point (>0 = from south to north)
# hIceMean: mean ice thickness
# hSnowMean: mean snow thickness
# Area: ice cover fraction
# etaN: ocean surface elevation
# pLoad: surface pressure
# SeaIceLoad: load of sea ice on ocean surface
# useRealFreshWaterFlux: flag for using the sea ice load in the calculation of the ocean surface height
# uVel: zonal ocean velocity
# vVel: meridional ocean velocity
# uWind: zonal wind velocity
# vWind: meridional wind velocity
# fu: zonal stress on ocean surface (ice or atmopshere)
# fv: meridional stress on ocean surface (ice or atmopshere)

### output:
# uIce: zonal ice velocity
# vIce: meridional ice velocity
# fu: zonal stress on ocean surface (ice or atmopshere)
# fv: meridional stress on ocean surface (ice or atmopshere)


def dynsolver(uIce, vIce, uVel, vVel, uWind, vWind, hIceMean,
              hSnowMean, Area, etaN, pLoad, SeaIceLoad, useRealFreshWaterFlux,
              fu, fv, secondOrderBC, R_low):

    # set up mass per unit area
    SeaIceMassC = rhoIce * hIceMean

    # if SEAICEaddSnowMass (true)
    SeaIceMassC= SeaIceMassC + rhoSnow * hSnowMean
    SeaIceMassU = 0.5 * ( SeaIceMassC + np.roll(SeaIceMassC,1,1) )
    SeaIceMassV = 0.5 * ( SeaIceMassC + np.roll(SeaIceMassC,1,0) )

    # if SEAICE_maskRHS... (false)

    ##### set up forcing fields #####

    # compute surface stresses from wind, ocean and ice velocities
    tauX, tauY = get_dynforcing(uIce, vIce, uWind, vWind, uVel, vVel)

    # compute surface pressure at z = 0:
    # use actual sea surface height phiSurf for tilt computations
    phiSurf = gravity * etaN
    if useRealFreshWaterFlux:
        phiSurf = phiSurf + (pLoad + SeaIceLoad * gravity * seaIceLoadFac
                             ) * recip_rhoConst
    else:
        phiSurf = phiSurf + pLoad * recip_rhoConst

    # forcing by surface stress
    #if SEAICEscaleSurfStress (true)
    IceSurfStressX0 = tauX * 0.5 * (Area + np.roll(Area,1,1)) #forcex0 in F
    IceSurfStressY0 = tauY * 0.5 * (Area + np.roll(Area,1,0)) #forcey0 in F

    # add in tilt
    #if SEAICEuseTILT (true)
    IceSurfStressX0 = IceSurfStressX0 \
        - SeaIceMassU * recip_dxC * ( phiSurf - np.roll(phiSurf,1,1) )
    IceSurfStressY0 = IceSurfStressY0 \
        - SeaIceMassV * recip_dyC * ( phiSurf - np.roll(phiSurf,1,0) )

    # calculate ice strength
    press0 = SeaIceStrength * hIceMean * np.exp(-cStar * (1 - Area)) * iceMask

    #if SEAICEuseDYNAMICS (true)
    if useFreedrift:
        uIce, vIce = seaIceFreeDrift(hIceMean, uVel, vVel,
                                     IceSurfStressX0, IceSurfStressY0)

    # #ifdef ALLOW_OBCS
    # #call OBCS_APPLY_UVICE
    # #solver

    if useEVP:
        uIce, vIce = evp(
            uIce, vIce, uVel, vVel, hIceMean, Area,
            press0, secondOrderBC,
            IceSurfStressX0, IceSurfStressY0,
            SeaIceMassC, SeaIceMassU, SeaIceMassV, R_low)

    myTime = 0.
    myIter = 0
    if useLSR:
        uIce, vIce = lsr_solver(
            uIce, vIce, uVel, vVel, hIceMean, Area,
            press0, IceSurfStressX0, IceSurfStressY0,
            SeaIceMassC, SeaIceMassU, SeaIceMassV, R_low,
            myTime, myIter)

    if usePicard:
        uIce, vIce = picard_solver(
            uIce, vIce, uVel, vVel, hIceMean, Area,
            press0, IceSurfStressX0, IceSurfStressY0,
            SeaIceMassC, SeaIceMassU, SeaIceMassV, R_low,
            myTime, myIter)

    #if SEAICEuseJFNK
    #call SEAICE_JFNK

    # update stress on ocean surface
    fu, fv = ocean_stress(uIce, vIce, uVel, vVel, Area, fu, fv)

    # cap the ice velicity at 0.4 m/s to avoid CFL violations in open
    # water areas (drift of zero thickness ice)
    # uIce = np.clip(uIce, -0.4, 0.4)
    # vIce = np.clip(vIce, -0.4, 0.4)


    return uIce, vIce, fu, fv
