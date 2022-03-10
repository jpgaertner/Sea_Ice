from veros.core.operators import numpy as npx
from veros import veros_routine, veros_kernel, KernelOutput

from seaice_params import rhoIce, rhoSnow, useRealFreshWaterFlux, \
                            seaIceLoadFac, recip_rhoConst, gravity, \
                            useFreedrift, useEVP, useJFNK, useLSR, usePicard
from seaice_size import recip_dxC, recip_dyC


from seaice_freedrift import freedrift_solver
from seaice_evp import evp_solver
from seaice_implicit_solver import picard_solver, jfnk_solver
from seaice_lsr import lsr_solver
from seaice_get_dynforcing import get_dynforcing


# calculate sea ice mass from ice and snow thickness
@veros_kernel
def calc_SeaIceMass(state):

    # calculate sea ice mass centered around c, u, v points
    SeaIceMassC = rhoIce * state.variables.hIceMean \
                + rhoSnow * state.variables.hSnowMean
    SeaIceMassU = 0.5 * ( SeaIceMassC + npx.roll(SeaIceMassC,1,1) )
    SeaIceMassV = 0.5 * ( SeaIceMassC + npx.roll(SeaIceMassC,1,0) )

    return KernelOutput(SeaIceMassC = SeaIceMassC,
                        SeaIceMassU = SeaIceMassU,
                        SeaIceMassV = SeaIceMassV)

@veros_routine
def update_SeaIceMass(state):
    
    # retrieve sea ice mass centered around c, u, v points and update state object
    SeaIceMass = calc_SeaIceMass(state)
    state.variables.update(SeaIceMass)

# calculate surface forcing from wind 
@veros_kernel
def calc_SurfaceForcing(state):

    # compute surface stresses from wind and ice velocities
    tauX, tauY = get_dynforcing(state)

    # calculate forcing by surface stress
    WindForcingX = tauX * 0.5 * (state.variables.Area + npx.roll(state.variables.Area,1,1))
    WindForcingY = tauY * 0.5 * (state.variables.Area + npx.roll(state.variables.Area,1,0))

    # calculate actual sea surface height/ geopotential height anomaly
    # (equivalent to surface pressure)
    phiSurf = gravity * state.variables.etaN
    if useRealFreshWaterFlux:
        phiSurf = phiSurf + (state.variables.pLoad \
                    + state.variables.SeaIceLoad * gravity * seaIceLoadFac
                             ) * recip_rhoConst
    else:
        phiSurf = phiSurf + state.variables.pLoad * recip_rhoConst

    # add in tilt
    WindForcingX = WindForcingX - state.variables.SeaIceMassU \
                    * recip_dxC * ( phiSurf - npx.roll(phiSurf,1,1) )
    WindForcingY = WindForcingY - state.variables.SeaIceMassV \
                    * recip_dyC * ( phiSurf - npx.roll(phiSurf,1,0) )

    return KernelOutput(WindForcingX = WindForcingX,
                        WindForcingY = WindForcingY)

@veros_routine
def update_SurfaceForcing(state):

    # retrieve surface forcing and update state object
    SurfaceForcing = calc_SurfaceForcing(state)
    state.variables.update(SurfaceForcing)

# calculate sea ice cover fraction centered around velocity points
@veros_kernel
def calc_AreaWS(state):

    AreaW = 0.5 * (state.variables.Area + npx.roll(state.variables.Area,1,1))
    AreaS = 0.5 * (state.variables.Area + npx.roll(state.variables.Area,1,0))

    return KernelOutput(AreaW = AreaW, AreaS = AreaS)

@veros_routine
def update_AreaWS(state):

    # retrieve sea ice cover fraction centered around u,v points and update state object
    AreaWS = calc_AreaWS(state)
    state.variables.update(AreaWS)



# calculate ice velocities from surface forcing
@veros_kernel
def calc_IceVelocities(state):

    if useFreedrift:
        uIce, vIce = freedrift_solver(state)

    if useEVP:
        uIce, vIce = evp_solver(state)

    if useLSR:
        uIce, vIce = lsr_solver(state)

    if usePicard:
        uIce, vIce = picard_solver(state)

    if useJFNK:
        uIce, vIce = jfnk_solver(state)


    return KernelOutput(uIce = uIce, vIce = vIce)

@veros_routine
def update_IceVelocities(state):

    # retrieve ice velocities and update state object
    IceVelocities = calc_IceVelocities(state)
    state.variables.update(IceVelocities)