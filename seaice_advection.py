from veros.core.operators import numpy as npx
from veros.core.operators import update, at
from veros import veros_kernel, KernelOutput, veros_routine

from seaice_size import *
from seaice_params import *

from seaice_fluxlimit_adv_x import calc_ZonalFlux
from seaice_fluxlimit_adv_y import calc_MeridionalFlux


# calculate ice transports from ice velocities
@veros_kernel
def calc_Transport(state):

    # retrieve used cell faces
    xA = dyG * SeaIceMaskU
    yA = dxG * SeaIceMaskV

    # calculate ice transport
    uTrans = state.variables.uIce * xA
    vTrans = state.variables.vIce * yA

    return KernelOutput(uTrans = uTrans, vTrans = vTrans)

@veros_routine
def update_Transport(state):

    # retrieve ice transport
    Transport = calc_Transport(state)
    state.variables.update(Transport)

# calculate change in sea ice field due to advection
@veros_kernel
def calc_Advection(state, field):

    # make local copy of field prior to advective changes
    fieldLoc = field

    # calculate zonal advective fluxes
    ZonalFlux = calc_ZonalFlux(state, fieldLoc)

    # update field according to zonal fluxes
    if extensiveFld:
        fieldLoc = fieldLoc - deltaTtherm * maskInC * recip_rA \
                * ( npx.roll(ZonalFlux,-1,1) - ZonalFlux )
    else:
        fieldLoc = fieldLoc - deltaTtherm * maskInC * recip_rA * recip_hIceMean \
            * (( npx.roll(ZonalFlux,-1,1) - ZonalFlux )
            - ( npx.roll(state.variable.uTrans,-1,1) - state.variable.uTrans )
            * field)

    # calculate meridional advective fluxes
    MeridionalFlux = calc_MeridionalFlux(state, fieldLoc)

    # update field according to meridional fluxes
    if extensiveFld:
        fieldLoc = fieldLoc - deltaTtherm * maskInC * recip_rA \
            * ( npx.roll(MeridionalFlux,-1,0) - MeridionalFlux )
    else:
        fieldLoc = fieldLoc - deltaTtherm * maskInC * recip_rA * recip_hIceMean \
            * (( npx.roll(MeridionalFlux,-1,0) - MeridionalFlux )
            - ( npx.roll(state.variable.vTrans,-1,0) - state.variables.vTrans)
            * field)

    # apply mask
    fieldLoc = fieldLoc * iceMask

    return fieldLoc

@veros_kernel
def do_Advections(state):

    # retrieve change of ice thickness due to advection
    hIceMean = calc_Advection(state, state.variables.hIceMean)

    # retrieve change of snow thickness due to advection
    hSnowMean = calc_Advection(state, state.variables.hSnowMean)

    # retrieve change of sea ice cover fraction due to advection
    Area = calc_Advection(state, state.variables.Area)

    return KernelOutput(hIceMean = hIceMean, hSnowMean = hSnowMean, Area = Area)

@veros_routine
def update_Advection(state):

    # retrieve changes of ice fields due to advection and update state object
    Advection = do_Advections(state)
    state.variables.update(Advection)