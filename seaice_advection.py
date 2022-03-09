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

# calculate change in sea ice fields due to advection
@veros_kernel
def calc_Advection(state):

    #store fields prior to advective changes
    fields = npx.array([state.variables.hIceMean, state.variables.hSnowMean, state.variables.Area])
    fields_preAdv = update(fields, at[:], fields)

    # calculate zonal advective fluxes of hIceMean, hSnowMean, Area
    ZonalFlux = calc_ZonalFlux(state)

    # calculate meridional advective fluxes of hIceMean, hSnowMean, Area
    MeridionalFlux = calc_MeridionalFlux(state)

    # changes due to zonal fluxes
    if extensiveFld:
        fields = fields - deltaTtherm * maskInC * recip_rA \
            * ( npx.roll(ZonalFlux,-1,1) - ZonalFlux )
    else:
        fields= fields- deltaTtherm * maskInC * recip_rA * recip_hIceMean * (
            ( npx.roll(ZonalFlux,-1,1) - ZonalFlux )
            - ( npx.roll(state.variable.uTrans,-1,1) - state.variable.uTrans )
            * fields_preAdv)

    # changes due to meridional fluxes
    if extensiveFld:
        fields = fields - deltaTtherm * maskInC * recip_rA \
            * ( npx.roll(MeridionalFlux,-1,0) - MeridionalFlux )
    else:
        fields = fields - deltaTtherm * maskInC * recip_rA * recip_hIceMean * (
            ( npx.roll(MeridionalFlux,-1,0) - MeridionalFlux )
            - ( npx.roll(state.variable.vTrans,-1,0) - state.variables.vTrans)
            * fields_preAdv)

    # apply mask
    fields = fields * iceMask
    
    return KernelOutput(hIceMean = fields[0], hSnowMean = fields[1], Area = fields[2])

@veros_routine
def update_Advection(state):

    # retrieve change of hIceMean, hSnowMean, Area due to advection and update state object
    Advection = calc_Advection(state)
    state.variables.update(Advection)