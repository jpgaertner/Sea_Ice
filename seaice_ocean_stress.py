from veros.core.operators import numpy as npx
from veros import veros_kernel, veros_routine, KernelOutput

from seaice_params import *
from seaice_size import *

from seaice_fill_overlap import fill_overlap_uv
from dynamics_routines import ocean_drag_coeffs


# calculate stress on ocean surface from ocean and ice velocities
@veros_kernel
def calc_OceanStress(state):

    # get linear drag coefficient at c point
    cDrag = ocean_drag_coeffs(state, state.variables.uIce, state.variables.vIce)

    # introduce turning angle (default is zero)
    sinWat = npx.sin(npx.deg2rad(waterTurnAngle))
    cosWat = npx.cos(npx.deg2rad(waterTurnAngle))

    # calculate velocity of ice relative to ocean surface 
    du = state.variables.uIce - state.variables.uVel
    dv = state.variables.vIce - state.variables.vVel

    # interpolate to c points
    duAtC = 0.5 * (du + npx.roll(du,-1,1))
    dvAtC = 0.5 * (dv + npx.roll(dv,-1,0))

    # calculate forcing in u and v direction
    fuLoc = 0.5 * (cDrag + npx.roll(cDrag,1,1)) * cosWat * du \
        - npx.sign(fCori) * sinWat * 0.5 * (
            cDrag * dvAtC + npx.roll(cDrag * dvAtC,1,0) )
    fvLoc = 0.5 * (cDrag + npx.roll(cDrag,1,0)) * cosWat * dv \
        + npx.sign(fCori) * sinWat * 0.5 * (
            cDrag * duAtC + npx.roll(cDrag * duAtC,1,1) )

    # calculate ice cover area centered around u and v points
    areaW = 0.5 * (state.variables.Area + npx.roll(state.variables.Area,1,1))
    areaS = 0.5 * (state.variables.Area + npx.roll(state.variables.Area,1,0))

    # update forcing for ice covered area
    fu = (1 - areaW) * state.variables.fu + areaW * fuLoc
    fv = (1 - areaS) * state.variables.fv + areaS * fvLoc

    # fill overlaps
    fu, fv = fill_overlap_uv(fu,fv)


    return KernelOutput(fu = fu, fv = fv)

@veros_routine
def update_OceanStress(state):

    # retrieve zonal and meridional stresses on ocean surface
    OceanStress = calc_OceanStress(state)
    state.variables.update(OceanStress)