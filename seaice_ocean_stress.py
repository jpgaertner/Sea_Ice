from veros.core.operators import numpy as npx
from veros.core.operators import update, at

from seaice_params import *
from seaice_size import *

from seaice_fill_overlap import fill_overlap_uv
from dynamics_routines import ocean_drag_coeffs

def ocean_stress(uIce, vIce, uVel, vVel, Area, fu, fv):

    ### input
    # uIce: zonal ice velocity
    # vIce: meridional ice velocity
    # uVel: zonal ocean velocity
    # vVel: meridional ice velocity
    # Area: ice cover fraction
    # fu: zonal stress on ocean surface (ice or atmopshere/ wind)
    # fv: meridional stress on ocean surface (ice or atmopshere/ wind)

    ### output
    # fu: zonal stress on ocean surface (ice or atmopshere/ wind)
    # fv: meridional stress on ocean surface (ice or atmopshere/ wind)

    # get linear drag coefficient at c point
    cDrag = ocean_drag_coeffs(uIce, vIce, uVel, vVel)

    # introduce turning angle (default is zero)
    sinWat = npx.sin(npx.deg2rad(waterTurnAngle))
    cosWat = npx.cos(npx.deg2rad(waterTurnAngle))

    # calculate ice affected wind stress by averaging wind stress and
    # ice-ocean stress according to ice cover
    du = uIce - uVel
    dv = vIce - vVel
    duAtC = 0.5 * (du + npx.roll(du,-1,1))
    dvAtC = 0.5 * (dv + npx.roll(dv,-1,0))
    fuLoc = 0.5 * (cDrag + npx.roll(cDrag,1,1)) * cosWat * du \
        - npx.sign(fCori) * sinWat * 0.5 * (
            cDrag * dvAtC + npx.roll(cDrag * dvAtC,1,0) )
    fvLoc = 0.5 * (cDrag + npx.roll(cDrag,1,0)) * cosWat * dv \
        + npx.sign(fCori) * sinWat * 0.5 * (
            cDrag * duAtC + npx.roll(cDrag * duAtC,1,1) )

    areaW = 0.5 * (Area + npx.roll(Area,1,1))
    areaS = 0.5 * (Area + npx.roll(Area,1,0))
    fu = (1 - areaW) * fu + areaW * fuLoc
    fv = (1 - areaS) * fv + areaS * fvLoc

    # fill overlaps
    fu, fv = fill_overlap_uv(fu,fv)

    return fu, fv
