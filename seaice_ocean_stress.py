import numpy as np

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
    # fu: zonal stress on ocean surface (ice or atmopshere)
    # fv: meridional stress on ocean surface (ice or atmopshere)

    ### output
    # fu: zonal stress on ocean surface (ice or atmopshere)
    # fv: meridional stress on ocean surface (ice or atmopshere)

    # get linear drag coefficient at c point
    cDrag = ocean_drag_coeffs(uIce, vIce, uVel, vVel)

    # introduce turning angle (default is zero)
    sinWat = np.sin(np.deg2rad(waterTurnAngle))
    cosWat = np.cos(np.deg2rad(waterTurnAngle))

    # calculate ice affected wind stress by averaging wind stress and
    # ice-ocean stress according to ice cover
    du = uIce - uVel
    dv = vIce - vVel
    duAtC = 0.5 * (du + np.roll(du,-1,1))
    dvAtC = 0.5 * (dv + np.roll(dv,-1,0))
    fuLoc = 0.5 * (cDrag + np.roll(cDrag,1,1)) * cosWat * du \
        - np.sign(fCori) * sinWat * 0.5 * (
            cDrag * dvAtC + np.roll(cDrag * dvAtC,1,0) )
    fvLoc = 0.5 * (cDrag + np.roll(cDrag,1,0)) * cosWat * dv \
        + np.sign(fCori) * sinWat * 0.5 * (
            cDrag * duAtC + np.roll(cDrag * duAtC,1,1) )

    #stressFactor = 1, what is it, is it needed?
    areaW = 0.5 * (Area + np.roll(Area,1,1)) * stressFactor
    areaS = 0.5 * (Area + np.roll(Area,1,0)) * stressFactor
    fu = (1 - areaW) * fu + areaW * fuLoc
    fv = (1 - areaS) * fv + areaS * fvLoc

    # fill overlaps
    fu, fv = fill_overlap_uv(fu,fv)

    return fu, fv
