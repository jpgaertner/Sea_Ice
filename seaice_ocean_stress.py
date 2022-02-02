import numpy as np

from seaice_params import *
from seaice_size import *

from seaice_fill_overlap import fill_overlap
from seaice_ocean_drag_coeffs import ocean_drag_coeffs


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


def ocean_stress(uIce, vIce, uVel, vVel, Area, fu, fv):

    # get linear drag coefficient at c point
    cDrag = ocean_drag_coeffs(uIce, vIce, uVel, vVel)

    # introduce turning angle (default is zero)
    sinWat = np.sin(np.deg2rad(waterTurnAngle))
    cosWat = np.cos(np.deg2rad(waterTurnAngle))

    # calculate ice affected wind stress by averaging wind stress and ice-ocean stress according to ice cover
    fuLoc = 0.5 * (cDrag[OLy:-OLy,OLx:-OLx] + cDrag[OLy:-OLy,OLx+1:-OLx+1]) * cosWat * (uIce[OLy:-OLy,OLx:-OLx] - uVel[OLy:-OLy,OLx:-OLx]) - np.sign(fCori[OLy:-OLy,OLx:-OLx]) * sinWat * 0.5 * (cDrag[OLy:-OLy,OLx:-OLx] * 0.5 * (vIce[OLy:-OLy,OLx:-OLx] - vVel[OLy:-OLy,OLx:-OLx] + vIce[OLy+1:-OLy+1,OLx:-OLx] - vVel[OLy+1:-OLy+1,OLx:-OLx]) + cDrag[OLy:-OLy,OLx:-OLx] * 0.5 * (vIce[OLy:-OLy,OLx-1:-OLx-1] - vVel[OLy:-OLy,OLx-1:-OLx-1] + vIce[OLy+1:-OLy+1,OLx-1:-OLx-1] - vVel[OLy+1:-OLy+1,OLx-1:-OLx-1]))
    fvLoc = 0.5 * (cDrag[OLy:-OLy,OLx:-OLx] + cDrag[OLy-1:-OLy-1,OLx:-OLx]) * cosWat * (vIce[OLy:-OLy,OLx:-OLx] - vVel[OLy:-OLy,OLx:-OLx]) + np.sign(fCori[OLy:-OLy,OLx:-OLx]) * sinWat * 0.5 * (cDrag[OLy:-OLy,OLx:-OLx] * 0.5 * (uIce[OLy:-OLy,OLx:-OLx] - uVel[OLy:-OLy,OLx:-OLx] + uIce[OLy:-OLy,OLx+1:-OLx+1] - uVel[OLy:-OLy,OLx+1:-OLx+1]) + cDrag[OLy:-OLy,OLx:-OLx] * 0.5 * (uIce[OLy-1:-OLy-1,OLx:-OLx] - uVel[OLy-1:-OLy-1,OLx:-OLx] + uIce[OLy-1:-OLy-1,OLx+1:-OLx+1] - uVel[OLy-1:-OLy-1,OLx+1:-OLx+1]))
    areaW = 0.5 * (Area[OLy:-OLy,OLx:-OLx] + Area[OLy:-OLy,OLx-1:-OLx-1]) * stressFactor #stressFactor = 1, what is it, is it needed? 
    areaS = 0.5 * (Area[OLy:-OLy,OLx:-OLx] + Area[OLy-1:-OLy-1,OLx:-OLx]) * stressFactor
    fu[OLy:-OLy,OLx:-OLx] = (1 - areaW) * fu[OLy:-OLy,OLx:-OLx] + areaW * fuLoc
    fv[OLy:-OLy,OLx:-OLx] = (1 - areaS) * fv[OLy:-OLy,OLx:-OLx] + areaS * fvLoc

    # fill overlaps
    fu = fill_overlap(fu)
    fv = fill_overlap(fv)


    return fu, fv