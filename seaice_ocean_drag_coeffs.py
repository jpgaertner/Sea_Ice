import numpy as np

from seaice_params import *
from seaice_size import *

### input:
# uIce: zonal ice velocity
# vIce: meridional ice velocity
# uVel: zonal ocean velocity
# vVel: meridional ice velocity

### output:
# cDrag: linear drag coefficient between ice and ocean at c point (this coefficient creates a linear relationship between ice-ocean stress difference and ice-ocean velocity difference)


def ocean_drag_coeffs(uIce, vIce, uVel, vVel):

    # get ice-water drag coefficient times density
    dragCoeff = np.ones(uIce.shape) * waterIceDrag * rhoConst
    south = np.where(fCori < 0)
    dragCoeff[south] = waterIceDrag_south * rhoConst

    # calculate linear drag coefficient
    cDrag = np.ones(uIce.shape) * cDragMin
    du = (uIce - uVel)*maskInW
    dv = (vIce - vVel)*maskInS
    tmpVar = 0.25 * ( du**2 + np.roll(du,-1,1)**2
                    + dv**2 + np.roll(dv,-1,0)**2 )

    tmp = np.where(dragCoeff**2 * tmpVar > cDragMin**2)
    cDrag[tmp] = dragCoeff[tmp] * np.sqrt(tmpVar[tmp])
    cDrag = cDrag * iceMask


    return cDrag
