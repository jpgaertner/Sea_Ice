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
    dragCoeff = np.ones((sNy+2*OLy,sNx+2*OLx)) * waterIceDrag * rhoConst
    south = np.where(fCori < 0)
    dragCoeff[south] = waterIceDrag_south * rhoConst

    # calculate linear drag coefficient
    cDrag = np.zeros((sNy+2*OLy,sNx+2*OLx))
    cDrag[:-1,:-1] = dragCoeff[:-1,:-1] * np.sqrt(0.25 * (((uIce[:-1,:-1] - uVel[:-1,:-1])**2 * maskInW[:-1,:-1] + (uIce[:-1,1:] - uVel[:-1,1:])**2 * maskInW[:-1,1:]) + ((vIce[:-1,:-1] - vVel[:-1,:-1])**2 * maskInS[:-1,:-1] + (vIce[1:,:-1] - vVel[1:,:-1])**2 * maskInS[1:,:-1])))
    tmp = np.where(dragCoeff * cDrag <= cDragMin**2)
    cDrag[tmp] = cDragMin
    cDrag = cDrag * hIceMeanMask


    return cDrag