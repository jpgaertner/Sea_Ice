import numpy as np

from seaice_size import sNy, sNx, OLx, OLy, iceMask
from seaice_fill_overlap import fill_overlap


##### calculates value at z point by averaging c point values #####

def c_point_to_z_point(Cfield):

    Zfield = np.zeros((sNy+2*OLy,sNx+2*OLx))

    sumNorm = iceMask[OLy:-OLy,OLx:-OLx] + iceMask[OLy:-OLy,OLx-1:-OLx-1] + iceMask[OLy-1:-OLy-1,OLx:-OLx] + iceMask[OLy-1:-OLy-1,OLx-1:-OLx-1]
    tmp = np.where(sumNorm > 0)
    sumNorm[tmp] = 1 / sumNorm[tmp]

    Zfield[OLy:-OLy,OLx:-OLx] = sumNorm * (Cfield[OLy:-OLy,OLx:-OLx] + Cfield[OLy:-OLy,OLx-1:-OLx-1] + Cfield[OLy-1:-OLy-1,OLx:-OLx] + Cfield[OLy-1:-OLy-1,OLx-1:-OLx-1])
    
    Zfield = fill_overlap(Zfield)


    return Zfield