import numpy as np

from seaice_size import iceMask
from seaice_fill_overlap import fill_overlap


##### calculates value at z point by averaging c point values #####

def c_point_to_z_point(Cfield):

    sumNorm = iceMask + np.roll(iceMask,1,1)
    sumNorm = sumNorm + np.roll(sumNorm,1,0)
    sumNorm = np.where(sumNorm>0,1./sumNorm,0.)
    Zfield =             Cfield + np.roll(Cfield,1,1)
    Zfield = sumNorm * ( Zfield + np.roll(Zfield,1,0) )

    # Zfield = fill_overlap(Zfield)


    return Zfield
