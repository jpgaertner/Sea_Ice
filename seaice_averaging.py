from veros.core.operators import numpy as npx

from seaice_size import iceMask

# calculates value at z point by averaging c point values


def c_point_to_z_point(Cfield, noSlip = True):

    sumNorm = iceMask + npx.roll(iceMask,1,1)
    sumNorm = sumNorm + npx.roll(sumNorm,1,0)
    if noSlip:
        sumNorm = npx.where(sumNorm>0,1./sumNorm,0.)
    else:
        sumNorm = npx.where(sumNorm==4.,0.25,0.)

    Zfield =             Cfield + npx.roll(Cfield,1,1)
    Zfield = sumNorm * ( Zfield + npx.roll(Zfield,1,0) )


    return Zfield
