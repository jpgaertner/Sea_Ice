import numpy as np

from seaice_params import *
from seaice_size import *

### input 
# uFld: zonal ice field velocity
# vFld: meridional ice Field velocity
# secondOrderBC: flag

### output
# e11: 1,1 component of strain rate tensor
# e22: 2,2 component of strain rate tensor
# e12: 1,2 component of strain rate tensor


def strainrates(uFld, vFld, secondOrderBC):

    # initializations
    noSlipFac = 1
    dudx = np.zeros((sNy+2*OLy,sNx+2*OLx))
    dudy = np.zeros((sNy+2*OLy,sNx+2*OLx))
    dvdx = np.zeros((sNy+2*OLy,sNx+2*OLx))
    dvdy = np.zeros((sNy+2*OLy,sNx+2*OLx))
    uave = np.zeros((sNy+2*OLy,sNx+2*OLx))
    vave = np.zeros((sNy+2*OLy,sNx+2*OLx))

    # abbreviations at c points
    dudx[:-1,:-1] = (uFld[:-1,1:] - uFld[:-1,:-1]) * recip_dxF[:-1,:-1]
    uave[:-1,:-1] = 0.5 * (uFld[:-1,:-1] + uFld[:-1,1:])
    dvdy[:-1,:-1] = (vFld[1:,:-1] - vFld[:-1,:-1]) * recip_dyF[:-1,:-1]
    vave[:-1,:-1] = 0.5 * (vFld[:-1,:-1] + vFld[1:,:-1])

    # evaluate strain rates at c points
    e11 = np.ones((sNy+2*OLy,sNx+2*OLx))
    e22 = np.ones((sNy+2*OLy,sNx+2*OLx))
    e11[:-1,:-1] = (dudx[:-1,:-1] + vave[:-1,:-1] * k2AtC[:-1,:-1]) * maskInC[:-1,:-1]
    e22[:-1,:-1] = (dvdy[:-1,:-1] + uave[:-1,:-1] * k1AtC[:-1,:-1]) * maskInC[:-1,:-1]

    # abbreviations at z points
    dudy[1:,1:] = (uFld[1:,1:] - uFld[:-1,1:]) * recip_dyU[1:,1:]
    uave[1:,1:] = 0.5 * (uFld[1:,1:] + uFld[:-1,1:])
    dvdx[1:,1:] = (vFld[1:,1:] - vFld[1:,:-1]) * recip_dxV[1:,1:]
    vave[1:,1:] = 0.5 * (vFld[1:,1:] + vFld[1:,:-1])

    # evaluate strain rate at z points
    e12 = np.ones((sNy+2*OLy,sNx+2*OLx))
    hFacU = SeaIceMaskU[1:,1:] - SeaIceMaskU[:-1,1:]
    hFacV = SeaIceMaskV[1:,1:] - SeaIceMaskV[1:,:-1]
    e12[1:,1:] = 0.5 * (dudy[1:,1:] + dvdx[1:,1:] - k1AtZ[1:,1:] * vave[1:,1:] - k2AtZ[1:,1:] * uave[1:,1:]) * iceMask[1:,1:] * iceMask[1:,:-1] * iceMask[:-1,1:] * iceMask[:-1,:-1] + noSlipFac * (2 * uave[1:,1:] * recip_dyU[1:,1:] * hFacU + 2 * vave[1:,1:] * recip_dxV[1:,1:] * hFacV)

    if secondOrderBC:
        hFacU = (SeaIceMaskU[2:-1,2:-1] - SeaIceMaskU[1:-2,2:-1]) / 3
        hFacV = (SeaIceMaskV[2:-1,2:-1] - SeaIceMaskV[2:-1,1:-2]) / 3
        hFacU = hFacU * (SeaIceMaskU[:-3,2:-1] * SeaIceMaskU[1:-2,2:-1] + SeaIceMaskU[3:,2:-1] * SeaIceMaskU[2:-1,2:-1])
        hFacV = hFacV * (SeaIceMaskV[2:-1,:-3] * SeaIceMaskV[2:-1,1:-2] + SeaIceMaskV[2:-1,3:] * SeaIceMaskV[2:-1,2:-1])

        e12[2:-1,2:-1] = e12[2:-1,2:-1] + 0.5 * (recip_dyU[2:-1,2:-1] * (6 * uave[2:-1,2:-1] - uFld[:-3,2:-1] * SeaIceMaskU[1:-2,2:-1] - uFld[3:,2:-1] * SeaIceMaskU[2:-1,2:-1]) * hFacU + recip_dxV[2:-1,2:-1] * (6 * vave[2:-1,2:-1] - vFld[2:-1,:-3] * SeaIceMaskV[2:-1,1:-2] - vFld[2:-1,3:] * SeaIceMaskV[2:-1,2:-1]) * hFacV)
        

    return e11, e22, e12