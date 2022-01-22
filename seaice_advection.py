import numpy as np

from seaice_size import *
from seaice_params import *

from seaice_fluxlimit_adv_x import fluxlimit_adv_x
from seaice_fluxlimit_adv_y import fluxlimit_adv_y
from seaice_fill_overlap import fill_overlap

# calculates the tendency of a sea ice field due to advection

### input
# uFld: zonal velocity component
# vFld: meridional velocity component
# uTrans: volume transport at U points
# vTrans: volume transport at V points
# iceFld: sea ice field (mean thickness of ice or snow, or area)
# r_hFld: reciprocal of ice thickness
# extensiveFld: indicates to advect an "extensive" type of ice field

### output
# gFld: advection tendency


def advection(uFld, vFld, uTrans, vTrans, iceFld, r_hFld, extensiveFld):

    # make local copy of sea-ice field
    localTij = iceFld.copy()

    # mask West & South
    # ifdef ALLOW_OBCS
    maskLocW = SeaIceMaskU * maskInW
    maskLocS = SeaIceMaskV * maskInS


    ##### calculate advective flux in x direction #####

    # advective flux in x direction
    afx = fluxlimit_adv_x(uFld, localTij, uTrans, deltaTtherm, maskLocW)

    # update the local seaice field
    if extensiveFld:
        localTij[:,1:-1] = localTij[:,1:-1] - deltaTtherm * maskInC[:,1:-1] * recip_rA[:,1:-1] * (afx[:,2:] - afx[:,1:-1])
    else:
        localTij[:,1:-1] = localTij[:,1:-1] - deltaTtherm * maskInC[:,1:-1] * recip_rA[:,1:-1] * r_hFld[:,1:-1] * ((afx[:,2:] - afx[:,1:-1]) - (uTrans[:,2:] - uTrans[:,1:-1]) * iceFld[:,1:-1])


    ##### calculate advective flux in y direction #####

    # advective flux in y direction
    afy = fluxlimit_adv_y(vFld, localTij, vTrans, deltaTtherm, maskLocS)

    # update the local seaice field
    if extensiveFld:
        localTij[1:-1,:] = localTij[1:-1,:] - deltaTtherm * maskInC[1:-1,:] * recip_rA[1:-1,:] * (afy[2:,:] - afy[1:-1,:])
    else:
        localTij[1:-1,:] = localTij[1:-1,:] - deltaTtherm * maskInC[1:-1,:] * recip_rA[1:-1,:] * r_hFld[1:-1,:] * ((afy[2:,:] - afy[1:-1,:]) - (vTrans[2:,:] - vTrans[1:-1,:]) * iceFld[1:-1,:])


    # explicit advection is done, store tendency in gFld
    gFld = (localTij - iceFld) / deltaTtherm
    gFld = fill_overlap(gFld)


    return gFld