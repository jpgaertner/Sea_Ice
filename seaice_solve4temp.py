#Calculates heat fluxes and temperature of the ice surface (see Hibler, MWR, 108, 1943-1973, 1980)

import numpy as np
from seaice_size import *
from seaice_params import *

sNx = 1
sNy = 1
OLx = 2
OLy = 2

### input
#hIceActual     : actual ice thickness [m]
#hSnowActual    : actual snow thickness [m]
#TSurfIn        : surface temperature of ice/snow [K]
#TempFrz        : freezing temperature [K]
#ug             : atmospheric wind speed [m/s]
#SWDown         : shortwave radiative downward flux [W/m2]
#LWDown         : longwave radiative downward flux [W/m2]
#ATemp          : atmospheric temperature [K]
#aqh            : atmospheric specific humidity [g/kg]

### output
#TSurfOut       : updated surface temperature of ice/snow [K]
#F_io_net       : upward conductive heat flux through sea ice and snow [W/m2]
#F_ia_net       : net heat flux divergence at the sea ice/snow surface
#                   including ice conductive fluxes and atmospheric fluxes [W/m2]
#F_ia           : upward seaice/snow surface heat flux to atmosphere [W/m^2]
#IcePenetSW     : short wave heat flux arriving at the ocean-ice interface (+ = upward) [W/m^2]
#FWsublim       : fresh water (mass) flux due to sublimation (+ = upward) [kg/sm2]


def solve4temp(hIceActual, hSnowActual, TSurfIn, TempFrz, ug, SWDown, LWDown, ATemp, aqh):


    ##### define local constants used for calculations #####

    # coefficients for the saturation vapor pressure equation
    aa1 = 2663.5
    aa2 = 12.537
    bb1 = 0.622
    bb2 = 1 - bb1
    Ppascals = 100000
    lnTen = np.log(10)
    cc0 = 10**aa2
    cc1 = cc0 * aa1 * bb1 * Ppascals * lnTen
    cc2 = cc0 * bb2
    
    d1 = seaice_dalton * cpAir * rhoAir #sensible heat constant
    d1i = seaice_dalton * lhSublim * rhoAir #ice latent heat constant

    Tmelt = celsius2K #melting temperature of ice

    SurfMeltTemp = Tmelt + wetAlbTemp #temperature threshold for wet albedo


    ##### initializations #####

    # initialize output arrays
    TSurfOut = TSurfIn.copy()
    F_ia = np.zeros((sNx+2*OLx, sNy+2*OLy))
    #F_ia_net
    F_io_net = np.zeros((sNx+2*OLx, sNy+2*OLy))
    IcePenetSW = np.zeros((sNx+2*OLx, sNy+2*OLy)) #the shortwave radiative flux at the ocean-ice interface (+ = upwards)
    FWsublim = np.zeros((sNx+2*OLx, sNy+2*OLy))
    
    iceOrNot = (hIceActual > 0)
    
    effConduct = np.zeros((sNx+2*OLx, sNy+2*OLy)) #effective conductivity of ice and snow combined
    dFia_dTs = np.zeros((sNx+2*OLx, sNy+2*OLy)) #derivative of F_ia w.r.t snow/ice surf. temp
    absorbedSW = np.zeros((sNx+2*OLx, sNy+2*OLy)) #shortwave radiative flux convergence in the sea ice
    qhice = np.zeros((sNx+2*OLx, sNy+2*OLy)) #saturation vapor pressure of snow/ice surface
    dqh_dTs = np.zeros((sNx+2*OLx, sNy+2*OLy)) #derivative of qhice w.r.t snow/ice surf. temp
    F_lh = np.zeros((sNx+2*OLx, sNy+2*OLy)) #latent heat flux (sublimation) (+ = upward)
    F_lwu = np.zeros((sNx+2*OLx, sNy+2*OLy)) #upward long-wave surface heat flux (+ = upward)
    F_sens = np.zeros((sNx+2*OLx, sNy+2*OLy)) #sensible surface heat flux (+ = upward)
    F_c = np.zeros((sNx+2*OLx, sNy+2*OLy)) #conductive heat flux through ice and snow (+ = upward)

    # make local copies of downward longwave radiation, surface and atmospheric temperatures
    TSurfLoc = TSurfIn.copy()
    #TSurfLoc = np.minimum(celsius2k + maxTIce, TSurfIn)
    LWDownLoc = np.maximum(minLwDown, LWDown)
    ATempLoc = np.maximum(celsius2K + minTAir, ATemp)


    ##### determine fixed (relative to surface temperature) forcing term in heat budget #####

    isSnow = np.where(hSnowActual > 0)
    isIce = np.where(iceOrNot == True)

    d3 = np.ones((sNx+2*OLx,sNy+2*OLy)) * iceEmiss * stefBoltz
    d3[isSnow] = snowEmiss * stefBoltz

    LWDownLoc[isIce] = iceEmiss * LWDownLoc[isIce]
    LWDownLoc[isSnow] = snowEmiss * LWDownLoc[isSnow]


    ##### determine albedo #####

    albIce = np.zeros((sNx+2*OLx, sNy+2*OLy))
    albSnow = np.zeros((sNx+2*OLx, sNy+2*OLy))
    alb = np.zeros((sNx+2*OLx, sNy+2*OLy))

    for j in range(0, sNy+2*OLy):
        for i in range(0, sNx+2*OLx):
            if (iceOrNot[i,j]):
                if fCori[i,j] < 0:
                    if TSurfLoc[i,j] >= SurfMeltTemp:
                        albIce[i,j] = wetIceAlb_south
                        albSnow[i,j] = wetSnowAlb_south
                    else:
                        albIce[i,j] = dryIceAlb_south
                        albSnow[i,j] = drySnowAlb_south
                else:
                    if TSurfLoc[i,j] >= SurfMeltTemp:
                        albIce[i,j] = wetIceAlb
                        albSnow[i,j] = wetSnowAlb
                    else:
                        albIce[i,j] = dryIceAlb
                        albSnow[i,j] = drySnowAlb

                if hSnowActual[i,j] > hCut:
                # shortwave optically thick snow
                    alb[i,j] = albSnow[i,j]
                elif hSnowActual[i,j] == 0:
                    alb[i,j] = albIce[i,j]
                else:
                # use linear transition between snow and ice albedo
                    alb[i,j] = albIce[i,j] + hSnowActual[i,j] / hCut * (albSnow[i,j] - albIce[i,j])


    ##### determine the shortwave radiative flux arriving at the     #####
    #####  ice-ocean interface after scattering through snow and ice #####

                # if snow is present, all radiation is absorbed in the ice
                if hSnowActual[i,j] > 0:
                    penetSWFrac = 0
                else:
                    penetSWFrac = shortwave * np.exp(-1.5 * hIceActual[i,j])
                #the shortwave radiative flux at the ocean-ice interface (+ = upward)
                IcePenetSW[i,j] = -(1 - alb[i,j]) * penetSWFrac * SWDown[i,j]
                #the shortwave radiative flux convergence in the ice
                absorbedSW[i,j] = (1 - alb[i,j]) * (1 - penetSWFrac) * SWDown[i,j]
                
                # calculate the effective conductivity of the snow-ice system
                effConduct[i,j] = iceConduct * snowConduct / (snowConduct * hIceActual[i,j] + iceConduct * hSnowActual[i,j])


    ##### calculate the fluxes #####

    t1 = TSurfLoc[isIce]
    def fluxes(t1):

        t2 = t1 * t1
        t3 = t2 * t1
        t4 = t2 * t2

        # calculate the saturation vapor pressure in the snow/ ice-atmosphere boundary layer
        mm_log10pi = - aa1 / t1 + aa2
        mm_pi = 10**mm_log10pi

        qhice[isIce] = bb1 * mm_pi / (Ppascals - (1 - bb1) * mm_pi)
        # a constant for the saturation vapor pressure derivative
        cc3t = 10**(aa1 / t1)
        dqh_dTs[isIce] = cc1 * cc3t / ((cc2 - cc3t * Ppascals)**2 * t2)

        # calculate the fluxes based on the surface temperature
        F_c[isIce] = effConduct[isIce] * (TempFrz[isIce] - t1)
        F_lh[isIce] = d1i * ug[isIce] * (qhice[isIce] - aqh[isIce])

        F_lwu[isIce] = t4 * d3[isIce]
        F_sens[isIce] = d1 * ug[isIce] * (t1 - ATempLoc[isIce])
        F_ia[isIce] = - LWDownLoc[isIce] - absorbedSW[isIce] + F_lwu[isIce] + F_sens[isIce] + F_lh[isIce]
        dFia_dTs[isIce] = 4 * d3[isIce] * t3 + d1 * ug[isIce] + d1i * ug[isIce] * dqh_dTs[isIce]

        return F_c, F_lh, F_ia, dFia_dTs

    # save TSurfLoc because it is changed for finding the fluxes
    TSurfIter = TSurfLoc.copy()

    for i in range(5):
        F_c, F_lh, F_ia, dFia_dTs = fluxes(t1)

        # update surface temperature as solution of F_c = F_ia + d/dT (F_c - F_ia) * delta T
        TSurfIter[isIce] = TSurfIter[isIce] + (F_c[isIce] - F_ia[isIce]) / (effConduct[isIce] + dFia_dTs[isIce])
        #TSurfLoc = np.maximum(TSurfLoc, celsius2K + minTIce)
        #TSurfLoc = np.minimum(TSurfLoc, Tmelt)

        # recalculate the fluxes based on the adjusted surface temperature
        t1 = TSurfIter[isIce]

    TSurfLoc[isIce] = TSurfLoc[isIce] + (F_c[isIce] - F_ia[isIce]) / (effConduct[isIce] + dFia_dTs[isIce])

    # case 1: F_c <= 0
    # F_io_net is already set up as zero everywhere
    F_ia_net = F_ia.copy()
    # case 2: F_c > 0
    upCondflux = np.where(F_c > 0)
    F_io_net[upCondflux] = F_c[upCondflux]
    F_ia_net[upCondflux] = 0

    # save updated surface temperature as output and finalize the flux terms
    TSurfOut[isIce] = TSurfLoc[isIce]
    FWsublim[isIce] = F_lh[isIce] / lhSublim


    return TSurfOut, F_io_net, F_ia_net, F_ia, IcePenetSW, FWsublim