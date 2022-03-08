import numpy as np

from seaice_size import *
from seaice_params import *

from seaice_solve4temp import solve4temp

# # for the 1d test
# sNx = 1
# sNy = 1
# OLx = 2
# OLy = 2

### input
# hIceMean: mean ice thickness [m3/m2] (= hIceActual * Area with Area the
#   sea ice cover fraction and hIceActual = Vol_ice / x_len y_len)
# iceMask: contains geometry of the set up
# hSnowMean: mean snow thickness [m3/m2]
# Area: sea ice cover fraction (0 <= Area <= 1)
# salt: surface salinity of the ocean [g/kg]
# TIceSnow: ice/ snow surface temp for each category [K]
# precip: precipition [m/s] (order 1e-6)
# snowPrecip: snowfall [m/s]
# evap: evaporation over ocean [m/s] (same order, =0 for now)
# runoff: runoff into ocean [m/s]   = 0 for now
# wspeed: wind speed [m/s]
# theta: potential temperature of the ocean surface [K]
# Qnet: surface net heat flux out of the ocean (open water or under ice) [W/m2]
# Qsw: surface shortwave heat flux out of the ocean (open water or under ice) [W/m2]
# SWDown: shortwave downward radiation [W/m2]
# LWDown: longwave downward radiation [W/m2]
# ATemp: atmospheric temperature [K]
# aqh: atmospheric specific humidity [g/kg]

### output
# hIceMean
# hSnowMean
# Area
# TIceSnow
# saltflux: salt flux into the ocean
# EmPmR: evaporation minus precipitation minus runoff (freshwater flux to ocean) - direction?
# Qnet
# Qsw
# seaIceLoad: sea ice + snow load on the sea surface


def growth(hIceMean, hSnowMean, Area, os_hIceMean, os_hSnowMean, salt, TIceSnow, precip,
        snowPrecip, evap, runoff, wspeed, theta, Qnet, Qsw, SWDown,
        LWDown, ATemp, aqh):

    ##### constants and initializations #####

    # XXX: salt
    # ifdef ALLOW_SALT_PLUME ...

    # heat fluxes in [W/m2]

    # sea ice/snow surface heat flux to atmosphere (+ = upward)
    # F_ia

    # net heat flux divergence at the
    # sea ice/snow surface including sea ice conductive fluxes and
    # atmospheric fluxes
    # = 0: surface heat loss is balanced by upward conductive fluxes
    # < 0: net heat flux convergence at the ice/ snow surface
    # F_ia_net

    # the net heat flux divergence at the sea ice/snow
    # surface before snow is melted with any convergence
    # < 0: some snow (if present) will melt
    # F_ia_net_before_snow

    # net upward conductive heat flux
    # through sea ice and snow realized at the sea ice/snow surface (+ = upward)
    # F_io_net

    # heat flux from atmosphere to ocean (+ = upward)
    # F_ao

    # heat flux from ocean to the ice (change of mixed layer temperature) (+ = upward)
    # F_oi

    # snow accumulation rate over ice [m/s]
    SnowAccRateOverIce = np.zeros_like(iceMask)

    # freshwater flux due to sublimation [kg/m2] (+ = upward)
    # FWsublim

    hIceActual_mult = np.zeros(np.append(iceMask.shape,nITC))
    hSnowActual_mult = np.zeros(np.append(iceMask.shape,nITC))
    F_io_net_mult = np.zeros(np.append(iceMask.shape,nITC))
    F_ia_net_mult = np.zeros(np.append(iceMask.shape,nITC))
    F_ia_mult = np.zeros(np.append(iceMask.shape,nITC))
    qswi_mult = np.zeros(np.append(iceMask.shape,nITC))
    FWsublim_mult = np.zeros(np.append(iceMask.shape,nITC))

    # constants for converting heat fluxes into growth rates
    qi = 1 / (rhoIce * lhFusion)
    qs = 1 / (rhoSnow * lhFusion)

    # regularization values squared
    area_reg_sq = SEAICE_area_reg**2
    hice_reg_sq = SEAICE_hice_reg**2


    ##### save ice, snow thicknesses and area prior to thermodynamic #####
    #####   changes and regularize thicknesses                       #####

    # set lower boundaries
    hIceMean = np.maximum(hIceMean, 0)
    Area = np.maximum(Area, 0)

    noIce = np.where((hIceMean == 0) | (Area == 0))
    Area[noIce] = 0
    hSnowMean[noIce] = 0
    hIceMean[noIce] = 0

    # store mean ice and snow thickness and sea ice cover fraction
    # (prior to any thermodynamical changes)
    hIceMeanpreTH = hIceMean.copy()
    hSnowMeanpreTH = hSnowMean.copy()
    AreapreTH = Area.copy()

    # compute actual ice and snow thickness using the regularized area
    # and add lower boundary

    # case 1: hIceMeanpreTH = 0
    hIceActual = np.zeros_like(iceMask)
    hSnowActual = np.zeros_like(iceMask)
    recip_hIceActual = np.zeros_like(iceMask)

    # case 2: hIceMeanpreTH > 0
    isIce = np.where(hIceMeanpreTH > 0)
    regAreaSqrt =  np.sqrt(AreapreTH[isIce]**2 + area_reg_sq)
    recip_regAreaSqrt = 1 / regAreaSqrt

    # ice or snow thickness divided by Area does not work if Area -> 0,
    # therefore the regularization
    hIceActual[isIce] = hIceMeanpreTH[isIce] * recip_regAreaSqrt
    hIceActual = np.maximum(hIceActual, 0.05)
    recip_hIceActual = AreapreTH / np.sqrt(hIceMeanpreTH**2 + hice_reg_sq)
    hSnowActual[isIce] = hSnowMeanpreTH[isIce] * recip_regAreaSqrt


    ##### retrieve the air-sea heat and shortwave radiative fluxes and #####
    #####   calculate the corresponding ice growth rate for open water #####

    # set wind speed
    ug = np.maximum(eps, wspeed)

    # set fluxed in (qswo) and out (F_ao) of the ocean
    F_ao = Qnet.copy()
    qswo = Qsw.copy()

    # the fraction of shortwave radiation that passes the ocean surface layer
    swFracPassTopOcean = 0

    # XXX: maybe as output for the ocean model
    # qswo_below_first_layer = qswo * swFracPassTopOcean
    qswo_in_first_layer = qswo * (1 - swFracPassTopOcean)

    # IceGrowthRateOpenWater is also defined for Area > 0 as the area can
    # be only partially covered with ice
    IceGrowthRateOpenWater = qi * (F_ao - qswo + qswo_in_first_layer)


    ##### calculate surface temperature and heat fluxes ##### 

    # record prior ice surface temperatures
    TIce_mult = TIceSnow.copy()

    for l in range(0, nITC):
        # set relative thickness of ice and snow categories
        pFac = (2 * (l + 1) - 1) * recip_nITC
        # find actual snow and ice thickness within each category
        hIceActual_mult[:,:,l] = hIceActual * pFac
        hSnowActual_mult[:,:,l] = hSnowActual * pFac

    # calculate freezing temperature
    TempFrz = tempFrz0 + dTempFrz_dS * salt

    # calculate heat fluxes and ice/ snow surface temperature
    for l in range(0, nITC):
        TIce_mult[:,:,l], F_io_net_mult[:,:,l], F_ia_net_mult[:,:,l], \
            F_ia_mult[:,:,l], qswi_mult[:,:,l], FWsublim_mult[:,:,l] = (
        solve4temp(hIceActual_mult[:,:,l], hSnowActual_mult[:,:,l],
            TIce_mult[:,:,l], TempFrz, ug, SWDown, LWDown, ATemp, aqh))


    ##### evaluate precipitation as snow or rain #####

    # the precipitation rate over the ice which goes immediately into the
    # ocean (flowing through cracks in the ice). if the temperature is
    # above the freezing point, the precipitation remains wet and runs
    # into the ocean
    PrecipRateOverIceSurfaceToSea = precip.copy()

    # if there is ice and the temperature is below the freezing point,
    # the precipitation falls and accumulates as snow:
    tmp = np.where((AreapreTH > 0) & (TIce_mult[:,:,0] < celsius2K))
    SnowAccRateOverIce[tmp] = snowPrecip[tmp] * rhoFresh2rhoSnow
    PrecipRateOverIceSurfaceToSea[tmp] = 0

    # total snow accumulation over ice [m]
    SnowAccOverIce = SnowAccRateOverIce * AreapreTH * deltaTtherm


    ##### for every thickness category, record the ice surface #####
    #####  temperature and find the average flux across it     #####

    # update surface temperature and fluxes
    # multplying the fluxes with the area changes them from mean fluxes
    # for the ice part of the cell to mean fluxes for the whole cell
    TIceSnow = TIce_mult.copy()
    for i in range(nITC):
        TIceSnow[:,:,i] = np.where(hIceMean==0,np.nan,TIceSnow[:,:,i])
    F_io_net = np.sum(F_io_net_mult*recip_nITC, axis=2) * AreapreTH
    F_ia_net = np.sum(F_ia_net_mult*recip_nITC, axis=2) * AreapreTH
    F_ia = np.sum(F_ia_mult*recip_nITC, axis=2) * AreapreTH
    qswi = np.sum(qswi_mult*recip_nITC, axis=2) * AreapreTH
    FWsublim = np.sum(FWsublim_mult*recip_nITC, axis=2) * AreapreTH

    import matplotlib.pyplot as plt

    plt.pcolormesh(F_io_net)
    plt.colorbar()
    plt.show()


    ##### calculate growth rates of ice and snow #####

    # the ice growth rate beneath ice is given by the upward conductive
    # flux F_io_net and qi:
    IceGrowthRateUnderExistingIce = F_io_net * qi

    # the potential snow melt rate if all snow surface heat flux
    # convergence (F_ia_net < 0) goes to melting snow [m/s]
    PotSnowMeltRateFromSurf = - F_ia_net * qs

    # the thickness of snow that can be melted in one time step:
    PotSnowMeltFromSurf = PotSnowMeltRateFromSurf * deltaTtherm

    noPriorArea = np.where(AreapreTH == 0)
    IceGrowthRateUnderExistingIce[noPriorArea] = 0

    # if the heat flux convergence could melt more snow than is actually
    # there, the excess is used to melt ice

    # case 1: snow will remain after melting, i.e. all of the heat flux
    # convergence will be used up to melt snow

    # the actual surface heat flux convergence used to melt snow [W/m2]
    SurfHeatFluxConvergToSnowMelt = F_ia_net.copy()

    # the actual snow melt rate due to snow surface heat flux convergence [m/s]
    SnowMeltRateFromSurface = PotSnowMeltRateFromSurf.copy()

    # the actual thickness of snow to be melted by snow surface
    # heat flux convergence [m]
    SnowMeltFromSurface = PotSnowMeltFromSurf.copy()

    # case 2: all snow will be melted if the potential snow melt height is
    # larger or equal to the actual snow height. if there is an excess of
    # heat flux convergence after snow melting, it will be used to melt ice

    allSnowMelted = np.where(PotSnowMeltFromSurf >= hSnowMean)
    SnowMeltFromSurface[allSnowMelted] = hSnowMean[allSnowMelted]
    SnowMeltRateFromSurface[allSnowMelted] = \
        SnowMeltFromSurface[allSnowMelted] * recip_deltaTtherm
    SurfHeatFluxConvergToSnowMelt[allSnowMelted] = \
        - hSnowMean[allSnowMelted]         * recip_deltaTtherm / qs

    # the surface heat flux convergence is reduced by the amount that
    # is used for melting snow:
    F_ia_net = F_ia_net - SurfHeatFluxConvergToSnowMelt

    # the remaining heat flux convergence is used to melt ice:
    IceGrowthRateFromSurface = F_ia_net * qi

    # the total ice growth rate is then:
    NetExistingIceGrowthRate = IceGrowthRateUnderExistingIce + IceGrowthRateFromSurface


    ##### calculate the heat fluxes from the ocean to the sea ice #####

    tmpscal0 = 0.4
    tmpscal1 = 7 / tmpscal0
    tmpscal2 = stantonNr * uStarBase * rhoConst * heatCapacity

    # the ocean temperature cannot be lower than the freezing temperature
    surf_theta = np.maximum(theta, TempFrz)

    # mltf = mixed layer turbulence factor (determines how much of the temperature
    # difference is used for heat flux)
    mltf = 1 + (McPheeTaperFac - 1) / (1 + np.exp((AreapreTH - tmpscal0) * tmpscal1))

    F_oi = - tmpscal2 * (surf_theta - TempFrz) * mltf
    IceGrowthRateMixedLayer = F_oi * qi


    ##### calculate change in ice, snow thicknesses and area #####

    # calculate thickness derivatives of ice and snow
    dhIceMean_dt = NetExistingIceGrowthRate * AreapreTH + \
        IceGrowthRateOpenWater * (1 - AreapreTH) + IceGrowthRateMixedLayer
    dhSnowMean_dt = (SnowAccRateOverIce - SnowMeltRateFromSurface) * AreapreTH

    # XXX: salt
    # ifdef allow_salt_plume ...

    tmpscal0 =  0.5 * recip_hIceActual

    # ice growth open water (due to ocean-atmosphere fluxes)
    # reduce ice cover if the open water growth rate is negative
    dArea_oaFlux = tmpscal0 * IceGrowthRateOpenWater * (1 - AreapreTH)

    # increased ice cover if the open water growth rate is positive
    tmp = np.where((IceGrowthRateOpenWater > 0) & (
        (AreapreTH > 0) | (dhIceMean_dt > 0)))

    dArea_oaFlux[tmp] = IceGrowthRateOpenWater[tmp] \
        * (1 - AreapreTH[tmp])

    dArea_oaFlux[tmp] = np.where(fCori[tmp] < 0,
        dArea_oaFlux[tmp] * recip_h0_south,
        dArea_oaFlux[tmp] * recip_h0)

    # ice growth mixed layer (due to ocean-ice fluxes)
    # (if the ocean is warmer than the ice IceGrowthRateMixedLayer > 0.
    # the supercooled state of the ocean is ignored/ does not lead to ice
    # growth. ice growth is only due to fluxes calculated by solve4temp)
    dArea_oiFlux = np.where(IceGrowthRateMixedLayer <= 0,
        tmpscal0 * IceGrowthRateMixedLayer, 0)

    # ice growth over ice (due to ice-atmosphere fluxes)
    # (NetExistingIceGrowthRate leads to vertical and lateral melting but
    # only to vertical growing. lateral growing is covered by
    # IceGrowthRateOpenWater)
    dArea_iaFlux = np.where((NetExistingIceGrowthRate <= 0) & (hIceMeanpreTH > 0),
        tmpscal0 * NetExistingIceGrowthRate * AreapreTH, 0)

    # total change in area
    dArea_dt = dArea_oaFlux + dArea_oiFlux + dArea_iaFlux


    # import matplotlib.pyplot as plt
    # plt.contourf(dhIceMean_dt[OLy:-OLy,OLx:-OLx])
    # plt.colorbar()
    # plt.show()

    ######  update ice, snow thickness and area #####

    Area = AreapreTH + dArea_dt * iceMask * deltaTtherm
    hIceMean = hIceMeanpreTH + dhIceMean_dt * iceMask * deltaTtherm
    hSnowMean = hSnowMeanpreTH + dhSnowMean_dt * iceMask * deltaTtherm



    # set boundaries:
    Area = np.clip(Area, 0, 1)
    hIceMean = np.maximum(hIceMean, 0)
    hSnowMean = np.maximum(hSnowMean, 0)

    noIce = np.where((hIceMean == 0) | (Area == 0))
    Area[noIce] = 0
    hSnowMean[noIce] = 0
    hIceMean[noIce] = 0

    # change of ice thickness due to conversion of snow to ice if snow
    # is submerged with water
    tmpscal0 = (hSnowMean * rhoSnow + hIceMean * rhoIce) * recip_rhoConst
    d_hIceMeanByFlood = np.maximum(0, tmpscal0 - hIceMean)
    hIceMean = hIceMean + d_hIceMeanByFlood
    hSnowMean = hSnowMean - d_hIceMeanByFlood * rhoice2rhosnow


    ##### calculate output to ocean #####

    # effective shortwave heating rate
    Qsw = qswi * AreapreTH + qswo * (1 - AreapreTH)

    # the actual ice volume change over the time step [m3/m2]
    ActualNewTotalVolumeChange = hIceMean - hIceMeanpreTH

    # the net melted snow thickness [m3/m2] (positive if the snow
    # thickness decreases/ melting occurs)
    ActualNewTotalSnowMelt = hSnowMeanpreTH + SnowAccOverIce - hSnowMean

    # the energy required to melt or form the new ice volume [J/m2]
    EnergyInNewTotalIceVolume = ActualNewTotalVolumeChange / qi

    # the net energy flux out of the ocean [J/m2]
    NetEnergyFluxOutOfOcean = (AreapreTH * (F_ia_net + F_io_net + qswi)
        + (1 - AreapreTH) * F_ao) * deltaTtherm

    # energy taken out of the ocean which is not used for sea ice growth [J].
    # If the net energy flux out of the ocean is balanced by the latent
    # heat of fusion, the temperature of the mixed layer will not change
    ResidualEnergyOutOfOcean = NetEnergyFluxOutOfOcean - EnergyInNewTotalIceVolume

    # total heat flux out of the ocean [W/m2]
    Qnet = ResidualEnergyOutOfOcean * recip_deltaTtherm

    # the freshwater contribution to (from) the ocean from melting (growing)
    # ice [m3/m2] (positive for melting)
    FreshwaterContribFromIce = - ActualNewTotalVolumeChange * rhoIce2rhoFresh

    # in the case of non-zero ice salinity, the freshwater contribution
    # is reduced by the salinity ration of ice to water
    saltInWater = np.where((salt > 0) & (salt > saltIce))
    FreshwaterContribFromIce[saltInWater] = - ActualNewTotalVolumeChange[saltInWater] \
        * rhoIce2rhoFresh * (1 - saltIce/salt[saltInWater])

    # if the liquid cell has a lower salinity than the specified salinity
    # of sea ice, then assume the sea ice is completely fresh
    # (if the water is fresh, no salty sea ice can form)

    tmpscal0 = np.minimum(saltIce, salt)
    saltflux = (ActualNewTotalVolumeChange + os_hIceMean) * tmpscal0 \
        * iceMask * rhoIce * recip_deltaTtherm

    # the freshwater contribution to the ocean from melting snow [m]
    FreshwaterContribFromSnowMelt = ActualNewTotalSnowMelt / rhoFresh2rhoSnow

    # evaporation minus precipitation minus runoff (freshwater flux to ocean)
    EmPmR = iceMask *  ((evap - precip) * (1 - AreapreTH) \
        - PrecipRateOverIceSurfaceToSea * AreapreTH - runoff - (
        FreshwaterContribFromIce + FreshwaterContribFromSnowMelt) \
            / deltaTtherm) * rhoFresh + iceMask * (
        os_hIceMean * rhoIce + os_hSnowMean * rhoSnow) \
            * recip_deltaTtherm

    # sea ice + snow load on the sea surface
    seaIceLoad = hIceMean * rhoIce + hSnowMean * rhoSnow
    # XXX: maybe introduce a cap if needed for ocean model



    return hIceMean, hSnowMean, Area, TIceSnow, saltflux, EmPmR, \
        Qsw, Qnet, seaIceLoad