import numpy as np

from seaice_size import *
from seaice_params import *

from seaice_solve4temp import solve4temp

sNx = 1
sNy = 1
OLx = 2
OLy = 2

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


def growth(hIceMean, hSnowMean, Area, salt, TIceSnow, precip,
        snowPrecip, evap, runoff, wspeed, theta, Qnet, Qsw, SWDown,
        LWDown, ATemp, aqh):

    ##### constants and initializations #####

    # #ifdef ALLOW_SALT_PLUME
    # IceGrowthRateInLeads #d(hIceMean)/dt from heat fluxes in the open water fraction of the grid cell
    # leadPlumeFraction #The fraction of salt released in leads by new ice production there
    # #                                           which is to be sent to the salt plume package


    # heat fluxes in [W/m2]

    # sea ice/snow surface heat flux to atmosphere (+ = upward)
    # F_ia

    # net heat flux divergence at the
    # sea ice/snow surface including sea ice conductive fluxes and
    # atmospheric fluxes
    # = 0: surface heat loss is balanced by upward conductive fluxes
    # < 0: net heat flux convergence at the ice/ snow surface
    F_ia_net = np.zeros_like(iceMask)

    # the net heat flux divergence at the sea ice/snow
    # surface before snow is melted with any convergence
    # < 0: some snow (if present) will melt
    # F_ia_net_before_snow

    # net upward conductive heat flux
    # through sea ice and snow realized at the sea ice/snow surface (+ = upward)
    F_io_net = np.zeros_like(iceMask)

    # heat flux from atmosphere to ocean (+ = upward)
    # F_ao

    # heat flux from ocean to the ice (change of mixed layer temperature) (+ = upward)
    F_oi = np.zeros_like(iceMask)


    # sea ice growth rates in [m/s]

    NetExistingIceGrowthRate = np.zeros_like(iceMask)
    IceGrowthRateUnderExistingIce = np.zeros_like(iceMask)
    IceGrowthRateFromSurface = np.zeros_like(iceMask)
    IceGrowthRateMixedLayer = np.zeros_like(iceMask)

    # change of ice thickness due to conversion of snow to ice if snow
    # is submerged with water [m]
    d_hIceMeanByFlood = np.zeros_like(iceMask)

    # the energy taken out of the ocean which is used for sea ice growth [J]
    ResidualEnergyOutOfOcean = np.zeros_like(iceMask)

    # snow accumulation rate over ice [m/s]
    SnowAccRateOverIce = np.zeros_like(iceMask)

    # total snow accumulation over ice [m]
    SnowAccOverIce = np.zeros_like(iceMask)

    # the precipitation rate over the ice which goes immediately into
    # the ocean (flowing through cracks in the ice)
    PrecipRateOverIceSurfaceToSea = np.zeros_like(iceMask)

    # the potential snow melt rate if all snow surface heat flux
    # convergences goes to melting snow [m/s]
    PotSnowMeltRateFromSurf = np.zeros_like(iceMask)

    # the potential thickness of snow which could be melted by snow
    # surface heat flux convergence [m]
    PotSnowMeltFromSurf = np.zeros_like(iceMask)

    # the actual snow melt rate due to snow surface heat flux convergence [m/s]
    SnowMeltRateFromSurface = np.zeros_like(iceMask)

    # the actual surface heat flux convergence used to melt snow [W/m2]
    SurfHeatFluxConvergToSnowMelt = np.zeros_like(iceMask)

    # the actual thickness of snow to be melted by snow surface
    # heat flux convergence [m]
    SnowMeltFromSurface = np.zeros_like(iceMask)

    # the freshwater contribution to the ocean from melting snow [m]
    FreshwaterContribFromSnowMelt = np.zeros_like(iceMask)

    # the freshwater contribution to (from) the ocean from melting (growing) ice [m]
    FreshwaterContribFromIce = np.zeros_like(iceMask)

    dArea_dt = np.zeros_like(iceMask)

    # reshwater flux due to sublimation [kg/m2] (+ = upward)
    # FWsublim

    hIceActual_mult = np.zeros(np.append(iceMask.shape,nITC))
    hSnowActual_mult = np.zeros(np.append(iceMask.shape,nITC))
    F_io_net_mult = np.zeros(np.append(iceMask.shape,nITC))
    F_ia_net_mult = np.zeros(np.append(iceMask.shape,nITC))
    F_ia_mult = np.zeros(np.append(iceMask.shape,nITC))
    qswi_mult = np.zeros(np.append(iceMask.shape,nITC))
    FWsublim_mult = np.zeros(np.append(iceMask.shape,nITC))

    # d(Area)/dt due to open water fluxes
    dArea_oaFlux = np.zeros_like(iceMask)

    # d(Area)/dt due to ocean-ice fluxes
    dArea_oiFlux = np.zeros_like(iceMask)

    # d(Area)/dt due to ice-atmosphere fluxes
    dArea_iaFlux = np.zeros_like(iceMask)

    # Cutoff for iceload, can be left out for now

    # constants for converting heat fluxes into growth rates
    qi = 1 / (rhoIce * lhFusion)
    qs = 1 / (rhoSnow * lhFusion)

    # regularization values squared
    area_reg_sq = SEAICE_area_reg**2
    hice_reg_sq = SEAICE_hice_reg**2


    ##### Store ice and snow state on onset and    ######
    ##### regularize actual snow and ice thickness ######

    # placeholders for extra bits from advection, used in budget
    d_hIcebyDyn = np.ones_like(iceMask)
    d_hSnowbyDyn = np.ones_like(iceMask)
    #d_hIcebyDyn (d_HEFFbyNEG in F) set up empty in seaice.h
    SIhIceMeanNeg = d_hIcebyDyn * SINegFac
    SIhSnowMeanNeg = d_hSnowbyDyn * SINegFac

    # set lower boundaries
    hIceMean = np.clip(hIceMean, 0, None)
    Area = np.clip(Area, 0, None)

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

    #case 2: hIceMeanpreTH > 0
    isIce = np.where(hIceMeanpreTH > 0)
    regAreaSqrt =  np.sqrt(AreapreTH[isIce]**2 + area_reg_sq)
    recip_regAreaSqrt = 1 / regAreaSqrt
    # hIceMeanpreTh / Area does not work if Area = 0, therefore the regularization
    hIceActual[isIce] = hIceMeanpreTH[isIce] * recip_regAreaSqrt
    hIceActual = np.clip(hIceActual, 0.05, None)
    recip_hIceActual = AreapreTH / np.sqrt(hIceMeanpreTH**2 + hice_reg_sq)
    #hSnowActual[isIce] = hSnowMeanpreTH[isIce] * recip_regAreaSqrt
    hSnowActual[isIce] = hSnowMeanpreTH[isIce] / AreapreTH[isIce]


    ##### Retrieve the air-sea heat and shortwave radiative fluxes        #####
    #####  and calculate the corresponding ice growth rate for open water #####

    # set wind speed
    ug = np.maximum(eps, wspeed)

    # set fluxed in (qswo) and out (F_ao) of the ocean
    F_ao = Qnet.copy()
    qswo = Qsw.copy()
    qswi = np.zeros_like(iceMask)

    swFracAbsTopOcean = 0 #-> qswo_in_first_layer = qswo? (l. 177)

    #qswo_below_first_layer = qswo * swFracAbsTopOcean
    qswo_in_first_layer = qswo * (1 - swFracAbsTopOcean)

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

    # if the temperature is above the freezing point, the precipitation
    # remains wet and runs into the ocean
    PrecipRateOverIceSurfaceToSea = precip.copy()

    # if there is ice and the temperature is below the freezing point,
    # the precipitation falls and accumulates as snow:
    tmp = np.where((AreapreTH > 0) & (TIce_mult[:,:,0] < celsius2K))
    SnowAccRateOverIce[tmp] = snowPrecip[tmp] * rhoFresh2rhoSnow
    PrecipRateOverIceSurfaceToSea[tmp] = 0

    SnowAccOverIce = SnowAccRateOverIce * AreapreTH * deltaTtherm


    ##### for every thickness category, record the ice surface #####
    #####  temperature and find the average flux across it     #####

    # update surface temperature and fluxes
    TIceSnow = TIce_mult.copy()
    for i in range(nITC):
        TIceSnow[:,:,i] = np.where(hIceMean==0,np.nan,TIceSnow[:,:,i])
    F_io_net = np.sum(F_io_net_mult*recip_nITC, axis=2)
    F_ia_net = np.sum(F_ia_net_mult*recip_nITC, axis=2)
    F_ia = np.sum(F_ia_mult*recip_nITC, axis=2)
    qswi = np.sum(qswi_mult*recip_nITC, axis=2)
    FWsublim = np.sum(FWsublim_mult*recip_nITC, axis=2)

    # the ice growth rate beneath ice is given by the upward conductive
    # flux F_io_net and qi:
    IceGrowthRateUnderExistingIce = F_io_net * qi

    # if there is a net heat flux convergence at the sea ice/snow surface
    # (F_ia_net < 0), snow can be melted:
    PotSnowMeltRateFromSurf = - F_ia_net * qs

    # the depth of snow that can be melted in one time step:
    PotSnowMeltFromSurf = PotSnowMeltRateFromSurf * deltaTtherm

    noPriorArea = np.where(AreapreTH == 0)
    IceGrowthRateUnderExistingIce[noPriorArea] = 0
    IceGrowthRateFromSurface[noPriorArea] = 0
    NetExistingIceGrowthRate[noPriorArea] = 0

    # if the heat flux convergence could melt more snow than is actually
    # there, the excess is used to melt ice:

    #HSNOW ACTUAL SHOULD NOT BE REGULARIZED FOR THIS (this is done by
    # using hSnowMean instead of hSnowActual)

    # leave it for now but try to do it without the regularization once
    # the program is running: multiplicate PotSnowMeltFromSurf with
    # the area, then compare with hSnowMean
    # when calculating dhIceMean_dt
    # and dhSnowMean_dt, the factor AreaPreTh can be dropped
    
    # case 1: snow will remain after melting, i.e. all of the heat flux
    # convergence will be used up to melt snow
    SnowMeltFromSurface = PotSnowMeltFromSurf.copy()
    SnowMeltRateFromSurface = PotSnowMeltRateFromSurf.copy()
    SurfHeatFluxConvergToSnowMelt = F_ia_net.copy()

    # case 2: all snow will be melted if the potential snow melt height is
    # larger or equal to the actual snow height. If there is an excess of
    # heat flux convergence after snow melting, it will be used to melt ice
    allSnowMelted = np.where(PotSnowMeltFromSurf >= hSnowActual)
    SnowMeltFromSurface[allSnowMelted] = hSnowActual[allSnowMelted]
    SnowMeltRateFromSurface[allSnowMelted] = \
        SnowMeltFromSurface[allSnowMelted] * recip_deltaTtherm
    SurfHeatFluxConvergToSnowMelt[allSnowMelted] = \
        - hSnowActual[allSnowMelted]       * recip_deltaTtherm / qs
    
    # the surface heat flux convergence is reduced by the amount that
    # is used for melting snow:
    F_ia_net = F_ia_net - SurfHeatFluxConvergToSnowMelt

    # the remaining heat flux convergence is used to melt ice:
    IceGrowthRateFromSurface = F_ia_net * qi

    # the total ice growth rate is then:
    NetExistingIceGrowthRate = IceGrowthRateUnderExistingIce + IceGrowthRateFromSurface
    

    ##### calculate the heat fluxes from the ocean to the sea ice #####

    tmpscal0 = 0.4 #inflection point
    tmpscal1 = 7 / tmpscal0 #steepness/ inflection point
    tmpscal2 = stantonNr * uStarBase * rhoConst * heatCapacity

    # the ocean temperature cannot be lower than the freezing temperature
    surf_theta = np.maximum(theta, TempFrz)

    # mltf = mixed layer turbulence factor (determines how much of the temperature
    # difference is used for heat flux)
    mltf = 1 + (McPheeTaperFac - 1) / (1 + np.exp((AreapreTH - tmpscal0) * tmpscal1))

    F_oi = - tmpscal2 * (surf_theta - TempFrz) * mltf
    IceGrowthRateMixedLayer = F_oi * qi


    ##### calculate d(Area)/dt #####

    # calculate thickness derivatives of ice and snow
    dhIceMean_dt = NetExistingIceGrowthRate * AreapreTH + \
        IceGrowthRateOpenWater * (1 - AreapreTH) + IceGrowthRateMixedLayer
    dhSnowMean_dt = (SnowAccRateOverIce - SnowMeltRateFromSurface) * AreapreTH

    # ifdef allow_salt_plume
    # ifdef salt_plume_in_leads
    # calculate leadPlumeFraction, IceGrowthRateInLeads and saltPlumeFlux

    tmpscal0 =  0.5 * recip_hIceActual

    # ice growth open water
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

    # ice growth mixed layer
    dArea_oiFlux = np.where(IceGrowthRateMixedLayer <= 0,
        tmpscal0 * IceGrowthRateMixedLayer, 0)

    # ice growth over ice (from ice atmosphere fluxes)
    dArea_iaFlux = np.where((NetExistingIceGrowthRate<= 0) & (hIceMeanpreTH > 0),
        tmpscal0 * NetExistingIceGrowthRate * AreapreTH, 0)

    #update area derivative
    dArea_dt = dArea_dt + dArea_oaFlux + dArea_oiFlux + dArea_iaFlux


    ######  update sea ice cover fraction and mean ice and snow thickness #####

    Area = AreapreTH + dArea_dt * iceMask * deltaTtherm
    hIceMean = hIceMeanpreTH + dhIceMean_dt * iceMask * deltaTtherm
    hSnowMean = hSnowMeanpreTH + dhSnowMean_dt * iceMask * deltaTtherm

    # set boundaries:
    Area = np.clip(Area, 0, 1)
    hIceMean = np.clip(hIceMean, 0, None)
    hSnowMean = np.clip(hSnowMean, 0, None)

    noIce = np.where((hIceMean == 0) | (Area <= 0))
    Area[noIce] = 0
    hSnowMean[noIce] = 0
    hIceMean[noIce] = 0


    ##### output to ocean #####

    Qsw = qswi * AreapreTH + qswo * (1 - AreapreTH)

    # convert snow to ice if submerged
    tmpscal0 = (hSnowMean * rhoSnow + hIceMean * rhoIce) * recip_rhoConst
    d_hIceMeanByFlood = np.maximum(0, tmpscal0 - hIceMean)
    hIceMean = hIceMean + d_hIceMeanByFlood
    hSnowMean = hSnowMean - d_hIceMeanByFlood * rhoice2rhosnow

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

    # if the net energy flux out of the ocean is balanced by the latent
    # heat of fusion, the temperature of the mixed layer will not change
    ResidualEnergyOutOfOcean = NetEnergyFluxOutOfOcean - EnergyInNewTotalIceVolume

    # total heat flux out of the ocean [W/m2]
    Qnet = ResidualEnergyOutOfOcean * recip_deltaTtherm

    # freshwater flux from melting [m3/m2] (positive if the ice
    # thickness decreases/ melting occurs)
    FreshwaterContribFromIce = - ActualNewTotalVolumeChange * rhoIce2rhoFresh

    # in the case of non-zero ice salinity, the freshwater contribution
    # is reduced by the salinity ration of ice to water
    saltInWater = np.where((salt > 0) & (salt > saltIce))
    FreshwaterContribFromIce[saltInWater] = - ActualNewTotalVolumeChange[saltInWater] \
        * rhoIce2rhoFresh * (1 - saltIce/salt[saltInWater])

    # if the liquid cell has a lower salinity than the specified salinity
    # of sea ice, then assume the sea ice is completely fresh
    # (if the water is fresh, no salty sea ice can form)

    salt = np.clip(salt, 0, None) #leave in for now, remove when code is running and can be compared

    tmpscal0 = np.minimum(saltIce, salt)
    saltflux = (ActualNewTotalVolumeChange + SIhIceMeanNeg) * tmpscal0 \
        * iceMask * rhoIce * recip_deltaTtherm

    # freshwater contribution from snow melt [m3/m2]
    FreshwaterContribFromSnowMelt = ActualNewTotalSnowMelt / rhoFresh2rhoSnow

    # evaporation minus precipitation minus runoff (freshwater flux to ocean)
    EmPmR = iceMask *  ((evap - precip) * (1 - AreapreTH) \
        - PrecipRateOverIceSurfaceToSea * AreapreTH - runoff - (
        FreshwaterContribFromIce + FreshwaterContribFromSnowMelt) \
            / deltaTtherm) * rhoFresh + iceMask * (
        SIhIceMeanNeg * rhoIce + SIhSnowMeanNeg * rhoSnow) \
            * recip_deltaTtherm

    # sea ice + snow load on the sea surface
    seaIceLoad = hIceMean * rhoIce + hSnowMean * rhoSnow
    # maybe introduce a cap later

    
    return hIceMean, hSnowMean, Area, TIceSnow, saltflux, EmPmR, \
        Qsw, Qnet, seaIceLoad