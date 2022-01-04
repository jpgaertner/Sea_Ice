
import numpy as np

from seaice_size import *
from seaice_params import *

from gendata import uWind, vWind, uVel, vVel, hIce_init

from seaice_dynsolver import dynsolver
from seaice_advdiff import advdiff
from seaice_reg_ridge import ridging
from seaice_growth import growth


### input
hIceMean = hIce_init.copy()
hSnowMean = np.zeros((sNx+2*OLx,sNy+2*OLy))
Area = np.ones((sNx+2*OLx,sNy+2*OLy))
TIceSnow = np.ones((sNx+2*OLx,sNy+2*OLy,nITC))*celsius2K

hIceMeanMask = np.ones((sNx+2*OLx,sNy+2*OLy))
uIce = np.ones((sNx+2*OLx,sNy+2*OLy)) # ice velocity
vIce = np.ones((sNx+2*OLx,sNy+2*OLy))
etaN = np.ones((sNx+2*OLx,sNy+2*OLy))
pLoad = np.ones((sNx+2*OLx,sNy+2*OLy))
SeaIceLoad = np.ones((sNx+2*OLx,sNy+2*OLy))
useRealFreshWaterFlux = True

salt = np.ones((sNx+2*OLx,sNy+2*OLy)) * 29
precip = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
snowPrecip = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
evap = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
runoff = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
wspeed = np.sqrt(uWind[0,:,:]**2 + vWind[0,:,:]**2)
theta = np.ones((sNx+2*OLx,sNy+2*OLy))*celsius2K - 1.96
Qnet = np.ones((sNx+2*OLx,sNy+2*OLy)) * 153.536072
Qsw = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
SWDown = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
LWDown = np.ones((sNx+2*OLx,sNy+2*OLy)) * 80
ATemp = np.ones((sNx+2*OLx,sNy+2*OLy)) * celsius2K - 20.16
aqh = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0


timesteps = 14

ice = [None] * timesteps
snow = [None] * timesteps
area = [None] * timesteps
days = [None] * timesteps


for i in range(1):

    ice[i] = hIceMean
    snow[i] = hIceMean
    area[i] = Area
    days[i] = i


    uIce, vIce = dynsolver(uIce, vIce, uVel, vVel, uWind[0,:,:], vWind[0,:,:], hIceMean, hSnowMean, Area, etaN, pLoad, SeaIceLoad, useRealFreshWaterFlux)

    hIceMean, hSnowMean, Area = advdiff(uIce, vIce, hIceMean, hSnowMean, hIceMeanMask, Area)

    hIceMean, hSnowMean, Area, TIceSnow = ridging(hIceMean, hSnowMean, Area, TIceSnow)

    hIceMean, hSnowMean, Area, TIceSnow, saltflux, EvPrecRun, Qsw, Qnet, seaIceLoad = growth(hIceMean, hIceMeanMask, hSnowMean, Area, salt, TIceSnow, precip, snowPrecip, evap, runoff, wspeed, theta, Qnet, Qsw, SWDown, LWDown, ATemp, aqh)


#print(ice[10])