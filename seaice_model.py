
import numpy as np
import matplotlib.pyplot as plt

from seaice_size import *
from seaice_params import *

from gendata import uWind_gendata, vWind_gendata, uVel_gendata, vVel_gendata, hIce_init

from seaice_dynsolver import dynsolver
from seaice_advdiff import advdiff
from seaice_reg_ridge import ridging
from seaice_growth import growth
from seaice_fill_overlap import fill_overlap, fill_overlap3d


### input from gendata
hIceMean = np.ones((sNx+2*OLx,sNy+2*OLy))
uWind = np.zeros((32, sNx+2*OLx,sNy+2*OLy))*0
vWind = np.zeros((32, sNx+2*OLx,sNy+2*OLy))
uVel = np.zeros((sNx+2*OLx,sNy+2*OLy))
vVel = np.zeros((sNx+2*OLx,sNy+2*OLy))

hIceMean[OLx:-OLx,OLy:-OLy] = hIce_init
hIceMean = fill_overlap(hIceMean)
# uWind[:,OLx:-OLx,OLy:-OLy] = uWind_gendata
# uWind = fill_overlap3d(uWind)
vWind[:,OLx:-OLx,OLy:-OLy] = vWind_gendata
vWind = fill_overlap3d(vWind)
# uVel[OLx:-OLx,OLy:-OLy] = uVel_gendata
# uVel = fill_overlap(uVel)
# vVel[OLx:-OLx,OLy:-OLy] = vVel_gendata
# vVel = fill_overlap(vVel)


hSnowMean = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
Area = np.ones((sNx+2*OLx,sNy+2*OLy))
TIceSnow = np.ones((sNx+2*OLx,sNy+2*OLy,nITC))*celsius2K

uIce = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
vIce = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0

hIceMeanMask = np.ones((sNx+2*OLx,sNy+2*OLy))
etaN = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
pLoad = np.ones((sNx+2*OLx,sNy+2*OLy))
SeaIceLoad = np.ones((sNx+2*OLx,sNy+2*OLy))
useRealFreshWaterFlux = True

salt = np.ones((sNx+2*OLx,sNy+2*OLy)) * 29
precip = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
snowPrecip = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
evap = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
runoff = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
wspeed = np.ones((sNx+2*OLx,sNy+2*OLy))*2 #np.sqrt(uWind[0,:,:]**2 + vWind[0,:,:]**2)
theta = np.ones((sNx+2*OLx,sNy+2*OLy))*celsius2K - 1.96
Qnet = np.ones((sNx+2*OLx,sNy+2*OLy)) * 153.536072
Qsw = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
SWDown = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
LWDown = np.ones((sNx+2*OLx,sNy+2*OLy)) * 20
ATemp = np.ones((sNx+2*OLx,sNy+2*OLy)) * celsius2K - 20.16
aqh = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0


# timesteps = 32

# ice = [None] * timesteps
# snow = [None] * timesteps
# area = [None] * timesteps
# days = [None] * timesteps


for i in range(1):

    # ice[i] = hIceMean
    # snow[i] = hIceMean
    # area[i] = Area
    # days[i] = i


    uIce, vIce = dynsolver(uIce, vIce, uVel, vVel, uWind[0,:,:], vWind[0,:,:], hIceMean, hSnowMean, Area, etaN, pLoad, SeaIceLoad, useRealFreshWaterFlux)

    hIceMean, hSnowMean, Area = advdiff(uIce, vIce, hIceMean, hSnowMean, hIceMeanMask, Area)

    hIceMean, hSnowMean, Area, TIceSnow = ridging(hIceMean, hSnowMean, Area, TIceSnow)

    hIceMean, hSnowMean, Area, TIceSnow, saltflux, EvPrecRun, Qsw, Qnet, seaIceLoad = growth(hIceMean, hIceMeanMask, hSnowMean, Area, salt, TIceSnow, precip, snowPrecip, evap, runoff, wspeed, theta, Qnet, Qsw, SWDown, LWDown, ATemp, aqh)


# print(np.mean(hIceMean))
# print(np.mean(uWind))
# print(np.mean(vWind))
# print(np.max(uIce))
# print(np.max(vIce))
#plt.contourf(vIce[OLx:-OLx,OLy:-OLy])
plt.contourf(hIceMean[OLx:-OLx,OLy:-OLy])
#plt.contourf(vWind[0,OLx:-OLx,OLy:-OLy])
plt.colorbar()
plt.show()