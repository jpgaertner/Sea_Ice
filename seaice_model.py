
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
hIceMean = np.ones((sNy+2*OLy,sNx+2*OLx))
uWind = np.ones((32, sNy+2*OLy,sNx+2*OLx))
vWind = np.ones((32, sNy+2*OLy,sNx+2*OLx))
uVel = np.zeros((sNy+2*OLy,sNx+2*OLx))
vVel = np.zeros((sNy+2*OLy,sNx+2*OLx))

# hIceMean[OLy:-OLy,OLx:-OLx] = hIce_init
# hIceMean = fill_overlap(hIceMean)
# uWind[:,OLy:-OLy,OLx:-OLx] = uWind_gendata
# uWind = fill_overlap3d(uWind)
# vWind[:,OLy:-OLy,OLx:-OLx] = vWind_gendata
# vWind = fill_overlap3d(vWind)
# uVel[OLy:-OLy,OLx:-OLx] = uVel_gendata
# uVel = fill_overlap(uVel)
# vVel[OLy:-OLy,OLx:-OLx] = vVel_gendata
# vVel = fill_overlap(vVel)


hSnowMean = np.ones((sNy+2*OLy,sNx+2*OLx)) * 0
Area = np.ones((sNy+2*OLy,sNx+2*OLx))
TIceSnow = np.ones((sNy+2*OLy,sNx+2*OLx,nITC))*celsius2K

uIce = np.zeros((sNy+2*OLy,sNx+2*OLx))
vIce = np.zeros((sNy+2*OLy,sNx+2*OLx))

etaN = np.ones((sNy+2*OLy,sNx+2*OLx)) * 0
pLoad = np.ones((sNy+2*OLy,sNx+2*OLx)) * 0
SeaIceLoad = hIceMean * rhoIce + hSnowMean * rhoSnow
useRealFreshWaterFlux = True

salt = np.ones((sNy+2*OLy,sNx+2*OLx)) * 29
precip = np.ones((sNy+2*OLy,sNx+2*OLx)) * 0
snowPrecip = np.ones((sNy+2*OLy,sNx+2*OLx)) * 0
evap = np.ones((sNy+2*OLy,sNx+2*OLx)) * 0
runoff = np.ones((sNy+2*OLy,sNx+2*OLx)) * 0
wspeed = np.sqrt(uWind[0,:,:]**2 + vWind[0,:,:]**2)
theta = np.ones((sNy+2*OLy,sNx+2*OLx))*celsius2K - 1.62
Qnet = np.ones((sNy+2*OLy,sNx+2*OLx)) * 153.536072
Qsw = np.ones((sNy+2*OLy,sNx+2*OLx)) * 0
SWDown = np.ones((sNy+2*OLy,sNx+2*OLx)) * 0
LWDown = np.ones((sNy+2*OLy,sNx+2*OLx)) * 20
ATemp = np.ones((sNy+2*OLy,sNx+2*OLx)) * celsius2K - 20.16
aqh = np.ones((sNy+2*OLy,sNx+2*OLx)) * 0


# timesteps = 32

# ice = [None] * timesteps
# snow = [None] * timesteps
# area = [None] * timesteps
# days = [None] * timesteps


for i in range(2):

    # ice[i] = hIceMean
    # snow[i] = hIceMean
    # area[i] = Area
    # days[i] = i


    uIce, vIce = dynsolver(uIce, vIce, uVel, vVel, uWind[0,:,:], vWind[0,:,:], hIceMean, hSnowMean, Area, etaN, pLoad, SeaIceLoad, useRealFreshWaterFlux)

    hIceMean, hSnowMean, Area = advdiff(uIce, vIce, hIceMean, hSnowMean, Area)

    hIceMean, hSnowMean, Area, TIceSnow = ridging(hIceMean, hSnowMean, Area, TIceSnow)

    hIceMean, hSnowMean, Area, TIceSnow, saltflux, EvPrecRun, Qsw, Qnet, seaIceLoad = growth(hIceMean, hIceMeanMask, hSnowMean, Area, salt, TIceSnow, precip, snowPrecip, evap, runoff, wspeed, theta, Qnet, Qsw, SWDown, LWDown, ATemp, aqh)




# print(np.mean(hIceMean))
# print(np.mean(uWind))
# print(np.mean(vWind))
#print(np.max(uIce))
# print(np.max(vIce))
#plt.contourf(vIce[OLy:-OLy,OLx:-OLx])
plt.contourf(hIceMean[OLy:-OLy,OLx:-OLx])
#plt.contourf(vWind[0,OLy:-OLy,OLx:-OLx])
plt.colorbar()
plt.show()