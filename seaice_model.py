
import numpy as np

from seaice_size import *
from seaice_params import *

from gendata import uWind, vWind, uVel, vVel, hIce_init

from seaice_dynsolver import dynsolver
from seaice_advdiff import advdiff
from seaice_reg_ridge import ridging
from seaice_growth import growth


### input
hIceMean = np.ones((sNx+2*OLx,sNy+2*OLy))
hSnowMean = np.ones((sNx+2*OLx,sNy+2*OLy))
Area = np.ones((sNx+2*OLx,sNy+2*OLy))
TIceSnow = np.ones((sNx+2*OLx,sNy+2*OLy))*celsius2K

hIceMeanMask = np.ones((sNx+2*OLx,sNy+2*OLy))
uIce = np.ones((sNx+2*OLx,sNy+2*OLy)) # ice velocity
vIce = np.ones((sNx+2*OLx,sNy+2*OLy))
etaN = np.ones((sNx+2*OLx,sNy+2*OLy))
pLoad = np.ones((sNx+2*OLx,sNy+2*OLy))
SeaIceLoad = np.ones((sNx+2*OLx,sNy+2*OLy))
useRealFreshWaterFlux = True

uVel = np.ones((sNx+2*OLx,sNy+2*OLy)) # ocean velocity 
vVel = np.ones((sNx+2*OLx,sNy+2*OLy))
uWind = np.ones((sNx+2*OLx,sNy+2*OLy))
vWind = np.ones((sNx+2*OLx,sNy+2*OLy))

salt = np.ones((sNx+2*OLx,sNy+2*OLy)) * 29
precip = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
snowPrecip = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
evap = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
runoff = np.ones((sNx+2*OLx,sNy+2*OLy)) * 0
wspeed = np.ones((sNx+2*OLx,sNy+2*OLy))
theta = np.ones((sNx+2*OLx,sNy+2*OLy))
Qnet = np.ones((sNx+2*OLx,sNy+2*OLy))
Qsw = np.ones((sNx+2*OLx,sNy+2*OLy))
SWDown = np.ones((sNx+2*OLx,sNy+2*OLy))
LWDown = np.ones((sNx+2*OLx,sNy+2*OLy))
ATemp = np.ones((sNx+2*OLx,sNy+2*OLy))
aqh = np.ones((sNx+2*OLx,sNy+2*OLy))


uIce, vIce = dynsolver(uIce, vIce, uVel, vVel, uWind, vWind, hIceMean, hSnowMean, Area, etaN, pLoad, SeaIceLoad, useRealFreshWaterFlux)

hIceMean, hSnowMean, Area = advdiff(uIce, vIce, hIceMean, hSnowMean, hIceMeanMask, Area)

hIceMean, hSnowMean, Area, TIceSnow = ridging(hIceMean, hSnowMean, Area, TIceSnow)

hIceMean, hSnowMean, Area, TIceSnow, saltflux, EvPrecRun, Qsw, Qnet, seaIceLoad = growth(hIceMean, hIceMeanMask, hSnowMean, Area, salt, TIceSnow, precip, snowPrecip, evap, 
runoff, wspeed, theta, Qnet, Qsw, SWDown, LWDown, ATemp, aqh)
