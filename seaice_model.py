import numpy as np
import matplotlib.pyplot as plt

from seaice_size import *
from seaice_params import *

from gendata import uWind_gendata, vWind_gendata, \
    uVel_gendata, vVel_gendata, hIce_init

from seaice_dynsolver import dynsolver
from seaice_advdiff import advdiff
from seaice_reg_ridge import clean_up_advection, ridging
from seaice_growth import growth
from seaice_fill_overlap import fill_overlap, fill_overlap3d


### input from gendata
windspd = 10
waterspd= 0.0
hIceMean = np.ones((sNy+2*OLy,sNx+2*OLx))
uWind = np.ones((32, sNy+2*OLy,sNx+2*OLx))*windspd
vWind = np.ones((32, sNy+2*OLy,sNx+2*OLx))*windspd
uVel = np.zeros((sNy+2*OLy,sNx+2*OLx)) + waterspd
vVel = np.zeros((sNy+2*OLy,sNx+2*OLx)) + waterspd

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
Area = np.ones((sNy+2*OLy,sNx+2*OLx)) * iceMask
TIceSnow = np.ones((sNy+2*OLy,sNx+2*OLx,nITC))*celsius2K

uIce = np.ones((sNy+2*OLy,sNx+2*OLx)) * 0.0
vIce = np.ones((sNy+2*OLy,sNx+2*OLx)) * 0.0

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
theta = np.zeros((sNy+2*OLy,sNx+2*OLx))*celsius2K - 1.62
Qnet = np.ones((sNy+2*OLy,sNx+2*OLx)) * 153.536072
Qsw = np.ones((sNy+2*OLy,sNx+2*OLx)) * 0
SWDown = np.ones((sNy+2*OLy,sNx+2*OLx)) * 0
LWDown = np.ones((sNy+2*OLy,sNx+2*OLx)) * 20
ATemp = np.ones((sNy+2*OLy,sNx+2*OLx)) * celsius2K - 20.16
aqh = np.ones((sNy+2*OLy,sNx+2*OLx)) * 0
fu = np.zeros((sNy+2*OLy,sNx+2*OLx))
fv = np.zeros((sNy+2*OLy,sNx+2*OLx))

R_low = np.ones((sNy+2*OLy,sNx+2*OLx)) * -1000

secondOrderBC = False


useFreedrift = False
useEVP = not useFreedrift

plt.close('all')
monFreq = 5
nTimeSteps = 10
for i in range(nTimeSteps):

    uIce, vIce, fu, fv = dynsolver(uIce, vIce, uVel, vVel,
                                uWind[0,:,:], vWind[0,:,:],
                                hIceMean, hSnowMean, Area, etaN,
                                pLoad, SeaIceLoad,
                                useRealFreshWaterFlux, useFreedrift, useEVP,
                                fu, fv, secondOrderBC, R_low)

    hIceMean, hSnowMean, Area = advdiff(uIce, vIce, hIceMean,
                                        hSnowMean, Area)

    hIceMean, hSnowMean, Area, TIceSnow, os_hIceMean, os_hSnowMean \
        = clean_up_advection(hIceMean, hSnowMean, Area, TIceSnow)

    Area = ridging(Area)

    hIceMean, hSnowMean, Area, TIceSnow, saltflux, EvPrecRun, \
        Qsw_out, Qnet_out, seaIceLoad \
        = growth(hIceMean, hSnowMean, Area, os_hIceMean, os_hSnowMean,
                salt, TIceSnow, precip, snowPrecip, evap, 
                runoff, wspeed, theta, Qnet, Qsw, SWDown, LWDown,
                ATemp, aqh)


    printMonitor = monFreq>0 and (np.mod(i,monFreq)==0 or i==nTimeSteps-1)
    print('Time step %04i'%i)
    if printMonitor:
        print('Time step %4s, %11s, %11s, %11s, %11s, %11s'%(
            ' ','hIceMean','hSnowMean','Area','uIce','vIce'))
        print('mean      %4i, %11.4e, %11.4e, %11.4e, %11.4e, %11.4e'%(
            i,hIceMean.mean(),hSnowMean.mean(),Area.mean(),uIce.mean(),
            vIce.mean()))
        print('min       %4i, %11.4e, %11.4e, %11.4e, %11.4e, %11.4e'%(
            i,hIceMean.min(),hSnowMean.min(),Area.min(),uIce.min(),vIce.min()))
        print('max       %4i, %11.4e, %11.4e, %11.4e, %11.4e, %11.4e'%(
            i,hIceMean.max(),hSnowMean.max(),Area.max(),uIce.max(),vIce.max()))
        print('std       %4i, %11.4e, %11.4e, %11.4e, %11.4e, %11.4e'%(
            i,hIceMean.std(),hSnowMean.std(),Area.std(),uIce.std(),vIce.std()))

# need this for my plots
def sq(a):
    import numpy as np
    a = np.squeeze(a)
    masked_array=np.ma.masked_where(a==0., a)
    return masked_array

fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5),sharex=True,sharey=True,)
csf0=ax[0].pcolormesh(sq(uIce[OLy:-OLy,OLx:-OLx]))
ax[0].set_title('uIce')
csf1=ax[1].pcolormesh(sq(vIce[OLy:-OLy,OLx:-OLx]))
ax[1].set_title('vIce')
csf2=ax[2].pcolormesh( hIceMean[OLy:-OLy,OLx:-OLx]*
                        sq(iceMask[OLy:-OLy,OLx:-OLx]))
ax[2].set_title('hIce')
#plt.contourf(vWind[0,OLy:-OLy,OLx:-OLx])
plt.colorbar(csf0,ax=ax[0],orientation='horizontal')
plt.colorbar(csf1,ax=ax[1],orientation='horizontal')
plt.colorbar(csf2,ax=ax[2],orientation='horizontal')

# fig, axs = plt.subplots(1,3, figsize = (10,3))
# ax1 = axs[0].pcolormesh(uIce[OLy:-OLy,OLx:-OLx])
# axs[0].set_title('uIce')
# plt.colorbar(ax1, ax = axs[0])
# ax2 = axs[1].pcolormesh(vIce[OLy:-OLy,OLx:-OLx])
# axs[1].set_title('vIce')
# plt.colorbar(ax2, ax = axs[1])
# ax3 = axs[2].pcolormesh(hIceMean[OLy:-OLy,OLx:-OLx])
# axs[2].set_title('hIceMean')
# plt.colorbar(ax3, ax = axs[2])
# fig.tight_layout()
# plt.show()
# =======
# # print(np.mean(hIceMean))
# # print(np.mean(uWind))
# # print(np.mean(vWind))
# # print(np.max(uIce))
# # print(np.max(vIce))
# plt.close('all')
# fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5),sharex=True,sharey=True,)
# csf0=ax[0].pcolormesh(sq(uIce[OLy:-OLy,OLx:-OLx]))
# ax[0].set_title('uIce')
# csf1=ax[0].pcolormesh(sq(vIce[OLy:-OLy,OLx:-OLx]))
# ax[1].set_title('vIce')
# csf2=ax[2].pcolormesh(sq(hIceMean[OLy:-OLy,OLx:-OLx]))
# ax[2].set_title('hIce')
# #plt.contourf(vWind[0,OLy:-OLy,OLx:-OLx])
# plt.colorbar(csf0,ax=ax[0],orientation='horizontal')
# plt.colorbar(csf1,ax=ax[1],orientation='horizontal')
# plt.colorbar(csf2,ax=ax[2],orientation='horizontal')

plt.show()
