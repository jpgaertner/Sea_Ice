from veros import runtime_settings
from seaice_params import backend
runtime_settings.backend = backend
from veros.core.operators import numpy as npx

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
windspd = 5
waterspd= 0.0
hIceMean = npx.ones((ny+2*oly,nx+2*olx))*0.3
uWind = npx.ones((32, ny+2*oly,nx+2*olx))*windspd
vWind = npx.ones((32, ny+2*oly,nx+2*olx))*windspd
uVel = npx.zeros((ny+2*oly,nx+2*olx)) + waterspd
vVel = npx.zeros((ny+2*oly,nx+2*olx)) + waterspd

x = (npx.arange(nx+2*olx)+0.5)*gridcellWidth;
y = (npx.arange(ny+2*oly)+0.5)*gridcellWidth;
xx,yy = npx.meshgrid(x,y);
hice = 0.3 + 0.005*(npx.sin(60./1000.e3*xx) + npx.sin(30./1000.e3*yy))
#hIceMean = 0.5*(hice + hice.transpose())

# hIceMean[oly:-oly,olx:-olx] = hIce_init
# hIceMean = fill_overlap(hIceMean)
# uWind[:,oly:-oly,olx:-olx] = uWind_gendata
# uWind = fill_overlap3d(uWind)
# vWind[:,oly:-oly,olx:-olx] = vWind_gendata
# vWind = fill_overlap3d(vWind)
# uVel[oly:-oly,olx:-olx] = uVel_gendata
# uVel = fill_overlap(uVel)
# vVel[oly:-oly,olx:-olx] = vVel_gendata
# vVel = fill_overlap(vVel)


hSnowMean = npx.ones((ny+2*oly,nx+2*olx)) * 0.1
Area = npx.ones((ny+2*oly,nx+2*olx)) * 0.9
TIceSnow = npx.ones((ny+2*oly,nx+2*olx,nITC))*celsius2K

uIce = npx.ones((ny+2*oly,nx+2*olx)) * 0.0
vIce = npx.ones((ny+2*oly,nx+2*olx)) * 0.0

etaN = npx.ones((ny+2*oly,nx+2*olx)) * 0
pLoad = npx.ones((ny+2*oly,nx+2*olx)) * 0
SeaIceLoad = hIceMean * rhoIce + hSnowMean * rhoSnow
useRealFreshWaterFlux = True # ??? parameter?

salt = npx.ones((ny+2*oly,nx+2*olx)) * 29
precip = npx.ones((ny+2*oly,nx+2*olx)) * 0
snowPrecip = npx.ones((ny+2*oly,nx+2*olx)) * 0
evap = npx.ones((ny+2*oly,nx+2*olx)) * 0
runoff = npx.ones((ny+2*oly,nx+2*olx)) * 0
wspeed = npx.sqrt(uWind[0,:,:]**2 + vWind[0,:,:]**2)
theta = npx.zeros((ny+2*oly,nx+2*olx))*celsius2K - 1.62
Qnet = npx.ones((ny+2*oly,nx+2*olx)) * 153.536072
Qsw = npx.ones((ny+2*oly,nx+2*olx)) * 0
SWDown = npx.ones((ny+2*oly,nx+2*olx)) * 0
LWDown = npx.ones((ny+2*oly,nx+2*olx)) * 20
ATemp = npx.ones((ny+2*oly,nx+2*olx)) * celsius2K - 20.16
aqh = npx.ones((ny+2*oly,nx+2*olx)) * 0
fu = npx.zeros((ny+2*oly,nx+2*olx))
fv = npx.zeros((ny+2*oly,nx+2*olx))

R_low = npx.ones((ny+2*oly,nx+2*olx)) * -1000

plt.close('all')
monFreq = 5
nTimeSteps = 2
nIter0 = 0

for i in range(nTimeSteps):

    myIter = nIter0 + i
    myTime = myIter*deltaTtherm
    uIce, vIce, fu, fv = dynsolver(uIce, vIce, uVel, vVel,
                                   uWind[0,:,:], vWind[0,:,:],
                                   hIceMean, hSnowMean, Area, etaN,
                                   pLoad, SeaIceLoad,
                                   useRealFreshWaterFlux,
                                   fu, fv, secondOrderBC, R_low,
                                   myTime, myIter)

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

    printMonitor = monFreq>0 and (npx.mod(myIter,monFreq)==0
                                  or myIter==nTimeSteps-1)
    print('Time step %04i'%myIter)
    if printMonitor:
        print('Time step %4s, %11s, %11s, %11s, %11s, %11s'%(
            ' ','hIceMean','hSnowMean','Area','uIce','vIce'))
        print('mean      %4i, %11.4e, %11.4e, %11.4e, %11.4e, %11.4e'%(
            myIter,hIceMean.mean(),hSnowMean.mean(),Area.mean(),
            uIce.mean(),vIce.mean()))
        print('min       %4i, %11.4e, %11.4e, %11.4e, %11.4e, %11.4e'%(
            myIter,hIceMean.min(),hSnowMean.min(),Area.min(),
            uIce.min(),vIce.min()))
        print('max       %4i, %11.4e, %11.4e, %11.4e, %11.4e, %11.4e'%(
            myIter,hIceMean.max(),hSnowMean.max(),Area.max(),
            uIce.max(),vIce.max()))
        print('std       %4i, %11.4e, %11.4e, %11.4e, %11.4e, %11.4e'%(
            myIter,hIceMean.std(),hSnowMean.std(),Area.std(),
            uIce.std(),vIce.std()))

# need this for my plots
def sq(a):
    import numpy as npx
    a = npx.squeeze(a)
    masked_array=npx.ma.masked_where(a==0., a)
    return masked_array

from dynamics_routines import strainrates
import matplotlib.colors as mcolors
mynorm = mcolors.LogNorm(vmin=1e-12,vmax=1e-5)

e11,e22,e12=strainrates(uIce,vIce)
divergence = (e11+e22)*iceMask
# use area weighted average of squares of e12 (more accurate)
e12Csq = rAz * e12**2
e12Csq =                     e12Csq + npx.roll(e12Csq,-1,0)
e12Csq = 0.25 * recip_rA * ( e12Csq + npx.roll(e12Csq,-1,1) )
shear = npx.sqrt((e11-e22) ** 2 + 4.*e12Csq)*iceMask

fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5),sharex=True,sharey=True,)
csf0=ax[0].pcolormesh(sq(uIce[oly:-oly,olx:-olx]))
ax[0].set_title('uIce')
# csf0=ax[0].pcolormesh(sq(shear[oly:-oly,olx:-olx]),norm=mynorm)
# ax[0].set_title('shear')
csf0=ax[0].pcolormesh(sq(uIce-vIce.transpose())[oly:-oly,olx:-olx]) #
# csf0=ax[0].pcolormesh(sq(vIce-vIce[:,::-1])[oly:-oly,olx:-olx],
#                       vmin=-1e-7,vmax=1e-7)
ax[0].set_title('uIce-vIce.transpose()')
csf1=ax[1].pcolormesh(sq(vIce[oly:-oly,olx:-olx]))
ax[1].set_title('vIce')
csf2=ax[2].pcolormesh( hIceMean[oly:-oly,olx:-olx]*
                        sq(iceMask[oly:-oly,olx:-olx]))
ax[2].set_title('hIce')
#plt.contourf(vWind[0,oly:-oly,olx:-olx])
plt.colorbar(csf0,ax=ax[0],orientation='horizontal')
plt.colorbar(csf1,ax=ax[1],orientation='horizontal')
plt.colorbar(csf2,ax=ax[2],orientation='horizontal')

# fig, axs = plt.subplots(1,3, figsize = (10,3))
# ax1 = axs[0].pcolormesh(uIce[oly:-oly,olx:-olx])
# axs[0].set_title('uIce')
# plt.colorbar(ax1, ax = axs[0])
# ax2 = axs[1].pcolormesh(vIce[oly:-oly,olx:-olx])
# axs[1].set_title('vIce')
# plt.colorbar(ax2, ax = axs[1])
# ax3 = axs[2].pcolormesh(hIceMean[oly:-oly,olx:-olx])
# axs[2].set_title('hIceMean')
# plt.colorbar(ax3, ax = axs[2])
# fig.tight_layout()
# plt.show()
# =======
# # print(npx.mean(hIceMean))
# # print(npx.mean(uWind))
# # print(npx.mean(vWind))
# # print(npx.max(uIce))
# # print(npx.max(vIce))
# plt.close('all')
# fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5),sharex=True,sharey=True,)
# csf0=ax[0].pcolormesh(sq(uIce[oly:-oly,olx:-olx]))
# ax[0].set_title('uIce')
# csf1=ax[0].pcolormesh(sq(vIce[oly:-oly,olx:-olx]))
# ax[1].set_title('vIce')
# csf2=ax[2].pcolormesh(sq(hIceMean[oly:-oly,olx:-olx]))
# ax[2].set_title('hIce')
# #plt.contourf(vWind[0,oly:-oly,olx:-olx])
# plt.colorbar(csf0,ax=ax[0],orientation='horizontal')
# plt.colorbar(csf1,ax=ax[1],orientation='horizontal')
# plt.colorbar(csf2,ax=ax[2],orientation='horizontal')

plt.show()
