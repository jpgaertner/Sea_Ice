import numpy as np

from seaice_size import *

# densities [g/m3]
rhoIce = 900 
rhoSnow = 330
rhoFresh = 1000
recip_rhoFresh = 1 / rhoFresh
rhoConst = 1026 #constant reference density for sea water (Boussinesq)
recip_rhoConst = 1 / rhoConst
rhoice2rhosnow     = rhoIce / rhoSnow
rhoIce2rhoFresh = rhoIce / rhoFresh
rhoFresh2rhoSnow = rhoFresh / rhoSnow

# albedos
dryIceAlb = 0.75
dryIceAlb_south = 0.75
wetIceAlb = 0.66
wetIceAlb_south = 0.66
drySnowAlb = 0.84
drySnowAlb_south = 0.84
wetSnowAlb = 0.7
wetSnowAlb_south = 0.7
wetAlbTemp = 0

lhFusion = 3.34 * 10**5
lhEvap = 2.5 * 10**6
lhSublim = lhEvap + lhFusion
cpAir = 1004
rhoAir = 1.3
stefBoltz = 5.67e-8
emissivity = 0.97
iceEmiss = emissivity
snowEmiss = emissivity

iceConduct = 2.1656
snowConduct = 0.31
hCut = 0.15 #cut off snow thickness, used for calculating albedo
shortwave = 0.3 #penetration shortwave radiation factor

heatCapacity = 3986.0

tempFrz0 = -1.96
dTempFrz_dS = 0

saltIce = 0 #salinity of sea ice

minLwDown = 60 #minimum downward longwave radiation
maxTIce = 30 #maximum ice temperature
minTIce = -50 #minimum ice temperature
minTAir = -50 #minimum air temperature

# coefficients for flux computations/bulk formulae
seaice_dalton = 0.00175

# parametrization values:
SEAICE_area_reg = 0.15
SEAICE_hice_reg = 0.10

#The change of mean ice thickness due to out-of-bounds values following
#sea ice dynamics and advection
d_hIcebyDyn = np.ones((sNy,sNx))
d_hSnowbyDyn = np.ones((sNy,sNx))

SINegFac = 1 #value is actually one, but what is it?
swFracAbsTopOcean = 0 #the fraction of incoming shortwave radiation absorbed in the uppermost ocean grid cell

celsius2K = 273.16

deltaTtherm = 86400/2 #timestep for thermodynamic equations [s]
recip_deltaTtherm = 1 / deltaTtherm
deltaTdyn = deltaTtherm #timestep for dynamic equations [s]
recip_deltaTdyn = 1 / deltaTdyn

#constants needed for McPhee formulas for calculating turbulent ocean fluxes:
stantonNr = 0.0056 #stanton number
uStarBase = 0.0125 #typical friction velocity beneath sea ice [m/s]
#McPheeTaperFac = 12.5 #tapering factor
McPheeTaperFac = 0.92

# lead closing parameters:
h0 = 0.5 # the thickness of new ice formed in open water [m]
recip_h0 = 1 / h0
h0_south = 0.5
recip_h0_south = 1 / h0_south

airTurnAngle = 0 #turning angle of air-ice interfacial stress
waterTurnAngle = 0 #turning angle of the water-ice interfacial stress

eps = 1e-8
eps_sq = eps**2
si_eps = 1e-5
area_floor = si_eps

airOceanDrag = 0.0012 #air-ocean drag coefficient
airIceDrag = 0.0012 #air-ice drag coefficient
airIceDrag_south = 0.001 
waterIceDrag = 0.0055 #water-ice drag coefficient
waterIceDrag_south = 0.0055
cDragMin = 0.25 #minimum of liniear drag coefficient between ice and ocean
stressFactor = 1

seaIceLoadFac = 1 #factor to scale (and turn off) seaIceLoading
gravity = 9.8156

PlasDefCoeff = 2 #coefficient for plastic deformation, related to the relation of the critical stresses needed for plastic deformation when pushing ice together vs. pulling it apart
evpTauRelax = -1

deltaMin = eps
evpAlphaMin = 5
pressReplFac = 1

cStar = 20

# basal drag parameters
basalDragU0 = 5e-5
basalDragK1 = 8
basalDragK2 = 0
cBasalStar = cStar