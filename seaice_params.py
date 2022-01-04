import numpy as np

from seaice_size import *

# densities [g/m3]
rhoIce = 910 
rhoSnow = 330
rhoFresh = 999.8
recip_rhoFresh = 1 / rhoFresh
rhoConst = 1027.0 #constant reference density for sea water (Boussinesq)
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

#ifdef ALLOW_EXF
#Use parameters that have already been set in data.exf:
#else:
lhFusion = 3.34 * 10**5
lhEvap = 2.5 * 10**6
lhSublim = lhEvap + lhFusion
F_lh_cap = True
cpAir = 1004
rhoAir = 1.3
stefBoltz = 5.67e-8
emissivity = 0.97
iceEmiss = emissivity
snowEmiss = emissivity

iceConduct = 2.1656
snowConduct = 0.31
hCut = 0.2 #cut off snow thickness, used for calculating albedo
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
d_hIcebyDyn = np.ones((sNx, sNy))
d_hSnowbyDyn = np.ones((sNx, sNy))

SINegFac = 1 #value is actually one, but what is it?
swFracAbsTopOcean = 0 #the fraction of incoming shortwave radiation absorbed in the uppermost ocean grid cell

celsius2K = 273.16

deltatTherm = 86400/2 #timestep for thermodynamic equations [s]
recip_deltatTherm = 1 / deltatTherm

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


############# dynsolver ##############

airTurnAngle = 0 #turning angle of air-ice interfacial stress
waterTurnAngle = 0 #turning angle of the water-ice interfacial stress 
eps = 1e-8
eps_sq = eps**2
si_eps = 1e-5
area_floor = si_eps

airOceanDrag = 0.001 #air-ocean drag coefficient
airIceDrag = 0.001 #air-ice drag coefficient
airIceDrag_south = 0.001 
waterIceDrag = 0.0055 #water-ice drag coefficient
waterIceDrag_south = 0.0055

seaice_strenght = 2.75e4 #'pstar'?
pStar = seaice_strenght
seaice_cStar = 20 #sea ice strength parameter

zetaMax = 2.5e8 #factor determining the maximum viscosity (?)
zetaMin = 0 #lower bound for viscosity

linearIterMax = 1500 #number of allowed linear solver iterations for implicit (JFNK and Picard)
nonLinearIterMax = 2 #number of allowed non-linear solver iterations for implicit solvers (JFNK and Picard)
#can have different values (ifdef... )

seaIceEccen = 2 #sea-ice eccentricity of the elliptical yield curve

deltatDyn = 1 #value? #timestep for dynamic solver [s]
recip_deltatDyn = 1 / deltatDyn

#### sea ice dynsolver ####
seaIceLoadFac = 1 #factor to scale (and turn off) seaIceLoading
gravity = 9.8156

#### sea ice lsr ####
seaIceOLx = OLx - 2
seaIceOLy = OLy - 2
LSRrelaxU = 0.95 #relaxation parameter for LSR-solver: U/V-component
LSRrelaxV = 0.95

