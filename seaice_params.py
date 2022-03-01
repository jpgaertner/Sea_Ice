backend = 'numpy' # flag which backend to use (numpy or jax)

# densities [kg/m3]
rhoIce = 910        # density of ice
rhoSnow = 330       # density of snow
rhoFresh = 999.8    # density of fresh water
recip_rhoFresh = 1 / rhoFresh
rhoAir = 1.2        # density of air
rhoConst = 1027     # constant reference density of sea water (Boussineq approximation)
recip_rhoConst = 1 / rhoConst
rhoice2rhosnow     = rhoIce / rhoSnow
rhoIce2rhoFresh = rhoIce / rhoFresh
rhoFresh2rhoSnow = rhoFresh / rhoSnow


##### constants used in growth and solve4temp #####

# albedos
dryIceAlb = 0.75        # albedo of dry ice
dryIceAlb_south = 0.75  # albedo of dry ice in the southern hemisphere
wetIceAlb = 0.66        # albedo of wet ice
wetIceAlb_south = 0.66  # albedo of wet ice in the southern hemisphere
drySnowAlb = 0.84       # albedo of dry snow
drySnowAlb_south = 0.84 # albedo of dry snow in the southern hemisphere
wetSnowAlb = 0.7        # albdeo of wet snow
wetSnowAlb_south = 0.7  # albedo of wet snow in the southern hemisphere
wetAlbTemp = 0          # temperature [°C] above which the wet albedos are used

# latent heat constants of water
lhFusion = 3.34e5               # latent heat of fusion
lhEvap = 2.5e6                  # latent heat of evaporation
lhSublim = lhEvap + lhFusion    # latent heat of sublimation

cpAir = 1005    # specific heat of air

stefBoltz = 5.67e-8 # stefan boltzman constant

# emissivities
emissivity = 0.95       # longwave ocean emissivity
iceEmiss = emissivity   # longwave ice emissivity
snowEmiss = emissivity  # longwave snow emissivity

# conductivities
iceConduct = 2.1656 # sea ice conductivity
snowConduct = 0.31  # snow conductivity

hCut = 0.15 # cut off snow thickness (for h >= hCut the snow is shortwave opaque)

shortwave = 0.3 # ice penetration shortwave radiation factor

heatCapacity = 3986.0   # heat capacity of water

tempFrz0 = -1.96    # freezing temperature [°C]
dTempFrz_dS = 0     # derivative of freezing temperature w.r.t. salinity

saltIce = 0 # salinity of sea ice

# boundaries for longwave radiation and temperature
minLwDown = 60  # minimum downward longwave radiation
maxTIce = 30    # maximum ice temperature
minTIce = -50   # minimum ice temperature
minTAir = -50   # minimum air temperature

seaice_dalton = 0.00175 # dalton number/ sensible heat transfer coefficient

# regularization values for area and ice thickness
area_reg = 0.15
hice_reg = 0.10
area_reg_sq = area_reg**2
hice_reg_sq = hice_reg**2

celsius2K = 273.15 # conversion from [K] to [°C]

# timesteps [s]
deltaTtherm = 7200          # timestep for thermodynamic equations
recip_deltaTtherm = 1 / deltaTtherm
deltaTdyn = deltaTtherm     # timestep for dynamic equations
recip_deltaTdyn = 1 / deltaTdyn

# constants for McPhee formula for calculating turbulent ocean heat fluxes
stantonNr = 0.0056      # stanton number
uStarBase = 0.0125      # typical friction velocity beneath sea ice [m/s]
McPheeTaperFac = 12.5   # tapering factor

# lead closing parameter/ demarcation thickness between thin and thick ice
# ('thin ice' = open water)
h0 = 0.5
recip_h0 = 1 / h0
h0_south = 0.5
recip_h0_south = 1 / h0_south


##### constants in advection routines #####

airTurnAngle = 0    # turning angle of air-ice stress
waterTurnAngle = 0  # turning angle of the water-ice stress

# minimum wind speed [m/s]
eps = 1e-10         
eps_sq = eps**2

si_eps = 1e-5       # 'minimum' ice thickness [m] (smaller ice thicknesses are set to zero)
area_floor = si_eps # minimum ice cover fraction if ice is present

# drag coefficients
airIceDrag = 0.0012         # air-ice drag coefficient
airIceDrag_south = 0.001
waterIceDrag = 0.0055       # water-ice drag coefficient
waterIceDrag_south = 0.0055
cDragMin = 0.25             # minimum of linear ice-ocean drag coefficient

seaIceLoadFac = 1   # factor to scale (and turn off) sea ice loading

gravity = 9.8156    # gravitational acceleration

PlasDefCoeff = 2    # coefficient for plastic deformation/ axes ratio of the
    #elliptical yield curve of the relationship between the principal stress
    # components (sigma_1, sigma_2) (related to the relation of the critical
    # stresses needed for plastic deformation when pushing ice together vs.
    # pulling it apart)

deltaMin = 2.e-9    # regularization value for delta

pressReplFac = 0    # interpolator between press0 and regularized pressure

cStar = 20  # sea ice strength parameter

# basal drag parameters
basalDragU0 = 5e-5
basalDragK1 = 8
basalDragK2 = 0
cBasalStar = cStar

tensileStrFac = 0 # sea ice tensile strength factor

SeaIceStrength = 27.5e3 # sea ice strength P^star

zetaMaxfac = 2.5e8  # factor determining the maximum viscosity [s]
zetaMin = 0         # minimum viscosity

noSlip = True # flag whether to use no slip conditions


# flag which solver to use
useFreedrift = False
useEVP       = False
useLSR       = False
usePicard    = False
useJFNK      = False

useEVP = True
# useLSR = True
# usePicard=True
# useJFNK = True

