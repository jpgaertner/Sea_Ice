from veros.state import VerosState
from veros.settings import Setting
from veros.variables import Variable
from veros import veros_routine
from veros.core.operators import numpy as npx
from veros.core.operators import update, at

from seaice_size import *
from seaice_params import *

from seaice_fill_overlap import fill_overlap, fill_overlap3d
from gendata import uwind, vwind, uo, vo, hice

dimensions = dict(x=nx+2*olx, y=ny+2*oly, z=nITC)
dims = ("x","y")

var_meta = dict(
    hIceMean        = Variable("Mean ice thickness", dims, "m"),
    hSnowMean       = Variable("Mean snow thickness", dims, "m"),
    Area            = Variable("Sea ice cover fraction", dims, " "),
    TIceSnow        = Variable("Ice/ snow temperature", ("x","y","z"), "K"),
    uIce            = Variable("Zonal ice velocity", dims, "m/s"),
    vIce            = Variable("Merdidional ice velocity", dims, "m/s"),
    uVel            = Variable("Zonal ocean surface velocity", dims, "m/s"),
    vVel            = Variable("Merdional ocean surface velocity", dims, "m/s"),
    uWind           = Variable("Zonal wind velocity", dims, "m/s"),
    vWind           = Variable("Merdional wind velocity", dims, "m/s"),
    wSpeed          = Variable("Total wind speed", dims, "m/s"),
    fu              = Variable("Zonal stress on ocean surface", dims, "N/m^2"),
    fv              = Variable("Meridional stress on ocean surface", dims, "N/m^2"),
    WindForcingX    = Variable("Zonal forcing on ice by wind stress", dims, "N"),
    WindForcingY    = Variable("Meridional forcing on ice by wind stress", dims, "N"),
    uTrans          = Variable("Zonal ice transport", dims, "m^2/s"),
    vTrans          = Variable("Meridional ice transport", dims, "m^2/s"),
    etaN            = Variable("Ocean surface elevation", dims, "m"),
    pLoad           = Variable("Surface pressure", dims, "P"),
    SeaIceLoad      = Variable("Load of sea ice on ocean surface", dims, "kg/m^2"),
    salt            = Variable("Ocean surface salinity", dims, "g/kg"),
    theta           = Variable("Ocean surface temperature", dims, "K"),
    Qnet            = Variable("Net heat flux out of the ocean", dims, "W/m^2"),
    Qsw             = Variable("Shortwave heatflux into the ocean", dims, "W/m^2"),
    SWDown          = Variable("Downward shortwave radiation", dims, "W/m^2"),
    LWDown          = Variable("Downward longwave radiation", dims, "W/m^2"),
    ATemp           = Variable("Atmospheric temperature", dims, "K"),
    aqh             = Variable("Atmospheric specific humidity", dims, "g/kg"),
    precip          = Variable("Precipitation rate", dims, "m/s"),
    snowPrecip      = Variable("Snowfall rate", dims, "m/s"),
    evap            = Variable("Evaporation rate over open ocean", dims, "m/s"),
    runoff          = Variable("Runoff into ocean", dims, "m/s"),
    EmPmR           = Variable("Evaporation minus precipitation minus runoff", dims, "m/s"),
    saltflux        = Variable("Salt flux into the ocean", dims, "m/s"),
    R_low           = Variable("Sea floor depth (<0)", dims, "m"),
    SeaIceMassC     = Variable("Sea ice mass centered around c point", dims, "kg"),
    SeaIceMassU     = Variable("Sea ice mass centered around u point", dims, "kg"),
    SeaIceMassV     = Variable("Sea ice mass centered around v point", dims, "kg"),
    SeaIceStrength  = Variable("Ice Strength", dims, "N/m"),
    press           = Variable("Ice Pressure", dims, "P"), #??? unit? difference to SeaIceStrength?
    zeta            = Variable("Bulk ice viscosity", dims, "Ns/m^2"),
    eta             = Variable("Shear ice viscosity", dims, "Ns/m^2"),
    os_hIceMean     = Variable("Overshoot of ice thickness from advection", dims, "m"),
    os_hSnowMean    = Variable("Overshoot of snow thickness from advection", dims, "m/s"),
    AreaW           = Variable("Sea ice cover fraction centered around u point", dims, " "),
    AreaS           = Variable("Sea ice cover fraction centered around v point", dims, " "),
    sigma11         = Variable("Stress tensor element", dims, "N/m^2"),
    sigma22         = Variable("Stress tensor element", dims, "N/m^2"),
    sigma12         = Variable("Stress tensor element", dims, "N/m^2"),
    tauX            = Variable("Zonal surface stress", dims, "N/m^2"),
    tauY            = Variable("Meridional surface stress", dims, "N/m^2")
)

sett_meta = dict(
    deltaTtherm     = Setting(0, float, "Timestep for thermodynamic equations"),
    deltaTdyn       = Setting(0, float, "Timestep for dynamic equations"),
    nx              = Setting(0, int, "Grid points in zonal direction"),
    ny              = Setting(0, int, "Grid points in meridional direction"),
    noSlip          = Setting(False, bool, "flag whether to use no slip condition"),
    secondOrderBC   = Setting(False, bool, "flag whether to use second order appreoximation for boundary conditions"),
    useFreedrift    = Setting(False, bool, "flag whether to use freedrift solver"),
    useEVP          = Setting(False, bool, "flag whether to use EVP solver"),
    useLSR          = Setting(False, bool, "flag whether to use LSR solver"),
    usePicard       = Setting(False, bool, "flag whether to use Picard solver"),
    useJNFK         = Setting(False, bool, "flag whether to use JNFK solver"),
    pStar  = Setting(27.5e3, float, "Standart sea ice strength"),
    cStar           = Setting(20, float, "Sea ice strength parameter"),
    PlasDefCoeff    = Setting(2, float, "Axes ratio of the elliptical yield curve"),
    h0              = Setting(0.5, float, "Lead closing parameter")
)

ones2d = npx.ones((nx+2*olx,ny+2*oly))
ones3d = npx.ones((nx+2*olx,ny+2*oly,nITC))
onesWind = npx.ones((32,nx+2*olx,ny+2*oly))
def copy(x):
    return update(x, at[:,:], x)


uWind_gen = copy(onesWind)
uWind_gen = update(uWind_gen, at[:,oly:-oly,olx:-olx], uwind)
uWind_gen = fill_overlap3d(uWind_gen)

vWind_gen = copy(onesWind)
vWind_gen = update(vWind_gen, at[:,oly:-oly,olx:-olx], vwind)
vWind_gen = fill_overlap3d(vWind_gen)

uVel_gen = copy(ones2d)
uVel_gen = update(uVel_gen, at[oly:-oly,olx:-olx], uo)
uVel_gen = fill_overlap(uVel_gen)

vVel_gen = copy(ones2d)
vVel_gen = update(vVel_gen, at[oly:-oly,olx:-olx], vo)
vVel_gen = fill_overlap(vVel_gen)

hIce_gen = copy(ones2d)
hIce_gen = update(hIce_gen, at[oly:-oly,olx:-olx], hice)
hIce_gen = fill_overlap(hIce_gen)


@veros_routine
def set_inits(state):
    state.variables.hIceMean    = hIce_gen
    state.variables.hSnowMean   = ones2d * 0
    state.variables.Area        = ones2d * 1
    state.variables.TIceSnow    = ones3d * 273.0
    state.variables.SeaIceLoad  = ones2d * (rhoIce * state.variables.hIceMean
                                            + rhoSnow * state.variables.hSnowMean)
    state.variables.uWind       = uWind_gen[0,:,:]
    state.variables.vWind       = vWind_gen[0,:,:]
    state.variables.wSpeed      = npx.sqrt(state.variables.uWind**2 + state.variables.vWind**2)
    state.variables.uVel        = uVel_gen
    state.variables.vVel        = vVel_gen
    state.variables.salt        = ones2d * 30
    state.variables.theta       = ones2d * celsius2K - 1.62
    state.variables.Qnet        = ones2d * 255.28928198201896
    state.variables.Qsw         = ones2d * 0
    state.variables.SWDown      = ones2d * 0
    state.variables.LWDown      = ones2d * 180
    state.variables.ATemp       = ones2d * 253
    state.variables.R_low       = ones2d * -1000
    state.variables.R_low = update(state.variables.R_low, at[:,-1], 0)
    state.variables.R_low = update(state.variables.R_low, at[-1,:], 0)
    state.variables.precip      = ones2d * 0


state = VerosState(var_meta, sett_meta, dimensions)
state.initialize_variables()
set_inits(state)