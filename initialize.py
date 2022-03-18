from veros.state import VerosState
from veros.settings import Setting
from veros.variables import Variable
from veros import veros_routine, veros_kernel, KernelOutput
from veros.core.operators import numpy as npx

from seaice_size import *
from seaice_params import *

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

init_values = dict(
    hIceMean_init   = ones2d * 1.3,
    hSnowMean_init  = ones2d * 0.1,
    Area_init       = ones2d * 0.9,
    TIceSnow_init   = ones3d * celsius2K,
    uWind_init      = ones2d * 0, #5,
    vWind_init      = ones2d * 0, #5,
    wSpeed_init     = ones2d * 2, #npx.sqrt(5**2+5**2),
    SeaIceLoad_init = ones2d * (rhoIce * 1.3 + rhoSnow * 0.2),
    salt_init       = ones2d * 29,
    theta_init      = ones2d * celsius2K - 1.66,
    Qnet_init       = ones2d * 173.03212617345582,
    Qsw_init        = ones2d * 0,
    SWDown_init     = ones2d * 0,
    LWDown_init     = ones2d * 180,
    ATemp_init      = ones2d * 253.,
    precip_init     = ones2d * 0,
    R_low_init      = ones2d * -1000
)

@veros_routine
def set_inits(state):
    state.variables.hIceMean    = init_values["hIceMean_init"]
    state.variables.hSnowMean   = init_values["hSnowMean_init"]
    state.variables.Area        = init_values["Area_init"]
    state.variables.TIceSnow    = init_values["TIceSnow_init"]
    state.variables.uWind       = init_values["uWind_init"]
    state.variables.vWind       = init_values["vWind_init"]
    state.variables.SeaIceLoad  = init_values["SeaIceLoad_init"]
    state.variables.salt        = init_values["salt_init"]
    state.variables.theta       = init_values["theta_init"]
    state.variables.Qnet        = init_values["Qnet_init"]
    state.variables.Qsw         = init_values["Qsw_init"]
    state.variables.SWDown      = init_values["SWDown_init"]
    state.variables.LWDown      = init_values["LWDown_init"]
    state.variables.ATemp       = init_values["ATemp_init"]
    state.variables.wSpeed      = init_values["wSpeed_init"]
    state.variables.R_low       = init_values["R_low_init"]
    state.variables.precip      = init_values["precip_init"]


state = VerosState(var_meta, sett_meta, dimensions)
state.initialize_variables()

set_inits(state)
#print(state.variables.hIceMean)