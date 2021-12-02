import numpy as np

path = "C:/Users/Jan/Documents/MITgcm/verification/1D_ocean_ice_column/input/"
AirTemp = np.fromfile(path + "atemp_1x1_one_year")
LWDown = np.fromfile(path + "dlwrf_1x1_one_year")
SWDown = np.fromfile(path + "dswrf_1x1_one_year")
uWind = np.fromfile(path + "u_1ms_1x1_one_year")
vWind = np.fromfile(path + "u_1ms_1x1_one_year")


print(f"AirTemp{AirTemp[:10]}")
print(f"LWDown{LWDown[:10]}")
print(f"SWDown{SWDown[:10]}")
print(f"uWind{uWind[:10]}")
print(f"vWind{vWind[:10]}")