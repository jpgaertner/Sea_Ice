import numpy as np

path = "/Users/jgaertne/MITgcm/verification/1D_ocean_ice_column/input/"
AirTemp = np.fromfile(path + "atemp_1x1_one_year", dtype = ">f4")
LWDown = np.fromfile(path + "dlwrf_1x1_one_year", dtype = ">f4")
SWDown = np.fromfile(path + "dswrf_1x1_one_year", dtype = ">f4")
uWind = np.fromfile(path + "u_1ms_1x1_one_year", dtype = ">f4")
vWind = np.fromfile(path + "u_1ms_1x1_one_year", dtype = ">f4")


print(f"AirTemp{AirTemp[:10]}")
print(f"LWDown{LWDown[:10]}")
print(f"SWDown{SWDown[:10]}")
print(f"uWind{uWind[:10]}")
print(f"vWind{vWind[:10]}")