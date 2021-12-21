import numpy as np

path = "/Users/jgaertne/MITgcm/verification/1D_ocean_ice_column/input/"
AirTemp_forcing = np.fromfile(path + "atemp_1x1_one_year", dtype = ">f4")
LWDown_forcing = np.fromfile(path + "dlwrf_1x1_one_year", dtype = ">f4")
SWDown_forcing = np.fromfile(path + "dswrf_1x1_one_year", dtype = ">f4")
uWind_forcing = np.fromfile(path + "u_1ms_1x1_one_year", dtype = ">f4")
vWind_forcing = np.fromfile(path + "u_1ms_1x1_one_year", dtype = ">f4")


print(f"AirTemp{AirTemp_forcing[:10]}")
print(f"LWDown{LWDown_forcing[:10]}")
print(f"SWDown{SWDown_forcing[:10]}")
print(f"uWind{uWind_forcing[:10]}")
print(f"vWind{vWind_forcing[:10]}")

print(np.shape(AirTemp_forcing))