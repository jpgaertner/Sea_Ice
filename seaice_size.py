#Declaring the size of the underlying grid

# The design here supports a three-dimensional model grid
# with indices I,J and K. The three-dimensional domain
# is comprised of nPx*nSx blocks (or tiles) of size sNx
# along the first (left-most index) axis, nPy*nSy blocks
# of size sNy along the second axis and one block of size
# Nr along the vertical (third) axis.
# Blocks/tiles have overlap regions of size OLx and OLy
# along the dimensions that are subdivided.


import numpy as np

sNx =   1 #Number of X points in tile
sNy =   1 #Number of Y points in tile
OLx =   2 #Tile overlap extent in X
OLy =   2 #Tile overlap extent in Y
#nSx =   1 #Number of tiles per process in X
#nSy =   1 #Number of tiles per process in Y
#tiles are subprocesses that can work with the same memory
#nPx =   1 #Number of processes to use in X
#nPy =   1 #Number of processes to use in Y
#different processes for working with multiple cpus
#Nx  = sNx*nSx*nPx #Number of points in X for the full domain
#Ny  = sNy*nSy*nPy #Number of points in Y for the full domain
Nr  =  26 #Number of points in vertical direction

#nITC is the number of ice thickness categories to allocate.
#Instead of assuming just one thickness for a grid cell, a distribution of nITC thicknesses (centered around the mean value and containing equal fractions of the area) is assumed.
#Bitz et al. (2001, JGR) suggest a minimum of nITD = 5
nITC = 1
recip_nITC = 1 / nITC

#grid descriptor variables
dxC = np.ones((sNx+2*OLx, sNy+2*OLy)) #distance between two adjacent cell centers in x direction across western cell wall [m]
dyC = np.ones((sNx+2*OLx, sNy+2*OLy)) #distance between two adjacent cell centers in y direction across southern cell wall [m]
dxG = np.ones((sNx+2*OLx, sNy+2*OLy)) #distance between cell faces (cell width) in x direction along southern cell wall [m]
dyG = np.ones((sNx+2*OLx, sNy+2*OLy)) #distance between cell faces (cell width) in y direction along western cell wall [m]
dxF = np.ones((sNx+2*OLx, sNy+2*OLy)) #distance between cell faces (cell width) in x direction through cell center [m]
dyF = np.ones((sNx+2*OLx, sNy+2*OLy)) #distance between cell faces (cell width) in y direction through cell center [m]
dxV = np.ones((sNx+2*OLx, sNy+2*OLy)) #distance between two adjacent v points in x direction across south-west corner of the cell [m]
dyU = np.ones((sNx+2*OLx, sNy+2*OLy)) #distance between two adjacent u points in y direction across south-west corner of the cell [m]
recip_dxC = 1 / dxC
recip_dyC = 1 / dyC
recip_dxG = 1 / dxG
recip_dyG = 1 / dyG
recip_dxF = 1 / dxF
recip_dyF = 1 / dyF
recip_dxV = 1 / dxV
recip_dyU = 1 / dyU

rAz = np.ones((sNx+2*OLx, sNy+2*OLy))

fCori = np.ones((sNx+2*OLx, sNy+2*OLy))*1e-4 #coriolis parameter at grid center point
fCoriG = np.ones((sNx+2*OLx, sNy+2*OLy))*1e-4 #coriolis parameter at grid corner point (south west corner?)
