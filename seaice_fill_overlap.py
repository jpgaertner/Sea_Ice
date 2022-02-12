import numpy as np

#from seaice_size import OLx, OLy

### input:
# A: field of size (sNy+2*OLy, sNx+2*OLx) where [OLy:-OLy,OLx:-OLx] (=[OLy:sNy+OLy,OLx:sNx+OLx]) is the actual cell

### output:
# A with filled overlaps (using the values from the actual cell)
# this requires that the cell size is larger than the overlap (sN > OL)!

# sNx = 15 #Number of X points in tile
# sNy = 15 #Number of Y points in tile
# OLx = 2 #Tile overlap extent in X
# OLy = 2 #Tile overlap extent in Y

def fill_overlap(A):
    from seaice_size import OLy, OLx
    A[:OLy,:]  = A[-2*OLy:-OLy,:]
    A[-OLy:,:] = A[OLy:2*OLy,:]
    A[:,:OLx]  = A[:,-2*OLx:-OLx]
    A[:,-OLx:] = A[:,OLx:2*OLx]

    return A


def fill_overlap3d(A):
    from seaice_size import OLy, OLx
    A[:,:OLy,:]  = A[:,-2*OLy:-OLy,:]
    A[:,-OLy:,:] = A[:,OLy:2*OLy,:]
    A[:,:,:OLx]  = A[:,:,-2*OLx:-OLx]
    A[:,:,-OLx:] = A[:,:,OLx:2*OLx]

    return A

def fill_overlap_uv(U,V):
    return fill_overlap(U), fill_overlap(V)
