from veros.core.operators import update, at

### input:
# field of size (sNy+2*OLy, sNx+2*OLx) where [OLy:-OLy,OLx:-OLx] (=[OLy:-OLy,OLx:-OLx]) is the actual cell

### output:
# field with filled overlaps (using the values from the actual cell)
# this requires that the cell size is larger than the overlap (sN > OL)!


def fill_overlap(A):
    from seaice_size import OLy, OLx
    A = update(A, at[:OLy,:], A[-2*OLy:-OLy,:])
    A = update(A, at[-OLy:,:], A[OLy:2*OLy,:])
    A = update(A, at[:,:OLx], A[:,-2*OLx:-OLx])
    A = update(A, at[:,-OLx:], A[:,OLx:2*OLx])

    return A


def fill_overlap3d(A):
    from seaice_size import OLy, OLx
    A = update(A, at[:,:OLy,:], A[:,-2*OLy:-OLy,:])
    A = update(A, at[:,-OLy:,:], A[:,OLy:2*OLy,:])
    A = update(A, at[:,:,:OLx], A[:,:,-2*OLx:-OLx])
    A = update(A, at[:,:,-OLx:], A[:,:,OLx:2*OLx])

    return A


def fill_overlap_uv(U,V):
    return fill_overlap(U), fill_overlap(V)
