from veros.core.operators import update, at

### input:
# field of size (ny+2*oly, nx+2*olx) where [oly:-oly,olx:-olx] (=[oly:-oly,olx:-olx]) is the actual cell

### output:
# field with filled overlaps (using the values from the actual cell)
# this requires that the cell size is larger than the overlap (sN > OL)!


def fill_overlap(A):
    from seaice_size import oly, olx
    A = update(A, at[:oly,:], A[-2*oly:-oly,:])
    A = update(A, at[-oly:,:], A[oly:2*oly,:])
    A = update(A, at[:,:olx], A[:,-2*olx:-olx])
    A = update(A, at[:,-olx:], A[:,olx:2*olx])

    return A


def fill_overlap3d(A):
    from seaice_size import oly, olx
    A = update(A, at[:,:oly,:], A[:,-2*oly:-oly,:])
    A = update(A, at[:,-oly:,:], A[:,oly:2*oly,:])
    A = update(A, at[:,:,:olx], A[:,:,-2*olx:-olx])
    A = update(A, at[:,:,-olx:], A[:,:,olx:2*olx])

    return A


def fill_overlap_uv(U,V):
    return fill_overlap(U), fill_overlap(V)
