import torch
from . import CellType


def velocityVorticity(U, flags):
    r""" Calculates the velocity Vorticity on 2D (with boundary cond modifications). This is
    essentially a replica of VelocityDivergence.

    Arguments:
        U (Tensor): input vel field (should  be 2 , indicating 2D)
        flags (Tensor): input occupancy grid
    Output:
        UDiv (Tensor) : output divergence (scalar field).
    """
    # Check sizes
    vorticity = torch.zeros_like(flags).type(U.type())
    assert (U.dim() == 5 and flags.dim() == 5 and vorticity.dim() == 5), \
        "Dimension mismatch"
    assert flags.size(1) == 1, "flags is not scalar"
    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    z = 2
    y = 3
    x = 4

    is_3d = (U.size(1) == 3)
    if (not is_3d):
        assert d == 1, "2D velocity field but zdepth > 1"
        assert (U.size(1) == 2), "2D velocity field must have only 2 channels"

    assert (U.size(0) == bsz and U.size(2) == d and
            U.size(3) == h and U.size(4) == w), "Size mismatch"

    assert (U.is_contiguous() and flags.is_contiguous() and
            vorticity.is_contiguous()), "Input is not contiguous"

    #Uijk : Velocity in ijk
    # Uijk_p : Velocity in (i+1),(j+1),(k+1)

    # We call this only on fluid cells. Non-fluid cells have a zero divergence
    #isFluid = flags.eq(CellType.TypeFluid)
    #noFluid = isFluid.ne(CellType.TypeFluid)

    if (not is_3d):
        Uijk = U.narrow(x, 1, w - 2).narrow(y, 1, h - 2)
        Uijk_p = Uijk.clone()
        Uijk_p[:, 0] = U.narrow(x, 2, w - 2).narrow(y, 1, h - 2).select(1, 0)
        Uijk_p[:, 1] = U.narrow(x, 1, w - 2).narrow(y, 2, h - 2).select(1, 1)
    else:
        Uijk = U.narrow(x, 1, w - 2).narrow(y, 1, h - 2).narrow(z, 1, d - 2)
        Uijk_p = Uijk.clone()
        Uijk_p[:, 0] = U.narrow(x, 2, w - 2).narrow(y, 1, h - 2).narrow(z, 1, d - 2).select(1, 0)
        Uijk_p[:, 1] = U.narrow(x, 1, w - 2).narrow(y, 2, h - 2).narrow(z, 1, d - 2).select(1, 1)
        Uijk_p[:, 2] = U.narrow(x, 1, w - 2).narrow(y, 1, h - 2).narrow(z, 2, d - 2).select(1, 2)

    vor = -(Uijk.select(1, 0) - Uijk_p.select(1, 0)) + \
        (Uijk.select(1, 1) - Uijk_p.select(1, 1))

    if (is_3d):
        vor += Uijk.select(1, 2) - Uijk_p.select(1, 2)
    if (not is_3d):
        vorticity[:, :, :, 1:(h - 1), 1:(w - 1)] = vor.view(bsz, 1, d, h - 2, w - 2)
    else:
        vorticity[:, :, 1:(d - 1), 1:(h - 1), 1:(w - 1)] = div.view(bsz, 1, d - 2, h - 2, w - 2)

    # Set div to 0 in obstacles
    mask_obst = flags.eq(CellType.TypeObstacle)
    vorticity.masked_fill_(mask_obst, 0)

    return vorticity
