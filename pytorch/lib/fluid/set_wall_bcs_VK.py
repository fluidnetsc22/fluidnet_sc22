import torch
from . import CellType


def setWallVKBcs(U, flags, BC_V):
    r"""Sets velocity V input tangentially in the wall.

    Arguments:
        U (Tensor): Input velocity.
        flags (Tensor): Input occupancy grid.
    Output:
        U (Tensor): Output velocity (with enforced BCs).
    """
    cuda = torch.device('cuda')
    assert (U.dim() == 5 and flags.dim() == 5), 'Dimension mismatch'
    assert flags.size(1) == 1, 'flags is not a scalar'
    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    is3D = (U.size(1) == 3)
    if (not is3D):
        assert d == 1, '2D velocity field but zdepth > 1'
        assert U.size(1) == 2, '2D velocity field must have only 2 channels'

    assert (U.size(0) == bsz and U.size(2) == d and U.size(3) == h and U.size(4) == w),\
        'Size mismatch'
    assert (U.is_contiguous() and flags.is_contiguous()), 'Input is not contiguous'

    # Hard Coded, new BC U_Scale = 0.05
    u_scale = BC_V

    i = torch.arange(start=0, end=w, dtype=torch.long, device=cuda) \
        .view(1, w).expand(bsz, d, h, w)
    j = torch.arange(start=0, end=h, dtype=torch.long, device=cuda) \
        .view(1, h, 1).expand(bsz, d, h, w)
    k = torch.zeros_like(i)
    if (is3D):
        k = torch.arange(start=0, end=d, dtype=torch.long, device=cuda) \
            .view(1, d, 1, 1).expand(bsz, d, h, w)

    zero = torch.zeros_like(i)
    zeroBy = torch.zeros(i.size(), dtype=torch.uint8, device=cuda)

    idx_b = torch.arange(start=0, end=bsz, dtype=torch.long, device=cuda) \
        .view(bsz, 1, 1, 1).expand(bsz, d, h, w)

    mCont = torch.ones_like(zeroBy)

    if (not is3D):
        U[:, 1, :, 1:-1, 1] = u_scale
        U[:, 1, :, 1:-1, -1] = u_scale
        U[:, 1, :, 1:-1, 2] = u_scale
        U[:, 1, :, 1:-1, -2] = u_scale
    else:
        U[:, 1, 1:-1, 1:-1, :2] = u_scale
        U[:, 1, 1:-1, 1:-1, -2:] = u_scale

    return U
