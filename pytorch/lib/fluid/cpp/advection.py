import torch
import fluidnet_cpp

def __check_advection_method__(method):
    assert (method == 'eulerFluidNet' or method == 'maccormackFluidNet'), \
            'Error: Advection method not supported. Options are: \
                     maccormackFluidNet, eulerFluidNet'

def correctScalar(dt, src, div, flags):
    maskFluid = flags.eq(1)
    src.masked_scatter_(maskFluid, \
            (src + dt*0.5*src*div).masked_select(maskFluid))

def advectScalar(dt, src, U, flags, method = 'maccormackFluidNet', boundary_width = 1,
        sample_outside_fluid = False, maccormack_strength = 0.75):
    r"""Advects scalar field src by the input vel field U

    Arguments:
        dt (float): timestep in seconds.
        src (Tensor): input scalar field, to be advected.
            Shape is (batch,1,D,H,W) with D=1 for 2D
            and D>1 for 3D simulations.
        U (Tensor): input velocity field.
            Shape is (batch,2/3,D,H,W) with D=1 for 2D
            and D>1 for 3D simulations.
        flags (Tensor): Input occupancy grid.
        method (string, optional): Sets the method of advection.
            Options are eulerFluidNet and maccormackFluidNet.
            Defaults to maccormackFluidNet.
        boundary_width (int, optional): width of fluid domain boundary.
            Defaults to 1.
        sample_outside_fluid(bool, optional): For density advection, we do not want
            to advect values inside non-fluid cells and so this should be set to false.
            For other quantities (like temperature), this should be true.
            Defaults to ''False''.
        maccormack_strength (float, optional): A strength parameter that
            will make the advection eularian (with values interpolating in between). A
            value of 1 (which implements the update from An Unconditionally Stable
            MaCormack Method) tends to add too much high-frequency detail.
    """
    #Check sizes
    __check_advection_method__(method)

    assert src.dim() == 5 and U.dim() == 5 and flags.dim() == 5, "Dimension mismatch"
    assert flags.size(1) == 1, "flags is not scalar"

    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    is3D = U.size(1) == 3
    if (not is3D):
       assert d == 1, "2D velocity field but zdepth > 1"
       assert U.size(1) == 2, "2D velocity field must have only 2 channels"

    assert U.size(0) == bsz and U.size(2) == d and \
               U.size(3) == h and U.size(4) == w, "Size mismatch"
    assert U.is_contiguous() and flags.is_contiguous() and \
             src.is_contiguous(), "Input is not contiguous"

    s_dst = fluidnet_cpp.advect_scalar(dt, src, U, flags, method,
            boundary_width, sample_outside_fluid, maccormack_strength)
    return s_dst

def advectVelocity(dt, orig, U, flags, method = 'maccormackFluidNet', boundary_width = 1,
        maccormack_strength = 0.75):
    r"""Advects velocity field orig by velocity field U

    Arguments:
        dt (float): timestep in seconds.
        orig (Tensor): velocity field to be advected.
            Shape is (batch,2/3,D,H,W) with D=1 for 2D
            and D>1 for 3D simulations.
        U (Tensor): non-divergent velocity field to advect orig.
            Shape is (batch,2/3,D,H,W) with D=1 for 2D
            and D>1 for 3D simulations.
        flags (Tensor): Input occupancy grid.
        method (string, optional): Sets the method of advection.
            Options are eulerFluidNet and maccormackFluidNet.
            Defaults to maccormackFluidNet.
        boundary_width (int, optional): width of fluid domain boundary.
            Defaults to 1.
        maccormack_strength (float, optional): A strength parameter that
            will make the advection eularian (with values interpolating in between). A
            value of 1 (which implements the update from An Unconditionally Stable
            MaCormack Method) tends to add too much high-frequency detail.
    """

    #Check sizes

    assert U.dim() == 5 and orig.dim() == 5 and flags.dim() == 5, "Dimension mismatch"
    assert flags.size(1) == 1, "flags is not scalar"

    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)


    Open_mask = flags.eq(16)
    Openboundary = (Open_mask.sum()>1)

    #print("Boundary ", Openboundary)

    is3D = U.size(1) == 3
    if (not is3D):
       assert d == 1, "2D velocity field but zdepth > 1"
       assert orig.size(1) == 2, "2D velocity field must have only 2 channels"
       assert U.size(1) == 2, "2D velocity field must have only 2 channels"

    assert U.size(0) == bsz and U.size(2) == d and \
               U.size(3) == h and U.size(4) == w, "Size mismatch"
    assert orig.size(0) == bsz and orig.size(2) == d and \
               orig.size(3) == h and orig.size(4) == w, "Size mismatch"
    assert U.is_contiguous() and orig.is_contiguous() and flags.is_contiguous(), "Input is not contiguous"


    U_dst = fluidnet_cpp.advect_vel(dt, orig, U, flags, method,
            boundary_width, Openboundary,maccormack_strength)


    return U_dst
