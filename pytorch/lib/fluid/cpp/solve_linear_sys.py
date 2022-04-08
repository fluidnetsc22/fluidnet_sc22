import torch
import fluidnet_cpp

def solveLinearSystemJacobi(flags, div, is_3d=False, p_tol=1e-5, max_iter=1000, verbose=True):
    r"""Solves the linear system using the Jacobi method.
        Note: Since we don't receive a velocity field, we need to receive the is3D
        flag from the caller.

    Arguments:
        flags (Tensor): Input occupancy grid.
        div (Tensor): The velocity divergence.
        is_3d (Tensor, optional): If True, a 3D domain is expected.
        p_tol (float, optional): ||p - p_prev|| termination condition.
            Defaults 1e-5.
        max_iter (int, optional): Maximum number of Jacobi iterations.
            Defaults 1000.
        verbose (bool, optional). Defaults False.
    Output:
        p (Tensor): Pressure field
        p_tol: Maximum residual accross all batches.

    """
    #Check sizes

    assert div.dim() == 5 and flags.dim() == 5, "Dimension mismatch"
    assert flags.size(1) == 1, "flags is not scalar"

    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    assert div.is_same_size(flags), "Size mismatch"

    assert flags.is_contiguous() and div.is_contiguous(), "Input is not contiguous"

    p, p_tol = fluidnet_cpp.solve_linear_system_Jacobi(flags, div, is_3d, \
            p_tol, max_iter, verbose)

    return p, p_tol


def solveLinearSystemJacobi_Density(flags, div, density, is_3d=False, p_tol=1e-5, max_iter=1000, verbose=True):
    r"""Solves the linear system using the Jacobi method.
        Note: Since we don't receive a velocity field, we need to receive the is3D
        flag from the caller.

    Arguments:
        flags (Tensor): Input occupancy grid.
        div (Tensor): The velocity divergence.
        is_3d (Tensor, optional): If True, a 3D domain is expected.
        p_tol (float, optional): ||p - p_prev|| termination condition.
            Defaults 1e-5.
        max_iter (int, optional): Maximum number of Jacobi iterations.
            Defaults 1000.
        verbose (bool, optional). Defaults False.
    Output:
        p (Tensor): Pressure field
        p_tol: Maximum residual accross all batches.

    """
    #Check sizes

    assert div.dim() == 5 and flags.dim() == 5, "Dimension mismatch"
    assert flags.size(1) == 1, "flags is not scalar"

    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    assert div.is_same_size(flags), "Size mismatch"

    assert flags.is_contiguous() and div.is_contiguous(), "Input is not contiguous"

    div_rho = div * (1-density)

    p, p_tol = fluidnet_cpp.solve_linear_system_Jacobi(flags, div_rho, is_3d, \
            p_tol, max_iter, verbose)

    return p, p_tol


def solveLinearSystemPCG(flags, div, inflow,is_3d=False, p_tol=1e-5, max_iter=1000, verbose=False):
    r"""Solves the linear system using the Jacobi method.
        Note: Since we don't receive a velocity field, we need to receive the is3D
        flag from the caller.

    Arguments:
        flags (Tensor): Input occupancy grid.
        div (Tensor): The velocity divergence.
        is_3d (Tensor, optional): If True, a 3D domain is expected.
        p_tol (float, optional): ||p - p_prev|| termination condition.
            Defaults 1e-5.
        max_iter (int, optional): Maximum number of Jacobi iterations.
            Defaults 1000.
        verbose (bool, optional). Defaults False.
    Output:
        p (Tensor): Pressure field
        p_tol: Maximum residual accross all batches.

    """
    #Check sizes

    assert div.dim() == 5 and flags.dim() == 5, "Dimension mismatch"
    assert flags.size(1) == 1, "flags is not scalar"

    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    assert div.is_same_size(flags), "Size mismatch"

    assert flags.is_contiguous() and div.is_contiguous(), "Input is not contiguous"

    p, p_tol = fluidnet_cpp.solve_linear_system_PCG(flags, div, inflow, is_3d, \
            p_tol, max_iter, verbose)

    return p, p_tol

def solveLinearSystemCG(flags, pressure, div, A_val, I_A, J_A,is_3d=False, p_tol=1e-5, max_iter=1000, verbose=False):
    r"""Solves the linear system using the Jacobi method.
        Note: Since we don't receive a velocity field, we need to receive the is3D
        flag from the caller.

    Arguments:
        flags (Tensor): Input occupancy grid.
        div (Tensor): The velocity divergence.
        is_3d (Tensor, optional): If True, a 3D domain is expected.
        p_tol (float, optional): ||p - p_prev|| termination condition.
            Defaults 1e-5.
        max_iter (int, optional): Maximum number of Jacobi iterations.
            Defaults 1000.
        verbose (bool, optional). Defaults False.
    Output:
        p (Tensor): Pressure field
        p_tol: Maximum residual accross all batches.

    """
    #Check sizes

    assert div.dim() == 5 and flags.dim() == 5, "Dimension mismatch"
    assert flags.size(1) == 1, "flags is not scalar"

    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    assert div.is_same_size(flags), "Size mismatch"

    assert flags.is_contiguous() and div.is_contiguous(), "Input is not contiguous"

    residual = 0.0

    div_vec_t = torch.FloatTensor(torch.flatten(div[0,0].cpu()))
    div_vec = div_vec_t.clone()


    fluidnet_cpp.solve_linear_system_CG(flags, div_vec, A_val, I_A, J_A, pressure, residual, is_3d, \
            p_tol, max_iter, verbose)

    #print("End sol sys, residual = ", residual)

    return residual
