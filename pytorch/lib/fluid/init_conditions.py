import torch
import math
import numpy as np
from scipy.stats import multivariate_normal


def createVKBCs(batch_dict, density_val, u_scale, rad):
    r"""Creates masks to enforce an inlet at the domain bottom wall.
    Modifies batch_dict inplace.
    Arguments:
        batch_dict (dict): Input tensors (p, UDiv, flags, density)
        density_val (float): Inlet density.
        u_scale (float); Inlet velocity.
        rad (float): radius of inlet circle (centered around midpoint of wall)
    """

    # Jet length (jl -a)
    jl = 4
    # Jet first cell point
    a = 1

    flags = batch_dict['flags']

    cuda = torch.device('cuda')
    # batch_dict at input: {p, UDiv, flags, density, Ustar, Div_input,VK, simutype}
    #assert len(batch_dict) == 8, "Batch must contain 8 tensors (p, UDiv, flags, density, flags_inflow, Ustar, Div input, VK, simutype)"
    UDiv = batch_dict['U']
    density = batch_dict['density']
    UBC = UDiv.clone().fill_(0)
    UBCInvMask = UDiv.clone().fill_(1)

    # Single density value
    densityBC = density.clone().fill_(0)
    densityBCInvMask = density.clone().fill_(1)

    assert UBC.dim() == 5, 'UBC must have 5 dimensions'
    assert UBC.size(0) == 1, 'Only single batches allowed (inference)'

    xdim = UBC.size(4)
    ydim = UBC.size(3)
    zdim = UBC.size(2)
    is3D = (UBC.size(1) == 3)

    if not is3D:
        assert zdim == 1, 'For 2D, zdim must be 1'
    centerX = xdim // 2
    centerZ = max(zdim // 2, 1.0)
    # Remember that floor (5.6 = 5, -7.1 = -7)
    plumeRad = math.floor(xdim * rad)

    y = 1
    if (not is3D):
        #vec = (0,1)
        vec = torch.arange(0, 2, device=cuda).float()
    else:
        vec = torch.arange(0, 3, device=cuda).float()
        vec[2] = 0

    vec.mul_(u_scale)

    # Equal to = vector size H, then reshaped to a matrix of size (H,1) and expanded
    index_x = torch.arange(0, xdim, device=cuda).view(xdim).expand_as(density[0][0])
    index_y = torch.arange(0, ydim, device=cuda).view(ydim, 1).expand_as(density[0][0])
    if (is3D):
        index_z = torch.arange(0, zdim, device=cuda).view(zdim, 1, 1).expand_as(density[0][0])

    if (not is3D):
        index_ten = torch.stack((index_x, index_y), dim=0)
    else:
        index_ten = torch.stack((index_x, index_y, index_z), dim=0)

    indx_circle = index_ten[:, :, a:jl]
    maskInside = (indx_circle[1] <= jl)

    # Inside the plume. Set the BCs.

    # It is clearer to just multiply by mask (casted into Float)
    maskInside_f = maskInside.float().clone()

    UBC[:, :, :, a:jl] = maskInside_f * vec.view(1, 3, 1, 1, 1).expand_as(UBC[:, :, :, a:jl]).float()
    UBCInvMask[:, :, :, a:jl].masked_fill_(maskInside, 0)

    densityBC[:, :, :, a:jl].masked_fill_(maskInside, density_val)
    densityBCInvMask[:, :, :, a:jl].masked_fill_(maskInside, 0)

    # Outside the plume. Set the velocity to zero and leave density alone.
    maskOutside = (maskInside == 0)
    UBC[:, :, :, a:jl].masked_fill_(maskOutside, 0)
    UBCInvMask[:, :, :, a:jl].masked_fill_(maskOutside, 0)

    # Outflow
    indx_circle = index_ten[:, :, -jl:]
    maskInside = (indx_circle[1] >= ydim - jl)

    # Inside the plume. Set the BCs.

    # It is clearer to just multiply by mask (casted into Float)
    maskInside_f = maskInside.float().clone()

    UBC[:, :, :, -jl:] = maskInside_f * vec.view(1, 3, 1, 1, 1).expand_as(UBC[:, :, :, -jl:]).float()
    UBCInvMask[:, :, :, -jl:].masked_fill_(maskInside, 0)
    densityBCInvMask[:, :, :, -jl:].masked_fill_(maskInside, 0)

    # Outside the plume. Set the velocity to zero and leave density alone.
    maskOutside = (maskInside == 0)
    UBC[:, :, :, -jl:].masked_fill_(maskOutside, 0)
    UBCInvMask[:, :, :, -jl:].masked_fill_(maskOutside, 0)

    # Insert the new tensors in the batch_dict.
    batch_dict['UBC'] = UBC
    batch_dict['UBCInvMask'] = UBCInvMask
    batch_dict['densityBC'] = densityBC
    batch_dict['densityBCInvMask'] = densityBCInvMask


def createStepBCs(batch_dict, density_val, u_scale, rad, resX, Long_S_X):
    r"""Creates masks to enforce an inlet at the domain bottom wall.
    Modifies batch_dict inplace.
    Arguments:
        batch_dict (dict): Input tensors (p, UDiv, flags, density)
        density_val (float): Inlet density.
        u_scale (float); Inlet velocity.
        rad (float): radius of inlet circle (centered around midpoint of wall)
    """

    # Jet length (jl -a)
    jl = 4
    # Jet first cell point
    a = 1

    flags = batch_dict['flags']

    cuda = torch.device('cuda')

    UDiv = batch_dict['U']
    density = batch_dict['density']
    UBC = UDiv.clone().fill_(0)
    UBCInvMask = UDiv.clone().fill_(1)

    # Single density value
    densityBC = density.clone().fill_(0)
    densityBCInvMask = density.clone().fill_(1)

    assert UBC.dim() == 5, 'UBC must have 5 dimensions'
    assert UBC.size(0) == 1, 'Only single batches allowed (inference)'

    xdim = UBC.size(4)
    ydim = UBC.size(3)
    zdim = UBC.size(2)
    is3D = (UBC.size(1) == 3)

    if not is3D:
        assert zdim == 1, 'For 2D, zdim must be 1'
    centerX = xdim // 2
    centerZ = max(zdim // 2, 1.0)
    # Remember that floor (5.6 = 5, -7.1 = -7)
    plumeRad = math.floor(xdim * rad)

    y = 1
    if (not is3D):
        #vec = (0,1)
        vec = torch.arange(0, 2, device=cuda).float()
    else:
        vec = torch.arange(0, 3, device=cuda).float()
        vec[2] = 0

    # Equal to = vector size H, then reshaped to a matrix of size (H,1) and expanded
    index_x = torch.arange(0, xdim, device=cuda).view(xdim).expand_as(density[0][0])
    index_y = torch.arange(0, ydim, device=cuda).view(ydim, 1).expand_as(density[0][0])
    if (is3D):
        index_z = torch.arange(0, zdim, device=cuda).view(zdim, 1, 1).expand_as(density[0][0])

    if (not is3D):
        index_ten = torch.stack((index_x, index_y), dim=0)
    else:
        index_ten = torch.stack((index_x, index_y, index_z), dim=0)

    h_s = np.int(xdim / 2.0)

    input_U = -6.0 * u_scale * (index_x) * (index_x - h_s) / (h_s * h_s)
    input_U[:, :, h_s:-1] = 0.0
    output_U = -3.0 * u_scale * (index_x) * (index_x - 2.0 * h_s) / (4. * (h_s * h_s))

    BC_Uy = input_U
    BC_Uy[:, ydim // 2:, :] = output_U[:, ydim // 2:, :]
    BC_Uy[:, jl:ydim - jl - 1, :] = 0.0

    BC_Ux = torch.zeros_like(BC_Uy)

    BC_U = torch.stack((BC_Ux, BC_Uy), dim=0)

    maskInside_in = (index_y[0, :, :] <= jl)
    maskInside_out = (index_y[0, :, :] >= ydim - jl)

    maskInside = maskInside_in + maskInside_out
    # Inside the plume. Set the BCs.

    # It is clearer to just multiply by mask (casted into Float)
    maskInside_f = maskInside.float().clone()

    UBC = BC_U.unsqueeze(0)

    densityBC[:, :, :, a:jl].masked_fill_(maskInside[a:jl], density_val)
    densityBCInvMask[:, :, :, a:jl].masked_fill_(maskInside[a:jl], 0)

    densityBC[:, :, :, -jl - 1:].masked_fill_(maskInside[-jl - 1:], density_val)
    densityBCInvMask[:, :, :, -jl - 1:].masked_fill_(maskInside[-jl - 1:], 0)

    # Outside the plume. Set the velocity to zero and leave density alone.

    maskOutside = (maskInside == 0)
    UBC.masked_fill_(maskOutside, 0)
    UBCInvMask.masked_fill_(maskOutside, 0)

    # Outflow
    indx_circle = index_ten[:, :, -jl:]
    maskInside = (indx_circle[1] >= ydim - jl)

    # Inside the plume. Set the BCs.

    # It is clearer to just multiply by mask (casted into Float)
    maskInside_f = maskInside.float().clone()

    # DEBUG
    UBCInvMask[:, :, :, -jl:].masked_fill_(maskInside, 0)
    densityBCInvMask[:, :, :, -jl:].masked_fill_(maskInside, 0)

    # Outside the plume. Set the velocity to zero and leave density alone.

    maskOutside = (maskInside == 0)
    # Insert the new tensors in the batch_dict.
    batch_dict['UBC'] = UBC
    batch_dict['UBCInvMask'] = UBCInvMask
    batch_dict['densityBC'] = densityBC
    batch_dict['densityBCInvMask'] = densityBCInvMask


def createPlumeBCs(batch_dict, density_val, u_scale, rad):
    r"""Creates masks to enforce an inlet at the domain bottom wall.
    Modifies batch_dict inplace.
    Arguments:
        batch_dict (dict): Input tensors (p, UDiv, flags, density)
        density_val (float): Inlet density.
        u_scale (float); Inlet velocity.
        rad (float): radius of inlet circle (centered around midpoint of wall)
    """

    # Jet length (jl -a)
    jl = 3
    # Jet first cell point
    a = 1

    flags = batch_dict['flags']

    cuda = torch.device('cuda')
    # batch_dict at input: {p, UDiv, flags, density, Ustar, Div_input}
    UDiv = batch_dict['U']
    density = batch_dict['density']
    UBC = UDiv.clone().fill_(0)
    UBCInvMask = UDiv.clone().fill_(1)

    # Single density value
    densityBC = density.clone().fill_(0)
    densityBCInvMask = density.clone().fill_(1)

    assert UBC.dim() == 5, 'UBC must have 5 dimensions'
    assert UBC.size(0) == 1, 'Only single batches allowed (inference)'

    xdim = UBC.size(4)
    ydim = UBC.size(3)
    zdim = UBC.size(2)
    is3D = (UBC.size(1) == 3)
    # if not is3D:
    #    assert zdim == 1, 'For 2D, zdim must be 1'
    centerX = xdim // 2
    centerZ = max(zdim // 2, 1.0)
    # Remember that floor (5.6 = 5, -7.1 = -7)
    plumeRad = math.floor(xdim * rad)

    y = 1
    if (not is3D):
        #vec = (0,1)
        vec = torch.arange(0, 2, device=cuda).float()
    else:
        vec = torch.arange(0, 3, device=cuda).float()
        vec[2] = 0

    # vec = vec * u_scale (vinj)
    vec.mul_(u_scale)
    print("V INJ = ", vec[1])
    print("Scale", u_scale)

    # Equal to = vector size H, then reshaped to a matrix of size (H,1) and expanded
    index_x = torch.arange(0, xdim, device=cuda).view(xdim).expand_as(density[0][0])
    index_y = torch.arange(0, ydim, device=cuda).view(ydim, 1).expand_as(density[0][0])
    if (is3D):
        index_z = torch.arange(0, zdim, device=cuda).view(zdim, 1, 1).expand_as(density[0][0])

    if (not is3D):
        index_ten = torch.stack((index_x, index_y), dim=0)
    else:
        index_ten = torch.stack((index_x, index_y, index_z), dim=0)

    indx_circle = index_ten[:, :, a:jl]
    if (not is3D):
        indx_circle[0] -= centerX
        maskInside = (indx_circle[0].pow(2) <= plumeRad * plumeRad)
    else:
        indx_circle[0] -= centerX
        indx_circle[2] -= centerZ
        maskInside = ((indx_circle[0].pow(2) + indx_circle[2].pow(2)) <= plumeRad * plumeRad)

    # Inside the plume. Set the BCs.

    # It is clearer to just multiply by mask (casted into Float)
    maskInside_f = maskInside.float().clone()

    # Tan H try:
    delt = 5
    ind_x = torch.arange(0, xdim).view(xdim).float()

    if (not is3D):
        UBC[:, :, :, a:jl] = maskInside_f * vec.view(1, 2, 1, 1, 1).expand_as(UBC[:, :, :, a:jl]).float()
    else:
        UBC[:, :, :, a:jl] = maskInside_f * vec.view(1, 3, 1, 1, 1).expand_as(UBC[:, :, :, a:jl]).float()

    UBCInvMask[:, :, :, a:jl].masked_fill_(maskInside, 0)

    densityBC[:, :, :, a:jl].masked_fill_(maskInside, density_val)
    densityBCInvMask[:, :, :, a:jl].masked_fill_(maskInside, 0)

    # Outside the plume. Set the velocity to zero and leave density alone.

    maskOutside = (maskInside == 0)
    UBC[:, :, :, a:jl].masked_fill_(maskOutside, 0)
    UBCInvMask[:, :, :, a:jl].masked_fill_(maskOutside, 0)

    # Only inside the domain
    UBC = torch.where(flags == 2 * torch.ones_like(flags), torch.zeros_like(UBC).float(), UBC)
    UBCInvMask = torch.where(flags == 2 * torch.ones_like(flags), torch.zeros_like(UBCInvMask).float(), UBCInvMask)

    densityBC = torch.where(flags == 2 * torch.ones_like(flags), torch.zeros_like(densityBC).float(), densityBC)
    densityBCInvMask = torch.where(
        flags == 2 * torch.ones_like(flags),
        torch.zeros_like(densityBCInvMask).float(),
        densityBCInvMask)

    # Insert the new tensors in the batch_dict.
    batch_dict['UBC'] = UBC
    batch_dict['UBCInvMask'] = UBCInvMask
    batch_dict['densityBC'] = densityBC
    batch_dict['densityBCInvMask'] = densityBCInvMask


def createCilinder(batch_dict):
    """
    Creates a cilinder in the flags. It will be located in the point x = 64 and y = 80.
    Radius = 10
    """
    flags = batch_dict['flags']
    resX = flags.size(4)
    resY = flags.size(3)

    # Here, we just impose initial conditions.
    # Upper layer rho2, vel = 0
    # Lower layer rho1, vel = 0

    centerX = 64
    centerY = 80

    radCyl = 10

    X = torch.arange(0, resX, device=cuda).view(resX).expand((1, resY, resX))
    Y = torch.arange(0, resY, device=cuda).view(resY, 1).expand((1, resY, resX))

    dist_from_center = (X - centerX).pow(2) + (Y - centerY).pow(2)
    mask_cylinder = dist_from_center <= radCyl * radCyl

    flags = flags.masked_fill_(mask_cylinder, 2)


def createBubble(batch_dict, mconf, rho1, height, radCyl):
    r"""Adds a bubble of radius R at the height h on the center of the domain
    Arguments:
        batch_dict (dict): Input tensors (p, UDiv, flags, density)
        mconf (dict): configuration dict (to set thickness and amplitude of interface).
        rho1 (float): Bubble density.
        height (float): Bubble's relative size.
    """

    cuda = torch.device('cuda')

    # batch_dict at input: {p, UDiv, flags, density, UStar, divergency input}
    assert len(batch_dict) == 7, "Batch must contain 7 tensors (p, UDiv, flags, density,Ustar, divergency input)"
    UDiv = batch_dict['U']
    flags = batch_dict['flags']

    resX = UDiv.size(4)
    resY = UDiv.size(3)
    resZ = UDiv.size(2)
    if resZ > 1:
        is3D = True
    else:
        is3D = False

    # Old BC Distribution
    density = batch_dict['density']

    X = torch.arange(0, resX, device=cuda).view(resX).expand((1, resZ, resY, resX)).float()
    Y = torch.arange(0, resY, device=cuda).view(resY, 1).expand((1, resZ, resY, resX)).float()
    Z = torch.arange(0, resZ, device=cuda).view(resZ, 1, 1).expand((1, resZ, resY, resX)).float()

    if is3D:
        coord = torch.cat((X, Y, Z), dim=0).unsqueeze(0)
    else:
        coord = torch.cat((X, Y), dim=0).unsqueeze(0)

    normalized_x = (coord[:, 0].float() / np.float(resX - 1))
    normalized_y = (coord[:, 1].float() / np.float(resY - 1))
    if is3D:
        normalized_z = (coord[:, 2].float() / np.float(resZ - 1))

    centerX = resX // 2
    centerZ = resZ // 2
    centerY = np.round(resY * height)

    thick = 1.0

    if is3D:

        dist_from_center = ((X - centerX).pow(2) + (Y - centerY).pow(2) + (Z - centerZ).pow(2)).pow(0.5)
        density = ((-1 + torch.tanh((dist_from_center - radCyl) * thick)) / -2 * rho1).unsqueeze(1)

    else:

        dist_from_center = (X - centerX).pow(2) + (Y - centerY).pow(2)
        mask_cylinder = dist_from_center <= radCyl * radCyl

        density = density.masked_fill_(mask_cylinder, 2)

    batch_dict['density'] = density
    batch_dict['flags'] = flags


def createRayleighTaylorBCs(batch_dict, mconf, rho1, rho2):
    r"""Creates masks to enforce a Rayleigh-Taylor instability initial conditions.
    Top fluid has a density rho1 and lower one rho2. rho1 > rho2 to trigger instability.
    Modifies batch_dict inplace.
    Arguments:
        batch_dict (dict): Input tensors (p, UDiv, flags, density)
        mconf (dict): configuration dict (to set thickness and amplitude of interface).
        rho1 (float): Top fluid density.
        rho2 (float): Lower fluid density.
    """

    cuda = torch.device('cuda')

    assert len(batch_dict) == 7, "Batch must contain 7 tensors (p, UDiv, flags, density,Ustar, divergency input)"
    UDiv = batch_dict['U']
    flags = batch_dict['flags']

    resX = UDiv.size(4)
    resY = UDiv.size(3)
    resZ = UDiv.size(2)
    if resZ > 1:
        is3D = True
    else:
        is3D = False

    # Old BC Distribution

    X = torch.arange(0, resX, device=cuda).view(resX).expand((1, resZ, resY, resX))
    Y = torch.arange(0, resY, device=cuda).view(resY, 1).expand((1, resZ, resY, resX))
    Z = torch.arange(0, resZ, device=cuda).view(resZ, 1, 1).expand((1, resZ, resY, resX))

    if is3D:
        coord = torch.cat((X, Y, Z), dim=0).unsqueeze(0)
    else:
        coord = torch.cat((X, Y), dim=0).unsqueeze(0)

    normalized_x = (coord[:, 0].float() / np.float(resX - 1))
    normalized_y = (coord[:, 1].float() / np.float(resY - 1))
    if is3D:
        normalized_z = (coord[:, 2].float() / np.float(resZ - 1))

    # Atwood number
    #A = ((1+rho2) - (1+rho1)) / ((1+rho2) + (1+rho1))
    thick = mconf['perturbThickness']
    ampl = mconf['perturbAmplitude']
    h = mconf['height']

    if is3D:
        teta_cos_x = math.pi * normalized_x
        teta_cos_z = math.pi * normalized_z

        cos_field = (1 + torch.cos(4.0 * teta_cos_x)) * (1 + torch.cos(4 * teta_cos_z))

        field_4 = torch.ones_like(teta_cos_z).float() * math.pi / 4.0
        field3_4 = torch.ones_like(teta_cos_z).float() * 3.0 * math.pi / 4.0

        cos_field = torch.where(teta_cos_x < field_4, torch.zeros_like(cos_field).float(), cos_field)
        cos_field = torch.where(teta_cos_z < field_4, torch.zeros_like(cos_field).float(), cos_field)
        cos_field = torch.where(teta_cos_x > field3_4, torch.zeros_like(cos_field).float(), cos_field)
        cos_field = torch.where(teta_cos_z > field3_4, torch.zeros_like(cos_field).float(), cos_field)

        delta = 1 / resX
        circ = 0.05

        mu = [0.5, 0.5]
        sigma = np.array([[circ, 0.0], [0.0, circ]])

        x, y = np.mgrid[0:1:delta, 0:1:delta]
        pos = np.dstack((x, y))
        gauss_field = multivariate_normal(mu, sigma).pdf(pos) * 0.5
        gauss_field_t = torch.from_numpy(gauss_field).view(
            resZ, 1, resX).expand_as(
            flags[0][0]).unsqueeze(0).float().cuda()

        cos_field_25 = (-0.5 * ((torch.cos(2.0 * teta_cos_x[0,
                                                            :,
                                                            0])) + torch.cos(2.0 * teta_cos_z[0,
                                                                                              :,
                                                                                              0]))).view(1,
                                                                                                         resZ,
                                                                                                         1,
                                                                                                         resX).expand(1,
                                                                                                                      resZ,
                                                                                                                      resY,
                                                                                                                      resX)

        density = 0.5 * (rho2 + rho1 + (rho2 - rho1) * torch.tanh(thick *
                         (normalized_y - (h + ampl * ((cos_field_25)))))).unsqueeze(0)

    else:
        teta_cos = 2.0 * math.pi * normalized_x

        density = 0.5 * (rho2 + rho1 + (rho2 - rho1) * torch.tanh(thick *
                         (normalized_y - (h - ampl * torch.cos(teta_cos))))).unsqueeze(1)

    batch_dict['density'] = density
    batch_dict['flags'] = flags
