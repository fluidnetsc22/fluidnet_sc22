import torch


def set_inflow_bc(div, U, flags, batch_dict):

    cuda = torch.device('cuda')
    assert (U.dim() == 5 and flags.dim() == 5), 'Dimension mismatch'
    assert flags.size(1) == 1, 'flags is not a scalar'
    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    is3D = (U.size(1) == 3)

    assert (U.size(0) == bsz and U.size(2) == d and U.size(3) == h and U.size(4) == w),\
        'Size mismatch'
    assert (U.is_contiguous() and flags.is_contiguous()), 'Input is not contiguous'

    density_mask_0 = 1 - batch_dict['densityBCInvMask']
    density_mask_0 = torch.where(
        flags == 2 * torch.ones_like(flags),
        torch.zeros_like(density_mask_0).float(),
        density_mask_0)
    density_mask = torch.cat(
        (torch.zeros_like(density_mask_0[:, :, :, 0, :].unsqueeze(3)), density_mask_0[:, :, :, :-1, :]), dim=3)

    corrected_div = (density_mask[:, 0] * (batch_dict['UBC'][:, 1])).unsqueeze(1)

    div[:, :, :, :] -= corrected_div

    div = torch.where(density_mask_0 == 1, torch.zeros_like(div).float(), div)
    div = torch.where(flags == 2 * torch.ones_like(flags), torch.zeros_like(div).float(), div)
