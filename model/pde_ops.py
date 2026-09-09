"""Coordinate-differentiable sampling and generator-aligned PML geometry."""
import torch
import torch.nn.functional as F


def source_radial_coordinate(dz, dx, length_scale, mode='legacy'):
    """Metre-valued radial feature, not necessarily geometric distance.

    squared: q=(dz²+dx²)/L, so q/L and k*q are dimensionless. Unlike (kr)²,
    this keeps the phase range comparable at r=L. It is a quadratic feature,
    NOT the original radial wave phase; no coordinate sqrt or detach is used.
    """
    if mode == 'legacy':
        return torch.sqrt(dz.square() + dx.square() + 1e-12)
    if mode != 'squared':
        raise ValueError(f'Unknown source_radius_mode: {mode}')
    if length_scale <= 0:
        raise ValueError('Source radial length scale must be positive')
    return (dz.square() + dx.square()) / length_scale


def sample_cubic_features(field, coordinates):
    """C2 cubic B-spline sampling, [B,C,H,W] and pixel [B,N,(z,x)].

    Grid values are spline coefficients (smoothing, not interpolating samples).
    Replicated coefficient padding keeps the extension C2 at the grid edges.
    Only cell selection is discrete; all polynomial weights retain gradients.
    """
    if coordinates.ndim == 2:
        coordinates = coordinates.unsqueeze(0).expand(field.shape[0], -1, -1)
    if coordinates.shape[0] != field.shape[0]:
        raise ValueError('Coordinate and field batch sizes must match')
    base = torch.floor(coordinates)
    t = coordinates - base
    weights = ((1-t)**3 / 6, (3*t**3-6*t**2+4) / 6,
               (-3*t**3+3*t**2+3*t+1) / 6, t**3 / 6)
    base = base.long()
    batch = torch.arange(field.shape[0], device=field.device)[:, None]
    result = 0
    for iz in range(4):
        z = (base[..., 0] + iz - 1).clamp(0, field.shape[2]-1)
        for ix in range(4):
            x = (base[..., 1] + ix - 1).clamp(0, field.shape[3]-1)
            value = field[batch, :, z, x]
            weight = weights[iz][..., 0] * weights[ix][..., 1]
            result = result + value * weight.unsqueeze(-1)
    return result


def pml_profiles(args, y, nz, nx):
    """Map original fine-grid PML ramps to cropped physical coordinates.

    Generator nodes are one-based, and saved nodes are fine[::stride].
    Thus the interface lies half a generation-grid spacing before the first
    physical node (or before the first right/bottom PML node).
    """
    if not args.pml or args.pml_total == 0:
        return y[..., 0] * 0, y[..., 1] * 0
    dh = float(args.dh)
    fine_dh = float(getattr(args, 'pde_generation_dh', dh))
    if fine_dh <= 0 or dh <= 0 or not 0 <= args.pml_crop <= args.pml_total:
        raise ValueError('Invalid PML geometry')
    width = args.pml_total * dh
    remaining = (args.pml_total - args.pml_crop) * dh
    z, x = y[..., 0], y[..., 1]
    left = remaining - fine_dh / 2
    right = nx * dh - remaining - fine_dh / 2
    bottom = nz * dh - remaining - fine_dh / 2
    lx = F.relu((left-x)/width) + F.relu((x-right)/width)
    lz = F.relu((z-bottom)/width)
    if args.boundary_type != 'free_surface':
        lz = lz + F.relu((remaining-fine_dh/2-z)/width)
    return lx, lz
