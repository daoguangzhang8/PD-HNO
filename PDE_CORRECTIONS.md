# PDE corrections

The spatial branch features now use a tensor-product cubic B-spline with
replicated coefficient padding. This is C2 smoothing, not interpolating the
original feature values. Coordinate gradients flow through both the local
branch features and the multiscale FiLM features. Polynomial weights use native
PyTorch operations, including higher-order backward, rather than grid_sample
coordinate backward. Per-sample query coordinates are supported.

Parameter names and shapes are unchanged: existing checkpoints load strictly.
Their predictions change because spatial sampling changed. Previously exported
validation files describe the old forward/loss implementation; they are not
updated automatically. Re-evaluation must not skip these old results using the
evaluation script's resume mechanism.

PML stretching follows the dataset generator beta = 2*pi*1.79*f0/f. The mass
term still uses omega = 2*pi*f*1e-3. Coefficient derivatives are retained.
Ramps use the original full PML width and the cropped-to-fine-grid mapping.
For the current 145x150, dh=20, pml_total=20, pml_crop=15,
pde_generation_dh=10 configuration, interfaces are x=95, x=2895 and z=2795 m.
The top is free of PML; disabling pml gives zero damping.

Old loss_normalizers are reference scales from the old implementation, not a
calibration of the corrected loss. Recalibrate before using a new epoch-zero
baseline, and do not splice old/new loss curves as the same objective.
Training configuration still points at the existing resume checkpoint; no
training process is restarted by these source changes.

Validation: run `python -m unittest test_pde_derivatives -v` in the PyTorch
environment. Tests cover first/second sampling derivatives, C2 continuity at
cell boundaries, fine-grid PML geometry, disabled PML and an independent
complex divergence reference with loss backward. Additionally, checkpoint
1000 was strictly loaded and a small CPU full-model loss backward including
continuous-frequency PDE checked for finite Branch/FiLM/Trunk gradients.
Production GPU batch-size memory/performance has not been benchmarked.

These fixes concern the continuous PDE operator. They do not make a continuous
operator identical to the generator's fine-grid 9-point discrete stencil or
eliminate errors caused by wavefield downsampling.
