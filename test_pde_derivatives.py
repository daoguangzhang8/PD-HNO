"""Regression checks for spatial derivatives, PML mapping and loss backward."""
import unittest
from types import SimpleNamespace

import torch

from model.pde_ops import sample_cubic_features, pml_profiles
from model.PI_DeepOnet import Pi_DeepONet


class DerivativeTests(unittest.TestCase):
    def test_sampler_gradients_and_grid_node_continuity(self):
        torch.manual_seed(7)
        field = torch.randn(2, 2, 5, 6, dtype=torch.double, requires_grad=True)
        y = torch.tensor([[[2.2, 2.3]], [[1.3, 3.1]]], dtype=torch.double,
                         requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(sample_cubic_features, (field, y)))
        self.assertTrue(torch.autograd.gradgradcheck(sample_cubic_features, (field, y)))
        seconds = []
        for offset in (-1e-7, 1e-7):
            q = torch.tensor([[[2.+offset, 2.3]]], dtype=torch.double, requires_grad=True)
            value = sample_cubic_features(field[:1], q).sum()
            g = torch.autograd.grad(value, q, create_graph=True)[0]
            seconds.append(torch.autograd.grad(g[..., 0].sum(), q)[0])
        torch.testing.assert_close(seconds[0], seconds[1], atol=1e-5, rtol=1e-5)

    def test_pml_fine_grid_mapping(self):
        a = SimpleNamespace(pml=True, pml_total=20, pml_crop=15, dh=20.,
                            pde_generation_dh=10., boundary_type='free_surface')
        y = torch.tensor([[[0., 0.], [2880., 2980.], [1000., 1000.]]])
        lx, lz = pml_profiles(a, y, 145, 150)
        torch.testing.assert_close(lx, torch.tensor([[95/400, 85/400, 0.]]))
        torch.testing.assert_close(lz, torch.tensor([[0., 85/400, 0.]]))
        a.pml = False
        lx, lz = pml_profiles(a, y, 145, 150)
        self.assertEqual(float((lx+lz).sum()), 0.)

    def test_complex_divergence_reference_and_backward(self):
        a = SimpleNamespace(pml=True, pml_total=20, pml_crop=15, dh=20.,
                            pde_generation_dh=10., boundary_type='free_surface',
                            default_freq=10., pde_attenuation_alpha=0.)
        y = torch.tensor([[[20., 40.], [2850., 2950.], [1000., 1200.]]],
                         dtype=torch.double, requires_grad=True)
        amplitude = torch.tensor(1.2, dtype=torch.double, requires_grad=True)
        z, x = y[..., 0], y[..., 1]
        u = amplitude * torch.complex((z*z+x*x)/1e6, z*x/1e6)
        vel = torch.full((1, 1, 145, 150), 1.5, dtype=torch.double)
        bg = torch.zeros(1, 2, 145, 150, dtype=torch.double)
        actual = Pi_DeepONet._compute_pde_residual(
            SimpleNamespace(args=a), vel, y, bg,
            torch.stack([u.real, u.imag], -1), return_pointwise=True)
        # Independent fine-grid ramps, before expanding the complex arithmetic.
        sx = 1-1j*(2*torch.pi*1.79)*(torch.relu((95-x)/400)+torch.relu((x-2895)/400))**2
        sz = 1-1j*(2*torch.pi*1.79)*torch.relu((z-2795)/400)**2
        def derivative(value, axis):
            return torch.complex(
                torch.autograd.grad(value.real.sum(), y, create_graph=True, retain_graph=True)[0][..., axis],
                torch.autograd.grad(value.imag.sum(), y, create_graph=True, retain_graph=True)[0][..., axis])
        residual = (derivative(sx/sz*derivative(u, 0), 0)
                    + derivative(sz/sx*derivative(u, 1), 1)
                    + sx*sz*(2*torch.pi*10*1e-3/1.5)**2*u)
        torch.testing.assert_close(actual, residual.abs().square(), rtol=1e-10, atol=1e-15)
        actual.mean().backward()
        self.assertTrue(torch.isfinite(amplitude.grad))
        self.assertGreater(abs(amplitude.grad.item()), 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
