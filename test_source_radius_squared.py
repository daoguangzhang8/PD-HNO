import unittest
import numpy as np
import torch
from config import Args
from model.pde_ops import source_radial_coordinate
from model.PI_DeepOnet import Pi_DeepONet


class RadiusTests(unittest.TestCase):
    def test_squared_coordinate_hessian_and_gradgrad(self):
        scale=float(np.hypot(2880,2980))
        y=torch.tensor([[0.,0.],[1.,2.],[100.,100.]],dtype=torch.double,requires_grad=True)
        def feature(y):return source_radial_coordinate(y[:,0],y[:,1],scale,'squared')
        self.assertTrue(torch.autograd.gradcheck(feature,(y,)))
        self.assertTrue(torch.autograd.gradgradcheck(feature,(y,)))
        g=torch.autograd.grad(feature(y).sum(),y,create_graph=True)[0]
        for axis in (0,1):
            h=torch.autograd.grad(g[:,axis].sum(),y,retain_graph=True)[0][:,axis]
            torch.testing.assert_close(h,torch.full_like(h,2/scale))
        print(f'Squared radial Hessian={2/scale:.9g}; normalized Hessian={2/scale**2:.9g}')

    def test_legacy_unchanged_and_mode_validation(self):
        z=torch.tensor([0.,1.]);x=torch.tensor([0.,2.])
        torch.testing.assert_close(source_radial_coordinate(z,x,10.),(z*z+x*x+1e-12).sqrt(),rtol=0,atol=0)
        with self.assertRaises(ValueError):source_radial_coordinate(z,x,10.,'typo')

    def test_full_model_source_pde_backward(self):
        torch.set_num_threads(2);torch.manual_seed(123)
        a=Args();a.nz=145;a.nx=150;a.source_radius_mode='squared'
        a.film_smooth_weight=0.;a.enable_continuous_frequency_pde=False
        a.continuous_pde_weight=0.
        m=Pi_DeepONet(a).eval()
        v=torch.full((1,1,145,150),2.)
        bg=torch.randn(1,2,145,150);labels=torch.zeros_like(bg)
        y=torch.tensor([[[0.,400.],[.01,400.],[20.,400.],[500.,600.]]])
        f=torch.tensor([10.]);src=torch.tensor([[0.,400.]])
        losses=m.loss(v,y,bg,labels,1.,1.,0.,freq_batch=f,source_coord_batch=src)
        self.assertTrue(torch.isfinite(losses[0]))
        losses[0].backward()
        for name,p in m.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertGreater(m.relative_source_encoder[0].weight.grad.abs().sum().item(),0.)
        print('Squared source full-model PDE=',losses[1].item(),'all parameter gradients finite')


if __name__=='__main__':unittest.main()
