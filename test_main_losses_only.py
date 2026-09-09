"""Data/PDE-only objective keeps FiLM conditioning trainable."""
import unittest
from unittest.mock import patch
import torch
from config import Args
from model.PI_DeepOnet import Pi_DeepONet


class MainLossesOnlyTests(unittest.TestCase):
    def test_single_and_ddp_loss_paths_skip_auxiliary_computation(self):
        torch.set_num_threads(2)
        torch.manual_seed(128)
        args=Args();args.nz=args.nx=64
        args.film_smooth_weight=0.
        args.enable_continuous_frequency_pde=False
        args.continuous_pde_weight=0.
        model=Pi_DeepONet(args).eval()
        velocity=torch.full((1,1,64,64),2.)
        background=torch.randn(1,2,64,64)
        labels=torch.zeros_like(background)
        frequency=torch.tensor([10.])
        source=torch.tensor([[0.,400.]])
        for mode in ('loss','compute_loss'):
            with self.subTest(mode=mode):
                model.zero_grad(set_to_none=True)
                y=torch.tensor([[[501.,603.],[601.,703.],[801.,903.]]],requires_grad=True)
                with patch.object(model,'film_frequency_smoothness_loss',side_effect=AssertionError('disabled FiLM loss called')), \
                     patch.object(model,'prepare_continuous_frequency_batch',side_effect=AssertionError('disabled continuous PDE called')):
                    if mode=='loss':
                        losses=model.loss(velocity,y,background,labels,1.,1.,0.,d=0.,
                                          freq_batch=frequency,source_coord_batch=source,
                                          use_continuous_pde=True)
                    else:
                        pred=model(velocity,y,background,freq_batch=frequency,source_coord_batch=source)
                        losses=model.compute_loss(pred,velocity,y,background,labels,y,1.,1.,0.,d=0.,
                                                  freq_batch=frequency,source_coord_batch=source)
                    total,pde,data,regularizer,envelope,continuous=losses
                    torch.testing.assert_close(total,data+pde,rtol=0,atol=0)
                    for auxiliary in (regularizer,envelope,continuous):
                        self.assertEqual(auxiliary.item(),0.)
                        self.assertFalse(auxiliary.requires_grad)
                    total.backward()
                grad=model.film_freq_encoder[0].weight.grad
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(),0.)
                self.assertTrue(model.use_relative_source_encoding)
                del losses,total,pde,data,regularizer,envelope,continuous


if __name__=='__main__':unittest.main()
