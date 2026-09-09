"""CPU/Gloo eight-rank equivalence; production PDE, full PML, complex FNO."""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
from contextlib import nullcontext
import tempfile
from pathlib import Path
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_four_rank_training import SmallPhysics
from model.net_module import SpectralConv2d
from model.point_averaging import coordinate_accumulation_batches
from model.distributed_complex import wrap_complex_safe_ddp, average_complex_gradients


class ComplexPhysics(SmallPhysics):
    def __init__(self):
        super().__init__()
        self.args.pml_crop = 0
        self.spectral = SpectralConv2d(1, 1, 2, 2)

    def forward(self, v, y, bg, freq_batch=None):
        features = self.spectral(v)
        condition = torch.stack((features.square().mean((1, 2, 3)), bg.mean((1, 2, 3))), -1)
        scale, bias = self.film(condition).chunk(2, -1)
        h = torch.sin(self.trunk(y / 1000)) * (1 + scale[:, None]) + bias[:, None]
        return self.output(torch.sin(h))


def worker(rank, size, init_path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_path, rank=rank, world_size=size)
    try:
        torch.manual_seed(131 + rank)
        model = ComplexPhysics()
        ddp = wrap_complex_safe_ddp(model)
        reference = ComplexPhysics() if rank == 0 else None
        if reference is not None:
            reference.load_state_dict(model.state_dict())
            refopt = torch.optim.Adam(reference.parameters(), lr=2e-4, weight_decay=1e-4)
        opt = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(77)
        # Full uncut PML on the actual saved-grid dimensions.
        velocity = 1.5 + torch.rand(64, 1, 160, 180, generator=generator)
        background = torch.randn(64, 2, 160, 180, generator=generator)
        labels = torch.randn(64, 2, 160, 180, generator=generator)
        frequency = torch.linspace(3, 25, 64)
        local = slice(rank * 8, (rank + 1) * 8)
        batches = []
        for count in (1500, 1500, 1500, 900, 375):
            indices = torch.randint(28800, (count,), generator=generator)
            batches.append((torch.stack((indices // 180, indices % 180), -1).float() * 20,))
        max_gradient_error, max_parameter_error, updates = 0., 0., 0
        for batch, weight, _, end in coordinate_accumulation_batches(batches, 2):
            points = batch[0]
            y = points[None].expand(8, -1, -1).clone().requires_grad_(True)
            with nullcontext() if end else ddp.no_sync():
                prediction = ddp(velocity[local], y, background[local], freq_batch=frequency[local])
                losses = model.compute_loss(prediction, velocity[local], y, background[local],
                    labels[local], y, 1., 1., 0., freq_batch=frequency[local])
                (losses[0] * weight).backward()
            average = torch.stack([loss.detach() for loss in losses[:3]])
            dist.all_reduce(average)
            average /= size
            if rank == 0:
                all_y = points[None].expand(64, -1, -1).clone().requires_grad_(True)
                refpred = reference(velocity, all_y, background, freq_batch=frequency)
                reflosses = reference.compute_loss(refpred, velocity, all_y, background, labels,
                    all_y, 1., 1., 0., freq_batch=frequency)
                (reflosses[0] * weight).backward()
                torch.testing.assert_close(average, torch.stack([x.detach() for x in reflosses[:3]]),
                                           atol=1e-6, rtol=2e-5)
                del refpred, reflosses, all_y
            if end:
                average_complex_gradients(model)
                if rank == 0:
                    for p, q in zip(model.parameters(), reference.parameters()):
                        max_gradient_error = max(max_gradient_error, (p.grad - q.grad).abs().max().item())
                        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4)
                    refopt.step()
                    refopt.zero_grad(set_to_none=True)
                opt.step()
                opt.zero_grad(set_to_none=True)
                for index, p in enumerate(model.parameters()):
                    expected = list(reference.parameters())[index].detach().clone() if rank == 0 else torch.empty_like(p)
                    dist.broadcast(torch.view_as_real(expected) if expected.is_complex() else expected, src=0)
                    max_parameter_error = max(max_parameter_error, (p - expected).abs().max().item())
                    torch.testing.assert_close(p, expected, atol=2e-6, rtol=2e-5)
                updates += 1
            del prediction, losses, y
        if rank == 0:
            print(f'8 ranks x BS8 == BS64; full PML; y BS1500; tail900/375; '
                  f'complex FNO; Adam updates={updates}; max_grad_abs={max_gradient_error:.9g}; '
                  f'max_param_abs={max_parameter_error:.9g}', flush=True)
    finally:
        dist.destroy_process_group()


class EightRankTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv('RUN_DISTRIBUTED_TESTS') == '1', 'opt-in local communication')
    def test_full_production_model_reducer_and_checkpoint_names(self):
        from config import Args
        from model.PI_DeepOnet import Pi_DeepONet
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as root:
            dist.init_process_group('gloo', init_method='file://' + str(Path(root) / 'init'),
                                    rank=0, world_size=1)
            try:
                args = Args()
                args.nz = args.nx = 64
                args.pml_crop = 0
                args.source_radius_mode = 'squared'
                args.film_smooth_weight = 0.
                args.enable_continuous_frequency_pde = False
                args.continuous_pde_weight = 0.
                model = Pi_DeepONet(args)
                model._init_weights()
                original_keys = list(model.state_dict())
                ddp = wrap_complex_safe_ddp(model)
                optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
                velocity = torch.full((1, 1, 64, 64), 2.)
                background = torch.randn(1, 2, 64, 64)
                source = torch.tensor([[0., 400.]])
                frequency = torch.tensor([5.])
                for _ in range(2):
                    y = torch.tensor([[[0., 400.], [1200., 1200.], [500., 600.]]], requires_grad=True)
                    prediction = ddp(velocity, y, background, freq_batch=frequency, source_coord_batch=source)
                    losses = model.compute_loss(prediction, velocity, y, background, background,
                        y, 1., 1., 0., freq_batch=frequency, source_coord_batch=source)
                    losses[0].backward()
                    average_complex_gradients(model)
                    for name, p in model.named_parameters():
                        if p.requires_grad:
                            self.assertIsNotNone(p.grad, name)
                            self.assertTrue(torch.isfinite(p.grad).all(), name)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    del prediction, losses, y
                self.assertEqual(list(model.state_dict()), original_keys)
            finally:
                dist.destroy_process_group()

    @unittest.skipUnless(os.getenv('RUN_DISTRIBUTED_TESTS') == '1', 'opt-in local communication')
    def test_eight_ranks(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(worker, args=(8, str(Path(root) / 'init')), nprocs=8, join=True)


if __name__ == '__main__':
    unittest.main()
