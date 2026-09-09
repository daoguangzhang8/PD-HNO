import ast
from pathlib import Path
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from model.point_averaging import (
    coordinate_accumulation_batches, point_mean, weighted_microbatch_loss,
)


class PointAveragingTests(unittest.TestCase):
    def test_counts_follow_actual_grid_and_tail(self):
        for total in (10875, 14400, 28800):
            loader = DataLoader(TensorDataset(torch.arange(total)), batch_size=1500)
            groups = list(coordinate_accumulation_batches(loader, 2))
            self.assertEqual(sum(len(x[0][0]) for x in groups), total)
            weights = []
            for _, data_weight, pde_weight, end in groups:
                self.assertEqual(data_weight, pde_weight)
                weights.append(data_weight)
                if end:
                    self.assertAlmostEqual(sum(weights), 1.)
                    weights = []
            self.assertFalse(weights)
        loader = DataLoader(TensorDataset(torch.arange(1875)), batch_size=1500)
        self.assertEqual([x[1] for x in coordinate_accumulation_batches(loader, 2)], [.8, .2])

    def test_gradients_and_updates_equal_concatenated_group(self):
        # Exercise model BS, short groups, and different data/PDE point sets.
        for bs in (1, 8, 32):
            for accumulation in (1, 2, 3):
                for extra in (0, 2):
                    x = torch.linspace(-1, 2, 11, dtype=torch.float64)
                    loader = DataLoader(TensorDataset(x), batch_size=4)
                    parameter = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
                    reference = parameter.detach().clone().requires_grad_()
                    opt = torch.optim.Adam([parameter], lr=.001)
                    ref_opt = torch.optim.Adam([reference], lr=.001)
                    group_data, group_pde = [], []
                    for batch, dw, pw, end in coordinate_accumulation_batches(loader, accumulation, extra):
                        data_x = batch[0].repeat(bs, 1)
                        pde_x = torch.cat([data_x, torch.full((bs, extra), 3., dtype=torch.float64)], dim=1)
                        data_loss = (parameter * data_x - 1).square().mean()
                        pde_loss = (parameter.square() * pde_x + .3).square().mean()
                        loss = 2 * data_loss + 3 * pde_loss
                        weighted_microbatch_loss(loss, pde_loss, 3, dw, pw).backward()
                        group_data.append(data_x)
                        group_pde.append(pde_x)
                        if end:
                            ref_loss = (2 * (reference * torch.cat(group_data, 1) - 1).square().mean()
                                        + 3 * (reference.square() * torch.cat(group_pde, 1) + .3).square().mean())
                            ref_loss.backward()
                            torch.testing.assert_close(parameter.grad, reference.grad)
                            opt.step()
                            ref_opt.step()
                            torch.testing.assert_close(parameter, reference)
                            opt.zero_grad()
                            ref_opt.zero_grad()
                            group_data, group_pde = [], []
                    self.assertIsNone(parameter.grad)

    def test_epoch_means_include_model_and_point_counts(self):
        values, counts, all_points = [], [], []
        for bs, points, value in ((32, 1500, 2.), (32, 375, 10.), (3, 1500, 7.)):
            values.append(value)
            counts.append(bs * points)
            all_points.extend([value] * (bs * points))
        self.assertAlmostEqual(point_mean(values, counts), sum(all_points) / len(all_points))
        self.assertAlmostEqual(point_mean([2., 10.], [1500, 375]), 3.6)

    def test_both_training_paths_use_weighting(self):
        tree = ast.parse(Path(__file__).with_name('model').joinpath('train.py').read_text())
        for name in ('_train_stage', 'train_single'):
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            calls = [node.func.id for node in ast.walk(function)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
            self.assertIn('coordinate_accumulation_batches', calls)
            self.assertIn('weighted_microbatch_loss', calls)
            self.assertIn('point_mean', calls)
            self.assertNotIn('np.mean(batch_', ast.unparse(function))

    def test_ddp_paths_use_same_group_boundaries(self):
        tree = ast.parse(Path(__file__).with_name('model').joinpath('train_distributed.py').read_text())
        for name in ('_train_worker_impl', '_run_stage_training_loop'):
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = [n.func.id for n in ast.walk(function)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            self.assertIn('coordinate_accumulation_batches', calls)
            self.assertIn('average_complex_gradients', calls)
            self.assertNotIn('_flush_ddp_accumulation', calls)
            self.assertNotIn('np.mean(batch_', ast.unparse(function))


if __name__ == '__main__':
    unittest.main()
