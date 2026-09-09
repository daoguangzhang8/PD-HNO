"""Run the actual four validation-loop bodies with a small higher-derivative loss."""
import ast
import gc
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref

import torch


class TinyPhysics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.5))
        self.loss_refs = []

    def loss(self, vel, y, background, labels, *args, **kwargs):
        y.requires_grad_(True)
        u = torch.sin(self.weight * y).sum(-1)
        first = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), y, create_graph=True)[0]
        pde, data = second.square().mean(), u.square().mean()
        self.loss_refs.extend([weakref.ref(pde), weakref.ref(data)])
        zero = y.new_zeros(())
        return pde + data, pde, data, zero, zero, zero


class ValidationCleanupTests(unittest.TestCase):
    def test_all_validation_loop_bodies_release_graph_and_preserve_metrics(self):
        root = Path(__file__).resolve().parent
        checked = 0
        for filename in ('model/train.py', 'model/train_distributed.py'):
            tree = ast.parse((root / filename).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.For):
                    continue
                if "dataloader['valid_y']" != ast.unparse(node.iter):
                    continue
                if 'loss_f_valid' not in ast.unparse(node):
                    continue
                checked += 1
                with self.subTest(file=filename, line=node.lineno):
                    tiny = TinyPhysics()
                    coords = torch.tensor([[.3, .7], [.8, .2]])
                    dummy = torch.zeros(1, 1, 2, 2)
                    ref = tiny.loss(dummy, coords[None].clone(), dummy, dummy)
                    expected_pde, expected_data = ref[1].item(), ref[2].item()
                    del ref
                    values = dict(
                        model=SimpleNamespace(module=tiny) if 'distributed' in filename else tiny,
                        batch=(coords,), device=torch.device('cpu'),
                        vel_batch=dummy, UU0_batch=dummy, labels_batch=dummy,
                        freq_batch=None, source_coord_batch=None,
                        a=1., b=1., c=0., d=0., data_norm_coe=1.,
                        pde_norm_coe=1., env_norm_coe=1.,
                        batch_u_loss=[], batch_f_loss=[], vb_u_loss=[], vb_f_loss=[],
                        batch_point_counts=[], batch_pde_point_counts=[],
                    )
                    body = ast.fix_missing_locations(ast.Module(body=node.body, type_ignores=[]))
                    for _ in range(2):
                        exec(compile(body, filename, 'exec'), values)
                        for name in ('loss_f_valid', 'loss_u_valid', 'y_batch'):
                            self.assertNotIn(name, values)
                    gc.collect()
                    self.assertTrue(all(ref() is None for ref in tiny.loss_refs))
                    data_values = values['batch_u_loss'] or values['vb_u_loss']
                    pde_values = values['batch_f_loss'] or values['vb_f_loss']
                    self.assertEqual(data_values, [expected_data] * 2)
                    self.assertEqual(pde_values, [expected_pde] * 2)
                    self.assertEqual(values['batch_point_counts'], [2, 2])
                    if 'distributed' not in filename:
                        self.assertEqual(values['batch_pde_point_counts'], [2, 2])
                    self.assertIsNone(tiny.weight.grad)
        self.assertEqual(checked, 4)


if __name__ == '__main__':
    unittest.main()
