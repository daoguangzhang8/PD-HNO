import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

import run_main_with_overrides as runner
from model import compact_output as output


class CompactOutputTests(unittest.TestCase):
    def test_exact_1000_epoch_schedule_and_no_validation_write(self):
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(a=1., b=1., NIter=1001, save_doc=root,
                validate_every=50, save_fig_every=51, save_model_every=100)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.Adam(model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
            history = []
            with patch.object(output, 'plot_history') as loss_plot, \
                 patch.object(output, 'plot_training_fields') as wave_plot, \
                 patch('test.evaluate_training_unseen_frequency') as marmousi, \
                 patch('builtins.print') as log:
                for epoch in range(1, 1001):
                    output.finish_epoch(args, model, optimizer, scheduler, epoch, history,
                        {}, ['frequency', 'source'], dict(data=2., pde=3., env=1.), None,
                        'cpu', 10., 2e-4, 1., 2.,
                        .5 if epoch % 50 == 0 else np.nan,
                        .7 if epoch % 50 == 0 else np.nan)
                    if epoch == 50:
                        self.assertEqual([p.name for p in Path(root).iterdir()], ['loss_history.npy'])
                        self.assertEqual(np.load(Path(root) / 'loss_history.npy').shape[0], 1)
                self.assertEqual(loss_plot.call_count, 20)
                self.assertEqual(wave_plot.call_count, 20)
                self.assertEqual([c.args[-1] for c in marmousi.call_args_list],
                    [1, 1] + [epoch for epoch in range(51, 1001, 51) for _ in range(2)])
                epoch_logs = [c for c in log.call_args_list if str(c.args[0]).startswith('[Epoch ')]
                self.assertEqual(len(epoch_logs), 1000)
            checkpoints = sorted(Path(root).glob('*.pth'))
            self.assertEqual([p.name for p in checkpoints],
                [f'checkpoint_epoch_{epoch:04d}.pth' for epoch in range(100, 1000, 100)] + ['checkpoint_final.pth'])
            self.assertFalse(list(Path(root).glob('*0000*')))
            self.assertFalse(list(Path(root).glob('*0001*')))
            values = np.load(Path(root) / 'loss_history.npy', allow_pickle=False)
            np.testing.assert_array_equal(values['epoch'], np.arange(1, 1001))
            np.testing.assert_array_equal(values['epoch'][np.isfinite(values['validation_data'])],
                                          np.arange(50, 1001, 50))
            np.testing.assert_allclose(values['total'], values['data'] + values['pde'])
            self.assertEqual(len(list(Path(root).iterdir())), 11)

    def test_cli_paths_and_periods(self):
        with tempfile.TemporaryDirectory() as root:
            argv = ['run', '--batch-size-v', '64', '--batch-size-y', '1500',
                '--accumulation-steps', '2', '--parallel', '--num-gpus', '8',
                '--gpu-ids', '0', '1', '2', '3', '4', '5', '6', '7',
                '--from-scratch', '--main-losses-only', '--compact-output',
                '--pml-crop', '0', '--source-radius-mode', 'squared', '--epochs', '1000',
                '--validate-every', '50', '--save-fig-every', '51', '--save-model-every', '100',
                '--nccl-timeout-minutes', '10', '--save-doc', root,
                '--data-dir', '/dataset/openfwi', '--marmousi-freq-dir', '/dataset/freq',
                '--marmousi-source-dir', '/dataset/source']
            with patch('sys.argv', argv), patch.object(runner.main2, 'main') as launch:
                runner.main()
            args = launch.call_args.args[0]
            self.assertEqual((args.start_epoch, args.NIter), (1, 1001))
            self.assertEqual((args.validate_every, args.save_fig_every, args.save_model_every), (50, 51, 100))
            self.assertEqual(args.vel_filename, '/dataset/openfwi/freesurface_full_5sources_velocity.npy')
            self.assertEqual(args.source_coord_filename, '/dataset/openfwi/source_grid_coords.npy')
            self.assertEqual(args.marmousi_eval_data_dir, '/dataset/freq')
            self.assertEqual(args.marmousi_unseen_source_eval_data_dir, '/dataset/source')
            self.assertEqual(args.nccl_timeout_minutes, 10)
            self.assertEqual(args.source_radius_mode, 'squared')
            self.assertTrue(args.enable_marmousi_eval and args.enable_marmousi_unseen_source_eval)
            snapshot = json.loads((Path(root) / 'runtime_config.json').read_text())
            self.assertEqual(snapshot['batch_size_v'], 64)

    def test_plotting_uses_existing_plot_data_layout_and_overwrites(self):
        class Tiny(torch.nn.Module):
            def forward(self, v, y, bg, **kwargs):
                return torch.zeros(y.shape[0], y.shape[1], 2)
        with tempfile.TemporaryDirectory() as root:
            data = dict(y_pred=torch.tensor([[0., 0.], [0., 20.], [20., 0.], [20., 20.]]))
            for tag in ('pred', 'test'):
                data['vel_' + tag] = torch.ones(1, 1, 2, 2)
                data['UU0_' + tag] = torch.zeros(1, 2, 2, 2)
                data['labels_' + tag] = torch.ones(1, 2, 2, 2)
            args = SimpleNamespace(save_doc=root, batch_size=3)
            model = Tiny().train()
            for epoch in (51, 102):
                output.plot_training_fields(model, args, data, 'cpu', epoch)
                output.plot_history([(epoch, 2., 1., 1., 2., 3., np.nan, np.nan, 2e-4, 10.)],
                                    Path(root) / 'loss_curve.png')
            self.assertTrue(model.training)
            self.assertEqual(sorted(p.name for p in Path(root).iterdir()), ['loss_curve.png', 'wavefields.png'])

    def test_nccl_default_ten_minutes_without_launching(self):
        from datetime import timedelta
        path = Path(__file__).with_name('model') / 'utils.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'setup_distributed')
        namespace = dict(os=Mock(), torch=Mock(), dist=Mock(), print=Mock())
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        namespace['setup_distributed'](0, 8, device_id=0)
        self.assertEqual(namespace['dist'].init_process_group.call_args.kwargs['timeout'], timedelta(minutes=10))

    def test_interrupted_checkpoint_preserves_state_and_marks_partial_epoch(self):
        with tempfile.TemporaryDirectory() as root:
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            model(torch.ones(1, 1)).sum().backward()
            optimizer.step()
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
            args = SimpleNamespace(save_doc=root, start_epoch=1)
            state = dict(model=model, optimizer=optimizer, scheduler=scheduler, first_flag=False,
                data_norm_coe=2., pde_norm_coe=3., env_norm_coe=1., last_completed_epoch=17,
                i=18, step_counter=3, optimizer_step_in_progress=False)
            path = output.save_interrupted_checkpoint(args, state, RuntimeError('synthetic failure'))
            saved = torch.load(path, weights_only=False)
            self.assertEqual(path.name, 'checkpoint_interrupted.pth')
            self.assertEqual(saved['epoch'], 17)
            self.assertEqual(saved['interrupted_epoch'], 18)
            self.assertFalse(saved['exact_mid_epoch_resume'])
            self.assertTrue(saved['optimizer_state_dict']['state'])
            for name, value in model.state_dict().items():
                torch.testing.assert_close(saved['model_state_dict'][name], value)

    def test_worker_saves_on_exception_without_starting_process_group(self):
        import model.train_distributed as training
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(save_doc=root, start_epoch=1)
            def broken_worker(rank, world_size, args, device_ids):
                model = torch.nn.Linear(1, 1)
                optimizer = torch.optim.Adam(model.parameters())
                last_completed_epoch = 0
                i = 1
                raise RuntimeError('injected forward failure')
            with patch.object(training, '_train_worker_impl', broken_worker):
                with self.assertRaisesRegex(RuntimeError, 'injected forward failure'):
                    training._train_worker(0, 8, args, list(range(8)))
            self.assertTrue((Path(root) / 'checkpoint_interrupted.pth').is_file())

    def test_marmousi_compact_path_skips_duplicate_plots_and_metrics(self):
        import test as marmousi_test
        args = SimpleNamespace(compact_output=True, _source_list=[2],
            _cli=SimpleNamespace(dataset='marmousi'), training_frequencies=[5.], save_doc='/unused')
        data = dict(vel=torch.ones(1, 2, 2) * 2000, background=torch.zeros(1, 2, 2, 2),
            wavefield=torch.ones(1, 2, 2, 2), freq=torch.tensor([10.]), n_samples=1, n_sources=1)
        model = torch.nn.Linear(1, 1).train()
        metrics = dict(r2=.1, relative_l2=.5)
        with patch.object(marmousi_test, 'evaluate_single', return_value=(np.zeros((2, 2, 2)), metrics, metrics)), \
             patch.object(output, 'plot_fields') as plot, \
             patch.object(marmousi_test, 'save_metrics', return_value={'ok': True}) as save, \
             patch.object(marmousi_test, 'plot_results_by_frequency') as duplicate, \
             patch.object(marmousi_test, 'save_frequency_metrics') as duplicate_metrics:
            result = marmousi_test.evaluate_loaded_data(args, model, data, 'cpu', 51, {})
            self.assertEqual(result, {'ok': True})
            self.assertEqual(plot.call_count, 1)
            self.assertEqual(save.call_args.kwargs['metrics_filename'], 'metrics_latest.json')
            self.assertFalse(save.call_args.kwargs['write_latest'])
            duplicate.assert_not_called()
            duplicate_metrics.assert_not_called()
            self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
