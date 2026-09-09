import os
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import main2
import run_main_with_overrides as runner
from model.utils import get_available_gpus
from model.train_distributed import _validate_ddp_alignment_options,_ddp_batch_size_v


class LaunchTests(unittest.TestCase):
    def test_eight_card_global_64_full_pml(self):
        with tempfile.TemporaryDirectory() as root:
            argv = ['run', '--batch-size-v', '64', '--save-doc', root, '--parallel',
                    '--num-gpus', '8', '--gpu-ids', '0', '1', '2', '3', '4', '5', '6', '7',
                    '--from-scratch', '--pml-crop', '0', '--last-epoch', '1000',
                    '--main-losses-only']
            with patch('sys.argv', argv), patch.object(runner.main2, 'main') as launch:
                runner.main()
            a = pickle.loads(pickle.dumps(launch.call_args.args[0]))
            self.assertEqual(_ddp_batch_size_v(a, 8), 8)
            self.assertEqual(a.batch_size, 1500)
            self.assertEqual(a.accumulation_steps, 2)
            self.assertEqual(a.pml_active, 20)
            self.assertFalse(a.ddp_scale_lr)

    def test_uuid_mapping_follows_cuda_order(self):
        inventory='GPU-aaa, 0, 24564, 24000\nGPU-bbb, 3, 24564, 23000\n'
        with patch('torch.cuda.is_available',return_value=True), \
             patch('torch.cuda.device_count',return_value=2), \
             patch('torch.cuda.get_device_properties',side_effect=[SimpleNamespace(uuid='GPU-bbb'),SimpleNamespace(uuid='GPU-aaa')]), \
             patch('subprocess.check_output',return_value=inventory):
            self.assertEqual(get_available_gpus(23500),[1])

    def test_detection_failure_does_not_invent_free_memory(self):
        with patch('torch.cuda.is_available',return_value=True),patch('subprocess.check_output',side_effect=RuntimeError('failure')):
            self.assertEqual(get_available_gpus(),[])

    def test_missing_or_wrong_selected_cards_never_fallback(self):
        a=SimpleNamespace(use_parallel=True,num_gpus=4,min_gpu_memory=23000,device_ids=None)
        with patch.object(main2,'get_available_gpus',return_value=[0]):
            with self.assertRaises(RuntimeError):main2.main(a)
        self.assertTrue(a.use_parallel)

    def test_runtime_options_survive_pickle(self):
        with tempfile.TemporaryDirectory() as root:
            argv=['run','--batch-size-v','32','--save-doc',root,'--parallel','--num-gpus','4',
                  '--gpu-ids','0','1','2','3','--from-scratch','--pml-crop','15',
                  '--last-epoch','1000','--main-losses-only','--save-every','10']
            with patch('sys.argv',argv),patch.object(runner.main2,'main') as launch:
                runner.main()
            a=pickle.loads(pickle.dumps(launch.call_args.args[0]))
            self.assertEqual(_ddp_batch_size_v(a,4),8)
            self.assertEqual(a.accumulation_steps,2)
            self.assertEqual(a.pml_active,5)
            self.assertEqual(a.save_fig_every,10)
            self.assertEqual(a.save_model_every,10)
            self.assertFalse(a.if_load_model)
            self.assertFalse(a.enable_continuous_frequency_pde)
            self.assertEqual(a.film_smooth_weight,0.)
            self.assertTrue((Path(root)/'runtime_config.json').is_file())

    def test_auxiliary_paths_rejected_explicitly(self):
        for changes in [dict(enable_continuous_frequency_pde=True),dict(film_smooth_weight=.01),
                        dict(ddp_scale_lr=True),dict(ddp_split_batch_size_v=False)]:
            with self.subTest(changes=changes):
                with self.assertRaises((ValueError,NotImplementedError)):
                    _validate_ddp_alignment_options(SimpleNamespace(**changes))


if __name__=='__main__':unittest.main()
