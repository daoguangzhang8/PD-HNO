"""Run main2 with per-run, non-persistent training configuration overrides."""

import argparse
import json
import os
from types import SimpleNamespace

# Must be set before scientific libraries are imported by spawned workers.
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')

import main2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size-v', type=int, required=True)
    parser.add_argument('--save-doc', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--pml-crop', type=int)
    parser.add_argument('--last-epoch', type=int)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num-gpus', type=int, default=4)
    parser.add_argument('--gpu-ids', type=int, nargs='+', help='Process-local CUDA indices after CUDA_VISIBLE_DEVICES')
    parser.add_argument('--main-losses-only', action='store_true')
    parser.add_argument('--save-every', type=int)
    parser.add_argument('--validate-every', type=int)
    parser.add_argument('--save-fig-every', type=int)
    parser.add_argument('--save-model-every', type=int)
    parser.add_argument('--epochs', type=int, help='Exact training epoch count, numbered 1..epochs (compact DDP)')
    parser.add_argument('--compact-output', action='store_true')
    parser.add_argument('--batch-size-y', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--nccl-timeout-minutes', type=int, default=10)
    parser.add_argument('--data-dir', help='Directory containing the five training NPY files')
    parser.add_argument('--marmousi-freq-dir')
    parser.add_argument('--marmousi-source-dir')
    parser.add_argument('--min-free-memory-mib', type=int)
    parser.add_argument('--master-port', type=int)
    parser.add_argument('--source-radius-mode', choices=('legacy', 'squared'))
    parser.add_argument('--activation-offload', action='store_true',
                        help='Store autograd saved tensors in CPU RAM; compute remains on GPU')
    options = parser.parse_args()

    base_args = main2.Args

    # Freeze inherited class attributes into a pickleable per-run value object.
    RuntimeArgs = SimpleNamespace(**{k: getattr(base_args, k) for k in dir(base_args)
                                    if not k.startswith('_') and not callable(getattr(base_args,k))})
    RuntimeArgs.batch_size_v = options.batch_size_v
    RuntimeArgs.save_doc = options.save_doc
    RuntimeArgs.device = options.device
    RuntimeArgs.nccl_timeout_minutes = options.nccl_timeout_minutes
    RuntimeArgs.compact_output = options.compact_output
    for option, attribute in (
        ('batch_size_y', 'batch_size'), ('accumulation_steps', 'accumulation_steps'),
        ('validate_every', 'validate_every'), ('save_fig_every', 'save_fig_every'),
        ('save_model_every', 'save_model_every'),
    ):
        value = getattr(options, option)
        if value is not None:
            if value <= 0: parser.error(f'{option} must be positive')
            setattr(RuntimeArgs, attribute, value)
    if options.nccl_timeout_minutes <= 0: parser.error('NCCL timeout must be positive')
    if options.data_dir:
        RuntimeArgs.load_path = os.path.abspath(options.data_dir)
        RuntimeArgs.training_data_dir = RuntimeArgs.load_path
        for attribute, filename in (
            ('vel_filename', 'freesurface_full_5sources_velocity.npy'),
            ('backgroundfield_filename', 'freesurface_full_5sources_background.npy'),
            ('wavefield_filename', 'freesurface_full_5sources_wavefield.npy'),
            ('freq_filename', 'freesurface_full_5sources_freq_used.npy'),
            ('source_coord_filename', 'source_grid_coords.npy'),
        ):
            setattr(RuntimeArgs, attribute, os.path.join(RuntimeArgs.load_path, filename))
    if options.marmousi_freq_dir:
        RuntimeArgs.marmousi_eval_data_dir = os.path.abspath(options.marmousi_freq_dir)
    if options.marmousi_source_dir:
        RuntimeArgs.marmousi_unseen_source_eval_data_dir = os.path.abspath(options.marmousi_source_dir)
    RuntimeArgs.activation_offload = options.activation_offload
    if options.source_radius_mode is not None:
        RuntimeArgs.source_radius_mode = options.source_radius_mode
    if options.from_scratch:
        RuntimeArgs.if_load_model = False
        RuntimeArgs.resume_checkpoint = ''
        RuntimeArgs.resume_epoch = 0
        RuntimeArgs.resume_restore_optimizer = False
        RuntimeArgs.resume_restore_scheduler = False
        RuntimeArgs.fixed_data_normalizer = None
        RuntimeArgs.fixed_pde_normalizer = None
        RuntimeArgs.fixed_env_normalizer = 1.0
    if options.parallel:
        RuntimeArgs.use_parallel = True
        RuntimeArgs.num_gpus = options.num_gpus
        RuntimeArgs.device_ids = options.gpu_ids
    if options.main_losses_only:
        RuntimeArgs.a = RuntimeArgs.b = 1.
        RuntimeArgs.c = RuntimeArgs.d = RuntimeArgs.film_smooth_weight = 0.
        RuntimeArgs.enable_continuous_frequency_pde = False
        RuntimeArgs.continuous_pde_weight = 0.
    if options.save_every is not None:
        if options.save_every <= 0: parser.error('save-every must be positive')
        RuntimeArgs.save_model_every = RuntimeArgs.save_fig_every = options.save_every
    if options.min_free_memory_mib is not None:
        RuntimeArgs.min_gpu_memory = options.min_free_memory_mib
    if options.master_port is not None:
        if not 1 <= options.master_port <= 65535: parser.error('invalid master-port')
        os.environ['MASTER_PORT'] = str(options.master_port)
    if RuntimeArgs.use_parallel:
        if options.activation_offload: parser.error('DDP activation offload is not validated')
        from model.train_distributed import _validate_ddp_alignment_options, _ddp_batch_size_v
        _validate_ddp_alignment_options(RuntimeArgs)
        _ddp_batch_size_v(RuntimeArgs, RuntimeArgs.num_gpus)

    if options.pml_crop is not None:
        if not 0 <= options.pml_crop <= RuntimeArgs.pml_total:
            parser.error('pml-crop must be between zero and pml_total')
        RuntimeArgs.pml_crop = options.pml_crop
        RuntimeArgs.pml_active = RuntimeArgs.pml_total - options.pml_crop
    if options.last_epoch is not None:
        if options.last_epoch < 0:
            parser.error('last-epoch must be nonnegative')
        RuntimeArgs.NIter = options.last_epoch + 1
    if options.epochs is not None:
        if options.last_epoch is not None or options.epochs <= 0:
            parser.error('--epochs must be positive and cannot accompany --last-epoch')
        if not options.compact_output or not options.parallel or not options.from_scratch:
            parser.error('--epochs currently requires --compact-output --parallel --from-scratch')
        RuntimeArgs.start_epoch = 1
        RuntimeArgs.NIter = options.epochs + 1
    if options.compact_output:
        if not options.parallel or not options.from_scratch or not options.main_losses_only:
            parser.error('compact output requires fresh, non-staged DDP with --main-losses-only')
        if getattr(RuntimeArgs, 'staged_training', False): parser.error('compact staged DDP is not supported')
        RuntimeArgs.enable_marmousi_eval = RuntimeArgs.enable_marmousi_unseen_source_eval = True
        # Do not silently overwrite a previous experiment.
        if os.path.exists(os.path.join(options.save_doc, 'runtime_config.json')):
            parser.error('output directory already has runtime_config.json; use a new directory')
    os.makedirs(options.save_doc, exist_ok=True)
    snapshot = {key: getattr(RuntimeArgs, key) for key in dir(RuntimeArgs)
                if not key.startswith('_') and not callable(getattr(RuntimeArgs, key))}
    with open(os.path.join(options.save_doc, 'runtime_config.json'), 'w') as handle:
        json.dump(snapshot, handle, indent=2, default=str)
    print(f'[Runtime] batch_size_v={RuntimeArgs.batch_size_v}, pml_crop={RuntimeArgs.pml_crop}, '
          f'pml_active={RuntimeArgs.pml_active}, epochs={getattr(RuntimeArgs, "start_epoch", 0)}..{RuntimeArgs.NIter-1}', flush=True)

    if options.activation_offload:
        print('[Memory] CPU activation offload enabled; GPU computation, unchanged batch size', flush=True)
        with main2.torch.autograd.graph.save_on_cpu(pin_memory=True):
            main2.main(RuntimeArgs)
    else:
        main2.main(RuntimeArgs)


if __name__ == '__main__':
    print('*******************************************')
    print(' START TRAINING Pi_DeepONet (runtime config)')
    print('*******************************************')
    main()
