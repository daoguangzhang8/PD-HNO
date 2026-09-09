"""
单机多卡分布式训练模块 (Single-Machine Multi-GPU Distributed Training)

使用方法:
    python main2.py  (根据 config.py 中的 use_parallel 自动选择单卡/多卡)
"""

import os
import copy
import time
import json
import random
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
from contextlib import nullcontext
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from Labconfig import *
from model.utils import (
    setup_distributed,
    cleanup_distributed,
    wrap_model_for_distributed,
    is_main_process,
    reduce_tensor,
    build_epoch_velocity_gradient_prob,
    sample_shared_y_ran_from_epoch_prob,
)
from model.dataloader import (
    dataset_condition_flags,
    make_conditioned_dataset,
    prepare_external_val_dataset,
    prepare_training_dataloaders,
    set_dataloader_epoch_seed,
    unpack_conditioned_batch,
)
from model.PI_DeepOnet import Pi_DeepONet
from model.FNO import FNO
from model.point_averaging import coordinate_accumulation_batches, point_mean
from model.distributed_complex import broadcast_module_state, average_complex_gradients
from model.plotting import plot_loss, test_plot, plot_sinlge, fine_tuning
from model.utils import count_parameters, WarmupScheduler
from test import (
    evaluate_training_unseen_frequency,
    prepare_training_unseen_frequency_evaluator,
)


def _ddp_lr(args, base_lr, world_size):
    """Return the per-rank learning rate for DDP."""
    return base_lr * world_size if getattr(args, 'ddp_scale_lr', False) else base_lr


def _ddp_batch_size_v(args, world_size):
    """Return per-rank velocity batch size; optionally keep global batch close to single-GPU."""
    batch_size_v = int(args.batch_size_v)
    if int(world_size) <= 0 or batch_size_v <= 0:
        raise ValueError('world_size 和 batch_size_v 必须为正整数')
    if getattr(args, 'ddp_split_batch_size_v', True):
        if batch_size_v < world_size or batch_size_v % world_size != 0:
            raise ValueError(
                f'batch_size_v={batch_size_v} 必须能被 world_size={world_size} 整除，'
                '否则 DDP 全局 batch 与单卡语义不一致'
            )
        return batch_size_v // world_size
    return batch_size_v


def _ddp_continuous_batch_spec(local_batch_size, world_size, rank, ratio):
    """Partition the single-GPU continuous-PDE sub-batch across DDP ranks."""
    global_batch_size = int(local_batch_size) * int(world_size)
    global_count = min(
        global_batch_size,
        max(1, int(round(global_batch_size * float(ratio)))),
    )
    quotient, remainder = divmod(global_count, int(world_size))
    local_count = quotient + (1 if int(rank) < remainder else 0)
    scale = (float(world_size) * local_count / global_count) if local_count else 0.0
    return local_count, scale


def _validate_ddp_alignment_options(args):
    """Reject paths whose single-card/DDP equivalence is not yet tested."""
    if getattr(args, 'sampling_strategy', 'original') != 'original':
        raise NotImplementedError('严格 DDP 对齐暂不支持 Sobol 训练路径')
    if bool(getattr(args, 'use_y_ran', False)):
        raise NotImplementedError('严格 DDP 对齐暂不支持 y_ran 自适应采样路径')
    if getattr(args, 'branch2_type', 'fno') != 'fno':
        raise NotImplementedError('严格 DDP 对齐暂不支持含 BatchNorm 的 Branch2 路径')
    if int(getattr(args, 'accumulation_steps', 1)) <= 0:
        raise ValueError('accumulation_steps 必须为正整数')
    if bool(getattr(args, 'enable_continuous_frequency_pde', False)):
        raise NotImplementedError('当前严格 DDP 仅支持 Data+PDE；连续频率子批次尚未全局对齐，请关闭')
    if float(getattr(args, 'film_smooth_weight', 0.0)) != 0.0:
        raise NotImplementedError('当前严格 DDP 仅支持 Data+PDE；FiLM 正则分母尚未全局对齐，请置零')
    if getattr(args, 'ddp_scale_lr', False) or not getattr(args, 'ddp_split_batch_size_v', True):
        raise ValueError('严格对齐要求 ddp_scale_lr=False, ddp_split_batch_size_v=True')


def _seed_ddp_rank(base_seed, rank):
    """Seed all RNGs deterministically while keeping rank-local random streams."""
    rank_seed = int(base_seed) + int(rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    return rank_seed


def _assert_coordinate_batches_aligned(coord_batches, device, rank, world_size):
    """Fail fast when ranks do not use identical coordinate batches."""
    signature = [float(len(coord_batches))]
    for batch in coord_batches:
        tensor = batch[0] if isinstance(batch, (tuple, list)) else batch
        flat = tensor.detach().to(device=device, dtype=torch.float64).reshape(-1)
        weights = torch.arange(1, flat.numel() + 1, device=device, dtype=torch.float64)
        signature.extend([
            float(flat.numel()),
            float(flat.sum()),
            float(flat.square().sum()),
            float((flat * weights).sum()),
        ])
    values = torch.tensor(signature, device=device, dtype=torch.float64)
    minimum = values.clone()
    maximum = values.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if not torch.equal(minimum, maximum):
        raise RuntimeError(
            f'DDP 坐标 batch 在 rank={rank}/{world_size} 间不一致；'
            '请检查 Halton seed 和 train_y generator'
        )


def _manual_average_gradients(module, world_size, bucket_mb=32):
    """Average gradients accumulated entirely under ``no_sync()``."""
    del bucket_mb  # Kept in the public signature for backward compatibility.
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        value = torch.view_as_real(parameter.grad) if parameter.grad.is_complex() else parameter.grad
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        parameter.grad.div_(float(world_size))


def _flush_ddp_accumulation(model, optimizer, pending_steps, accumulation_steps,
                            world_size):
    """Synchronize and apply an incomplete accumulation window at epoch end."""
    pending_steps = int(pending_steps)
    if pending_steps == 0:
        return
    _manual_average_gradients(model.module, world_size)
    correction = float(accumulation_steps) / float(pending_steps)
    for parameter in model.module.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(correction)
    optimizer.step()
    optimizer.zero_grad()


def _build_scheduler(args, optimizer, lr, warmup_epochs=None):
    """Mirror the single-GPU scheduler selection in DDP."""
    scheduler_type = getattr(args, 'scheduler_type', 'plateau')
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=getattr(args, 'cosine_T_0', 1000),
            T_mult=getattr(args, 'cosine_T_mult', 2),
            eta_min=getattr(args, 'cosine_eta_min', 1e-6),
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.factor, patience=args.patience, min_lr=args.min_lr
        )

    use_warmup = getattr(args, 'use_warmup', False)
    warmup_scheduler = None
    if use_warmup:
        warmup_scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=args.warmup_epochs if warmup_epochs is None else warmup_epochs,
            base_lr=lr,
            warmup_start_lr=lr / 10.,
            warmup_strategy="linear",
            after_scheduler=None
        )

    return scheduler, warmup_scheduler, scheduler_type, use_warmup


def _get_resume_spec(args):
    """返回 DDP 续训路径和 checkpoint 对应的全局 epoch。"""
    if not getattr(args, 'if_load_model', False):
        return None

    checkpoint_path = str(getattr(args, 'resume_checkpoint', '')).strip()
    if not checkpoint_path:
        raise ValueError('if_load_model=True 时必须在 config 中设置 resume_checkpoint')
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'DDP 续训 checkpoint 不存在: {checkpoint_path}')

    resume_epoch = getattr(args, 'resume_epoch', None)
    if resume_epoch is None or int(resume_epoch) < 0:
        raise ValueError('if_load_model=True 时必须设置非负的 resume_epoch')
    return checkpoint_path, int(resume_epoch)


@torch.no_grad()
def _broadcast_module_state(module):
    """从 rank 0 广播参数和 buffer，支持 BatchNorm 等非参数状态。"""
    broadcast_module_state(module)


def _move_optimizer_state_to_device(optimizer, device):
    """optimizer.load_state_dict 后确保 Adam 状态位于本 rank 的 GPU。"""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device, non_blocking=True)


def _save_normalizer_calibration(args, raw_data_loss, raw_pde_loss, raw_total_loss):
    """保存基于 checkpoint 快照的无归一化损失与反推的两个 normalizer。"""
    reference_dir = os.path.abspath(
        getattr(args, 'normalizer_reference_output_dir', '') or
        os.path.dirname(getattr(args, 'resume_checkpoint', ''))
    )
    reference_epoch = int(getattr(args, 'normalizer_reference_epoch', 0))
    data_log_path = os.path.join(reference_dir, 'loss_data_log.npy')
    pde_log_path = os.path.join(reference_dir, 'loss_pde_log.npy')
    if not os.path.isfile(data_log_path) or not os.path.isfile(pde_log_path):
        raise FileNotFoundError(
            '无法读取用于反推 normalizer 的历史 loss npy：'
            f'{data_log_path}, {pde_log_path}'
        )

    reference_data_log = np.load(data_log_path)
    reference_pde_log = np.load(pde_log_path)
    if reference_epoch >= len(reference_data_log) or reference_epoch >= len(reference_pde_log):
        raise IndexError(
            f'参考 epoch={reference_epoch} 超出历史 loss 长度：'
            f'data={len(reference_data_log)}, pde={len(reference_pde_log)}'
        )

    reference_data_loss = float(reference_data_log[reference_epoch])
    reference_pde_loss = float(reference_pde_log[reference_epoch])
    if reference_data_loss <= 0.0 or reference_pde_loss <= 0.0:
        raise ValueError(
            '参考的归一化 loss 必须为正值，'
            f'但得到 data={reference_data_loss}, pde={reference_pde_loss}'
        )

    data_normalizer = float(raw_data_loss) / reference_data_loss
    pde_normalizer = float(raw_pde_loss) / reference_pde_loss
    result = {
        'checkpoint': os.path.abspath(getattr(args, 'resume_checkpoint', '')),
        'reference_output_dir': reference_dir,
        'reference_epoch': reference_epoch,
        'raw_data_loss': float(raw_data_loss),
        'raw_pde_loss': float(raw_pde_loss),
        'raw_total_loss': float(raw_total_loss),
        'reference_normalized_data_loss': reference_data_loss,
        'reference_normalized_pde_loss': reference_pde_loss,
        'data_normalizer': data_normalizer,
        'pde_normalizer': pde_normalizer,
    }
    output_path = os.path.join(args.save_doc, 'normalizer_calibration.json')
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print('=' * 60)
    print('无归一化 loss 标定完成（未执行 backward / optimizer.step）')
    print(f'raw Data loss: {raw_data_loss:.10e}')
    print(f'raw PDE loss : {raw_pde_loss:.10e}')
    print(f'参考 epoch {reference_epoch}: Data={reference_data_loss:.10e}, PDE={reference_pde_loss:.10e}')
    print(f'反推 Data normalizer: {data_normalizer:.10e}')
    print(f'反推 PDE normalizer : {pde_normalizer:.10e}')
    print(f'标定结果已保存: {output_path}')
    print('=' * 60)


def _train_worker(rank, world_size, args, device_ids):
    """Best-effort rank-zero recovery, without collectives during exception handling."""
    import signal
    previous_handler = None
    if rank == 0:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Worker received signal {signum}')
        previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        _train_worker_impl(rank, world_size, args, device_ids)
    except BaseException as error:
        if rank == 0:
            # mp.spawn terminates peer workers when another rank fails. Guard
            # against a second SIGTERM interrupting our local recovery write.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            trace = error.__traceback__
            state = None
            while trace is not None:
                if trace.tb_frame.f_code is _train_worker_impl.__code__:
                    state = trace.tb_frame.f_locals
                    break
                trace = trace.tb_next
            if state is not None:
                try:
                    from model.compact_output import save_interrupted_checkpoint
                    save_interrupted_checkpoint(args, state, error)
                except BaseException as save_error:
                    print(f'[Interrupted] checkpoint save failed: {type(save_error).__name__}: {save_error}', flush=True)
        raise
    finally:
        if rank == 0 and previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)


def _train_worker_impl(rank, world_size, args, device_ids):
    """
    分布式训练工作进程 (每个 GPU 运行一个)

    Args:
        rank: 当前进程的 rank (0, 1, 2, ...)
        world_size: 总进程数 (等于 GPU 数量)
        args: 配置参数对象
        device_ids: rank 到物理 GPU 编号的映射
    """
    device_id = int(device_ids[rank])
    normalizer_calibration_only = bool(
        getattr(args, 'normalizer_calibration_only', False)
    )
    # 初始化分布式环境
    setup_distributed(rank, world_size, device_id=device_id,
                      timeout_minutes=int(getattr(args, 'nccl_timeout_minutes', 10)))
    device = torch.device(f'cuda:{device_id}')
    args.device = device_id  # 确保绘图等函数使用正确的 GPU
    _validate_ddp_alignment_options(args)
    ddp_seed = int(getattr(args, 'ddp_seed', 20260410))
    coordinate_seed = int(getattr(args, 'ddp_coordinate_seed', 20260412))
    _seed_ddp_rank(ddp_seed, rank)

    if is_main_process(rank):
        print("=" * 60)
        print(f"单机多卡分布式训练模式")
        print(f"GPU 数量: {world_size}; rank→GPU 映射: {list(device_ids)}")
        print(f"=" * 60)

    compact_output = bool(getattr(args, 'compact_output', False))
    compact_history = []
    resume_spec = _get_resume_spec(args)
    start_epoch = int(getattr(args, 'start_epoch', 0))
    last_completed_epoch = start_epoch - 1
    optimizer_step_in_progress = False
    if resume_spec is not None:
        checkpoint_path, resume_epoch = resume_spec
        start_epoch = resume_epoch + 1
        last_completed_epoch = resume_epoch
        if start_epoch >= int(args.NIter):
            raise ValueError(
                f'续训起点为 epoch {start_epoch}，但 NIter={args.NIter}；'
                'NIter 必须是大于 resume_epoch 的总目标 epoch。'
            )
        if os.path.abspath(args.save_doc) == os.path.dirname(checkpoint_path):
            raise ValueError(
                'DDP 续训输出目录不能与初始 checkpoint 所在目录相同，'
                '否则可能覆盖原始权重。请修改 config.save_doc。'
            )

    if is_main_process(rank):
        os.makedirs(args.save_doc, exist_ok=True)
    # 显式指定物理 GPU，避免 NCCL 在非连续 GPU 映射下猜测 barrier 设备。
    dist.barrier(device_ids=[device_id])

    # ==========================================
    # 加载数据
    # ==========================================
    if is_main_process(rank):
        print("正在加载数据...")

    dataloader, plot_data = prepare_training_dataloaders(args, device)

    # 加载外部验证集 (只在主进程)
    ext_val_sets = {}
    if hasattr(args, 'ext_val_datasets') and is_main_process(rank):
        for name, config in args.ext_val_datasets.items():
            loader, p_data = prepare_external_val_dataset(
                args,
                prefix=config['prefix'],
                loc_target=config['loc_target'],
                y_pred_grid=plot_data["y_pred"]
            )
            ext_val_sets[name] = {"loader": loader, "plot_data": p_data}

    if compact_output:
        # Fixed validation membership/order avoids changes from shuffled drop_last.
        for key, batch_size in (('valid', args.valid_batch_size_v), ('valid_y', args.valid_batch_size)):
            dataloader[key] = DataLoader(dataloader[key].dataset, batch_size=batch_size,
                                         shuffle=False, drop_last=False, num_workers=0)

    # 替换为 DistributedSampler
    train_sampler = DistributedSampler(
        dataloader['train'].dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=ddp_seed,
        drop_last=True,
    )
    dataloader['train'] = DataLoader(
        dataloader['train'].dataset,
        batch_size=_ddp_batch_size_v(args, world_size),
        sampler=train_sampler,
        drop_last=True,
        num_workers=max(1, 4),
        pin_memory=True
    )

    if is_main_process(rank):
        print(f"已启用 DistributedSampler (world_size={world_size})")

    # 检测数据集是否包含频率信息
    has_freq, has_source_coords = dataset_condition_flags(dataloader['train'].dataset)

    # ==========================================
    # 创建模型
    # ==========================================
    model = Pi_DeepONet(args).to(device)

    if is_main_process(rank):
        if resume_spec is None:
            model._init_weights()
            print('从头初始化模型权重')
        else:
            checkpoint_path, resume_epoch = resume_spec
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' not in checkpoint:
                raise KeyError(f'checkpoint 缺少 model_state_dict: {checkpoint_path}')
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(
                f'从 checkpoint 恢复模型: {checkpoint_path} '
                f'(全局 epoch {resume_epoch})'
            )
        print(f"PI_DeepONet 模型总参数数量: {count_parameters(model)}")

    # 在 DDP 包装前同步 rank 0 的模型参数和 buffer。
    _broadcast_module_state(model)

    # 包装模型为 DDP
    model = wrap_model_for_distributed(model, device_id)
    if is_main_process(rank):
        print("模型已包装为 DistributedDataParallel")

    marmousi_evaluators = []
    if is_main_process(rank) and getattr(args, 'enable_marmousi_eval', False):
        marmousi_evaluators.append(prepare_training_unseen_frequency_evaluator(
            args, device
        ))
    if is_main_process(rank) and getattr(args, 'enable_marmousi_unseen_source_eval', False):
        marmousi_evaluators.append(prepare_training_unseen_frequency_evaluator(
            args, device,
            frequencies=args.marmousi_unseen_source_eval_frequencies,
            sources=args.marmousi_unseen_source_eval_sources,
            data_dir=args.marmousi_unseen_source_eval_data_dir,
            output_name=args.marmousi_unseen_source_eval_output_dir,
        ))

    fno = None
    if args.use_fno_as_label:
        fno = FNO(args).to(device)
        if args.fno_weights_path:
            fno.load_state_dict(torch.load(args.fno_weights_path, map_location=device)['model_state_dict'])
            if is_main_process(rank):
                print(f"已加载 FNO 权重: {args.fno_weights_path}")
        fno.eval()

    # ==========================================
    # 优化器与调度器
    # ==========================================
    ddp_lr = _ddp_lr(args, args.lr, world_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=ddp_lr, weight_decay=args.weight_decay)
    scheduler, warmup_scheduler, scheduler_type, use_warmup = _build_scheduler(args, optimizer, ddp_lr)

    resume_normalizers = None
    if resume_spec is not None:
        checkpoint_path, resume_epoch = resume_spec
        # 每个 rank 按本地设备加载 optimizer 状态；模型权重已由 rank 0 广播。
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if getattr(args, 'resume_restore_optimizer', True):
            if 'optimizer_state_dict' not in checkpoint:
                raise KeyError(f'checkpoint 缺少 optimizer_state_dict: {checkpoint_path}')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            _move_optimizer_state_to_device(optimizer, device)
        if getattr(args, 'resume_restore_scheduler', True):
            if 'scheduler_state_dict' not in checkpoint:
                raise KeyError(f'checkpoint 缺少 scheduler_state_dict: {checkpoint_path}')
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        resume_normalizers = checkpoint.get('loss_normalizers')
        del checkpoint

    if is_main_process(rank):
        print(f"DDP per-rank batch_size_v: {_ddp_batch_size_v(args, world_size)} "
              f"(global≈{_ddp_batch_size_v(args, world_size) * world_size}, single={args.batch_size_v})")
        print(f"DDP lr: {ddp_lr:.3e} (scale_lr={getattr(args, 'ddp_scale_lr', False)})")
        print(f"Scheduler: {scheduler_type}" + (f" (warmup {args.warmup_epochs} epochs)" if use_warmup else ""))
        if resume_spec is not None:
            print(
                f'续训状态已恢复: optimizer={getattr(args, "resume_restore_optimizer", True)}, '
                f'scheduler={getattr(args, "resume_restore_scheduler", True)}, '
                f'从 epoch {start_epoch} 开始'
            )
        if normalizer_calibration_only:
            print(
                '已启用 normalizer_calibration_only：仅计算一个 epoch 的无归一化 '
                'Data/PDE loss；不会执行 backward、optimizer.step 或保存训练权重。'
            )

    # ==========================================
    # 训练状态初始化
    # ==========================================
    loss_log, loss_pde_log, loss_data_log, loss_reg_log, loss_env_log = [], [], [], [], []
    loss_continuous_pde_log = []
    valid_u_loss, valid_f_loss = [], []

    # epoch-level 共享 y_ran 采样状态
    epoch_prob = None
    epoch_score = None

    a, b, c, d = args.a, args.b, args.c, args.d
    configured_data_normalizer = getattr(args, 'fixed_data_normalizer', None)
    configured_pde_normalizer = getattr(args, 'fixed_pde_normalizer', None)
    configured_env_normalizer = getattr(args, 'fixed_env_normalizer', 1.0)
    has_configured_normalizers = (
        configured_data_normalizer is not None and
        configured_pde_normalizer is not None
    )
    if has_configured_normalizers:
        configured_data_normalizer = float(configured_data_normalizer)
        configured_pde_normalizer = float(configured_pde_normalizer)
        configured_env_normalizer = float(configured_env_normalizer)
        if (
            configured_data_normalizer <= 0.0 or
            configured_pde_normalizer <= 0.0 or
            configured_env_normalizer <= 0.0
        ):
            raise ValueError('fixed_data/pde/env_normalizer 必须均为正数')

    # 显式固定值优先于 checkpoint；这样从旧 checkpoint 续训时不会再用首轮 1.0 占位标定。
    first_flag = resume_normalizers is None and not has_configured_normalizers
    pde_norm_coe = 1.
    data_norm_coe = 1.
    env_norm_coe = 1.
    if has_configured_normalizers:
        data_norm_coe = configured_data_normalizer
        pde_norm_coe = configured_pde_normalizer
        env_norm_coe = configured_env_normalizer
        if is_main_process(rank):
            print(
                '使用 config 中固定的 loss normalizer：'
                f'data={data_norm_coe:.10e}, pde={pde_norm_coe:.10e}, '
                f'env={env_norm_coe:.10e}'
            )
    elif resume_normalizers is not None:
        data_norm_coe = float(resume_normalizers['data'])
        pde_norm_coe = float(resume_normalizers['pde'])
        env_norm_coe = float(resume_normalizers['env'])
    elif resume_spec is not None and is_main_process(rank):
        print('注意：旧 checkpoint 未保存 loss_normalizers，将在续训首个 epoch 重新估计。')

    # ==========================================
    # 主训练循环
    # ==========================================
    optimizer.zero_grad()
    # 只在主进程显示进度条
    epoch_range = (
        range(start_epoch, start_epoch + 1)
        if normalizer_calibration_only else range(start_epoch, args.NIter)
    )
    if is_main_process(rank):
        pbar = tqdm(epoch_range, desc="Normalizer Calibration" if normalizer_calibration_only else "Training Progress", dynamic_ncols=True, disable=compact_output)
    else:
        pbar = epoch_range

    last_epoch = start_epoch - 1
    for i in pbar:
        epoch_started = time.perf_counter()
        last_epoch = i
        step_counter = 0
        _seed_ddp_rank(ddp_seed + i * world_size, rank)
        # 动态调整损失权重
        if args.if_adjust and i > args.adjust_from and (i - args.adjust_from) % args.adjust_every == 0:
            decay_times = i // args.adjust_every
            a = max(a * (args.adjust_speed ** (-decay_times)), 2e-1)
            b, c = 1, 0

        model.train()
        batch_loss, batch_u_loss, batch_f_loss, batch_r_loss, batch_env_loss = [], [], [], [], []
        batch_continuous_pde_loss = []
        batch_point_counts = []
        continuous_pde_active = (
            not normalizer_calibration_only
            and bool(getattr(args, 'enable_continuous_frequency_pde', False))
            and i >= int(getattr(args, 'continuous_pde_start_epoch', 1))
        )

        # 设置 epoch 以确保每个 epoch shuffle 不同
        dataloader['train'].sampler.set_epoch(i)

        # 预收集 coordinate batches（每 epoch 一次，避免在 velocity 循环内重复物化）
        set_dataloader_epoch_seed(dataloader['train_y'], coordinate_seed, i)
        coord_batches = list(dataloader['train_y'])
        if (
            bool(getattr(args, 'ddp_assert_coordinate_alignment', True))
            and i == start_epoch
        ):
            _assert_coordinate_batches_aligned(
                coord_batches, device=device, rank=rank, world_size=world_size
            )
        n_coord = len(coord_batches)

        if i == start_epoch and is_main_process(rank):
            microsteps = len(dataloader['train']) * n_coord
            updates = len(dataloader['train']) * (
                (n_coord + args.accumulation_steps - 1) // args.accumulation_steps
            )
            print(f'[DDP batches] velocity_batches_per_rank={len(dataloader["train"])}, '
                  f'coordinate_batches={n_coord}, microsteps_per_epoch={microsteps}, '
                  f'optimizer_updates_per_epoch={updates}', flush=True)

        # y_ran: 每 epoch 生成一次（epoch shared 路径）
        y_ran_epoch_shared = None
        if getattr(args, 'use_y_ran', False) and getattr(args, 'use_epoch_shared_y_ran', False):
            should_update_prob = (
                epoch_prob is None
                or args.y_ran_prob_update_every == 1
                or (args.y_ran_prob_update_every > 1 and i % args.y_ran_prob_update_every == 0)
            )
            if should_update_prob:
                with torch.no_grad():
                    epoch_prob, epoch_score = build_epoch_velocity_gradient_prob(
                        train_loader=dataloader['train'],
                        device=device,
                        use_max_mix=args.y_ran_use_max_mix,
                        mean_weight=args.y_ran_mean_weight,
                        max_weight=args.y_ran_max_weight,
                    )

            with torch.no_grad():
                y_ran_epoch_shared = sample_shared_y_ran_from_epoch_prob(
                    prob=epoch_prob,
                    args=args,
                    num_pts=args.y_ran_num_pts,
                    structure_ratio=args.y_ran_structure_ratio,
                    surface_ratio=args.y_ran_surface_ratio,
                    uniform_ratio=args.y_ran_uniform_ratio,
                    source_ratio=args.y_ran_source_ratio,
                    surface_depth_grids=args.y_ran_surface_depth_grids,
                )

        # 遍历训练数据
        for velocity_batch_index, batch_data in enumerate(dataloader['train']):
            vel_batch, UU0_batch, labels_batch, freq_batch, source_coord_batch = \
                unpack_conditioned_batch(batch_data)
            if freq_batch is not None:
                freq_batch = freq_batch.to(device)
            if source_coord_batch is not None:
                source_coord_batch = source_coord_batch.to(device)
            vel_batch, UU0_batch = vel_batch.to(device), UU0_batch.to(device)

            if args.use_fno_as_label:
                with torch.no_grad():
                    labels_batch = fno(vel_batch, UU0_batch).to(device)
            else:
                labels_batch = labels_batch.to(device)

            # y_ran: 使用 epoch 预生成或 per-model 生成
            if y_ran_epoch_shared is not None:
                y_ran = y_ran_epoch_shared.unsqueeze(0).expand(
                    vel_batch.shape[0], -1, -1
                ).clone().requires_grad_(True)
            elif getattr(args, 'use_y_ran', False):
                with torch.no_grad():
                    y_ran = model.module.generate_structure_aware_y_ran(vel_batch, num_pts=900)
            else:
                y_ran = None

            if velocity_batch_index:
                # Match single-card semantics: reshuffle coordinate blocks for
                # every velocity batch, not one fixed grouping for the epoch.
                coord_batches = list(dataloader['train_y'])
            # Strict DDP options require identical data/PDE coordinates on all ranks.
            for batch, point_weight, _, group_end in coordinate_accumulation_batches(
                coord_batches, args.accumulation_steps
            ):
                y_batch = batch[0].to(device)
                y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                # 拼接数据坐标和 PDE 采样坐标
                y_combined = torch.cat([y_batch, y_ran], dim=1) if y_ran is not None else y_batch
                y_combined.requires_grad_(True)

                # 标定时不走 DDP forward：无 backward 时避免 DDP reducer 等待梯度，
                # 且两个 rank 的模型已在包装前完成同步。
                if normalizer_calibration_only:
                    sync_context = nullcontext()
                    forward_model = model.module
                else:
                    sync_grad = group_end
                    sync_context = nullcontext() if sync_grad else model.no_sync()
                    forward_model = model

                with sync_context:
                    continuous_inputs = (
                        model.module.prepare_continuous_frequency_batch(
                            vel_batch, UU0_batch,
                            source_coord_batch=source_coord_batch,
                        )
                        if continuous_pde_active else None
                    )
                    forward_result = forward_model(
                        vel_batch, y_combined, UU0_batch, freq_batch=freq_batch,
                        source_coord_batch=source_coord_batch,
                        continuous_inputs=continuous_inputs,
                    )
                    if continuous_inputs is None:
                        Delta_U = forward_result
                        Delta_U_cont = None
                    else:
                        Delta_U, Delta_U_cont = forward_result

                    loss, loss_f, loss_u, loss_r, loss_env, loss_continuous_pde = model.module.compute_loss(
                        Delta_U, vel_batch, y_batch, UU0_batch, labels_batch,
                        y_combined, a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                        freq_batch=freq_batch, continuous_inputs=continuous_inputs,
                        Delta_U_cont=Delta_U_cont,
                        source_coord_batch=source_coord_batch,
                    )

                    if not normalizer_calibration_only:
                        backward_loss = loss * point_weight
                        backward_loss.backward()  # Only group_end triggers DDP averaging.
                        del backward_loss

                if not normalizer_calibration_only:
                    step_counter += 1
                    if group_end:
                        average_complex_gradients(model.module)
                        optimizer_step_in_progress = True
                        optimizer.step()
                        optimizer_step_in_progress = False
                        optimizer.zero_grad()

                batch_point_counts.append(vel_batch.shape[0] * y_batch.shape[1])
                batch_loss.append(loss.item())
                batch_u_loss.append(loss_u.item())
                batch_f_loss.append(loss_f.item())
                batch_r_loss.append(loss_r.item() if isinstance(loss_r, torch.Tensor) else loss_r)
                batch_env_loss.append(loss_env.item())
                batch_continuous_pde_loss.append(loss_continuous_pde.item())

                del loss, loss_f, loss_u, loss_r, loss_env, loss_continuous_pde
                del y_batch, Delta_U, Delta_U_cont, continuous_inputs, forward_result

        # Every group, including a short tail, has already synchronized and stepped.

        # 跨进程平均损失
        avg_loss = point_mean(batch_loss, batch_point_counts) if batch_loss else 0
        avg_u_loss = point_mean(batch_u_loss, batch_point_counts) if batch_u_loss else 0
        avg_f_loss = point_mean(batch_f_loss, batch_point_counts) if batch_f_loss else 0
        avg_r_loss = point_mean(batch_r_loss, batch_point_counts) if batch_r_loss else 0
        avg_continuous_pde_loss = (
            point_mean(batch_continuous_pde_loss, batch_point_counts) if batch_continuous_pde_loss else 0
        )

        loss_tensor = torch.tensor(
            [avg_loss, avg_u_loss, avg_f_loss, avg_r_loss, avg_continuous_pde_loss],
            device=device,
        )
        loss_tensor = reduce_tensor(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss, avg_u_loss, avg_f_loss, avg_r_loss, avg_continuous_pde_loss = loss_tensor.cpu().numpy()

        avg_env_loss = point_mean(batch_env_loss, batch_point_counts) if batch_env_loss else 0

        if normalizer_calibration_only:
            if is_main_process(rank):
                _save_normalizer_calibration(
                    args,
                    raw_data_loss=avg_u_loss,
                    raw_pde_loss=avg_f_loss,
                    raw_total_loss=avg_loss,
                )
            dist.barrier(device_ids=[device_id])
            cleanup_distributed()
            return

        if first_flag:
            data_norm_coe = avg_u_loss if avg_u_loss > 0 else 1.0
            pde_norm_coe = avg_f_loss if avg_f_loss > 0 else 1.0
            env_norm_coe = avg_env_loss if avg_env_loss > 0 else 1.0
            if compact_output and is_main_process(rank):
                print(f'[Loss baseline] method=first_training_epoch_point_mean '
                      f'data={data_norm_coe:.10e} pde={pde_norm_coe:.10e}; '
                      'not a frozen epoch-0 evaluation', flush=True)
            loss_log.append(a + b)
            loss_data_log.append(1.)
            loss_pde_log.append(1.)
            loss_env_log.append(1.)
            loss_reg_log.append(avg_r_loss)
            loss_continuous_pde_log.append(0.0)
            first_flag = False
        else:
            loss_log.append(avg_loss)
            loss_data_log.append(avg_u_loss)
            loss_pde_log.append(avg_f_loss)
            loss_env_log.append(avg_env_loss)
            loss_reg_log.append(avg_r_loss)
            loss_continuous_pde_log.append(avg_continuous_pde_loss)

        train_seconds = time.perf_counter() - epoch_started
        if compact_output and not np.isfinite([avg_loss, avg_u_loss, avg_f_loss]).all():
            raise FloatingPointError(f'Non-finite global losses at epoch {i}')

        # 更新进度条 (只在主进程)
        if is_main_process(rank):
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Total': f"{avg_loss:.4e}",
                'PDE': f"{loss_pde_log[-1]:.4e}",
                'Data': f"{loss_data_log[-1]:.4e}",
                'ContPDE': f"{loss_continuous_pde_log[-1]:.2e}",
                'FiLM': f"{loss_reg_log[-1]:.2e}",
                'LR': f"{current_lr:.2e}",
                'GPU': f"{world_size}"
            })

        # 学习率调度
        if use_warmup and warmup_scheduler is not None and i <= args.warmup_epochs:
            warmup_scheduler.step(i)
        elif scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_loss)

        # ==========================================
        # 验证环节 (只在主进程)
        # ==========================================
        last_completed_epoch = i
        if i % args.validate_every == 0 and is_main_process(rank):
            model.eval()
            batch_u_loss, batch_f_loss = [], []
            batch_point_counts = []

            for batch_data in dataloader['valid']:
                vel_batch, UU0_batch, labels_batch, freq_batch, source_coord_batch = \
                    unpack_conditioned_batch(batch_data)
                if freq_batch is not None:
                    freq_batch = freq_batch.to(device)
                if source_coord_batch is not None:
                    source_coord_batch = source_coord_batch.to(device)
                vel_batch = vel_batch.to(device)
                UU0_batch = UU0_batch.to(device)
                labels_batch = labels_batch.to(device)

                for batch in dataloader['valid_y']:
                    y_batch = batch[0].to(device)
                    y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    _, loss_f_valid, loss_u_valid, _, _, _ = model.module.loss(
                        vel_batch, y_batch, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                        freq_batch=freq_batch, source_coord_batch=source_coord_batch,
                    )
                    batch_point_counts.append(vel_batch.shape[0] * y_batch.shape[1])
                    batch_u_loss.append(loss_u_valid.item())
                    batch_f_loss.append(loss_f_valid.item())

                    # Keep scalar metrics only, not the coordinate derivative graph.
                    del loss_f_valid, loss_u_valid, y_batch

            valid_u_loss.append(point_mean(batch_u_loss, batch_point_counts) if batch_u_loss else 0.0)
            valid_f_loss.append(point_mean(batch_f_loss, batch_point_counts) if batch_f_loss else 1.0)

            if not compact_output:
                for marmousi_evaluator in marmousi_evaluators:
                    evaluate_training_unseen_frequency(
                        marmousi_evaluator, model.module, i
                    )

        if compact_output:
            if is_main_process(rank):
                from model.compact_output import finish_epoch
                validation_due = i % args.validate_every == 0
                finish_epoch(
                    args, model.module, optimizer, scheduler, i, compact_history, plot_data,
                    marmousi_evaluators, dict(data=data_norm_coe, pde=pde_norm_coe, env=env_norm_coe),
                    resume_spec, device, train_seconds, current_lr,
                    loss_data_log[-1], loss_pde_log[-1],
                    valid_u_loss[-1] if validation_due else np.nan,
                    valid_f_loss[-1] if validation_due else np.nan,
                )
            dist.barrier(device_ids=[device_id])
            continue

        # ==========================================
        # 可视化与绘图 (只在主进程)
        # ==========================================
        if i % args.save_fig_every == 0 and is_main_process(rank):
            vel_pred = plot_data["vel_pred"]
            UU0_pred = plot_data["UU0_pred"]
            labels_pred = plot_data["labels_pred"]
            vel_test = plot_data["vel_test"]
            UU0_test = plot_data["UU0_test"]
            labels_test = plot_data["labels_test"]
            freq_pred = plot_data["freq_valid"][0:1] if has_freq else None
            freq_test = plot_data["freq_train"][0:1] if has_freq else None
            source_coord_test = plot_data.get("source_coord_train")
            source_coord_pred = plot_data.get("source_coord_valid")

            plot_loss(i, args.save_doc, loss_log, loss_data_log, loss_pde_log, valid_u_loss, valid_f_loss)

            if i % (args.save_fig_every * 20) == 0 and i > 0 and args.if_finetune:
                if ext_val_sets:
                    marmousi_data = ext_val_sets['Marmousi']
                    v_m_test = marmousi_data["plot_data"]["v_test"]
                    u0_m_test = marmousi_data["plot_data"]["u0_test"]
                    lab_m_test = marmousi_data["plot_data"]["lab_test"]
                    dataloader_m_y_full = marmousi_data["loader"]

                    test_plot(args, model.module, fno, i, dataloader_m_y_full,
                              v_m_test, u0_m_test, lab_m_test, 'FT_Marmousi', if_fine_tune=True)

            test_plot(args, model.module, fno, i, dataloader["pred"],
                      vel_pred, UU0_pred, labels_pred, 'valid_without_fine_tune',
                      if_fine_tune=False, freq=freq_pred, source_coord=source_coord_pred)
            test_plot(args, model.module, fno, i, dataloader["test"],
                      vel_test, UU0_test, labels_test, 'train',
                      if_fine_tune=False, freq=freq_test, source_coord=source_coord_test)
            plot_sinlge(
                model.module, args, 6, vel_test, UU0_test, labels_test,
                freq=freq_test, source_coord=source_coord_test,
            )

        # ==========================================
        # 模型保存 (只在主进程)
        # ==========================================
        if i % args.save_model_every == 0 and is_main_process(rank):
            if isinstance(pbar, tqdm):
                pbar.write(f'>>> Epoch {i} | 保存 Checkpoint: Total Loss {loss_log[-1]:.4e} | PDE Loss {loss_pde_log[-1]:.4e}')

            checkpoint = {
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': i,
                'resumed_from': resume_spec[0] if resume_spec is not None else None,
                'loss_normalizers': {
                    'data': data_norm_coe,
                    'pde': pde_norm_coe,
                    'env': env_norm_coe,
                },
            }

            torch.save(checkpoint, os.path.join(args.save_doc, f'{args.filename}_PI_model_{i}epoch_weights_{args.nz}.pth'))
            np.save(os.path.join(args.save_doc, 'loss_log.npy'), loss_log)
            np.save(os.path.join(args.save_doc, 'loss_data_log.npy'), loss_data_log)
            np.save(os.path.join(args.save_doc, 'loss_pde_log.npy'), loss_pde_log)
            np.save(os.path.join(args.save_doc, 'loss_continuous_pde_log.npy'), loss_continuous_pde_log)
            np.save(os.path.join(args.save_doc, 'loss_film_smooth_log.npy'), loss_reg_log)
            np.save(os.path.join(args.save_doc, 'loss_env_log.npy'), loss_env_log)

        dist.barrier(device_ids=[device_id])

    # ==========================================
    # 最终保存与清理
    # ==========================================
    if compact_output:
        if is_main_process(rank):
            print(f'训练完成: epochs={start_epoch}..{last_epoch}; no duplicate final checkpoint', flush=True)
        cleanup_distributed()
        return
    if is_main_process(rank):
        checkpoint = {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': last_epoch,
            'resumed_from': resume_spec[0] if resume_spec is not None else None,
            'loss_normalizers': {
                'data': data_norm_coe,
                'pde': pde_norm_coe,
                'env': env_norm_coe,
            },
        }
        torch.save(checkpoint, os.path.join(args.save_doc, f'{args.filename}_PI_model_final_weights_{args.nz}.pth'))
        print(f"训练完成! 模型已保存到 {args.save_doc}")

    cleanup_distributed()


def train_distributed(args, device_ids=None):
    """
    单机多卡分布式训练入口函数

    使用 mp.spawn 内部启动多进程，无需 torchrun

    Args:
        args: 配置参数对象 (config.Args)
        device_ids: 可选物理 GPU 列表；省略时兼容 [0, ..., num_gpus-1]
    """
    if device_ids is None:
        device_ids = list(range(getattr(args, 'num_gpus', 1)))
    device_ids = [int(device_id) for device_id in device_ids]
    if not device_ids or len(set(device_ids)) != len(device_ids):
        raise ValueError(f'DDP device_ids 必须是非空且不重复的 GPU 编号列表: {device_ids}')
    world_size = len(device_ids)
    _validate_ddp_alignment_options(args)
    _ddp_batch_size_v(args, world_size)

    print("=" * 60)
    print(f"启动单机多卡分布式训练")
    print(f"GPU 数量: {world_size}; 物理 GPU: {device_ids}")
    print("=" * 60)

    # 使用 mp.spawn 启动多进程
    mp.spawn(
        _train_worker,
        args=(world_size, args, device_ids),
        nprocs=world_size,
        join=True
    )


# ==============================================================================
# DDP 课程学习 (Staged Curriculum Training) 相关函数
# ==============================================================================

def _run_stage_training_loop(args, model, fno, device, rank, world_size,
                             dataloader, plot_data, has_freq,
                             optimizer, scheduler, warmup_scheduler,
                             stage_idx, stage_name, stage_niter, stage_warmup,
                             a, b, c, d, save_doc):
    """DDP 单阶段训练循环"""
    loss_log, loss_pde_log, loss_data_log, loss_reg_log, loss_env_log = [], [], [], [], []
    loss_continuous_pde_log = []
    valid_u_loss, valid_f_loss = [], []

    marmousi_evaluators = []
    if is_main_process(rank) and getattr(args, 'enable_marmousi_eval', False):
        marmousi_evaluators.append(prepare_training_unseen_frequency_evaluator(
            args, device, stage_tag=f'stage_{stage_idx}'
        ))
    if is_main_process(rank) and getattr(args, 'enable_marmousi_unseen_source_eval', False):
        marmousi_evaluators.append(prepare_training_unseen_frequency_evaluator(
            args, device, stage_tag=f'stage_{stage_idx}',
            frequencies=args.marmousi_unseen_source_eval_frequencies,
            sources=args.marmousi_unseen_source_eval_sources,
            data_dir=args.marmousi_unseen_source_eval_data_dir,
            output_name=args.marmousi_unseen_source_eval_output_dir,
        ))

    epoch_prob = None
    epoch_score = None

    first_flag = True
    pde_norm_coe = 1.
    data_norm_coe = 1.
    env_norm_coe = 1.

    optimizer.zero_grad()
    scheduler_type = getattr(args, 'scheduler_type', 'plateau')
    use_warmup = getattr(args, 'use_warmup', False)
    ddp_seed = int(getattr(args, 'ddp_seed', 20260410)) + stage_idx * 1_000_000
    coordinate_seed = (
        int(getattr(args, 'ddp_coordinate_seed', 20260412))
        + stage_idx * 1_000_000
    )

    if is_main_process(rank):
        pbar = tqdm(range(stage_niter), desc=f"Stage {stage_idx} [{stage_name}]", dynamic_ncols=True)
    else:
        pbar = range(stage_niter)

    for i in pbar:
        step_counter = 0
        _seed_ddp_rank(ddp_seed + i * world_size, rank)
        if args.if_adjust and i > args.adjust_from and (i - args.adjust_from) % args.adjust_every == 0:
            decay_times = i // args.adjust_every
            a = max(a * (args.adjust_speed ** (-decay_times)), 2e-1)
            b, c = 1, 0

        model.train()
        batch_loss, batch_u_loss, batch_f_loss, batch_r_loss, batch_env_loss = [], [], [], [], []
        batch_continuous_pde_loss = []
        batch_point_counts = []
        continuous_pde_active = (
            bool(getattr(args, 'enable_continuous_frequency_pde', False))
            and i >= int(getattr(args, 'continuous_pde_start_epoch', 1))
        )

        dataloader['train'].sampler.set_epoch(i)

        set_dataloader_epoch_seed(dataloader['train_y'], coordinate_seed, i)
        coord_batches = list(dataloader['train_y'])
        if bool(getattr(args, 'ddp_assert_coordinate_alignment', True)) and i == 0:
            _assert_coordinate_batches_aligned(
                coord_batches, device=device, rank=rank, world_size=world_size
            )
        n_coord = len(coord_batches)

        # y_ran: 每 epoch 生成一次（epoch shared 路径）
        y_ran_epoch_shared = None
        if getattr(args, 'use_y_ran', False) and getattr(args, 'use_epoch_shared_y_ran', False):
            should_update_prob = (
                epoch_prob is None
                or args.y_ran_prob_update_every == 1
                or (args.y_ran_prob_update_every > 1 and i % args.y_ran_prob_update_every == 0)
            )
            if should_update_prob:
                with torch.no_grad():
                    epoch_prob, epoch_score = build_epoch_velocity_gradient_prob(
                        train_loader=dataloader['train'],
                        device=device,
                        use_max_mix=args.y_ran_use_max_mix,
                        mean_weight=args.y_ran_mean_weight,
                        max_weight=args.y_ran_max_weight,
                    )

            with torch.no_grad():
                y_ran_epoch_shared = sample_shared_y_ran_from_epoch_prob(
                    prob=epoch_prob, args=args,
                    num_pts=args.y_ran_num_pts,
                    structure_ratio=args.y_ran_structure_ratio,
                    surface_ratio=args.y_ran_surface_ratio,
                    uniform_ratio=args.y_ran_uniform_ratio,
                    source_ratio=args.y_ran_source_ratio,
                    surface_depth_grids=args.y_ran_surface_depth_grids,
                )

        for velocity_batch_index, batch_data in enumerate(dataloader['train']):
            vel_batch, UU0_batch, labels_batch, freq_batch, source_coord_batch = \
                unpack_conditioned_batch(batch_data)
            if freq_batch is not None:
                freq_batch = freq_batch.to(device)
            if source_coord_batch is not None:
                source_coord_batch = source_coord_batch.to(device)
            vel_batch, UU0_batch = vel_batch.to(device), UU0_batch.to(device)

            if args.use_fno_as_label:
                with torch.no_grad():
                    labels_batch = fno(vel_batch, UU0_batch).to(device)
            else:
                labels_batch = labels_batch.to(device)

            # y_ran: 使用 epoch 预生成或 per-model 生成
            if y_ran_epoch_shared is not None:
                y_ran = y_ran_epoch_shared.unsqueeze(0).expand(
                    vel_batch.shape[0], -1, -1
                ).clone().requires_grad_(True)
            elif getattr(args, 'use_y_ran', False):
                with torch.no_grad():
                    y_ran = model.module.generate_structure_aware_y_ran(vel_batch, num_pts=900)
            else:
                y_ran = None

            if velocity_batch_index:
                coord_batches = list(dataloader['train_y'])
            # Strict DDP options require identical data/PDE coordinates on all ranks.
            for batch, point_weight, _, group_end in coordinate_accumulation_batches(
                coord_batches, args.accumulation_steps
            ):
                y_batch = batch[0].to(device)
                y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                y_combined = torch.cat([y_batch, y_ran], dim=1) if y_ran is not None else y_batch
                y_combined.requires_grad_(True)

                sync_grad = group_end
                sync_context = nullcontext() if sync_grad else model.no_sync()

                with sync_context:
                    continuous_inputs = (
                        model.module.prepare_continuous_frequency_batch(
                            vel_batch, UU0_batch,
                            source_coord_batch=source_coord_batch,
                        )
                        if continuous_pde_active else None
                    )
                    forward_result = model(
                        vel_batch, y_combined, UU0_batch, freq_batch=freq_batch,
                        source_coord_batch=source_coord_batch,
                        continuous_inputs=continuous_inputs,
                    )
                    if continuous_inputs is None:
                        Delta_U = forward_result
                        Delta_U_cont = None
                    else:
                        Delta_U, Delta_U_cont = forward_result

                    loss, loss_f, loss_u, loss_r, loss_env, loss_continuous_pde = model.module.compute_loss(
                        Delta_U, vel_batch, y_batch, UU0_batch, labels_batch,
                        y_combined, a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                        freq_batch=freq_batch, continuous_inputs=continuous_inputs,
                        Delta_U_cont=Delta_U_cont,
                        source_coord_batch=source_coord_batch,
                    )

                    backward_loss = loss * point_weight
                    backward_loss.backward()
                    del backward_loss

                step_counter += 1
                if group_end:
                    average_complex_gradients(model.module)
                    optimizer.step()
                    optimizer.zero_grad()

                batch_point_counts.append(vel_batch.shape[0] * y_batch.shape[1])
                batch_loss.append(loss.item())
                batch_u_loss.append(loss_u.item())
                batch_f_loss.append(loss_f.item())
                batch_r_loss.append(loss_r.item() if isinstance(loss_r, torch.Tensor) else loss_r)
                batch_env_loss.append(loss_env.item())
                batch_continuous_pde_loss.append(loss_continuous_pde.item())

                del loss, loss_f, loss_u, loss_r, loss_env, loss_continuous_pde
                del y_batch, Delta_U, Delta_U_cont, continuous_inputs, forward_result

        # Every group, including a short tail, has already synchronized and stepped.

        # ---- 跨进程平均损失 ----
        avg_loss = point_mean(batch_loss, batch_point_counts) if batch_loss else 0
        avg_u_loss = point_mean(batch_u_loss, batch_point_counts) if batch_u_loss else 0
        avg_f_loss = point_mean(batch_f_loss, batch_point_counts) if batch_f_loss else 0
        avg_r_loss = point_mean(batch_r_loss, batch_point_counts) if batch_r_loss else 0
        avg_env_loss = point_mean(batch_env_loss, batch_point_counts) if batch_env_loss else 0
        avg_continuous_pde_loss = (
            point_mean(batch_continuous_pde_loss, batch_point_counts) if batch_continuous_pde_loss else 0
        )

        loss_tensor = torch.tensor(
            [avg_loss, avg_u_loss, avg_f_loss, avg_r_loss, avg_continuous_pde_loss],
            device=device,
        )
        loss_tensor = reduce_tensor(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss, avg_u_loss, avg_f_loss, avg_r_loss, avg_continuous_pde_loss = loss_tensor.cpu().numpy()

        if first_flag:
            data_norm_coe = avg_u_loss if avg_u_loss > 0 else 1.0
            pde_norm_coe = avg_f_loss if avg_f_loss > 0 else 1.0
            env_norm_coe = avg_env_loss if avg_env_loss > 0 else 1.0
            loss_log.append(a + b)
            loss_data_log.append(1.)
            loss_pde_log.append(1.)
            loss_env_log.append(1.)
            loss_reg_log.append(avg_r_loss)
            loss_continuous_pde_log.append(0.0)
            first_flag = False
        else:
            loss_log.append(avg_loss)
            loss_data_log.append(avg_u_loss)
            loss_pde_log.append(avg_f_loss)
            loss_env_log.append(avg_env_loss)
            loss_reg_log.append(avg_r_loss)
            loss_continuous_pde_log.append(avg_continuous_pde_loss)

        if is_main_process(rank):
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Total': f"{avg_loss:.4e}",
                'PDE': f"{loss_pde_log[-1]:.4e}",
                'Data': f"{loss_data_log[-1]:.4e}",
                'ContPDE': f"{loss_continuous_pde_log[-1]:.2e}",
                'FiLM': f"{loss_reg_log[-1]:.2e}",
                'LR': f"{current_lr:.2e}",
                'GPU': f"{world_size}"
            })

        if use_warmup and warmup_scheduler is not None and i <= stage_warmup:
            warmup_scheduler.step(i)
        elif scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_loss)

        # ---- 验证 (仅主进程) ----
        if i % args.validate_every == 0 and is_main_process(rank):
            model.eval()
            vb_u_loss, vb_f_loss = [], []
            batch_point_counts = []

            for batch_data in dataloader['valid']:
                vel_batch, UU0_batch, labels_batch, freq_batch, source_coord_batch = \
                    unpack_conditioned_batch(batch_data)
                if freq_batch is not None:
                    freq_batch = freq_batch.to(device)
                if source_coord_batch is not None:
                    source_coord_batch = source_coord_batch.to(device)
                vel_batch = vel_batch.to(device)
                UU0_batch = UU0_batch.to(device)
                labels_batch = labels_batch.to(device)

                for batch in dataloader['valid_y']:
                    y_batch = batch[0].to(device)
                    y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    _, loss_f_valid, loss_u_valid, _, _, _ = model.module.loss(
                        vel_batch, y_batch, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                        freq_batch=freq_batch, source_coord_batch=source_coord_batch,
                    )
                    batch_point_counts.append(vel_batch.shape[0] * y_batch.shape[1])
                    vb_u_loss.append(loss_u_valid.item())
                    vb_f_loss.append(loss_f_valid.item())

                    del loss_f_valid, loss_u_valid, y_batch

            valid_u_loss.append(point_mean(vb_u_loss, batch_point_counts) if vb_u_loss else 0.0)
            valid_f_loss.append(point_mean(vb_f_loss, batch_point_counts) if vb_f_loss else 1.0)

            for marmousi_evaluator in marmousi_evaluators:
                evaluate_training_unseen_frequency(
                    marmousi_evaluator, model.module, i
                )

        # ---- 可视化 (仅主进程) ----
        if i % args.save_fig_every == 0 and is_main_process(rank):
            vel_pred = plot_data["vel_pred"]
            UU0_pred = plot_data["UU0_pred"]
            labels_pred = plot_data["labels_pred"]
            vel_test = plot_data["vel_test"]
            UU0_test = plot_data["UU0_test"]
            labels_test = plot_data["labels_test"]
            freq_pred = plot_data["freq_valid"][0:1] if has_freq else None
            freq_test = plot_data["freq_train"][0:1] if has_freq else None
            source_coord_test = plot_data.get("source_coord_train")
            source_coord_pred = plot_data.get("source_coord_valid")

            plot_loss(i, save_doc, loss_log, loss_data_log, loss_pde_log, valid_u_loss, valid_f_loss,
                      suffix=f'_stage{stage_idx}')

            test_plot(args, model.module, fno, i, dataloader["pred"],
                      vel_pred, UU0_pred, labels_pred, f'valid_stage{stage_idx}',
                      if_fine_tune=False, freq=freq_pred, source_coord=source_coord_pred)
            test_plot(args, model.module, fno, i, dataloader["test"],
                      vel_test, UU0_test, labels_test, f'train_stage{stage_idx}',
                      if_fine_tune=False, freq=freq_test, source_coord=source_coord_test)
            plot_sinlge(
                model.module, args, 6, vel_test, UU0_test, labels_test,
                freq=freq_test, source_coord=source_coord_test,
            )

        # ---- 模型保存 (仅主进程) ----
        if i % args.save_model_every == 0 and is_main_process(rank):
            if isinstance(pbar, tqdm):
                pbar.write(f'>>> Stage {stage_idx} Epoch {i} | Total Loss {loss_log[-1]:.4e} | PDE Loss {loss_pde_log[-1]:.4e}')

            checkpoint = {
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'stage': stage_idx,
                'epoch_in_stage': i,
            }
            torch.save(checkpoint, os.path.join(
                save_doc, f'{args.filename}_stage{stage_idx}_{i}epoch_weights_{args.nz}.pth'))
            np.save(os.path.join(save_doc, f'loss_log_stage{stage_idx}.npy'), loss_log)
            np.save(os.path.join(save_doc, f'loss_data_log_stage{stage_idx}.npy'), loss_data_log)
            np.save(os.path.join(save_doc, f'loss_pde_log_stage{stage_idx}.npy'), loss_pde_log)
            np.save(os.path.join(save_doc, f'loss_continuous_pde_log_stage{stage_idx}.npy'), loss_continuous_pde_log)
            np.save(os.path.join(save_doc, f'loss_film_smooth_log_stage{stage_idx}.npy'), loss_reg_log)
            np.save(os.path.join(save_doc, f'loss_env_log_stage{stage_idx}.npy'), loss_env_log)

        dist.barrier(device_ids=[device.index])

    # ---- 阶段结束：保存最终权重 ----
    if is_main_process(rank):
        final_path = os.path.join(save_doc, f'{args.filename}_stage{stage_idx}_final_weights_{args.nz}.pth')
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'stage': stage_idx,
        }, final_path)
        print(f'✅ Stage {stage_idx} [{stage_name}] 完成，权重已保存: {final_path}')

        np.save(os.path.join(save_doc, f'loss_log_stage{stage_idx}.npy'), loss_log)
        np.save(os.path.join(save_doc, f'loss_data_log_stage{stage_idx}.npy'), loss_data_log)
        np.save(os.path.join(save_doc, f'loss_pde_log_stage{stage_idx}.npy'), loss_pde_log)
        np.save(os.path.join(save_doc, f'loss_continuous_pde_log_stage{stage_idx}.npy'), loss_continuous_pde_log)
        np.save(os.path.join(save_doc, f'loss_film_smooth_log_stage{stage_idx}.npy'), loss_reg_log)
        np.save(os.path.join(save_doc, f'loss_env_log_stage{stage_idx}.npy'), loss_env_log)

    return model


def _train_stage_ddp(args, model, fno, device, rank, world_size,
                     stage_idx, stage_config, save_doc,
                     base_vel_filename, base_bg_filename, base_wf_filename, base_freq_filename):
    """DDP 单阶段训练：数据加载、replay 合并、优化器初始化、训练循环"""
    stage_name = stage_config['name']
    freq_range = stage_config['freq_range']
    stage_niter = stage_config.get('NIter', args.NIter)
    stage_lr = stage_config.get('lr', args.lr)
    stage_warmup = stage_config.get('warmup_epochs', args.warmup_epochs)
    a = stage_config.get('a', args.a)
    b = stage_config.get('b', args.b)
    c = stage_config.get('c', args.c)
    d = getattr(args, 'd', 0.1)

    # ---- 1. 替换文件名 ----
    base_freq_tag = 'freq3to20'
    stage_freq_tag = f'freq{freq_range}'

    args.vel_filename = base_vel_filename.replace(base_freq_tag, stage_freq_tag)
    args.backgroundfield_filename = base_bg_filename.replace(base_freq_tag, stage_freq_tag)
    args.wavefield_filename = base_wf_filename.replace(base_freq_tag, stage_freq_tag)
    args.freq_filename = base_freq_filename

    original_load_path = args.load_path
    if 'data_dir' in stage_config:
        args.load_path = stage_config['data_dir']
        args.vel_filename = os.path.basename(args.vel_filename)
        args.backgroundfield_filename = os.path.basename(args.backgroundfield_filename)
        args.wavefield_filename = os.path.basename(args.wavefield_filename)
        args.freq_filename = os.path.basename(args.freq_filename)
    current_stage_load_path = args.load_path

    if is_main_process(rank):
        print(f'\n[*] Stage {stage_idx} [{stage_name}] 数据文件:')
        print(f'    load_path: {args.load_path}')
        print(f'    vel:   {args.vel_filename}')
        print(f'    bg:    {args.backgroundfield_filename}')
        print(f'    wf:    {args.wavefield_filename}')
        print(f'    freq:  {args.freq_filename}')

    # ---- 2. 加载上一阶段权重 ----
    if stage_idx > 0:
        prev_path = os.path.join(save_doc, f'{args.filename}_stage{stage_idx - 1}_final_weights_{args.nz}.pth')
        if is_main_process(rank):
            if os.path.exists(prev_path):
                print(f'[*] 加载上一阶段权重: {prev_path}')
                ckpt = torch.load(prev_path, map_location=device)
                model.module.load_state_dict(ckpt['model_state_dict'])
            else:
                print(f'⚠️ 未找到上一阶段权重: {prev_path}，将使用当前模型权重继续')
        dist.barrier()
        _broadcast_module_state(model.module)
    else:
        if is_main_process(rank):
            print(f'[*] Stage 0: 权重已在 _train_worker_staged 中初始化')

    # ---- 3. 加载数据 ----
    # 固定 seed 确保所有 rank 得到相同的 train/valid 划分
    np.random.seed(1)
    torch.manual_seed(0)
    dataloader, plot_data = prepare_training_dataloaders(args, device)

    # ---- 3.5 Replay ----
    replay_stages_list = stage_config.get('replay_stages', [])
    if replay_stages_list and stage_idx > 0:
        cur_vel_fn = args.vel_filename
        cur_bg_fn = args.backgroundfield_filename
        cur_wf_fn = args.wavefield_filename
        cur_freq_fn = args.freq_filename

        train_ds = dataloader['train'].dataset
        train_tensors = train_ds.tensors
        has_freq_replay, has_source_replay = dataset_condition_flags(train_ds)
        combined_vel, combined_UU0, combined_labels, combined_freq, combined_source = \
            unpack_conditioned_batch(train_tensors)

        for replay_idx in replay_stages_list:
            replay_config = args.stages[replay_idx]
            replay_freq_tag = f'freq{replay_config["freq_range"]}'

            args.vel_filename = base_vel_filename.replace(base_freq_tag, replay_freq_tag)
            args.backgroundfield_filename = base_bg_filename.replace(base_freq_tag, replay_freq_tag)
            args.wavefield_filename = base_wf_filename.replace(base_freq_tag, replay_freq_tag)
            args.freq_filename = base_freq_filename

            if 'data_dir' in replay_config:
                args.load_path = replay_config['data_dir']
                args.vel_filename = os.path.basename(args.vel_filename)
                args.backgroundfield_filename = os.path.basename(args.backgroundfield_filename)
                args.wavefield_filename = os.path.basename(args.wavefield_filename)
                args.freq_filename = os.path.basename(args.freq_filename)

            if is_main_process(rank):
                print(f'    [Replay] 加载 Stage {replay_idx} [{replay_config["name"]}] 数据: {args.load_path}/{args.vel_filename}')

            # 固定 seed 确保所有 rank 的 replay 数据划分一致
            np.random.seed(1)
            torch.manual_seed(0)
            replay_dl, _ = prepare_training_dataloaders(args, device)
            replay_ds = replay_dl['train'].dataset
            replay_tensors = replay_ds.tensors

            replay_ratio = replay_config.get('replay_ratio', 1.0)
            n_replay = replay_tensors[0].shape[0]
            if replay_ratio < 1.0:
                torch.manual_seed(42 + replay_idx)
                n_sample = max(1, int(n_replay * replay_ratio))
                perm = torch.randperm(n_replay)[:n_sample]
                replay_tensors = tuple(t[perm] for t in replay_tensors)
                if is_main_process(rank):
                    print(f'    [Replay] Stage {replay_idx}: {n_sample}/{n_replay} 样本 (ratio={replay_ratio})')

            replay_vel, replay_UU0, replay_labels, replay_freq, replay_source = \
                unpack_conditioned_batch(replay_tensors)
            if (replay_freq is None) != (combined_freq is None) or \
                    (replay_source is None) != (combined_source is None):
                raise ValueError('Replay 阶段的频率/震源条件与当前阶段不一致')
            combined_vel = torch.cat([combined_vel, replay_vel], dim=0)
            combined_UU0 = torch.cat([combined_UU0, replay_UU0], dim=0)
            combined_labels = torch.cat([combined_labels, replay_labels], dim=0)
            if combined_freq is not None:
                combined_freq = torch.cat([combined_freq, replay_freq], dim=0)
            if combined_source is not None:
                combined_source = torch.cat([combined_source, replay_source], dim=0)

        # 恢复文件名
        args.vel_filename = cur_vel_fn
        args.backgroundfield_filename = cur_bg_fn
        args.wavefield_filename = cur_wf_fn
        args.freq_filename = cur_freq_fn
        args.load_path = current_stage_load_path

        pin_mem = device.type == 'cuda'

        new_train_ds = make_conditioned_dataset(
            combined_vel, combined_UU0, combined_labels,
            combined_freq, combined_source,
        )

        replay_sampler = DistributedSampler(
            new_train_ds, num_replicas=world_size, rank=rank, shuffle=True,
            seed=int(getattr(args, 'ddp_seed', 20260410)) + stage_idx * 1_000_000,
            drop_last=True,
        )
        dataloader['train'] = DataLoader(
            new_train_ds,
            batch_size=_ddp_batch_size_v(args, world_size), sampler=replay_sampler, drop_last=True,
            pin_memory=pin_mem, num_workers=4, prefetch_factor=2,
        )

        if is_main_process(rank):
            print(f'    [Replay] 训练集合并完成: {combined_vel.shape[0]} 样本 (含 replay)')

    else:
        train_sampler = DistributedSampler(
            dataloader['train'].dataset,
            num_replicas=world_size, rank=rank, shuffle=True,
            seed=int(getattr(args, 'ddp_seed', 20260410)) + stage_idx * 1_000_000,
            drop_last=True,
        )
        dataloader['train'] = DataLoader(
            dataloader['train'].dataset,
            batch_size=_ddp_batch_size_v(args, world_size),
            sampler=train_sampler,
            drop_last=True,
            num_workers=4, pin_memory=True, prefetch_factor=2
        )

    has_freq, has_source_coords = dataset_condition_flags(dataloader['train'].dataset)

    if is_main_process(rank):
        print(f"已启用 DistributedSampler (world_size={world_size})")

    # ---- 4. 初始化优化器与调度器 ----
    ddp_lr = _ddp_lr(args, stage_lr, world_size)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=ddp_lr, weight_decay=args.weight_decay
    )
    scheduler, warmup_scheduler, scheduler_type, use_warmup = _build_scheduler(
        args, optimizer, ddp_lr, warmup_epochs=stage_warmup
    )

    if is_main_process(rank):
        print(f"DDP per-rank batch_size_v: {_ddp_batch_size_v(args, world_size)} "
              f"(global≈{_ddp_batch_size_v(args, world_size) * world_size}, single={args.batch_size_v})")
        print(f"DDP lr: {ddp_lr:.3e} (scale_lr={getattr(args, 'ddp_scale_lr', False)})")
        print(f"Scheduler: {scheduler_type}" + (f" (warmup {stage_warmup} epochs)" if use_warmup else ""))

    # ---- 5. 运行训练循环 ----
    model = _run_stage_training_loop(
        args, model, fno, device, rank, world_size,
        dataloader, plot_data, has_freq,
        optimizer, scheduler, warmup_scheduler,
        stage_idx, stage_name, stage_niter, stage_warmup,
        a, b, c, d, save_doc,
    )

    # 恢复原始 load_path，防止跨阶段污染
    args.load_path = original_load_path

    return model, save_doc


def _train_worker_staged(rank, world_size, args, device_ids):
    """分布式课程学习训练工作进程，在单个 mp.spawn 生命周期内依次执行所有阶段"""
    device_id = int(device_ids[rank])
    setup_distributed(rank, world_size, device_id=device_id,
                      timeout_minutes=int(getattr(args, 'nccl_timeout_minutes', 10)))
    device = torch.device(f'cuda:{device_id}')
    args.device = device_id
    _validate_ddp_alignment_options(args)
    _seed_ddp_rank(int(getattr(args, 'ddp_seed', 20260410)), rank)

    # 创建模型（全局一次，跨阶段复用）
    model = Pi_DeepONet(args).to(device)

    if is_main_process(rank):
        model._init_weights()
        print(f"[Stage DDP] PI_DeepONet 模型总参数数量: {count_parameters(model)}")

    _broadcast_module_state(model)

    model = wrap_model_for_distributed(model, device_id)

    fno = None
    if args.use_fno_as_label:
        fno = FNO(args).to(device)
        if args.fno_weights_path:
            fno.load_state_dict(torch.load(args.fno_weights_path, map_location=device)['model_state_dict'])
            if is_main_process(rank):
                print(f"已加载 FNO 权重: {args.fno_weights_path}")
        fno.eval()

    save_doc = args.save_doc
    if is_main_process(rank):
        os.makedirs(save_doc, exist_ok=True)

    base_vel_filename = args.vel_filename
    base_bg_filename = args.backgroundfield_filename
    base_wf_filename = args.wavefield_filename
    base_freq_filename = args.freq_filename

    stages = args.stages

    for stage_idx, stage_config in enumerate(stages):
        if is_main_process(rank):
            print(f'\n{"=" * 60}')
            print(f'>>> [DDP] 开始 Stage {stage_idx}: {stage_config["name"]} '
                  f'[{stage_config["freq_min"]}-{stage_config["freq_max"]} Hz]')
            print(f'{"=" * 60}')

        model, save_doc = _train_stage_ddp(
            args, model, fno, device, rank, world_size,
            stage_idx, stage_config, save_doc,
            base_vel_filename, base_bg_filename, base_wf_filename, base_freq_filename,
        )

    if is_main_process(rank):
        print(f'\n{"=" * 60}')
        print(f'全部 {len(stages)} 个阶段训练完毕！')
        print(f'{"=" * 60}')

    cleanup_distributed()


def train_distributed_staged(args, device_ids=None):
    """单机多卡分布式 + 课程学习训练入口函数"""
    if _get_resume_spec(args) is not None:
        raise NotImplementedError(
            '当前 resume_checkpoint 仅支持普通 DDP 训练；'
            '分阶段课程训练需要额外记录 stage/replay 状态，已拒绝静默错误续训。'
        )
    if device_ids is None:
        device_ids = list(range(getattr(args, 'num_gpus', 1)))
    device_ids = [int(device_id) for device_id in device_ids]
    if not device_ids or len(set(device_ids)) != len(device_ids):
        raise ValueError(f'DDP device_ids 必须是非空且不重复的 GPU 编号列表: {device_ids}')
    world_size = len(device_ids)
    _validate_ddp_alignment_options(args)
    _ddp_batch_size_v(args, world_size)

    stages = getattr(args, 'stages', [])
    print("=" * 60)
    print(f"启动单机多卡分布式课程学习训练")
    print(f"GPU 数量: {world_size}; 物理 GPU: {device_ids}")
    print(f"训练阶段数: {len(stages)}")
    for si, s in enumerate(stages):
        print(f'  Stage {si}: {s["name"]} | freq [{s["freq_min"]}-{s["freq_max"]}] Hz | '
              f'{s.get("NIter", "?")} epochs | lr={s.get("lr", "?")}')
    print("=" * 60)

    mp.spawn(
        _train_worker_staged,
        args=(world_size, args, device_ids),
        nprocs=world_size,
        join=True
    )
