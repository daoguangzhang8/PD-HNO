import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from Labconfig import *
from model.utils import *
from model.utils import build_epoch_velocity_gradient_prob, sample_shared_y_ran_from_epoch_prob
from model.dataloader import *
from model.PI_DeepOnet import Pi_DeepONet
from model.FNO import FNO
from model.point_averaging import (
    coordinate_accumulation_batches, point_mean, weighted_microbatch_loss,
)
from model.plotting import *
from test import (
    evaluate_training_unseen_frequency,
    prepare_training_unseen_frequency_evaluator,
)


def _get_resume_spec(args):
    """返回单卡续训 checkpoint 及其对应的全局 epoch。"""
    if not getattr(args, 'if_load_model', False):
        return None

    checkpoint_path = os.path.abspath(str(getattr(args, 'resume_checkpoint', '')).strip())
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'单卡续训 checkpoint 不存在: {checkpoint_path}')
    resume_epoch = getattr(args, 'resume_epoch', None)
    if resume_epoch is None or int(resume_epoch) < 0:
        raise ValueError('单卡续训必须设置非负的 resume_epoch')
    return checkpoint_path, int(resume_epoch)


def _move_optimizer_state_to_device(optimizer, device):
    """恢复 Adam 状态后确保所有张量位于目标 GPU。"""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device, non_blocking=True)


def _save_merged_resume_loss_logs(args, resume_spec, loss_logs):
    """合并原 checkpoint 所在实验与本次接替训练的完整 loss 曲线。"""
    names = (
        'loss_log', 'loss_data_log', 'loss_pde_log',
        'loss_continuous_pde_log', 'loss_film_smooth_log', 'loss_env_log',
    )
    if resume_spec is None:
        old_dir = None
    else:
        old_dir = os.path.dirname(resume_spec[0])

    for name, current_values in zip(names, loss_logs):
        current = np.asarray(current_values, dtype=np.float64)
        if old_dir is not None:
            old_path = os.path.join(old_dir, f'{name}.npy')
            if os.path.isfile(old_path):
                old = np.asarray(np.load(old_path), dtype=np.float64).reshape(-1)
                expected_start = int(getattr(args, 'resume_epoch', 0)) + 1
                if old.size != expected_start:
                    raise ValueError(
                        f'{old_path} 长度为 {old.size}，但续训起点要求 {expected_start}；'
                        '拒绝生成可能错位的合并曲线。'
                    )
                merged = np.concatenate([old, current])
            else:
                merged = current
        else:
            merged = current
        np.save(os.path.join(args.save_doc, f'merged_{name}.npy'), merged)

    merged_arrays = {
        name: np.load(os.path.join(args.save_doc, f'merged_{name}.npy'))
        for name in names
    }
    merged_length = len(merged_arrays['loss_log'])
    np.save(
        os.path.join(args.save_doc, 'merged_loss_epoch.npy'),
        np.arange(merged_length, dtype=np.int64),
    )

    # 生成一张覆盖完整 epoch 0..N 的基础 loss 曲线，后续可单独调整样式。
    epochs = np.arange(merged_length)
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(epochs, merged_arrays['loss_log'], label='Total')
    axes[0].plot(epochs, merged_arrays['loss_data_log'], label='Data')
    axes[0].plot(epochs, merged_arrays['loss_pde_log'], label='PDE')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Normalized loss')
    axes[0].set_title(f'Complete loss curve (epoch 0-{merged_length - 1})')
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, merged_arrays['loss_film_smooth_log'], label='FiLM smooth')
    axes[1].plot(epochs, merged_arrays['loss_continuous_pde_log'], label='Continuous PDE')
    axes[1].plot(epochs, merged_arrays['loss_env_log'], label='Envelope')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(
        os.path.join(args.save_doc, 'merged_loss_curve.png'),
        dpi=160,
        bbox_inches='tight',
    )
    plt.close(figure)

    # 同时提供三种主 loss 的独立完整曲线，便于后续单独调整绘图风格。
    individual_curves = (
        ('merged_loss_curve_total.png', 'Total loss', merged_arrays['loss_log']),
        ('merged_loss_curve_data.png', 'Data loss', merged_arrays['loss_data_log']),
        ('merged_loss_curve_pde.png', 'PDE loss', merged_arrays['loss_pde_log']),
    )
    for filename, title, values in individual_curves:
        single_figure, single_axis = plt.subplots(figsize=(12, 5))
        single_axis.plot(epochs, values)
        single_axis.set_yscale('log')
        single_axis.set_xlabel('Epoch')
        single_axis.set_ylabel('Normalized loss')
        single_axis.set_title(f'Complete {title} (epoch 0-{merged_length - 1})')
        single_axis.grid(True, alpha=0.25)
        single_figure.tight_layout()
        single_figure.savefig(
            os.path.join(args.save_doc, filename),
            dpi=160,
            bbox_inches='tight',
        )
        plt.close(single_figure)


def _train_stage(args, model, fno, device, stage_idx, stage_config,
                 base_vel_filename, base_bg_filename, base_wf_filename, base_freq_filename):
    """
    单阶段训练函数
    - stage_idx: 当前阶段编号 (0, 1, 2)
    - stage_config: dict, 包含 name/freq_range/NIter/lr 等
    - base_*_filename: 原始文件名模板，用于替换 freq 标签
    """
    stage_name = stage_config['name']
    freq_range = stage_config['freq_range']
    stage_niter = stage_config.get('NIter', args.NIter)
    stage_lr = stage_config.get('lr', args.lr)
    stage_warmup = stage_config.get('warmup_epochs', args.warmup_epochs)
    a = stage_config.get('a', args.a)
    b = stage_config.get('b', args.b)
    c = stage_config.get('c', args.c)
    d = getattr(args, 'd', 0.1)

    # ---- 1. 根据阶段替换数据文件名 ----
    # vel/bg/wf 文件名含 'freq3to20'，直接替换为 'freq{range}'
    # freq 文件名含 'freq_used'，替换为 'freq{range}_used'
    base_freq_tag = 'freq3to20'
    stage_freq_tag = f'freq{freq_range}'

    args.vel_filename = base_vel_filename.replace(base_freq_tag, stage_freq_tag)
    args.backgroundfield_filename = base_bg_filename.replace(base_freq_tag, stage_freq_tag)
    args.wavefield_filename = base_wf_filename.replace(base_freq_tag, stage_freq_tag)
    args.freq_filename = base_freq_filename  # freq 文件不含 stage 标签，保持原文件名

    # 如果 stage 配置了 data_dir，覆盖 load_path（课程学习独立数据路径）
    original_load_path = args.load_path
    if 'data_dir' in stage_config:
        args.load_path = stage_config['data_dir']
        # data_dir 直接指向数据文件所在目录，文件名不需要子目录前缀
        args.vel_filename = os.path.basename(args.vel_filename)
        args.backgroundfield_filename = os.path.basename(args.backgroundfield_filename)
        args.wavefield_filename = os.path.basename(args.wavefield_filename)
        args.freq_filename = os.path.basename(args.freq_filename)
    current_stage_load_path = args.load_path

    print(f'\n[*] Stage {stage_idx} [{stage_name}] 数据文件:')
    print(f'    load_path: {args.load_path}')
    print(f'    vel:   {args.vel_filename}')
    print(f'    bg:    {args.backgroundfield_filename}')
    print(f'    wf:    {args.wavefield_filename}')
    print(f'    freq:  {args.freq_filename}')

    # ---- 2. 课程学习：后续阶段加载前一阶段权重 ----
    save_doc = args.save_doc
    if stage_idx > 0:
        prev_path = os.path.join(save_doc, f'{args.filename}_stage{stage_idx - 1}_final_weights_{args.nz}.pth')
        if os.path.exists(prev_path):
            print(f'[*] 加载上一阶段权重: {prev_path}')
            ckpt = torch.load(prev_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            print(f'⚠️ 未找到上一阶段权重: {prev_path}，将使用当前模型权重继续')
    else:
        model._init_weights()
        print(f'[*] Stage 0: 从头初始化模型权重')

    # ---- 3. 加载数据 ----
    dataloader, plot_data = prepare_training_dataloaders(args, device)

    marmousi_evaluators = []
    if getattr(args, 'enable_marmousi_eval', False):
        marmousi_evaluators.append(prepare_training_unseen_frequency_evaluator(
            args, device, stage_tag=f'stage_{stage_idx}'
        ))
    if getattr(args, 'enable_marmousi_unseen_source_eval', False):
        marmousi_evaluators.append(prepare_training_unseen_frequency_evaluator(
            args, device, stage_tag=f'stage_{stage_idx}',
            frequencies=args.marmousi_unseen_source_eval_frequencies,
            sources=args.marmousi_unseen_source_eval_sources,
            data_dir=args.marmousi_unseen_source_eval_data_dir,
            output_name=args.marmousi_unseen_source_eval_output_dir,
        ))

    # ---- 3.5 Replay 前序阶段数据（防遗忘） ----
    replay_stages_list = stage_config.get('replay_stages', [])
    if replay_stages_list and stage_idx > 0:
        # 保存当前阶段的文件名（replay 后需恢复）
        cur_vel_fn = args.vel_filename
        cur_bg_fn = args.backgroundfield_filename
        cur_wf_fn = args.wavefield_filename
        cur_freq_fn = args.freq_filename

        # 从当前 DataLoader 提取训练集 Tensor
        train_ds = dataloader['train'].dataset
        train_tensors = train_ds.tensors
        has_freq_replay, has_source_replay = dataset_condition_flags(train_ds)
        combined_vel, combined_UU0, combined_labels, combined_freq, combined_source = \
            unpack_conditioned_batch(train_tensors)

        for replay_idx in replay_stages_list:
            replay_config = args.stages[replay_idx]
            replay_freq_tag = f'freq{replay_config["freq_range"]}'

            # 替换文件名为 replay 阶段
            args.vel_filename = base_vel_filename.replace(base_freq_tag, replay_freq_tag)
            args.backgroundfield_filename = base_bg_filename.replace(base_freq_tag, replay_freq_tag)
            args.wavefield_filename = base_wf_filename.replace(base_freq_tag, replay_freq_tag)
            args.freq_filename = base_freq_filename  # freq 文件不含 stage 标签，保持原文件名

            # 使用 replay 阶段的 data_dir
            if 'data_dir' in replay_config:
                args.load_path = replay_config['data_dir']
                args.vel_filename = os.path.basename(args.vel_filename)
                args.backgroundfield_filename = os.path.basename(args.backgroundfield_filename)
                args.wavefield_filename = os.path.basename(args.wavefield_filename)
                args.freq_filename = os.path.basename(args.freq_filename)

            print(f'    [Replay] 加载 Stage {replay_idx} [{replay_config["name"]}] 数据: {args.load_path}/{args.vel_filename}')

            replay_dl, _ = prepare_training_dataloaders(args, device)
            replay_ds = replay_dl['train'].dataset
            replay_tensors = replay_ds.tensors

            # 按 replay_ratio 随机采样子集
            replay_ratio = replay_config.get('replay_ratio', 1.0)
            n_replay = replay_tensors[0].shape[0]
            if replay_ratio < 1.0:
                n_sample = max(1, int(n_replay * replay_ratio))
                perm = torch.randperm(n_replay)[:n_sample]
                replay_tensors = tuple(t[perm] for t in replay_tensors)
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

        # 恢复当前阶段的文件名和 load_path
        args.vel_filename = cur_vel_fn
        args.backgroundfield_filename = cur_bg_fn
        args.wavefield_filename = cur_wf_fn
        args.freq_filename = cur_freq_fn
        args.load_path = current_stage_load_path
        

        # 重建训练 DataLoader
        pin_mem = device.type == 'cuda'
        num_workers = 4
        prefetch_factor = 2

        new_train_ds = make_conditioned_dataset(
            combined_vel, combined_UU0, combined_labels,
            combined_freq, combined_source,
        )

        dataloader['train'] = DataLoader(
            new_train_ds,
            batch_size=args.batch_size_v, shuffle=True, drop_last=True,
            pin_memory=pin_mem, num_workers=num_workers, prefetch_factor=prefetch_factor,
        )

        print(f'    [Replay] 训练集合并完成: {combined_vel.shape[0]} 样本 (含 replay)')

    # ---- 4. 初始化优化器与调度器 ----
    optimizer = optim.Adam(model.parameters(), lr=stage_lr, weight_decay=args.weight_decay)

    scheduler_type = getattr(args, 'scheduler_type', 'plateau')
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=getattr(args, 'cosine_T_0', 1000),
            T_mult=getattr(args, 'cosine_T_mult', 2),
            eta_min=getattr(args, 'cosine_eta_min', 1e-6),
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.factor, patience=args.patience, min_lr=args.min_lr
        )

    use_warmup = getattr(args, 'use_warmup', False) and stage_warmup > 0
    warmup_scheduler = None
    if use_warmup:
        warmup_scheduler = WarmupScheduler(
            optimizer, warmup_epochs=stage_warmup, base_lr=stage_lr,
            warmup_start_lr=stage_lr / 10., warmup_strategy="linear"
        )

    # ---- 5. 训练状态 ----
    loss_log, loss_pde_log, loss_data_log, loss_reg_log, loss_env_log = [], [], [], [], []
    loss_continuous_pde_log = []
    valid_u_loss, valid_f_loss = [], []
    first_flag = True
    pde_norm_coe, data_norm_coe, env_norm_coe = 1., 1., 1.

    # ---- 6. Sobol 引擎 ----
    use_sobol = getattr(args, 'sampling_strategy', 'original') == 'sobol'
    if use_sobol:
        sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        valid_sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        sobol_scale = torch.tensor([args.nz * args.dh, args.nx * args.dh], dtype=torch.float32, device=device)
        sobol_pts = getattr(args, 'sobol_points_per_epoch', 800)
        valid_sobol_pts = getattr(args, 'valid_sobol_points', 800)
        print(f"Sobol 模式: 每 epoch {sobol_pts} 点, 验证 {valid_sobol_pts} 点")

    # ---- 7. 主训练循环 ----
    optimizer.zero_grad()
    step_counter = 0
    pbar = tqdm(range(stage_niter), desc=f"Stage {stage_idx} [{stage_name}]", dynamic_ncols=True)

    # epoch-level 共享 y_ran 采样状态
    epoch_prob = None
    epoch_score = None

    for i in pbar:
        if args.if_adjust and i > args.adjust_from and (i - args.adjust_from) % args.adjust_every == 0:
            decay_times = i // args.adjust_every
            a = max(a * (args.adjust_speed ** (-decay_times)), 2e-1)
            b, c = 1, 0

        model.train()
        batch_loss, batch_u_loss, batch_f_loss, batch_r_loss, batch_env_loss = [], [], [], [], []
        batch_continuous_pde_loss = []
        batch_point_counts, batch_pde_point_counts = [], []
        continuous_pde_active = (
            bool(getattr(args, 'enable_continuous_frequency_pde', False))
            and i >= int(getattr(args, 'continuous_pde_start_epoch', 1))
        )

        if use_sobol:
            y_sobol_base = sobol_engine.draw(sobol_pts).to(device)
            y_sobol_base = y_sobol_base * sobol_scale

        # y_ran: 每 epoch 生成一次（epoch shared 路径）
        y_ran_epoch_shared = None
        if not use_sobol and getattr(args, 'use_y_ran', False) and getattr(args, 'use_epoch_shared_y_ran', False):
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

        has_freq, has_source_coords = dataset_condition_flags(dataloader['train'].dataset)
        for batch_data in dataloader['train']:
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

            if use_sobol:
                y_sobol = y_sobol_base.unsqueeze(0).expand(vel_batch.shape[0], -1, -1).clone()
                y_sobol.requires_grad_(True)

                loss, loss_f, loss_u, loss_r, loss_env, loss_continuous_pde = model.loss(
                    vel_batch, y_sobol, UU0_batch, labels_batch,
                    a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch,
                    y_ran=None, use_continuous_pde=continuous_pde_active,
                    source_coord_batch=source_coord_batch,
                )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                batch_point_counts.append(vel_batch.shape[0] * y_sobol.shape[1])
                batch_pde_point_counts.append(batch_point_counts[-1])
                batch_loss.append(loss.item())
                batch_u_loss.append(loss_u.item())
                batch_f_loss.append(loss_f.item())
                batch_r_loss.append(loss_r.item() if isinstance(loss_r, torch.Tensor) else loss_r)
                batch_env_loss.append(loss_env.item())
                batch_continuous_pde_loss.append(loss_continuous_pde.item())

                del loss, loss_f, loss_u, loss_r, loss_env, loss_continuous_pde, y_sobol

            else:
                # y_ran: 使用 epoch 预生成或 per-model 生成
                if y_ran_epoch_shared is not None:
                    y_ran = y_ran_epoch_shared.unsqueeze(0).expand(
                        vel_batch.shape[0], -1, -1
                    ).clone().requires_grad_(True)
                elif getattr(args, 'use_y_ran', False):
                    with torch.no_grad():
                        y_ran = model.generate_structure_aware_y_ran(vel_batch, num_pts=900)
                else:
                    y_ran = None

                extra_pde_points = 0 if y_ran is None else y_ran.shape[1]
                for batch, data_weight, pde_weight, group_end in coordinate_accumulation_batches(
                    dataloader['train_y'], args.accumulation_steps, extra_pde_points
                ):
                    y_batch = batch[0].to(device)
                    y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    loss, loss_f, loss_u, loss_r, loss_env, loss_continuous_pde = model.loss(
                        vel_batch, y_batch, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch,
                        y_ran=y_ran, use_continuous_pde=continuous_pde_active,
                        source_coord_batch=source_coord_batch,
                    )

                    backward_loss = weighted_microbatch_loss(
                        loss, loss_f, b, data_weight, pde_weight
                    )
                    backward_loss.backward()
                    del backward_loss

                    step_counter += 1
                    if group_end:
                        optimizer.step()
                        optimizer.zero_grad()


                    batch_point_counts.append(vel_batch.shape[0] * y_batch.shape[1])
                    batch_pde_point_counts.append(
                        vel_batch.shape[0] * (y_batch.shape[1] + extra_pde_points)
                    )
                    batch_loss.append(loss.item())
                    batch_u_loss.append(loss_u.item())
                    batch_f_loss.append(loss_f.item())
                    batch_r_loss.append(loss_r.item() if isinstance(loss_r, torch.Tensor) else loss_r)
                    batch_env_loss.append(loss_env.item())
                    batch_continuous_pde_loss.append(loss_continuous_pde.item())

                    del loss, loss_f, loss_u, loss_r, loss_env, loss_continuous_pde, y_batch

        # ---- 记录损失 ----
        avg_loss = point_mean(batch_loss, batch_point_counts) if batch_loss else 0
        if batch_loss:
            # Separate point sets when y_ran contributes only to the PDE term.
            avg_loss += b * (
                point_mean(batch_f_loss, batch_pde_point_counts)
                - point_mean(batch_f_loss, batch_point_counts)
            )

        if first_flag:
            data_norm_coe = point_mean(batch_u_loss, batch_point_counts) if batch_u_loss else 1.0
            pde_norm_coe = point_mean(batch_f_loss, batch_pde_point_counts) if batch_f_loss else 1.0
            env_norm_coe = point_mean(batch_env_loss, batch_point_counts) if batch_env_loss else 1.0
            loss_log.append(a + b)
            loss_data_log.append(1.)
            loss_pde_log.append(1.)
            loss_env_log.append(1.)
            loss_reg_log.append(point_mean(batch_r_loss, batch_point_counts) if batch_r_loss else 0)
            loss_continuous_pde_log.append(0.0)
            first_flag = False
        else:
            loss_log.append(avg_loss)
            loss_data_log.append(point_mean(batch_u_loss, batch_point_counts) if batch_u_loss else 0)
            loss_pde_log.append(point_mean(batch_f_loss, batch_pde_point_counts) if batch_f_loss else 0)
            loss_env_log.append(point_mean(batch_env_loss, batch_point_counts) if batch_env_loss else 0)
            loss_reg_log.append(point_mean(batch_r_loss, batch_point_counts) if batch_r_loss else 0)
            loss_continuous_pde_log.append(
                point_mean(batch_continuous_pde_loss, batch_point_counts) if batch_continuous_pde_loss else 0
            )

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Total': f"{avg_loss:.4e}",
            'PDE': f"{loss_pde_log[-1]:.4e}",
            'Data': f"{loss_data_log[-1]:.4e}",
            'ContPDE': f"{loss_continuous_pde_log[-1]:.2e}",
            'FiLM': f"{loss_reg_log[-1]:.2e}",
            'LR': f"{current_lr:.2e}"
        })

        if use_warmup and warmup_scheduler is not None and i <= stage_warmup:
            warmup_scheduler.step(i)
        elif scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_loss)

        # ---- 验证 ----
        if i % args.validate_every == 0:
            model.eval()
            batch_u_loss, batch_f_loss = [], []
            batch_point_counts, batch_pde_point_counts = [], []

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

                if use_sobol:
                    y_valid = valid_sobol_engine.draw(valid_sobol_pts).to(device)
                    y_valid = y_valid * sobol_scale
                    y_valid = y_valid.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    _, loss_f_valid, loss_u_valid, _, _, _ = model.loss(
                        vel_batch, y_valid, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                        freq_batch=freq_batch, source_coord_batch=source_coord_batch,
                    )
                    batch_point_counts.append(vel_batch.shape[0] * y_valid.shape[1])
                    batch_pde_point_counts.append(batch_point_counts[-1])
                    batch_u_loss.append(loss_u_valid.item())
                    batch_f_loss.append(loss_f_valid.item())

                    del loss_f_valid, loss_u_valid, y_valid
                else:
                    for batch in dataloader['valid_y']:
                        y_batch = batch[0].to(device)
                        y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                        _, loss_f_valid, loss_u_valid, _, _, _ = model.loss(
                            vel_batch, y_batch, UU0_batch, labels_batch,
                            a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                            freq_batch=freq_batch, source_coord_batch=source_coord_batch,
                        )
                        batch_point_counts.append(vel_batch.shape[0] * y_batch.shape[1])
                        batch_pde_point_counts.append(batch_point_counts[-1])
                        batch_u_loss.append(loss_u_valid.item())
                        batch_f_loss.append(loss_f_valid.item())

                        # .item() does not release the tensors' higher-order graph.
                        del loss_f_valid, loss_u_valid, y_batch

            valid_u_loss.append(point_mean(batch_u_loss, batch_point_counts) if batch_u_loss else 0.0)
            valid_f_loss.append(point_mean(batch_f_loss, batch_pde_point_counts) if batch_f_loss else 1.0)

            for marmousi_evaluator in marmousi_evaluators:
                evaluate_training_unseen_frequency(
                    marmousi_evaluator, model, i
                )

        # ---- 可视化 ----
        if i % args.save_fig_every == 0:
            vel_pred = plot_data["vel_pred"]
            UU0_pred = plot_data["UU0_pred"]
            labels_pred = plot_data["labels_pred"]
            vel_test = plot_data["vel_test"]
            UU0_test = plot_data["UU0_test"]
            labels_test = plot_data["labels_test"]
            has_freq_plot = plot_data.get("has_freq", False)
            freq_pred = plot_data["freq_valid"] if has_freq_plot else None
            freq_test = plot_data["freq_train"] if has_freq_plot else None
            source_coord_test = plot_data.get("source_coord_train")
            source_coord_pred = plot_data.get("source_coord_valid")

            plot_loss(i, save_doc, loss_log, loss_data_log, loss_pde_log, valid_u_loss, valid_f_loss,
                      suffix=f'_stage{stage_idx}')

            test_plot(args, model, fno, i, dataloader["pred"], vel_pred, UU0_pred, labels_pred,
                      f'valid_stage{stage_idx}', if_fine_tune=False, freq=freq_pred,
                      source_coord=source_coord_pred)
            test_plot(args, model, fno, i, dataloader["test"], vel_test, UU0_test, labels_test,
                      f'train_stage{stage_idx}', if_fine_tune=False, freq=freq_test,
                      source_coord=source_coord_test)
            plot_sinlge(
                model, args, 6, vel_test, UU0_test, labels_test,
                freq=freq_test, source_coord=source_coord_test,
            )

        # ---- 模型保存 ----
        if i % args.save_model_every == 0:
            pbar.write(f'>>> Stage {stage_idx} Epoch {i} | Total Loss {loss_log[-1]:.4e} | PDE Loss {loss_pde_log[-1]:.4e}')

            checkpoint = {
                'model_state_dict': model.state_dict(),
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

    # ---- 8. 阶段结束：保存最终权重 ----
    final_path = os.path.join(save_doc, f'{args.filename}_stage{stage_idx}_final_weights_{args.nz}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
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


def train_staged(args, device):
    """三阶段课程学习训练"""
    model = Pi_DeepONet(args).to(device)
    print(f"PI_DeepONet 模型总参数数量：{count_parameters(model)}")

    fno = None
    if args.use_fno_as_label:
        fno = FNO(args).to(device)
        if args.fno_weights_path:
            fno.load_state_dict(torch.load(args.fno_weights_path, map_location=device)['model_state_dict'])
            print(f"已加载 FNO 权重: {args.fno_weights_path}")
        fno.eval()

    save_doc = args.save_doc
    os.makedirs(save_doc, exist_ok=True)

    # 保存基础文件名模板（供各阶段替换 freq 标签用）
    base_vel_filename = args.vel_filename
    base_bg_filename = args.backgroundfield_filename
    base_wf_filename = args.wavefield_filename
    base_freq_filename = args.freq_filename

    # 检查外部验证集配置
    if hasattr(args, 'ext_val_datasets') and args.ext_val_datasets:
        print(f'⚠️ staged_training 模式下暂不支持外部验证集 (ext_val_datasets)，已忽略')

    stages = args.stages
    print(f'\n{"=" * 60}')
    print(f'三阶段渐进训练计划：共 {len(stages)} 个阶段')
    for si, s in enumerate(stages):
        print(f'  Stage {si}: {s["name"]} | freq [{s["freq_min"]}-{s["freq_max"]}] Hz | '
              f'{s.get("NIter", "?")} epochs | lr={s.get("lr", "?")}')
    print(f'{"=" * 60}\n')

    for stage_idx, stage_config in enumerate(stages):
        print(f'\n{"=" * 60}')
        print(f'>>> 开始 Stage {stage_idx}: {stage_config["name"]} '
              f'[{stage_config["freq_min"]}-{stage_config["freq_max"]} Hz]')
        print(f'{"=" * 60}')

        model = _train_stage(
            args, model, fno, device, stage_idx, stage_config,
            base_vel_filename, base_bg_filename, base_wf_filename, base_freq_filename
        )

    print(f'\n{"=" * 60}')
    print(f'全部 {len(stages)} 个阶段训练完毕！')
    print(f'{"=" * 60}')


def train_single(args, device):
    """原始单阶段训练（原有 train() 逻辑完整保留）"""
    resume_spec = _get_resume_spec(args)
    start_epoch = 0
    if resume_spec is not None:
        _, resume_epoch = resume_spec
        start_epoch = resume_epoch + 1
        if start_epoch >= int(args.NIter):
            raise ValueError(
                f'续训起点为 epoch {start_epoch}，但 NIter={args.NIter}；'
                'NIter 必须是大于 resume_epoch 的总目标 epoch。'
            )
        if os.path.abspath(args.save_doc) == os.path.dirname(resume_spec[0]):
            raise ValueError('单卡续训输出目录不能与初始 checkpoint 所在目录相同')

    os.makedirs(args.save_doc, exist_ok=True)

    dataloader, plot_data = prepare_training_dataloaders(args, device)
    print(
        '[Train config] '
        f'batch_size_v={args.batch_size_v}, batch_size_y={args.batch_size}, '
        f'accumulation_steps={args.accumulation_steps}, '
        f'train_batches={len(dataloader["train"])}, '
        f'coordinate_batches={len(dataloader["train_y"])}',
        flush=True,
    )

    # 加载外部验证集
    ext_val_sets = {}
    if hasattr(args, 'ext_val_datasets'):
        for name, config in args.ext_val_datasets.items():
            loader, p_data = prepare_external_val_dataset(
                args,
                prefix=config['prefix'],
                loc_target=config['loc_target'],
                y_pred_grid=plot_data["y_pred"]
            )
            ext_val_sets[name] = {"loader": loader, "plot_data": p_data}

    model = Pi_DeepONet(args).to(device)
    if resume_spec is None:
        model._init_weights()
        print('从头初始化模型权重')
    else:
        checkpoint_path, resume_epoch = resume_spec
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' not in checkpoint:
            raise KeyError(f'checkpoint 缺少 model_state_dict: {checkpoint_path}')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f'从 checkpoint 恢复模型: {checkpoint_path} (全局 epoch {resume_epoch})')
    print(f"PI_DeepONet 模型总参数数量：{count_parameters(model)}")

    marmousi_evaluators = []
    if getattr(args, 'enable_marmousi_eval', False):
        marmousi_evaluators.append(prepare_training_unseen_frequency_evaluator(
            args, device
        ))
    if getattr(args, 'enable_marmousi_unseen_source_eval', False):
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
            print(f"已加载 FNO 权重: {args.fno_weights_path}")
        fno.eval()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler 选择
    scheduler_type = getattr(args, 'scheduler_type', 'plateau')
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=getattr(args, 'cosine_T_0', 1000),
            T_mult=getattr(args, 'cosine_T_mult', 2),
            eta_min=getattr(args, 'cosine_eta_min', 1e-6),
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.factor, patience=args.patience, min_lr=args.min_lr
        )

    use_warmup = getattr(args, 'use_warmup', False)
    warmup_scheduler = None
    if use_warmup:
        warmup_scheduler = WarmupScheduler(
            optimizer, warmup_epochs=args.warmup_epochs,
            base_lr=args.lr, warmup_start_lr=args.lr / 10.,
            warmup_strategy="linear"
        )

    resume_normalizers = None
    if resume_spec is not None:
        checkpoint_path, _ = resume_spec
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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

    print(f"Scheduler: {scheduler_type}" + (f" (warmup {args.warmup_epochs} epochs)" if use_warmup else ""))
    if resume_spec is not None:
        print(
            f'续训状态已恢复: optimizer={getattr(args, "resume_restore_optimizer", True)}, '
            f'scheduler={getattr(args, "resume_restore_scheduler", True)}, '
            f'从 epoch {start_epoch} 开始'
        )

    loss_log, loss_pde_log, loss_data_log, loss_reg_log, loss_env_log = [], [], [], [], []
    loss_continuous_pde_log = []
    valid_u_loss, valid_f_loss = [], []

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

    first_flag = resume_normalizers is None and not has_configured_normalizers
    pde_norm_coe, data_norm_coe, env_norm_coe = 1., 1., 1.
    if has_configured_normalizers:
        data_norm_coe = configured_data_normalizer
        pde_norm_coe = configured_pde_normalizer
        env_norm_coe = configured_env_normalizer
        print(
            '使用 config 中固定的 loss normalizer：'
            f'data={data_norm_coe:.10e}, pde={pde_norm_coe:.10e}, '
            f'env={env_norm_coe:.10e}'
        )
    elif resume_normalizers is not None:
        data_norm_coe = float(resume_normalizers['data'])
        pde_norm_coe = float(resume_normalizers['pde'])
        env_norm_coe = float(resume_normalizers['env'])
    elif resume_spec is not None:
        print('注意：旧 checkpoint 未保存 loss_normalizers，将在续训首个 epoch 重新估计。')

    use_sobol = getattr(args, 'sampling_strategy', 'original') == 'sobol'
    if use_sobol:
        sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        valid_sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        sobol_scale = torch.tensor([args.nz * args.dh, args.nx * args.dh], dtype=torch.float32, device=device)
        sobol_pts = getattr(args, 'sobol_points_per_epoch', 800)
        valid_sobol_pts = getattr(args, 'valid_sobol_points', 800)
        print(f"Sobol 模式: 每 epoch {sobol_pts} 点, 验证 {valid_sobol_pts} 点")

    optimizer.zero_grad()
    pbar = tqdm(range(start_epoch, args.NIter), desc="Training Progress", dynamic_ncols=True)
    step_counter = 0

    # epoch-level 共享 y_ran 采样状态
    epoch_prob = None
    epoch_score = None

    for i in pbar:
        if args.if_adjust and i > args.adjust_from and (i - args.adjust_from) % args.adjust_every == 0:
            decay_times = i // args.adjust_every
            a = max(a * (args.adjust_speed ** (-decay_times)), 2e-1)
            b, c = 1, 0

        model.train()
        batch_loss, batch_u_loss, batch_f_loss, batch_r_loss, batch_env_loss = [], [], [], [], []
        batch_continuous_pde_loss = []
        batch_point_counts, batch_pde_point_counts = [], []
        continuous_pde_active = (
            bool(getattr(args, 'enable_continuous_frequency_pde', False))
            and i >= int(getattr(args, 'continuous_pde_start_epoch', 1))
        )

        if use_sobol:
            y_sobol_base = sobol_engine.draw(sobol_pts).to(device)
            y_sobol_base = y_sobol_base * sobol_scale

        # y_ran: 每 epoch 生成一次（epoch shared 路径）
        y_ran_epoch_shared = None
        if not use_sobol and getattr(args, 'use_y_ran', False) and getattr(args, 'use_epoch_shared_y_ran', False):
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

        has_freq, has_source_coords = dataset_condition_flags(dataloader['train'].dataset)
        for batch_data in dataloader['train']:
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

            if use_sobol:
                y_sobol = y_sobol_base.unsqueeze(0).expand(vel_batch.shape[0], -1, -1).clone()
                y_sobol.requires_grad_(True)

                loss, loss_f, loss_u, loss_r, loss_env, loss_continuous_pde = model.loss(
                    vel_batch, y_sobol, UU0_batch, labels_batch,
                    a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch,
                    y_ran=None, use_continuous_pde=continuous_pde_active,
                    source_coord_batch=source_coord_batch,
                )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                batch_point_counts.append(vel_batch.shape[0] * y_sobol.shape[1])
                batch_pde_point_counts.append(batch_point_counts[-1])
                batch_loss.append(loss.item())
                batch_u_loss.append(loss_u.item())
                batch_f_loss.append(loss_f.item())
                batch_r_loss.append(loss_r.item() if isinstance(loss_r, torch.Tensor) else loss_r)
                batch_env_loss.append(loss_env.item())
                batch_continuous_pde_loss.append(loss_continuous_pde.item())

                del loss, loss_f, loss_u, loss_r, loss_env, loss_continuous_pde, y_sobol

            else:
                # y_ran: 使用 epoch 预生成或 per-model 生成
                if y_ran_epoch_shared is not None:
                    y_ran = y_ran_epoch_shared.unsqueeze(0).expand(
                        vel_batch.shape[0], -1, -1
                    ).clone().requires_grad_(True)
                elif getattr(args, 'use_y_ran', False):
                    with torch.no_grad():
                        y_ran = model.generate_structure_aware_y_ran(vel_batch, num_pts=900)
                else:
                    y_ran = None

                extra_pde_points = 0 if y_ran is None else y_ran.shape[1]
                for batch, data_weight, pde_weight, group_end in coordinate_accumulation_batches(
                    dataloader['train_y'], args.accumulation_steps, extra_pde_points
                ):
                    y_batch = batch[0].to(device)
                    y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    loss, loss_f, loss_u, loss_r, loss_env, loss_continuous_pde = model.loss(
                        vel_batch, y_batch, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe, freq_batch=freq_batch,
                        y_ran=y_ran, use_continuous_pde=continuous_pde_active,
                        source_coord_batch=source_coord_batch,
                    )

                    backward_loss = weighted_microbatch_loss(
                        loss, loss_f, b, data_weight, pde_weight
                    )
                    backward_loss.backward()
                    del backward_loss

                    step_counter += 1
                    if group_end:
                        optimizer.step()
                        optimizer.zero_grad()

                        if step_counter == args.accumulation_steps:
                            peak = (torch.cuda.max_memory_allocated(device) / 1024**3
                                    if device.type == 'cuda' else 0.0)
                            print(f'[First optimizer step] epoch={i}, '
                                  f'peak_allocated_GiB={peak:.3f}', flush=True)

                    batch_point_counts.append(vel_batch.shape[0] * y_batch.shape[1])
                    batch_pde_point_counts.append(
                        vel_batch.shape[0] * (y_batch.shape[1] + extra_pde_points)
                    )
                    batch_loss.append(loss.item())
                    batch_u_loss.append(loss_u.item())
                    batch_f_loss.append(loss_f.item())
                    batch_r_loss.append(loss_r.item() if isinstance(loss_r, torch.Tensor) else loss_r)
                    batch_env_loss.append(loss_env.item())
                    batch_continuous_pde_loss.append(loss_continuous_pde.item())

                    del loss, loss_f, loss_u, loss_r, loss_env, loss_continuous_pde, y_batch

        avg_loss = point_mean(batch_loss, batch_point_counts) if batch_loss else 0
        if batch_loss:
            # Separate point sets when y_ran contributes only to the PDE term.
            avg_loss += b * (
                point_mean(batch_f_loss, batch_pde_point_counts)
                - point_mean(batch_f_loss, batch_point_counts)
            )

        if first_flag:
            data_norm_coe = point_mean(batch_u_loss, batch_point_counts) if batch_u_loss else 1.0
            pde_norm_coe = point_mean(batch_f_loss, batch_pde_point_counts) if batch_f_loss else 1.0
            env_norm_coe = point_mean(batch_env_loss, batch_point_counts) if batch_env_loss else 1.0
            loss_log.append(a + b)
            loss_data_log.append(1.)
            loss_pde_log.append(1.)
            loss_env_log.append(1.)
            loss_reg_log.append(point_mean(batch_r_loss, batch_point_counts) if batch_r_loss else 0)
            loss_continuous_pde_log.append(0.0)
            first_flag = False
        else:
            loss_log.append(avg_loss)
            loss_data_log.append(point_mean(batch_u_loss, batch_point_counts) if batch_u_loss else 0)
            loss_pde_log.append(point_mean(batch_f_loss, batch_pde_point_counts) if batch_f_loss else 0)
            loss_env_log.append(point_mean(batch_env_loss, batch_point_counts) if batch_env_loss else 0)
            loss_reg_log.append(point_mean(batch_r_loss, batch_point_counts) if batch_r_loss else 0)
            loss_continuous_pde_log.append(
                point_mean(batch_continuous_pde_loss, batch_point_counts) if batch_continuous_pde_loss else 0
            )

        current_lr = optimizer.param_groups[0]['lr']

        pbar.set_postfix({
            'Total': f"{avg_loss:.4e}",
            'PDE': f"{loss_pde_log[-1]:.4e}",
            'Data': f"{loss_data_log[-1]:.4e}",
            'ContPDE': f"{loss_continuous_pde_log[-1]:.2e}",
            'FiLM': f"{loss_reg_log[-1]:.2e}",
            'LR': f"{current_lr:.2e}"
        })

        if use_warmup and warmup_scheduler is not None and i <= args.warmup_epochs:
            warmup_scheduler.step(i)
        elif scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_loss)

        # ---- 验证 ----
        if i % args.validate_every == 0:
            model.eval()
            batch_u_loss, batch_f_loss = [], []
            batch_point_counts, batch_pde_point_counts = [], []

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

                if use_sobol:
                    y_valid = valid_sobol_engine.draw(valid_sobol_pts).to(device)
                    y_valid = y_valid * sobol_scale
                    y_valid = y_valid.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                    _, loss_f_valid, loss_u_valid, _, _, _ = model.loss(
                        vel_batch, y_valid, UU0_batch, labels_batch,
                        a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                        freq_batch=freq_batch, source_coord_batch=source_coord_batch,
                    )
                    batch_point_counts.append(vel_batch.shape[0] * y_valid.shape[1])
                    batch_pde_point_counts.append(batch_point_counts[-1])
                    batch_u_loss.append(loss_u_valid.item())
                    batch_f_loss.append(loss_f_valid.item())

                    del loss_f_valid, loss_u_valid, y_valid
                else:
                    for batch in dataloader['valid_y']:
                        y_batch = batch[0].to(device)
                        y_batch = y_batch.unsqueeze(0).expand(vel_batch.shape[0], -1, -1)

                        _, loss_f_valid, loss_u_valid, _, _, _ = model.loss(
                            vel_batch, y_batch, UU0_batch, labels_batch,
                            a, b, c, d, data_norm_coe, pde_norm_coe, env_norm_coe,
                            freq_batch=freq_batch, source_coord_batch=source_coord_batch,
                        )
                        batch_point_counts.append(vel_batch.shape[0] * y_batch.shape[1])
                        batch_pde_point_counts.append(batch_point_counts[-1])
                        batch_u_loss.append(loss_u_valid.item())
                        batch_f_loss.append(loss_f_valid.item())

                        # Release before the next validation/training forward.
                        del loss_f_valid, loss_u_valid, y_batch

            valid_u_loss.append(point_mean(batch_u_loss, batch_point_counts) if batch_u_loss else 0.0)
            valid_f_loss.append(point_mean(batch_f_loss, batch_pde_point_counts) if batch_f_loss else 1.0)

            for marmousi_evaluator in marmousi_evaluators:
                evaluate_training_unseen_frequency(
                    marmousi_evaluator, model, i
                )

        # ---- 可视化 ----
        if i % args.save_fig_every == 0 or i == start_epoch:
            vel_pred, UU0_pred, labels_pred = plot_data["vel_pred"], plot_data["UU0_pred"], plot_data["labels_pred"]
            vel_test, UU0_test, labels_test = plot_data["vel_test"], plot_data["UU0_test"], plot_data["labels_test"]
            has_freq = plot_data.get("has_freq", False)
            freq_pred = plot_data["freq_valid"] if has_freq else None
            freq_test = plot_data["freq_train"] if has_freq else None
            source_coord_test = plot_data.get("source_coord_train")
            source_coord_pred = plot_data.get("source_coord_valid")

            plot_loss(i, args.save_doc, loss_log, loss_data_log, loss_pde_log, valid_u_loss, valid_f_loss)

            # 每次输出训练图片时，同时保存截至当前全局 epoch 的完整 loss
            # （续训时包含 checkpoint 中 epoch 0..resume_epoch 的历史）。
            # 这样中途即可读取 merged_loss_*.npy 和 merged_loss_curve.png，
            # 不必等到整个训练结束。
            _save_merged_resume_loss_logs(
                args,
                resume_spec,
                (loss_log, loss_data_log, loss_pde_log, loss_continuous_pde_log,
                 loss_reg_log, loss_env_log),
            )

            test_plot(args, model, fno, i, dataloader["pred"], vel_pred, UU0_pred, labels_pred, 'valid_without_fine_tune', if_fine_tune=False, freq=freq_pred, source_coord=source_coord_pred)
            test_plot(args, model, fno, i, dataloader["test"], vel_test, UU0_test, labels_test, 'train', if_fine_tune=False, freq=freq_test, source_coord=source_coord_test)
            plot_sinlge(
                model, args, 6, vel_test, UU0_test, labels_test,
                freq=freq_test, source_coord=source_coord_test,
            )

        # ---- 模型保存 ----
        if i % args.save_model_every == 0:
            pbar.write(f'>>> Epoch {i} | 保存 Checkpoint: Total Loss {loss_log[-1]:.4e} | PDE Loss {loss_pde_log[-1]:.4e}')

            checkpoint = {
                'model_state_dict': model.state_dict(),
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

    # 最终保存
    checkpoint = {
        'model_state_dict': model.state_dict(),
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
    _save_merged_resume_loss_logs(
        args,
        resume_spec,
        (loss_log, loss_data_log, loss_pde_log, loss_continuous_pde_log,
         loss_reg_log, loss_env_log),
    )


def train(args):
    try:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

        if getattr(args, 'staged_training', False):
            train_staged(args, device)
        else:
            train_single(args, device)

    except Exception as e:
        print(f"训练过程中断出错: {e}")
        raise
