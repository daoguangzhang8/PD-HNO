"""
PI-DeepONet 外部测试集评估脚本

加载 external_test/ 中的测试数据，支持通过命令行指定震源位置和测试频率。
默认执行未见频率评估 v1：使用 output2 的 500 epoch 权重测试
Marmousi 的 10 Hz 和 20 Hz（这两个频率未出现在训练集中）。

Usage:
    python test.py
    python test.py --dataset marmousi --sources 2 --freqs 5,10,15,20,25
    python test.py --dataset overthrust --sources 0,2,4 --freqs 10 --weights output2/xxx.pth
    python test.py --dataset random --sources 2 --freqs 5,15,25 --finetune
"""

import argparse
import json
import os
import pickle
import re
from types import SimpleNamespace
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from model.utils import *
from model.PI_DeepOnet import Pi_DeepONet
from model.FNO import FNO
from model.plotting import calculate_regression_metrics, fine_tuning


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT2_WEIGHTS = os.path.join(
    PROJECT_ROOT,
    'output2',
    'PI_DeepONet_pde_PI_model_500epoch_weights_145.pth',
)
TRAINING_FREQUENCIES = [3, 5, 7, 9, 11, 12, 13, 15, 17, 18, 19, 21, 22, 23, 25]


# =====================================================================
# 命令行参数
# =====================================================================
def parse_cli():
    p = argparse.ArgumentParser(description='PI-DeepONet 外部测试集评估')
    p.add_argument('--dataset', type=str, default='marmousi',
                   choices=['marmousi', 'overthrust', 'random'],
                   help='测试数据集名称 (默认 marmousi)')
    p.add_argument('--sources', type=str, default='2',
                   help='震源索引, 逗号分隔 (默认 "2")')
    p.add_argument('--freqs', type=str, default='10,20',
                   help='测试频率, 逗号分隔 (默认未见频率 "10,20")')
    p.add_argument('--weights', type=str, default=DEFAULT_OUTPUT2_WEIGHTS,
                   help='模型权重路径 (默认 output2 的 500 epoch checkpoint)')
    p.add_argument('--device', type=int, default=None,
                   help='GPU 编号')
    p.add_argument('--finetune', action='store_true',
                   help='启用域适应微调')
    p.add_argument('--output', type=str, default=None,
                   help='输出目录 (默认 output_test_{dataset}_freq_{freqs})')
    p.add_argument('--test-data-dir', type=str, default=None,
                   help='外部测试数据目录；默认使用 ArgsTest.test_data_dir')
    return p.parse_args()


# =====================================================================
# 测试配置 (与 config.py 格式一致，可直接修改此处的默认值)
# =====================================================================
class ArgsTest:
    # ==========================================
    # 1. 路径与文件配置 (Paths & I/O)
    # ==========================================
    load_path = '/home/ShareData/zdg'         # 测试数据根目录
    weights_save_path = '/home/ShareData/zdg' # 模型权重保存根目录
    save_doc = 'output_test'                  # 结果输出文件夹名称 (会被 CLI --output 覆盖)
    filename = 'PI_DeepONet_pde'             # 保存的模型前缀名称

    # 测试数据路径
    test_data_dir = '/home/ShareData/zdg/external_test'  # 外部测试数据目录

    # ==========================================
    # 2. 硬件与设备配置 (Hardware & Device)
    # ==========================================
    device = 0                                # GPU 设备编号 (会被 CLI --device 覆盖)
    use_parallel = False

    # ==========================================
    # 3. 物理网格与边界条件 (Physical Grid & PML)
    # ==========================================
    dh = 20                                   # 空间网格间距 (m)，物理坐标 = 网格索引 * dh
    nx = 140                                  # 物理模型 x 方向网格数 (不含外延 PML)
    nz = 140                                  # 物理模型 z 方向网格数 (不含外延 PML)
    pml = True                                # 是否启用 PML 吸收边界
    pml_total = 20                            # PML 吸收层的总网格厚度
    pml_crop = 15                             # 裁剪/忽略的 PML 网格数
    pml_active = pml_total - pml_crop         # 剩余参与评估的 PML 网格数

    # 边界类型配置
    boundary_type = 'free_surface'            # 'free_surface' | 'full_pml'

    # ==========================================
    # 4. 测试数据筛选 (Test Data Selection)
    # ==========================================
    source_list = [2]                         # 默认震源列表 (会被 CLI --sources 覆盖)
    freq_list = [10, 20]                      # 默认未见频率列表 (会被 CLI --freqs 覆盖)
    training_frequencies = TRAINING_FREQUENCIES
    n_freq_ranges = 3                         # 合并数据来源的频段数量

    # ==========================================
    # 5. 模型权重配置 (Model Weights)
    # ==========================================
    model_weights_path = ''                   # 模型权重路径 (会被 CLI --weights 覆盖)

    # ==========================================
    # 5.5 网络架构 (Architecture)
    # ==========================================
    # 以下架构参数会在加载 checkpoint 后自动识别。
    branch2_type = 'fno'                      # Branch2 架构: 'fno' | 'resnet' | 'conv'
    use_trunk_freq_encoding = False
    trunk_freq_embed_dim = 8
    trunk_freq_num_bands = 3
    trunk_freq_norm_hz = 25.0
    use_wavenumber_encoding = False
    wavenumber_num_bands = 2
    wavenumber_init_scale = 1.0
    use_relative_source_encoding = False
    relative_source_embed_dim = 16
    relative_source_num_bands = 3
    relative_source_phase_num_bands = 2
    use_joint_source_frequency_fusion = False
    joint_source_embed_dim = 16
    joint_source_phase_num_bands = 2
    joint_source_init_gate = -2.0
    use_film_frequency_conditioning = False
    film_freq_hidden_dim = 32
    film_freq_init_gate = -2.0
    frequency_min_hz = 3.0
    frequency_max_hz = 25.0
    film_smooth_delta_hz = 1.0
    film_smooth_weight = 0.0
    default_freq = 5.0

    # ==========================================
    # 6. 评估与批处理配置 (Evaluation & Batch)
    # ==========================================
    batch_size = 1600                         # 推理时坐标采样批次大小
    in_channels = 2                           # 波场输入通道数 (实部 + 虚部)
    in_channels_vel = 1                       # 速度模型输入通道数
    input_shape_trunk = (batch_size, in_channels, 1, 2)
    input_shape_branch1 = (batch_size, in_channels_vel, nz, nx)
    input_shape_branch2 = (batch_size, in_channels, nz, nx)

    # ==========================================
    # 7. 微调与域适应 (Fine-Tuning)
    # ==========================================
    if_finetune = False                       # 是否启用域适应微调 (会被 CLI --finetune 覆盖)
    ft_NIter = 1000                           # 微调迭代步数
    ft_lr = 2e-5                              # 微调学习率
    ft_a = 0.                                 # 微调数据 Loss 权重
    ft_b = 1                                  # 微调 PDE Loss 权重
    ft_c = 0.00001                            # 微调正则化 Loss 权重
    weight_decay = 1e-4
    factor = 0.9
    patience = 20
    min_lr = 1e-6

    # ==========================================
    # 8. 损失函数权重 (Loss Weights)
    # ==========================================
    a = 1                                     # 数据拟合项权重
    b = 1                                     # PDE 物理残差项权重
    c = 0                                     # 正则化项权重
    d = 1                                     # 包络损失项权重

    # ==========================================
    # 9. Positional Encoding
    # ==========================================
    pe_max_scale = 12.0                       # PE 最高频率尺度

    # ==========================================
    # 10. 训练相关占位 (网络构建需要，测试中不使用)
    # ==========================================
    nvel_train = 100
    ny_train = 100
    sampling_mode = 'full_grid'
    halton_sample_ratio = 0.5
    sampling_strategy = 'original'
    use_y_ran = False
    use_epoch_shared_y_ran = True

    def __init__(self, cli_args):
        # CLI 覆盖默认值
        if cli_args.device is not None:
            self.device = cli_args.device
        if cli_args.finetune:
            self.if_finetune = True
        if cli_args.weights:
            self.model_weights_path = cli_args.weights
        if cli_args.output:
            self.save_doc = cli_args.output
        else:
            freq_tag = '_'.join(part.strip() for part in cli_args.freqs.split(','))
            self.save_doc = f'output_test_{cli_args.dataset}_freq_{freq_tag}'
        if cli_args.test_data_dir:
            self.test_data_dir = cli_args.test_data_dir

        self._cli = cli_args
        self._source_list = [int(s) for s in cli_args.sources.split(',')]
        self._freq_list = [float(f) for f in cli_args.freqs.split(',')]
        self._source_labels = list(self._source_list)


def load_checkpoint_state(model_path, map_location):
    """读取 checkpoint，并兼容普通 state_dict 与 model_state_dict 包装格式。"""
    try:
        checkpoint = torch.load(model_path, map_location=map_location, weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        # 部分历史 checkpoint 的训练状态含 NumPy scalar；权重本身来自本地受信
        # 训练目录时，回退到兼容读取，随后仍只取 model_state_dict。
        checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f'checkpoint 中未找到有效 state_dict: {model_path}')

    # 兼容由 DDP wrapper 保存、key 带 module. 前缀的 checkpoint。
    if state_dict and all(key.startswith('module.') for key in state_dict):
        state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}

    return checkpoint, state_dict


def configure_model_from_checkpoint(args, state_dict):
    """从参数 key/shape 自动恢复会影响模型结构的配置。"""
    keys = tuple(state_dict.keys())
    if any(key.startswith('branch2.0.') for key in keys):
        args.branch2_type = 'fno'
    elif any(key.startswith('branch2.net.') for key in keys):
        args.branch2_type = 'conv'
    elif any(key.startswith('branch2.stem.') for key in keys):
        args.branch2_type = 'resnet'
    else:
        raise ValueError('无法从 checkpoint 识别 branch2_type')

    freq_weight = state_dict.get('trunk_freq_encoder.0.weight')
    args.use_trunk_freq_encoding = freq_weight is not None
    if args.use_trunk_freq_encoding:
        freq_feature_dim = int(freq_weight.shape[1])
        if freq_feature_dim < 1 or (freq_feature_dim - 1) % 2 != 0:
            raise ValueError(f'无法识别频率编码维度: {freq_feature_dim}')
        args.trunk_freq_num_bands = (freq_feature_dim - 1) // 2
        args.trunk_freq_embed_dim = int(
            state_dict['trunk_freq_encoder.2.weight'].shape[0]
        )

    joint_weight = state_dict.get('joint_source_encoder.0.weight')
    args.use_joint_source_frequency_fusion = joint_weight is not None
    args.use_relative_source_encoding = any(
        key.startswith('relative_source_encoder.') for key in keys
    )
    if args.use_relative_source_encoding:
        args.relative_source_embed_dim = int(
            state_dict['relative_source_encoder.2.weight'].shape[0]
        )
    if args.use_joint_source_frequency_fusion:
        joint_input_dim = int(joint_weight.shape[1])
        if joint_input_dim < 3 or (joint_input_dim - 3) % 6 != 0:
            raise ValueError(f'无法识别联合震源编码维度: {joint_input_dim}')
        args.joint_source_phase_num_bands = (joint_input_dim - 3) // 6
        args.joint_source_embed_dim = int(
            state_dict['joint_source_encoder.2.weight'].shape[0]
        )

    # 先扣除震源相对编码，再从剩余维度恢复波数编码；否则 60 维模型中的
    # source=16+phase=12 会被错误识别为额外 7 个波数 band。
    base_trunk_dim = 16 + (
        args.trunk_freq_embed_dim if args.use_trunk_freq_encoding else 0
    )
    if args.use_relative_source_encoding:
        base_trunk_dim += (
            args.relative_source_embed_dim
            + 6 * args.relative_source_phase_num_bands
        )
    checkpoint_trunk_dim = int(state_dict['trunk.fc1.weight'].shape[1])
    wavenumber_dim = checkpoint_trunk_dim - base_trunk_dim
    if wavenumber_dim < 0 or wavenumber_dim % 4 != 0:
        raise ValueError(
            f'Trunk 输入维度不一致: checkpoint={checkpoint_trunk_dim}, '
            f'去除频率/震源编码后剩余={wavenumber_dim}'
        )
    args.use_wavenumber_encoding = wavenumber_dim > 0
    if args.use_wavenumber_encoding:
        args.wavenumber_num_bands = wavenumber_dim // 4

    film_freq_weight = state_dict.get('film_freq_encoder.0.weight')
    args.use_film_frequency_conditioning = film_freq_weight is not None
    if args.use_film_frequency_conditioning:
        args.film_freq_hidden_dim = int(film_freq_weight.shape[0])

    return {
        'branch2_type': args.branch2_type,
        'use_trunk_freq_encoding': args.use_trunk_freq_encoding,
        'trunk_freq_embed_dim': args.trunk_freq_embed_dim,
        'trunk_freq_num_bands': args.trunk_freq_num_bands,
        'trunk_freq_norm_hz': args.trunk_freq_norm_hz,
        'use_wavenumber_encoding': args.use_wavenumber_encoding,
        'wavenumber_num_bands': args.wavenumber_num_bands,
        'use_relative_source_encoding': args.use_relative_source_encoding,
        'relative_source_embed_dim': args.relative_source_embed_dim,
        'use_joint_source_frequency_fusion': args.use_joint_source_frequency_fusion,
        'joint_source_embed_dim': args.joint_source_embed_dim,
        'joint_source_phase_num_bands': args.joint_source_phase_num_bands,
        'use_film_frequency_conditioning': args.use_film_frequency_conditioning,
        'film_freq_hidden_dim': args.film_freq_hidden_dim,
    }


# =====================================================================
# 数据加载
# =====================================================================
def load_test_data(args):
    """
    加载外部测试数据，按 source/freq 筛选。

    数据格式 (gen_external_test.py 生成):
        {name}_velocity.npy   (n_total, nz_ext, nx_ext)
        {name}_wavefield.npy  (n_total * N_SRC, 2, nz_ext, nx_ext)  source-major
        {name}_background.npy 同 wavefield
        {name}_freq_used.npy  (n_total,)

    其中 n_total = n_models * n_freqs_all, N_SRC = 5
    wavefield 排序: [src0_sample0, src0_sample1, ..., src1_sample0, ...]
    """
    name = args._cli.dataset
    source_list = args._source_list
    freq_list = args._freq_list
    data_dir = args.test_data_dir

    vel = np.load(os.path.join(data_dir, f'{name}_velocity.npy'))
    wf = np.load(os.path.join(data_dir, f'{name}_wavefield.npy'))
    bg = np.load(os.path.join(data_dir, f'{name}_background.npy'))
    freq_all = np.load(os.path.join(data_dir, f'{name}_freq_used.npy'))

    n_total = vel.shape[0]  # n_models * n_freqs_all
    if wf.shape[0] % n_total != 0:
        raise ValueError(
            f'wavefield 第一维 {wf.shape[0]} 不能被 velocity 样本数 '
            f'{n_total} 整除，无法推断震源数量'
        )
    n_src = wf.shape[0] // n_total
    metadata_path = os.path.join(data_dir, 'generation_metadata.json')
    source_labels = list(range(n_src))
    source_labels_are_positions = False
    if os.path.isfile(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as file:
            metadata = json.load(file)
        source_labels = metadata.get('source_positions_orig', source_labels)
        source_labels_are_positions = 'source_positions_orig' in metadata
        if len(source_labels) != n_src:
            raise ValueError(
                f'generation_metadata 中震源数量 {len(source_labels)} 与 wavefield '
                f'数量 {n_src} 不一致'
            )
    nz_ext, nx_ext = vel.shape[1], vel.shape[2]
    freq_unique = np.unique(freq_all).tolist()

    print(f'[*] 数据集: {name}')
    print(f'    原始维度: vel={vel.shape}, wf={wf.shape}, freq={freq_all.shape}')
    print(f'    包含频率: {freq_unique}')

    # --- PML 裁切 (与训练逻辑一致) ---
    if args.pml:
        pc = args.pml_crop
        # ``slice(0, -0)`` 在 Python 中是空切片。物理约束需要保留完整
        # 160×180 PML 域时显式使用 ``pml_crop=0``，此处必须原样保留数组。
        if pc > 0:
            if args.boundary_type == 'free_surface':
                z_sl = slice(0, -pc)
            else:
                z_sl = slice(pc, -pc)
            x_sl = slice(pc, -pc)
        else:
            z_sl = slice(None)
            x_sl = slice(None)

        vel = vel[:, z_sl, x_sl]
        wf = wf[:, :, z_sl, x_sl]
        bg = bg[:, :, z_sl, x_sl]

        args.nz = vel.shape[1]
        args.nx = vel.shape[2]
        print(f'    PML 裁切后: {vel.shape[1]}×{vel.shape[2]}  (pml_crop={pc})')

    # --- 按 freq 筛选 ---
    missing_freqs = [
        requested for requested in freq_list
        if not np.any(np.isclose(freq_all, requested))
    ]
    if missing_freqs:
        raise ValueError(
            f'请求频率 {missing_freqs} 在数据中不存在 (可用: {freq_unique})'
        )

    freq_mask = np.zeros(freq_all.shape, dtype=bool)
    for requested in freq_list:
        freq_mask |= np.isclose(freq_all, requested)
    sample_idx = np.where(freq_mask)[0]
    if len(sample_idx) == 0:
        raise ValueError(f'freq_list={freq_list} 在数据中不存在 (可用: {freq_unique})')

    vel_sel = vel[sample_idx]              # (n_sel, nz, nx)
    freq_sel = freq_all[sample_idx]        # (n_sel,)

    # --- 按 source 筛选 (source-major 布局) ---
    wf_sel = []
    bg_sel = []
    for s in source_list:
        if s < 0 or s >= n_src:
            raise ValueError(f'震源索引 {s} 超出范围 [0, {n_src-1}]')
        s_idx = s * n_total + sample_idx
        wf_sel.append(wf[s_idx])
        bg_sel.append(bg[s_idx])
    wf_sel = np.concatenate(wf_sel, axis=0)  # (n_sources * n_sel, 2, nz, nx)
    bg_sel = np.concatenate(bg_sel, axis=0)

    # 为每个 source 复制对应的 freq
    freq_expanded = np.tile(freq_sel, len(source_list))

    selected_source_labels = [source_labels[s] for s in source_list]
    args._source_labels = selected_source_labels
    args._source_labels_are_positions = source_labels_are_positions
    print(
        f'    筛选: source_indices={source_list}, '
        f'source_positions={selected_source_labels}, freqs={freq_list}'
    )
    print(f'    筛选后: vel={vel_sel.shape}, wf={wf_sel.shape}')

    # 转 tensor
    vel_t = torch.from_numpy(vel_sel).float()
    wf_t = torch.from_numpy(wf_sel).float()
    bg_t = torch.from_numpy(bg_sel).float()
    freq_t = torch.from_numpy(freq_expanded).float()

    return {
        'vel': vel_t,
        'wavefield': wf_t,
        'background': bg_t,
        'freq': freq_t,
        'n_samples': len(sample_idx),
        'n_sources': len(source_list),
    }


# =====================================================================
# 评估与绘图
# =====================================================================
def evaluate_single(args, model, vel, UU0, label, freq_val, device):
    """对单个 (vel, bg) 组合在全网格上推理，返回预测和指标。"""
    model.eval()
    nz, nx = args.nz, args.nx

    # 坐标网格
    grid_z, grid_x = torch.meshgrid(
        torch.arange(nz), torch.arange(nx), indexing='ij')
    y_grid = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1).float() * args.dh
    loader = DataLoader(TensorDataset(y_grid), batch_size=args.batch_size, shuffle=False)

    vel_dev = vel.to(device)
    UU0_dev = UU0.unsqueeze(0).to(device) if UU0.dim() == 3 else UU0.to(device)
    freq_dev = torch.tensor([freq_val], device=device, dtype=torch.float32)

    pred_parts = []
    with torch.no_grad():
        for batch in loader:
            y_b = batch[0].to(device).unsqueeze(0)
            out = model(vel_dev, y_b, UU0_dev, freq_batch=freq_dev)
            pred_parts.append(out)

    pred = torch.cat(pred_parts, dim=1)
    pred_2d = pred[0].view(nz, nx, 2).cpu().numpy()
    true_2d = label.cpu().numpy().transpose(1, 2, 0)  # (2, nz, nx) -> (nz, nx, 2)

    # PML active 区域裁切 (用于指标计算)
    L = args.pml_active
    if args.boundary_type == 'free_surface':
        z_sl = slice(0, -L) if L > 0 else slice(None)
    else:
        z_sl = slice(L, -L) if L > 0 else slice(None)
    x_sl = slice(L, -L) if L > 0 else slice(None)

    pred_crop = pred_2d[z_sl, x_sl, :]
    true_crop = true_2d[z_sl, x_sl, :] if true_2d.shape[0] == nz else true_2d

    m_r = calculate_regression_metrics(pred_crop[:, :, 0], true_crop[:, :, 0])
    m_i = calculate_regression_metrics(pred_crop[:, :, 1], true_crop[:, :, 1])

    return pred_2d, m_r, m_i


def plot_results(args, dataset_name, results, epoch_num):
    """绘制评估结果图。每个子图: True / Pred / Error × (real, imag)。"""
    n = len(results)
    if n == 0:
        return

    os.makedirs(args.save_doc, exist_ok=True)

    fig_r, ax_r = plt.subplots(3, n, figsize=(4.2 * n, 11))
    fig_i, ax_i = plt.subplots(3, n, figsize=(4.2 * n, 11))
    if n == 1:
        ax_r, ax_i = ax_r[:, np.newaxis], ax_i[:, np.newaxis]

    fig_r.suptitle(f'{dataset_name} REAL | Epoch {epoch_num}', fontsize=14)
    fig_i.suptitle(f'{dataset_name} IMAG | Epoch {epoch_num}', fontsize=14)

    for col, entry in enumerate(results):
        src = entry['source']
        freq = entry['freq']
        true = entry['true']
        pred = entry['pred']
        m_r, m_i = entry['metrics_real'], entry['metrics_imag']
        tag = f'src{src}_{freq:.0f}Hz'

        # PML active crop for display
        L = args.pml_active
        z_sl = slice(0, -L) if args.boundary_type == 'free_surface' and L > 0 else (
            slice(L, -L) if L > 0 else slice(None))
        x_sl = slice(L, -L) if L > 0 else slice(None)

        t_r, p_r = true[z_sl, x_sl, 0], pred[z_sl, x_sl, 0]
        t_i, p_i = true[z_sl, x_sl, 1], pred[z_sl, x_sl, 1]
        e_r, e_i = t_r - p_r, t_i - p_i

        for (fig, axes, t, p, e, m, part) in [
            (fig_r, ax_r, t_r, p_r, e_r, m_r, 'real'),
            (fig_i, ax_i, t_i, p_i, e_i, m_i, 'imag'),
        ]:
            vm = max(np.abs(t).max(), 1e-12)
            em = max(np.abs(e).max(), 1e-12)

            im = axes[0, col].imshow(t, cmap='seismic', vmin=-vm, vmax=vm, aspect='equal')
            axes[0, col].set_title(f'{tag}\nTrue {part} | R²={m["r2"]:.4f}', fontsize=9)
            axes[0, col].axis('off')
            fig.colorbar(im, ax=axes[0, col], fraction=0.046)

            im = axes[1, col].imshow(p, cmap='seismic', vmin=-vm, vmax=vm, aspect='equal')
            axes[1, col].set_title(f'Pred {part}', fontsize=9)
            axes[1, col].axis('off')
            fig.colorbar(im, ax=axes[1, col], fraction=0.046)

            im = axes[2, col].imshow(e, cmap='bwr', vmin=-em, vmax=em, aspect='equal')
            axes[2, col].set_title(f'Error MSE={m["mse"]:.2e}', fontsize=9)
            axes[2, col].axis('off')
            fig.colorbar(im, ax=axes[2, col], fraction=0.046)

    for fig, suffix in [(fig_r, 'REAL'), (fig_i, 'IMAG')]:
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        path = os.path.join(args.save_doc, f'{dataset_name}_{suffix}_epoch_{epoch_num}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  已保存 {path}')


def plot_results_by_frequency(args, dataset_name, results, epoch_num):
    """每个频率分别输出 REAL/IMAG；五列震源共享场值和误差 colorbar。"""
    if not results:
        return

    os.makedirs(args.save_doc, exist_ok=True)
    frequencies = sorted({float(item['freq']) for item in results})
    for frequency in frequencies:
        group = [
            item for item in results
            if np.isclose(float(item['freq']), frequency)
        ]
        group.sort(key=lambda item: int(item['source']))
        n = len(group)
        prepared = []
        for entry in group:
            true = entry['true']
            pred = entry['pred']
            L = args.pml_active
            if args.boundary_type == 'free_surface':
                z_sl = slice(0, -L) if L > 0 else slice(None)
            else:
                z_sl = slice(L, -L) if L > 0 else slice(None)
            x_sl = slice(L, -L) if L > 0 else slice(None)

            prepared.append((entry, true[z_sl, x_sl], pred[z_sl, x_sl]))

        for component, channel in (('REAL', 0), ('IMAG', 1)):
            field_limit = max(
                max(np.abs(true[:, :, channel]).max(),
                    np.abs(pred[:, :, channel]).max())
                for _, true, pred in prepared
            )
            error_limit = max(
                np.abs(true[:, :, channel] - pred[:, :, channel]).max()
                for _, true, pred in prepared
            )
            field_limit = max(float(field_limit), 1e-12)
            error_limit = max(float(error_limit), 1e-12)
            fig, axes = plt.subplots(3, n, figsize=(3.8 * n, 10), squeeze=False)
            fig.suptitle(
                f'{dataset_name} {component} | {frequency:g} Hz | Epoch {epoch_num}',
                fontsize=15,
            )
            field_image = error_image = None
            for col, (entry, true, pred) in enumerate(prepared):
                src = int(entry['source'])
                truth = true[:, :, channel]
                prediction = pred[:, :, channel]
                error = truth - prediction
                metrics = entry['metrics_real' if channel == 0 else 'metrics_imag']
                images = (truth, prediction, error)
                titles = (
                    f'Source {src} | True',
                    f'Source {src} | Pred\nR²={metrics["r2"]:.4f}',
                    f'Source {src} | Error\nRelL2={metrics["relative_l2"]:.4f}',
                )
                for row, (image, title) in enumerate(zip(images, titles)):
                    if row < 2:
                        handle = axes[row, col].imshow(
                            image, cmap='seismic', vmin=-field_limit,
                            vmax=field_limit, aspect='equal'
                        )
                        field_image = handle
                    else:
                        handle = axes[row, col].imshow(
                            image, cmap='bwr', vmin=-error_limit,
                            vmax=error_limit, aspect='equal'
                        )
                        error_image = handle
                    axes[row, col].set_title(title, fontsize=9)
                    axes[row, col].axis('off')

            fig.subplots_adjust(right=0.91, top=0.92, hspace=0.18, wspace=0.08)
            field_cax = fig.add_axes([0.925, 0.39, 0.012, 0.48])
            error_cax = fig.add_axes([0.925, 0.08, 0.012, 0.23])
            fig.colorbar(field_image, cax=field_cax, label='True / Pred amplitude')
            fig.colorbar(error_image, cax=error_cax, label='Error amplitude')
            path = os.path.join(
                args.save_doc,
                f'{dataset_name}_freq_{frequency:g}Hz_{component}_epoch_{epoch_num}.png',
            )
            fig.savefig(path, dpi=180, bbox_inches='tight')
            plt.close(fig)
            print(f'  已保存按频率分组图: {path}')


def plot_diagonal_imag_predictions(args, dataset_name, results, epoch_num):
    """横向绘制 src0@5Hz ... src4@25Hz 的虚部预测，与旧图色标规则一致。"""
    requested_pairs = list(zip(range(5), (5.0, 10.0, 15.0, 20.0, 25.0)))
    selected = []
    for source, frequency in requested_pairs:
        match = next(
            (
                item for item in results
                if int(item['source']) == source
                and np.isclose(float(item['freq']), frequency)
            ),
            None,
        )
        if match is None:
            return
        selected.append(match)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.4), squeeze=False)
    fig.suptitle(
        f'{dataset_name} IMAG predictions | Epoch {epoch_num}', fontsize=15
    )
    L = args.pml_active
    if args.boundary_type == 'free_surface':
        z_sl = slice(0, -L) if L > 0 else slice(None)
    else:
        z_sl = slice(L, -L) if L > 0 else slice(None)
    x_sl = slice(L, -L) if L > 0 else slice(None)

    for col, (entry, (source, frequency)) in enumerate(
            zip(selected, requested_pairs)):
        truth = entry['true'][z_sl, x_sl, 1]
        prediction = entry['pred'][z_sl, x_sl, 1]
        # 与 MARMOUSI_IMAG_epoch_500.png 一致：Pred 使用对应 True 的对称范围，
        # 且每个 source-frequency 子图保留独立 colorbar。
        limit = max(float(np.abs(truth).max()), 1e-12)
        image = axes[0, col].imshow(
            prediction, cmap='seismic', vmin=-limit, vmax=limit,
            aspect='equal',
        )
        axes[0, col].set_title(
            f'src{source}_{frequency:g}Hz\nPred imag', fontsize=10
        )
        axes[0, col].axis('off')
        fig.colorbar(image, ax=axes[0, col], fraction=0.046, pad=0.035)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    path = os.path.join(
        args.save_doc,
        f'{dataset_name}_IMAG_diagonal_predictions_epoch_{epoch_num}.png',
    )
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  已保存对角组合虚部预测图: {path}')


def plot_diagonal_real_results(args, dataset_name, results, epoch_num):
    """按旧版布局绘制五个指定 source-frequency 组合的实部 True/Pred/Error。"""
    requested_pairs = list(zip(range(5), (5.0, 10.0, 15.0, 20.0, 25.0)))
    selected = []
    for source, frequency in requested_pairs:
        match = next(
            (
                item for item in results
                if int(item['source']) == source
                and np.isclose(float(item['freq']), frequency)
            ),
            None,
        )
        if match is None:
            return
        selected.append(match)

    fig, axes = plt.subplots(3, 5, figsize=(20, 12), squeeze=False)
    fig.suptitle(f'{dataset_name} REAL | Epoch {epoch_num}', fontsize=15)
    L = args.pml_active
    if args.boundary_type == 'free_surface':
        z_sl = slice(0, -L) if L > 0 else slice(None)
    else:
        z_sl = slice(L, -L) if L > 0 else slice(None)
    x_sl = slice(L, -L) if L > 0 else slice(None)

    for col, (entry, (source, frequency)) in enumerate(
            zip(selected, requested_pairs)):
        truth = entry['true'][z_sl, x_sl, 0]
        prediction = entry['pred'][z_sl, x_sl, 0]
        error = truth - prediction
        field_limit = max(float(np.abs(truth).max()), 1e-12)
        error_limit = max(float(np.abs(error).max()), 1e-12)
        mse = float(entry['metrics_real']['mse'])
        titles = (
            f'src{source}_{frequency:g}Hz\nTrue real | R²={entry["metrics_real"]["r2"]:.4f}',
            'Pred real',
            f'Error MSE={mse:.2e}',
        )
        for row, (values, title) in enumerate(zip((truth, prediction, error), titles)):
            limit = field_limit if row < 2 else error_limit
            image = axes[row, col].imshow(
                values,
                cmap='seismic' if row < 2 else 'bwr',
                vmin=-limit,
                vmax=limit,
                aspect='equal',
            )
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis('off')
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.035)

    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    path = os.path.join(
        args.save_doc,
        f'{dataset_name}_REAL_diagonal_true_pred_error_epoch_{epoch_num}.png',
    )
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  已保存对角组合实部 True/Pred/Error 图: {path}')


def save_frequency_metrics(args, model_path, epoch_num, architecture, results):
    """按频率分别保存五个震源的指标。"""
    frequencies = sorted({float(item['freq']) for item in results})
    for frequency in frequencies:
        group = [
            item for item in results
            if np.isclose(float(item['freq']), frequency)
        ]
        payload = {
            'evaluation': 'marmousi_by_frequency',
            'dataset': args._cli.dataset,
            'frequency_hz': frequency,
            'sources': [int(item['source']) for item in group],
            'checkpoint': os.path.abspath(model_path) if model_path else None,
            'epoch': epoch_num,
            'architecture': architecture,
            'results': [
                {
                    'source': int(item['source']),
                    'real': {
                        key: float(value)
                        for key, value in item['metrics_real'].items()
                    },
                    'imag': {
                        key: float(value)
                        for key, value in item['metrics_imag'].items()
                    },
                }
                for item in group
            ],
        }
        path = os.path.join(
            args.save_doc,
            f'{args._cli.dataset}_freq_{frequency:g}Hz_metrics.json',
        )
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print(f'  已保存按频率指标: {path}')


def save_metrics(args, model_path, epoch_num, architecture, results,
                 metrics_filename='metrics.json', write_latest=False):
    """保存机器可读的逐频率指标，便于后续比较不同 checkpoint。"""
    os.makedirs(args.save_doc, exist_ok=True)

    serializable_results = []
    for result in results:
        freq = float(result['freq'])
        serializable_results.append({
            'source': int(result['source']),
            'frequency_hz': freq,
            'seen_in_training': any(
                np.isclose(freq, train_freq) for train_freq in args.training_frequencies
            ),
            'real': {key: float(value) for key, value in result['metrics_real'].items()},
            'imag': {key: float(value) for key, value in result['metrics_imag'].items()},
        })

    by_frequency = {}
    for freq in sorted({item['frequency_hz'] for item in serializable_results}):
        items = [item for item in serializable_results if np.isclose(item['frequency_hz'], freq)]
        by_frequency[f'{freq:g}'] = {
            'seen_in_training': items[0]['seen_in_training'],
            'count': len(items),
            'real_r2_mean': float(np.mean([item['real']['r2'] for item in items])),
            'imag_r2_mean': float(np.mean([item['imag']['r2'] for item in items])),
            'real_relative_l2_mean': float(np.mean([
                item['real']['relative_l2'] for item in items
            ])),
            'imag_relative_l2_mean': float(np.mean([
                item['imag']['relative_l2'] for item in items
            ])),
        }

    payload = {
        'evaluation': 'unseen_frequency_v1',
        'dataset': args._cli.dataset,
        'sources': args._source_list,
        'requested_frequencies_hz': args._freq_list,
        'training_frequencies_hz': args.training_frequencies,
        'checkpoint': os.path.abspath(model_path) if model_path else 'in_memory_training_model',
        'epoch': epoch_num,
        'architecture': architecture,
        'summary': {
            'count': len(serializable_results),
            'real_r2_mean': float(np.mean([
                item['real']['r2'] for item in serializable_results
            ])),
            'imag_r2_mean': float(np.mean([
                item['imag']['r2'] for item in serializable_results
            ])),
            'by_frequency': by_frequency,
        },
        'results': serializable_results,
    }

    metrics_paths = [os.path.join(args.save_doc, metrics_filename)]
    if write_latest:
        metrics_paths.append(os.path.join(args.save_doc, 'metrics_latest.json'))
    for metrics_path in metrics_paths:
        with open(metrics_path, 'w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print(f'  已保存 {metrics_path}')
    return payload


def evaluate_loaded_data(args, model, data, device, epoch_num, architecture,
                         model_path=None, metrics_filename='metrics.json',
                         write_latest=False):
    """使用内存中的模型评估已加载数据；供独立测试和训练验证共同调用。"""
    vel = (data['vel'] / 1000.0).unsqueeze(1)
    wf = data['wavefield']
    bg = data['background']
    freq = data['freq']
    n_samples = data['n_samples']
    n_sources = data['n_sources']

    print(f'\n{"="*60}')
    print(f'Marmousi 未见频率评估 | epoch={epoch_num} | '
          f'{n_sources} sources × {n_samples} samples')
    print(f'{"="*60}')

    was_training = model.training
    model.eval()
    results = []
    try:
        source_labels = getattr(args, '_source_labels', args._source_list)
        for si, src in enumerate(source_labels):
            for fi in range(n_samples):
                idx = si * n_samples + fi
                freq_val = freq[idx].item()
                bg_i = bg[idx]
                label_i = wf[idx] - bg[idx]
                vel_single = vel[fi:fi + 1]

                pred_2d, m_r, m_i = evaluate_single(
                    args, model, vel_single, bg_i.unsqueeze(0), label_i,
                    freq_val, device,
                )
                results.append({
                    'source': src,
                    'freq': freq_val,
                    'true': label_i.numpy().transpose(1, 2, 0),
                    'pred': pred_2d,
                    'metrics_real': m_r,
                    'metrics_imag': m_i,
                })

                frequency_seen = any(
                    np.isclose(freq_val, train_freq)
                    for train_freq in args.training_frequencies
                )
                source_tag = (
                    'UNSEEN_SOURCE'
                    if getattr(args, '_source_labels_are_positions', False)
                    else 'SOURCE_INDEX'
                )
                seen_tag = 'FREQ_SEEN' if frequency_seen else 'FREQ_UNSEEN'
                print(f'  src={src} freq={freq_val:5.1f}Hz '
                      f'[{seen_tag}, {source_tag}] | '
                      f'REAL R²={m_r["r2"]:.4f} RelL2={m_r["relative_l2"]:.4f} | '
                      f'IMAG R²={m_i["r2"]:.4f} RelL2={m_i["relative_l2"]:.4f}')

        dataset_name = args._cli.dataset.upper()
        if getattr(args, 'compact_output', False):
            from model.compact_output import plot_fields
            for item in results:
                item['name'] = f'src={item["source"]}, {item["freq"]:g}Hz'
            plot_fields(results, os.path.join(args.save_doc, 'wavefields.png'), epoch_num)
            return save_metrics(
                args, model_path, epoch_num, architecture, results,
                metrics_filename='metrics_latest.json', write_latest=False,
            )
        plot_results_by_frequency(args, dataset_name, results, epoch_num)
        plot_diagonal_imag_predictions(
            args, dataset_name, results, epoch_num
        )
        plot_diagonal_real_results(args, dataset_name, results, epoch_num)
        metrics = save_metrics(
            args, model_path, epoch_num, architecture, results,
            metrics_filename=metrics_filename,
            write_latest=write_latest,
        )
        save_frequency_metrics(
            args, model_path, epoch_num, architecture, results
        )

        r2_r = metrics['summary']['real_r2_mean']
        r2_i = metrics['summary']['imag_r2_mean']
        print(f'评估完成: 平均 R² real={r2_r:.4f} imag={r2_i:.4f}; '
              f'结果目录: {args.save_doc}/')
        return metrics
    finally:
        if was_training:
            model.train()


def prepare_training_unseen_frequency_evaluator(
        training_args, device, stage_tag=None, *, frequencies=None,
        sources=None, data_dir=None, output_name=None):
    """训练开始时缓存一组 Marmousi 泛化测试数据，供验证节点复用。"""
    frequencies = list(
        frequencies if frequencies is not None else getattr(
            training_args, 'marmousi_eval_frequencies', [10, 20]
        )
    )
    sources = list(
        sources if sources is not None else getattr(
            training_args, 'marmousi_eval_sources', [2]
        )
    )
    cli = SimpleNamespace(
        device=device.index if isinstance(device, torch.device) else int(device),
        finetune=False,
        weights=None,
        output=None,
        test_data_dir=None,
        dataset='marmousi',
        sources=','.join(str(value) for value in sources),
        freqs=','.join(str(value) for value in frequencies),
    )
    eval_args = ArgsTest(cli)

    for name in (
        'dh', 'pml', 'pml_total', 'pml_crop', 'pml_active', 'boundary_type',
        'pe_max_scale', 'branch2_type', 'use_trunk_freq_encoding',
        'trunk_freq_embed_dim', 'trunk_freq_num_bands', 'trunk_freq_norm_hz',
        'use_wavenumber_encoding', 'wavenumber_num_bands',
        'wavenumber_init_scale', 'use_relative_source_encoding',
        'relative_source_embed_dim', 'relative_source_num_bands',
        'relative_source_phase_num_bands',
        'use_joint_source_frequency_fusion', 'joint_source_embed_dim',
        'joint_source_phase_num_bands', 'joint_source_init_gate',
        'use_film_frequency_conditioning',
        'film_freq_hidden_dim', 'film_freq_init_gate', 'frequency_min_hz',
        'frequency_max_hz', 'film_smooth_delta_hz', 'film_smooth_weight',
        'default_freq', 'source_radius_mode', 'compact_output',
    ):
        if hasattr(training_args, name):
            setattr(eval_args, name, getattr(training_args, name))

    eval_args.batch_size = int(getattr(
        training_args, 'marmousi_eval_batch_size', 1600
    ))
    eval_args.test_data_dir = (
        data_dir if data_dir is not None else getattr(
            training_args,
            'marmousi_eval_data_dir',
            '/home/sharedata/zdg/external_test',
        )
    )
    output_name = (
        output_name if output_name is not None else getattr(
            training_args,
            'marmousi_eval_output_dir',
            'marmousi_unseen_frequency',
        )
    )
    if stage_tag:
        output_name = os.path.join(output_name, str(stage_tag))
    eval_args.save_doc = os.path.join(training_args.save_doc, output_name)

    data = load_test_data(eval_args)
    architecture = {
        'branch2_type': eval_args.branch2_type,
        'use_trunk_freq_encoding': eval_args.use_trunk_freq_encoding,
        'trunk_freq_embed_dim': eval_args.trunk_freq_embed_dim,
        'trunk_freq_num_bands': eval_args.trunk_freq_num_bands,
        'trunk_freq_norm_hz': eval_args.trunk_freq_norm_hz,
        'use_wavenumber_encoding': eval_args.use_wavenumber_encoding,
        'wavenumber_num_bands': eval_args.wavenumber_num_bands,
        'use_joint_source_frequency_fusion': eval_args.use_joint_source_frequency_fusion,
        'joint_source_embed_dim': eval_args.joint_source_embed_dim,
        'joint_source_phase_num_bands': eval_args.joint_source_phase_num_bands,
        'use_film_frequency_conditioning': eval_args.use_film_frequency_conditioning,
        'film_freq_hidden_dim': eval_args.film_freq_hidden_dim,
    }
    print(f'[Marmousi Eval] 数据已缓存: freqs={frequencies}, sources={sources}, '
          f'output={eval_args.save_doc}')
    return {
        'args': eval_args,
        'data': data,
        'device': device,
        'architecture': architecture,
    }


def evaluate_training_unseen_frequency(evaluator, model, epoch_num):
    """在训练验证节点使用当前模型输出 Marmousi 10/20 Hz 预测。"""
    epoch_tag = str(epoch_num)
    return evaluate_loaded_data(
        evaluator['args'], model, evaluator['data'], evaluator['device'],
        epoch_num=epoch_tag,
        architecture=evaluator['architecture'],
        model_path=None,
        metrics_filename=f'metrics_epoch_{epoch_tag}.json',
        write_latest=True,
    )


# =====================================================================
# 主测试流程
# =====================================================================
def test(args):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print(f'PI-DeepONet 外部测试评估 | 设备: {device}')
    print(f'数据集: {args._cli.dataset}  震源: {args._source_list}  频率: {args._freq_list}')
    print(f'网格: nz={args.nz}, nx={args.nx}, dh={args.dh}, boundary={args.boundary_type}')
    print(f'PML: total={args.pml_total}, crop={args.pml_crop}, active={args.pml_active}')
    print(f'{"="*60}')

    # ---- 1. 加载测试数据 ----
    data = load_test_data(args)
    vel = data['vel'] / 1000.0          # 归一化, 与训练一致
    vel = vel.unsqueeze(1)              # (N, nz, nx) -> (N, 1, nz, nx)
    wf = data['wavefield']
    bg = data['background']
    freq = data['freq']
    n_samples = data['n_samples']
    n_sources = data['n_sources']

    # ---- 2. 加载模型 ----
    model_path = getattr(args, 'model_weights_path', None)
    if not model_path:
        raise ValueError('未指定权重路径，请使用 --weights 或在 config.py 中设置 model_weights_path')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'权重文件不存在: {model_path}')

    _, state_dict = load_checkpoint_state(model_path, map_location=device)
    architecture = configure_model_from_checkpoint(args, state_dict)
    print(f'Checkpoint 架构: {architecture}')

    model = Pi_DeepONet(args).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    epoch_match = re.search(r'(\d+)epoch', model_path)
    epoch_num = int(epoch_match.group(1)) if epoch_match else 0
    print(f'✅ 权重已加载: {model_path} (epoch={epoch_num})')

    # ---- 3. 微调 (可选) ----
    if args.if_finetune:
        print(f'\n[!] 域适应微调...')
        fno = FNO(args).to(device)
        fno.eval()
        labels = wf - bg
        # source-major 顺序复制速度场，以匹配 bg/labels/freq。
        vel_ft = vel[:n_samples].repeat(n_sources, 1, 1, 1).to(device)
        bg_ft = bg.to(device)
        lab_ft = labels.to(device)
        freq_ft = freq.to(device)

        # 构建坐标 dataloader
        grid_z, grid_x = torch.meshgrid(
            torch.arange(args.nz), torch.arange(args.nx), indexing='ij')
        y_grid = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1).float() * args.dh
        loader_y = DataLoader(TensorDataset(y_grid), batch_size=args.batch_size, shuffle=False)

        model = fine_tuning(args, model, fno, loader_y, vel_ft, bg_ft, lab_ft, freq=freq_ft)
        model.eval()
        print('  微调完成')

    # ---- 4. 使用与训练循环相同的公共评估接口 ----
    return evaluate_loaded_data(
        args, model, data, device, epoch_num, architecture,
        model_path=model_path,
    )


def main():
    cli = parse_cli()
    args = ArgsTest(cli)

    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(args.device).total_memory / (1024**3)
        print(f'GPU: {torch.cuda.get_device_name(args.device)} ({mem:.1f}GB)')

    test(args)


if __name__ == '__main__':
    print('*******************************************')
    print('       PI-DeepONet External Test           ')
    print('*******************************************')
    main()
