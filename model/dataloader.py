from Labconfig import *
from model.utils import Halton_Sample


def set_dataloader_epoch_seed(loader, base_seed, epoch):
    """Reset a shuffled DataLoader to a deterministic epoch-specific order.

    ``RandomSampler`` owns the generator that determines index order.  The
    DataLoader generator is reset as well because it controls worker seeds.
    Keeping both explicit prevents unrelated model initialization/random draws
    from changing coordinate batches on different DDP ranks.
    """
    epoch_seed = int(base_seed) + int(epoch)
    sampler = getattr(loader, 'sampler', None)
    if sampler is not None and hasattr(sampler, 'generator'):
        if sampler.generator is None:
            sampler.generator = torch.Generator()
        sampler.generator.manual_seed(epoch_seed)
    if getattr(loader, 'generator', None) is not None:
        loader.generator.manual_seed(epoch_seed + 1_000_000)
    return epoch_seed



def make_conditioned_dataset(vel, uu0, labels, freq=None, source_coords=None):
    """Build a TensorDataset with an unambiguous optional-condition order."""
    tensors = [vel, uu0, labels]
    if freq is not None:
        tensors.append(freq)
    if source_coords is not None:
        tensors.append(source_coords)
    return TensorDataset(*tensors)


def unpack_conditioned_batch(batch_data):
    """Return ``vel, uu0, labels, freq, source`` for every supported dataset."""
    if len(batch_data) not in (3, 4, 5):
        raise ValueError(f'条件训练 batch 应包含 3/4/5 个张量，实际为 {len(batch_data)}')
    vel, uu0, labels = batch_data[:3]
    freq = source_coords = None
    if len(batch_data) == 5:
        freq, source_coords = batch_data[3], batch_data[4]
    elif len(batch_data) == 4:
        condition = batch_data[3]
        if condition.ndim >= 2 and condition.shape[-1] == 2:
            source_coords = condition
        else:
            freq = condition
    return vel, uu0, labels, freq, source_coords


def dataset_condition_flags(dataset):
    """Return ``(has_freq, has_source)`` without relying only on tuple length."""
    tensors = dataset.tensors
    if len(tensors) == 5:
        return True, True
    if len(tensors) == 4:
        is_source = tensors[3].ndim >= 2 and tensors[3].shape[-1] == 2
        return not is_source, is_source
    return False, False


def Training_data(args, vel, UU_loc, UU0_loc, freq=None, source_coord_loc=None):
    """
    生成训练数据和验证数据（支持多震源并发训练）。

    Args:
        freq: 频率数据 [N_vel]，每个速度模型对应一个频率值。若为 None 则使用默认值。

    ``nvel_train`` 的含义：
        * 单震源：从 ``N_vel`` 个 (速度模型, 频率) 条目中抽取的样本数；
        * 多震源：从 ``N_vel × N_source`` 个
          (速度模型, 频率, 震源) 条目中抽取的总样本数。

    多震源时不能先选一批 velocity/frequency 索引再复制到所有震源，否则实际
    训练集只是 ``nvel_train × N_source`` 个重复索引，而不是从完整候选空间中
    选择。这里对每个震源独立无放回抽样，并使源间样本数之差不超过 1。
    """
    # 1. 基本参数准备
    nvel, ny, pml_crop = args.nvel_train, args.ny_train, args.pml_crop
    spatial_step = args.dh
    nz, nx = vel.shape[1], vel.shape[2]
    valid_num = int(args.valid_rate * nvel) + 1
    
    # source_coords 已注释 — 硬编码坐标与实际数据不匹配，待后续从数据中自动检测
    # source_coords = [
    #     [pml_crop//2 + 1, pml_crop//2 + 5], [pml_crop//2 + 1, pml_crop//2 + 20], [pml_crop//2 + 1, pml_crop//2 + 35],
    #     [pml_crop//2 + 1, pml_crop//2 + 50], [pml_crop//2 + 1, pml_crop//2 + 65]
    # ]
    loc_list = args.source_list

    # --- 核心处理逻辑封装（支持多震源数据拼接） ---
    def process_split(indices_by_source):
        vel_list, u_list, u0_list, labels_list = [], [], [], []
        freq_list, source_coord_list = [], []

        # 遍历所有被激活的震源
        for loci in loc_list:
            indices = indices_by_source[loci]
            if len(indices) == 0:
                continue

            # 1. 每个 (velocity/frequency, source) 对都是一个独立候选样本。
            #    单震源时这与原有逻辑完全等价；多震源时索引可因 source 而异。
            base_vel = vel[indices, :, :].unsqueeze(1)
            vel_list.append(base_vel)

            # 2. 提取对应震源的物理场数据 [count, 2, NZ, NX]
            u_current = UU_loc[loci][indices, :, :, :]
            u0_current = UU0_loc[loci][indices, :, :, :]

            # 3. 计算标签残差
            labels_current = u_current - u0_current

            u_list.append(u_current)
            u0_list.append(u0_current)
            labels_list.append(labels_current)

            # 4. 对应于该 source 的独立 frequency/velocity 索引
            if freq is not None:
                freq_list.append(freq[indices])
            if source_coord_loc is not None:
                source_coord_list.append(source_coord_loc[loci][indices])

        # 沿 Batch 维度 (dim=0) 拼接所有震源的数据
        # 最终的 Batch Size = count * len(loc_list)
        vel_out = torch.cat(vel_list, dim=0)
        u_out = torch.cat(u_list, dim=0)
        u0_out = torch.cat(u0_list, dim=0)
        labels_out = torch.cat(labels_list, dim=0)
        freq_out = torch.cat(freq_list, dim=0) if freq_list else None
        source_coord_out = torch.cat(source_coord_list, dim=0) if source_coord_list else None

        return vel_out, u_out, u0_out, labels_out, freq_out, source_coord_out

    # 2. 划分 train/valid 的样本索引。
    # 单震源保留旧语义：nvel_train 是该唯一 source 的样本数。
    if len(loc_list) == 1:
        if nvel > vel.shape[0]:
            raise ValueError(
                f'nvel_train={nvel} 超过单震源可用候选数 {vel.shape[0]}'
            )
        idx = np.random.choice(vel.shape[0], nvel, replace=False)
        selected_idx_set = set(idx.tolist())
        remaining_idx = np.asarray(
            [i for i in range(len(vel)) if i not in selected_idx_set], dtype=np.int64
        )
        if valid_num > len(remaining_idx):
            raise ValueError(
                f'验证样本数 {valid_num} 超过剩余单震源候选数 {len(remaining_idx)}'
            )
        # 保持旧版验证集的确定性顺序，避免改变既有单震源训练划分。
        train_indices_by_source = {loc_list[0]: idx}
        valid_indices_by_source = {loc_list[0]: remaining_idx[:valid_num]}
    else:
        n_sources = len(loc_list)
        n_candidates_per_source = vel.shape[0]
        n_candidates_total = n_sources * n_candidates_per_source
        if nvel > n_candidates_total:
            raise ValueError(
                f'nvel_train={nvel} 超过多震源可用候选数 {n_candidates_total} '
                f'({n_candidates_per_source} × {n_sources})'
            )

        def balanced_counts(total):
            """将 total 个样本均匀分配到激活震源；差值最多为 1。"""
            quotient, remainder = divmod(total, n_sources)
            counts = np.full(n_sources, quotient, dtype=np.int64)
            # 余数随机分配给不同 source，避免每次总由低编号震源获得额外样本。
            if remainder:
                counts[np.random.permutation(n_sources)[:remainder]] += 1
            return counts

        train_counts = balanced_counts(nvel)
        valid_counts = balanced_counts(valid_num)
        train_indices_by_source, valid_indices_by_source = {}, {}
        for source_pos, loci in enumerate(loc_list):
            required = int(train_counts[source_pos] + valid_counts[source_pos])
            if required > n_candidates_per_source:
                raise ValueError(
                    f'source={loci} 需要 {required} 个互异 train/valid 样本，'
                    f'但仅有 {n_candidates_per_source} 个候选'
                )
            # 同一 source 内 train/valid 无交集；不同 source 对应不同物理样本，可使用相同
            # velocity/frequency 索引。
            source_permutation = np.random.permutation(n_candidates_per_source)
            train_end = int(train_counts[source_pos])
            valid_end = train_end + int(valid_counts[source_pos])
            train_indices_by_source[loci] = source_permutation[:train_end]
            valid_indices_by_source[loci] = source_permutation[train_end:valid_end]

        train_desc = ', '.join(
            f'src{loci}={len(train_indices_by_source[loci])}' for loci in loc_list
        )
        valid_desc = ', '.join(
            f'src{loci}={len(valid_indices_by_source[loci])}' for loci in loc_list
        )
        print(
            f'[多震源均衡抽样] 候选总数={n_candidates_per_source} '
            f'(速度/频率) × {n_sources} (震源) = {n_candidates_total}; '
            f'训练总数={nvel}: {train_desc}; 验证总数={valid_num}: {valid_desc}'
        )

    # --- 3. 生成训练集数据 ---
    vel_train, UU_loc_train, UU0_train, labels, freq_train, source_coord_train = process_split(
        train_indices_by_source
    )
    
    # 训练集的坐标点 y_train（所有样本共享一份以节省内存）
    if getattr(args, 'sampling_mode', 'full_grid') == 'halton':
        # Halton 准随机采样
        total_pts = nz * nx
        ratio = getattr(args, 'halton_sample_ratio', 0.2)
        num_pts = max(1, int(total_pts * ratio))
        halton_indices = Halton_Sample(
            (nz, nx), num_pts, seed=int(getattr(args, 'data_seed', 1))
        )
        z_idx = torch.tensor([p[0] for p in halton_indices], dtype=torch.float32)
        x_idx = torch.tensor([p[1] for p in halton_indices], dtype=torch.float32)
        y_train = torch.stack([z_idx, x_idx], dim=1) * spatial_step
        print(f'[Halton] 采样点数: {num_pts}, y_train shape: {y_train.shape}')
    else:
        # 全网格采样（默认）
        x_c = torch.arange(0, nx)
        z_c = torch.arange(0, nz)
        grid_z, grid_x = torch.meshgrid(z_c, x_c, indexing='ij')
        y_train = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1).float() * spatial_step

    # --- 4. 生成验证集数据 ---
    vel_valid, UU_loc_valid, UU0_valid, labels_valid, freq_valid, source_coord_valid = process_split(
        valid_indices_by_source
    )
    y_valid = y_train  # 验证集坐标点与训练集保持一致

    return (
        vel_train, UU_loc_train, UU0_train, y_train, labels, freq_train, source_coord_train,
        vel_valid, UU_loc_valid, UU0_valid, y_valid, labels_valid, freq_valid, source_coord_valid
    )

def Test_data_single(args, loc_idx, vel_single, UU_loc_single, UU0_loc_single):
    """
    专门用于加载测试模型（如 Marmousi），支持自适应单震源或多震源并发输入。
    """
    # 1. 基本参数准备
    spatial_step = args.dh
    nz, nx = vel_single.shape[-2], vel_single.shape[-1]
    
    # 确保输入是 Tensor
    if isinstance(vel_single, np.ndarray):
        vel_single = torch.from_numpy(vel_single)
    if isinstance(UU_loc_single, np.ndarray):
        UU_loc_single = torch.from_numpy(UU_loc_single)
    if isinstance(UU0_loc_single, np.ndarray):
        UU0_loc_single = torch.from_numpy(UU0_loc_single)
        
    # ==========================================
    # 核心修改点：自适应维度推导
    # ==========================================
    # 2. 提取波场数据，利用 -1 自动推导包含的震源数量 (num_sources)
    u_current = UU_loc_single.view(-1, 2, nz, nx).float()
    u0_current = UU0_loc_single.view(-1, 2, nz, nx).float()
    
    num_sources = u_current.shape[0]  # 获取实际传进来的震源数量 (例如 1 或 5)
    
    # 3. 速度模型处理
    # 速度模型本身只有 1 个，为了能放进 Dataloader，必须复制成 num_sources 份与波场对齐
    vel_test = vel_single.view(1, 1, nz, nx).expand(num_sources, -1, -1, -1).float()
    
    # 4. 计算标签 (UU - UU0) -> [num_sources, 2, NZ, NX]
    labels_test = u_current - u0_current
    
    # ==========================================
    # 坐标点采样
    # ==========================================
    # 生成全空间网格
    x_c = torch.arange(0, nx)
    z_c = torch.arange(0, nz)
    grid_z, grid_x = torch.meshgrid(z_c, x_c, indexing='ij') 
    
    # 展平成 [NZ*NX, 2]
    y_grid = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1).float() * spatial_step
    
    # 同样地，网格坐标也需要扩展成 num_sources 份，变成 [num_sources, NZ*NX, 2]
    y_test = y_grid.unsqueeze(0).expand(num_sources, -1, -1)
    
    # 返回参数顺序与原函数保持一致
    return vel_test, u_current, u0_current, y_test, labels_test

def load_tensor_from_npy(base_path, filename):
    """通用的数据读取鲁棒接口"""
    path = os.path.join(base_path, filename)
    if not os.path.exists(path):
        path = filename  # Fallback
    return torch.tensor(np.load(path), dtype=torch.float32)

def prepare_training_dataloaders(args, device):
    """
    仅处理用于模型训练和内部验证的数据流
    """
    # 1. 基础训练数据读取
    vel_original = load_tensor_from_npy(args.load_path, args.vel_filename)
    UU0_original = load_tensor_from_npy(args.load_path, args.backgroundfield_filename)
    UU_original = load_tensor_from_npy(args.load_path, args.wavefield_filename)

    # 精确震源网格坐标：[N_base_model, N_source, 2]，坐标顺序为 [x, z]。
    source_grid_coords = None
    source_coord_filename = getattr(args, 'source_coord_filename', None)
    if source_coord_filename:
        source_coord_path = os.path.join(args.load_path, source_coord_filename)
        if not os.path.exists(source_coord_path):
            source_coord_path = source_coord_filename
        if os.path.exists(source_coord_path):
            source_grid_coords = torch.tensor(
                np.load(source_coord_path), dtype=torch.float32
            )
            if source_grid_coords.ndim != 3 or source_grid_coords.shape[-1] != 2:
                raise ValueError(
                    'source_grid_coords 必须为 [N_base_model, N_source, 2]，'
                    f'实际为 {tuple(source_grid_coords.shape)}'
                )
            print(
                f'已加载显式震源坐标: {source_coord_filename}, '
                f'shape: {tuple(source_grid_coords.shape)}, order=[x,z]'
            )
        elif bool(getattr(args, 'require_source_coords', False)):
            raise FileNotFoundError(f'显式震源坐标文件不存在: {source_coord_path}')
        else:
            print(f'⚠️ 震源坐标文件不存在: {source_coord_path}，将从 UU0 回退推断')

    # 加载频率数据（若文件存在）
    freq_filename = getattr(args, 'freq_filename', None)
    if freq_filename:
        freq_path = os.path.join(args.load_path, freq_filename)
        if os.path.exists(freq_path):
            freq = torch.tensor(np.load(freq_path), dtype=torch.float32)
            print(f'已加载频率数据: {freq_filename}, shape: {freq.shape}, 唯一值: {freq.unique().tolist()}')
        else:
            print(f'⚠️ 频率文件不存在: {freq_path}，将使用默认频率')
            freq = None
    else:
        freq = None

    # 2. PML 边界处理
    if args.pml:
        pml_crop = args.pml_crop
        # 根据边界类型确定切片范围
        if args.boundary_type == 'free_surface':
            z_slice = slice(0, -pml_crop or None)  # crop=0 保留完整网格
        else:  # 'full_pml'
            z_slice = slice(pml_crop, -pml_crop or None)

        x_slice = slice(pml_crop, -pml_crop or None)

        vel = vel_original[:, z_slice, x_slice]
        UU0 = UU0_original[:, :, z_slice, x_slice]
        UU = UU_original[:, :, z_slice, x_slice]

        # 更新 args.nz 和 args.nx 为切片后的实际尺寸
        args.nz = vel.shape[1]  # 实际的 z 维度
        args.nx = vel.shape[2]  # 实际的 x 维度
    else:
        vel, UU0, UU = vel_original, UU0_original, UU_original
        # 无 PML 时，使用原始数据尺寸
        args.nz = vel.shape[1]
        args.nx = vel.shape[2]

    # 3. 多频段数据重排: [freq0_all, freq1_all, ...] → [src0_all, src1_all, ...]
    n_freq = getattr(args, 'n_freq_ranges', 1)
    if vel.shape[0] % n_freq != 0:
        raise ValueError(f'velocity 数量 {vel.shape[0]} 不能被 n_freq_ranges={n_freq} 整除')
    n_vel_per_freq = vel.shape[0] // n_freq
    if UU0.shape[0] % vel.shape[0] != 0:
        raise ValueError(
            f'背景场数量 {UU0.shape[0]} 不是速度/频率数量 {vel.shape[0]} 的整数倍'
        )
    n_src = UU0.shape[0] // vel.shape[0]
    if UU.shape[0] != UU0.shape[0]:
        raise ValueError(f'wavefield/background 数量不一致: {UU.shape[0]} vs {UU0.shape[0]}')

    source_coord_loc = None
    if source_grid_coords is not None:
        expected_source_shape = (n_vel_per_freq, n_src, 2)
        if tuple(source_grid_coords.shape) != expected_source_shape:
            raise ValueError(
                f'震源坐标 shape {tuple(source_grid_coords.shape)} 与数据布局要求 '
                f'{expected_source_shape} 不一致'
            )
        # 保存网格包含 PML；网络输入已裁剪。转换 [x,z] grid → [z,x] metres。
        left_crop = args.pml_crop if args.pml else 0
        top_crop = (
            args.pml_crop
            if args.pml and getattr(args, 'boundary_type', 'full_pml') != 'free_surface'
            else 0
        )
        source_z = (source_grid_coords[..., 1] - top_crop) * float(args.dh)
        source_x = (source_grid_coords[..., 0] - left_crop) * float(args.dh)
        source_coords_zx_m = torch.stack([source_z, source_x], dim=-1)
        max_z = (vel.shape[-2] - 1) * float(args.dh)
        max_x = (vel.shape[-1] - 1) * float(args.dh)
        if (
            torch.any(source_coords_zx_m[..., 0] < 0)
            or torch.any(source_coords_zx_m[..., 0] > max_z)
            or torch.any(source_coords_zx_m[..., 1] < 0)
            or torch.any(source_coords_zx_m[..., 1] > max_x)
        ):
            raise ValueError(
                '裁剪后的震源坐标超出网络网格；请检查 pml_crop、dh 和坐标文件'
            )
        # 每个 source block 内顺序严格为 [stage, model]，与重排后的场一致。
        source_coord_loc = [
            source_coords_zx_m[:, source_id, :].repeat(n_freq, 1)
            for source_id in range(n_src)
        ]

    if n_freq > 1:
        # reshape: (n_freq * n_src * n_vel_per_freq, C, H, W) → (n_freq, n_src, n_vel_per_freq, C, H, W)
        UU0 = UU0.reshape(n_freq, n_src, n_vel_per_freq, *UU0.shape[1:])
        UU0 = UU0.permute(1, 0, 2, *range(3, UU0.dim())).contiguous().reshape(n_src * vel.shape[0], *UU0.shape[3:])
        UU = UU.reshape(n_freq, n_src, n_vel_per_freq, *UU.shape[1:])
        UU = UU.permute(1, 0, 2, *range(3, UU.dim())).contiguous().reshape(n_src * vel.shape[0], *UU.shape[3:])

    # 4. 震源拆分与训练集生成
    UU_loc = [UU[loc * len(vel) : (loc + 1) * len(vel), ...] for loc in range(n_src)]
    UU0_loc = [UU0[loc * len(vel) : (loc + 1) * len(vel), ...] for loc in range(n_src)]
    
    np.random.seed(1)
    vel_train, UU_loc_train, UU0_train, y_train, labels_train, freq_train, source_coord_train, \
    vel_valid, UU_loc_valid, UU0_valid, y_valid, labels_valid, freq_valid, source_coord_valid = Training_data(
        args, vel, UU_loc, UU0_loc, freq, source_coord_loc=source_coord_loc
    )
    print('vel_train', vel_train.shape)
    # 4. 物理场归一化
    vel_train = vel_train / 1000.
    vel_valid = vel_valid / 1000.

    # 绘图示例: 选取高频样本 (freq 值最大的样本)
    if freq_train is not None:
        hf_test_idx = freq_train.argmax().item()
        hf_pred_idx = freq_valid.argmax().item()
    else:
        hf_test_idx = len(vel_train) - 1
        hf_pred_idx = len(vel_valid) - 1

    vel_pred, UU0_pred, labels_pred = vel_valid[hf_pred_idx:hf_pred_idx+1], UU0_valid[hf_pred_idx:hf_pred_idx+1], labels_valid[hf_pred_idx:hf_pred_idx+1]
    vel_test, UU0_test, labels_test = vel_train[hf_test_idx:hf_test_idx+1], UU0_train[hf_test_idx:hf_test_idx+1], labels_train[hf_test_idx:hf_test_idx+1]
    source_coord_pred = source_coord_valid[hf_pred_idx:hf_pred_idx+1] if source_coord_valid is not None else None
    source_coord_test = source_coord_train[hf_test_idx:hf_test_idx+1] if source_coord_train is not None else None

    # 5. 生成坐标网格点
    x_coords, z_coords = torch.arange(0, args.nx), torch.arange(0, args.nz)
    grid_z, grid_x = torch.meshgrid(z_coords, x_coords, indexing='ij')
    points = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1)
    y_pred = points.float() * args.dh
    y_test = y_pred

    # 6. 构建 DataLoader
    pin_mem = False
    num_workers = 0

    train_ds = make_conditioned_dataset(
        vel_train, UU0_train, labels_train, freq_train, source_coord_train
    )
    valid_ds = make_conditioned_dataset(
        vel_valid, UU0_valid, labels_valid, freq_valid, source_coord_valid
    )

    train_loaders = {
        "train": DataLoader(train_ds,
                            batch_size=args.batch_size_v, shuffle=True, drop_last=True,
                            pin_memory=pin_mem, num_workers=num_workers,
                            generator=torch.Generator().manual_seed(
                                int(getattr(args, 'data_seed', 1))
                            )),
        "train_y": DataLoader(TensorDataset(y_train),
                              batch_size=args.batch_size, shuffle=True, pin_memory=pin_mem,
                              num_workers=num_workers,
                              generator=torch.Generator().manual_seed(
                                  int(getattr(args, 'ddp_coordinate_seed', 20260412))
                              )),
        "valid": DataLoader(valid_ds,
                            batch_size=args.valid_batch_size_v, shuffle=True, drop_last=True,
                            pin_memory=pin_mem, num_workers=num_workers,
                            generator=torch.Generator().manual_seed(
                                int(getattr(args, 'data_seed', 1)) + 100_000
                            )),
        "valid_y": DataLoader(TensorDataset(y_valid),
                              batch_size=args.valid_batch_size, shuffle=True, pin_memory=pin_mem,
                              num_workers=num_workers,
                              generator=torch.Generator().manual_seed(
                                  int(getattr(args, 'ddp_coordinate_seed', 20260412)) + 100_000
                              )),
        "pred": DataLoader(TensorDataset(y_pred), batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(TensorDataset(y_test), batch_size=args.batch_size, shuffle=False)
    }

    plot_data = {
        "vel_pred": vel_pred, "UU0_pred": UU0_pred, "labels_pred": labels_pred,
        "vel_test": vel_test, "UU0_test": UU0_test, "labels_test": labels_test,
        "y_pred": y_pred,
        "freq_train": freq_train[hf_test_idx:hf_test_idx+1] if freq_train is not None else None,
        "freq_valid": freq_valid[hf_pred_idx:hf_pred_idx+1] if freq_valid is not None else None,
        "has_freq": freq_train is not None,
        "source_coord_train": source_coord_test,
        "source_coord_valid": source_coord_pred,
        "has_source_coords": source_coord_train is not None,
    }
    
    return train_loaders, plot_data

def prepare_external_val_dataset(args, prefix, loc_target, y_pred_grid):
    """
    通用接口：用于动态加载和处理单个外部验证集（如 Marmousi, BP 等）
    """
    # 1. 读取特定前缀的数据 (文件名中尺寸需与实际数据匹配)
    grid_suffix = f'{args.nz}_{args.nx}_n1.npy'
    vel_ext = load_tensor_from_npy(args.load_path, f'{prefix}velocity_data_{grid_suffix}')
    UU0_ext = load_tensor_from_npy(args.load_path, f'{prefix}backgroundfield_data_freq5_1source_{grid_suffix}')
    UU_ext = load_tensor_from_npy(args.load_path, f'{prefix}wavefield_data_freq5_5sources_{grid_suffix}')

    # 2. PML 边界处理
    if args.pml:
        pml_crop = args.pml_crop
        # 根据边界类型确定切片范围
        if args.boundary_type == 'free_surface':
            z_slice = slice(0, -pml_crop or None)
        else:  # 'full_pml'
            z_slice = slice(pml_crop, -pml_crop or None)
        x_slice = slice(pml_crop, -pml_crop or None)

        vel_ext = vel_ext.unsqueeze(0)[:, z_slice, x_slice]
        UU0_ext = UU0_ext[:, :, z_slice, x_slice]
        UU_ext = UU_ext[:, :, z_slice, x_slice]
    else:
        vel_ext = vel_ext.unsqueeze(0)

    # 3. 截取目标震源位置
    num_samples = len(vel_ext) 
    # m_uu_single = UU_ext[loc_target * num_samples : (loc_target + 1) * num_samples]
    # m_uu0_single = UU0_ext[loc_target * num_samples : (loc_target + 1) * num_samples]
    # 兼容 loc_target 是列表（多震源）或整数（单震源）的情况
    if isinstance(loc_target, list):
        m_uu_single = torch.cat([UU_ext[loc * num_samples : (loc + 1) * num_samples] for loc in loc_target], dim=0)
        m_uu0_single = torch.cat([UU0_ext[loc * num_samples : (loc + 1) * num_samples] for loc in loc_target], dim=0)
        
        # 注意：如果你的速度场 v_ext 只有一份（形状如 [1, 1, Z, X]），
        # 在拼接成 Dataloader 之前，可能需要将其按震源数量复制对齐：
        # v_ext = v_ext.repeat(len(loc_target), 1, 1, 1) 
    else:
        m_uu_single = UU_ext[loc_target * num_samples : (loc_target + 1) * num_samples]
        m_uu0_single = UU0_ext[loc_target * num_samples : (loc_target + 1) * num_samples]

    # 4. 生成测试格式数据
    v_test, u_test, u0_test, y_test, lab_test = Test_data_single(
        args, loc_target, vel_ext, m_uu_single, m_uu0_single
    )

    # 5. 归一化对齐训练逻辑
    v_test = v_test / 1000.0

    # 6. 生成专用的 DataLoader 和绘图数据字典
    ext_loader = DataLoader(TensorDataset(y_pred_grid), batch_size=args.batch_size, shuffle=False)
    
    ext_plot_data = {
        "v_test": v_test, 
        "u0_test": u0_test, 
        "lab_test": lab_test
    }
    
    print(f'External dataset [{prefix}] ready: vel_shape {v_test.shape}')
    return ext_loader, ext_plot_data

# def extract_single_model_multi_source(args, vel_set, UU0_set, labels_set, target_model_idx=0):
#     """
#     从按震源顺序拼接的数据集中，提取出【指定索引】的一个速度模型及其对应的多震源波场数据。
    
#     Args:
#         args: 全局参数，需包含 args.source_list (例如 [0, 1, 2, 3, 4])
#         vel_set: 训练或验证集的速度场 Tensor [base_count * num_sources, 1, Z, X]
#         UU0_set: 背景波场 Tensor [base_count * num_sources, 2, Z, X]
#         labels_set: 真实标签 Tensor [base_count * num_sources, 2, Z, X]
#         base_count: 该集合基础速度模型的数量 (train集为 nvel_train, valid集为 valid_num)
#         target_model_idx: 指定要提取第几个速度模型 (0 <= target_model_idx < base_count)
        
#     Returns:
#         model_data_pack (dict): 包含画图所需的 vel, UU0_list, labels_list
#     """
#     num_sources = len(args.source_list)
#     base_count = num_sources // 5
#     # 防止索引越界
#     if target_model_idx >= base_count or target_model_idx < 0:
#         raise ValueError(f"指定的索引 {target_model_idx} 超出范围，该集合只有 {base_count} 个基础模型。")
    
#     # 1. 提取指定索引的速度模型 (扩展出 batch=1 的维度 [1, 1, Z, X])
#     vel_single = vel_set[target_model_idx].unsqueeze(0)
    
#     UU0_list = []
#     labels_list = []
    
#     # 2. 跨块跳跃提取该模型在所有震源下的波场数据
#     for s in range(num_sources):
#         # 核心索引公式：指定模型索引 + 震源索引 * 基础模型数量
#         target_idx = target_model_idx + s * base_count
        
#         UU0_list.append(UU0_set[target_idx].unsqueeze(0))      # [1, 2, Z, X]
#         labels_list.append(labels_set[target_idx].unsqueeze(0)) # [1, Z, X, 2] 或其它对应维度
        
#     # 3. 组装返回
#     model_data_pack = {
#         "vel": vel_single,
#         "UU0_list": UU0_list,
#         "labels_list": labels_list
#     }
    
#     return model_data_pack
