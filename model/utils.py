

from Labconfig import *
from torch.optim.lr_scheduler import _LRScheduler

def load_or_save_checkpoint(vit, optimizer, net_opt, DEVICE):
    """
    Load a checkpoint if it exists, otherwise save the initial checkpoint.

    Args:
        vit (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        net_opt (str): The path to the checkpoint file.
        isa (int): The current iteration number.
        DEVICE (torch.device): The device (CPU or GPU) to map the loaded checkpoint to.
    """
    # Load checkpoint if exists
    if os.path.exists(net_opt):
        chk = torch.load(net_opt, map_location=DEVICE)
        print('chk - keys -', chk.keys())
        vit.load_state_dict(chk['vit'])
        optimizer.load_state_dict(chk['opt'])

        # Move the optimizer's state tensors to the specified device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)

        #isa = chk['iteration'] + 1  # Resume from next iteration
        print(f"Resuming training from iteration isa")
    else:

        #seed_everything(42)
        # Save initial checkpoint
        torch.save({
            'vit': vit.state_dict(),
            'opt': optimizer.state_dict(),
            #'iteration': isa
        }, net_opt)
        print(f"Starting new training from iteration ")
class WarmupScheduler(_LRScheduler):
    """
    学习率热身调度器：先在warmup_epochs内从初始学习率（warmup_start_lr）线性/余弦增长到基础学习率（base_lr），
    之后可衔接其他调度器（如StepLR、CosineAnnealingLR等）。
    
    Args:
        optimizer: 优化器
        warmup_epochs: 热身轮次（如5、10）
        base_lr: 热身结束后的目标学习率（即优化器初始lr）
        warmup_start_lr: 热身起始学习率（通常为base_lr的1/10或1/100，如0.0001）
        warmup_strategy: 热身策略，可选"linear"（线性增长）或"cosine"（余弦增长）
        after_scheduler: 热身结束后使用的调度器（如StepLR，可选）
    """
    def __init__(self, optimizer, warmup_epochs, base_lr, warmup_start_lr=0.0, 
                 warmup_strategy="linear", after_scheduler=None):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_strategy = warmup_strategy
        self.after_scheduler = after_scheduler  # 热身结束后的调度器
        self.current_epoch = 0  # 记录当前轮次
        super().__init__(optimizer)
        
    def get_lr(self):
        # 热身阶段：调整学习率
        if self.current_epoch < self.warmup_epochs:
            if self.warmup_strategy == "linear":
                # 线性增长：lr = start_lr + (base_lr - start_lr) * (current_epoch / warmup_epochs)
                progress = self.current_epoch / self.warmup_epochs
                lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * progress
            elif self.warmup_strategy == "cosine":
                # 余弦增长：lr = start_lr + (base_lr - start_lr) * (1 - cos(pi * current_epoch / (2*warmup_epochs))) / 2
                progress = self.current_epoch / self.warmup_epochs
                lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * \
                     (1 - torch.cos(torch.tensor(torch.pi * progress / 2))) / 2
            else:
                raise ValueError(f"不支持的热身策略：{self.warmup_strategy}，可选'linear'或'cosine'")
            return [lr for _ in self.base_lrs]
        # 热身结束后：使用后续调度器
        else:
            if self.after_scheduler is not None:
                # 后续调度器的 epoch 从 0 开始计数（相对于热身结束）
                self.after_scheduler.step(self.current_epoch - self.warmup_epochs)
                return self.after_scheduler.get_lr()
            else:
                # 若无后续调度器，保持 base_lr
                return [self.base_lr for _ in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.current_epoch + 1
        self.current_epoch = epoch
        # 调用父类方法更新学习率
        super().step()

def count_parameters(model):
    """
    计算 PyTorch 模型的总参数数量
    Args:
        model: PyTorch 模型（nn.Module 子类）
    Returns:
        total_params: 模型总参数数量（int）
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# ==========================================
# 单机多卡并行训练工具函数 (Single-Machine Multi-GPU)
# ==========================================

def setup_distributed(rank, world_size, device_id=None, timeout_minutes=10):
    """
    初始化单机多卡分布式训练环境

    Args:
        rank: 当前进程的 rank (0, 1, 2, ...)
        world_size: 总进程数 (等于 GPU 数量)
        device_id: 当前 rank 使用的物理 GPU 编号；省略时兼容 rank -> GPU 映射
    """
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')

    # rank 是通信编号，device_id 才是物理 CUDA 设备编号。
    if device_id is None:
        device_id = rank
    torch.cuda.set_device(device_id)
    from datetime import timedelta
    # Rank 0 performs validation/plots while other ranks wait at epoch end.
    # Timeout ceiling only; no delay is inserted into normal training steps.
    dist.init_process_group('nccl', rank=rank, world_size=world_size,
                            timeout=timedelta(minutes=timeout_minutes))

    print(
        f"[Rank {rank}] 单机多卡训练环境初始化完成 | "
        f"physical GPU: {device_id} | world_size={world_size} | NCCL timeout={timeout_minutes}min"
    )


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("分布式训练环境已清理")


def get_available_gpus(min_memory_mb=10240, require_count=None):
    """Return process-local CUDA indices, matched to nvidia-smi by device UUID.

    Fail closed: never estimate free memory by subtracting this process's
    allocations from total memory, which ignores other users' processes.
    """
    import subprocess
    available = []
    try:
        if not torch.cuda.is_available():
            return []
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=uuid,index,memory.total,memory.free',
             '--format=csv,nounits,noheader'], encoding='utf-8', timeout=15)
        def canonical(uuid):
            return str(uuid).strip().lower().removeprefix('gpu-')
        physical = {}
        for line in output.strip().splitlines():
            uuid, index, total, free = [v.strip() for v in line.split(',')]
            physical[canonical(uuid)] = (int(index), int(total), int(free))
        for logical in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(logical)
            identity = canonical(prop.uuid)
            if identity not in physical:
                raise RuntimeError(f'Cannot map visible CUDA device {logical} UUID={identity}')
            index, total, free = physical[identity]
            print(f'CUDA {logical} -> physical GPU {index}, UUID={prop.uuid}, free={free} MiB')
            if free >= min_memory_mb:
                available.append(logical)
    except Exception as error:
        print(f'GPU detection failed; refusing device selection: {error}')
        return []
    if require_count is not None and len(available) < require_count:
        return []
    return available


def select_gpus_for_training(args):
    """
    根据配置参数自动选择 GPU

    Args:
        args: 配置参数对象，应包含:
            - use_parallel: 是否启用并行
            - num_gpus: 使用的 GPU 数量
            - min_gpu_memory: GPU 最小内存 (MB)
            - device: 单 GPU 模式下的设备编号

    Returns:
        tuple: (selected_gpus, world_size)
    """
    use_parallel = getattr(args, 'use_parallel', False)
    num_gpus = getattr(args, 'num_gpus', 1)
    min_gpu_memory = getattr(args, 'min_gpu_memory', 10240)

    if not use_parallel:
        # 单 GPU 模式
        selected_gpus = [args.device] if hasattr(args, 'device') else [0]
        print(f"单 GPU 模式: 使用 GPU {selected_gpus[0]}")
        return selected_gpus, 1

    # 并行模式: 自动检测可用 GPU
    available_gpus = get_available_gpus(min_memory_mb=min_gpu_memory)

    if len(available_gpus) < num_gpus:
        raise RuntimeError(f'可用 GPU 不足 ({len(available_gpus)} < {num_gpus})；拒绝单卡回退')

    # 选择指定数量的 GPU
    selected_gpus = available_gpus[:num_gpus]
    world_size = len(selected_gpus)

    print(f"单机多卡模式: 使用 {world_size} 个 GPU -> {selected_gpus}")
    return selected_gpus, world_size


def wrap_model_for_distributed(model, device_id):
    """
    将模型包装为分布式并行模型 (DDP)

    Args:
        model: 原始模型
        device_id: 当前进程使用的物理 GPU 编号

    Returns:
        DDP 包装后的模型
    """
    from model.distributed_complex import wrap_complex_safe_ddp

    model = model.to(device_id)
    model = wrap_complex_safe_ddp(model, device_id)

    return model


def is_main_process(rank=None):
    """
    检查当前是否为主进程

    Args:
        rank: 进程 rank，若为 None 则自动获取

    Returns:
        bool: 是否为主进程
    """
    if rank is None:
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            return True

    return rank == 0


def reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    """
    跨进程归约张量 (All-Reduce)

    Args:
        tensor: 待归约的张量
        op: 归约操作 (SUM, AVG, MAX, MIN)

    Returns:
        归约后的张量
    """
    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=op)
    if op == dist.ReduceOp.SUM:
        tensor /= dist.get_world_size()

    return tensor


def gather_tensor(tensor):
    """
    跨进程收集张量 (All-Gather)

    Args:
        tensor: 待收集的张量

    Returns:
        拼接后的张量
    """
    if not dist.is_initialized():
        return tensor

    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)

    return torch.cat(gathered, dim=0)


# 保留旧版函数以兼容
def setup(rank, world_size):
    """兼容旧版 setup 函数"""
    setup_distributed(rank, world_size)


def cleanup():
    """兼容旧版 cleanup 函数"""
    cleanup_distributed()

def calculate_regression_metrics(pred, true):
    """
    计算回归任务核心指标，包括 Relative L2 Loss
    Args:
        pred: 预测值数组 (nz, nx)
    	true: 真实值数组 (nz, nx)
    Returns:
        metrics: 字典，包含 mse, mae, r2, relative_l2
    """
    assert pred.shape == true.shape, "预测值和真实值维度必须一致"
    
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    
    # 1. 基础指标
    mse = np.mean((pred_flat - true_flat) ** 2)
    mae = np.mean(np.abs(pred_flat - true_flat))
    
    # 2. R² (决定系数)
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # 3. Relative L2 Loss (相对 L2 误差)
    # 公式: ||pred - true||2 / ||true||2
    # np.linalg.norm 默认计算 L2 范数
    norm_diff = np.linalg.norm(true_flat - pred_flat)
    norm_true = np.linalg.norm(true_flat)
    
    # 避免分母为 0
    relative_l2 = (norm_diff / norm_true) if norm_true != 0 else 0.0
    
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "relative_l2": relative_l2,
    }
def Halton_Sample(array_shape, num_samples, seed=None):
    """Return reproducible scrambled Halton grid indices when ``seed`` is set."""
    sampler = qmc.Halton(d=2, scramble=True, seed=seed)
    samples = sampler.random(num_samples)
    rows = (samples[:, 0] * array_shape[0]).astype(int)
    cols = (samples[:, 1] * array_shape[1]).astype(int)
    rows = np.clip(rows, 0, array_shape[0]-1)
    cols = np.clip(cols, 0, array_shape[1]-1)
    return list(zip(rows, cols))


def generate_random_points(batch_size, n_pts, range_max=None):
    if range_max is None:
        range_max = 72.0  # 默认值，建议使用实际的网格尺寸
    # 生成 [B, n_pts, 2] 的随机坐标，范围在 [0, 1] 之间
    y_random = torch.rand(batch_size, n_pts, 2)

    # 缩放到 [0, range_max] 范围
    y_random = y_random * range_max
    
    # 记得开启梯度跟踪，否则无法计算 PDE Loss 中的导数
    y_random.requires_grad_(True)
    
    return y_random # 输出形状 [B, n_pts, 2]
    
def get_local_physical_features(vel, y, eps=1e-3):
    # 关键点：在采样前，将坐标 y 从计算图中分离 (detach)
    # 这告诉 PyTorch：我只需要在坐标 y 处的值，不需要计算通过 grid_sample 对 y 的梯度
    y_detached = y.detach() 
    
    grid_center = y_detached.unsqueeze(2) 
    
    # 执行采样（现在不需要担心二阶导数问题了）
    v_center = F.grid_sample(vel, grid_center, align_corners=True).squeeze(-1)
    b1 = vel.shape[0]        # Batch size (B_v)
    b_pts = y.shape[1]
    # 同样地采样周围点...
    grid_right = grid_center + torch.tensor([eps, 0], device=y.device)
    grid_up    = grid_center + torch.tensor([0, eps], device=y.device)
    
    v_right = F.grid_sample(vel, grid_right.clamp(-1, 1), align_corners=True).squeeze(-1)
    v_up    = F.grid_sample(vel, grid_up.clamp(-1, 1), align_corners=True).squeeze(-1)
    
    dv_dx = (v_right - v_center) / eps
    dv_dz = (v_up - v_center) / eps
    
    vel_mean = torch.mean(vel, dim=[2, 3]).view(b1,-1) # [b1, 1, 1, 1]
    vel_std = torch.std(vel, dim=[2, 3]).view(b1,-1)  # [b1, 1, 1, 1]    
    vel_feature = vel_std * 1.5 + vel_mean
    vel_feature = vel_feature.unsqueeze(1).expand(-1,b_pts,-1).permute(0, 2, 1)
    physical_context = torch.cat([vel_feature,dv_dx, dv_dz], dim=1).permute(0, 2, 1)
    # print('physical_context', physical_context.shape)
    # 返回的是 detach 后的特征，它作为 Trunk 的“静态输入”
    return physical_context
    
def generate_weight(vel, type="energy"):
    """
    vel: [B, 1, H, W] 或 [B, N_pts, 1]
    """
    if type == "energy":
        # 权重与 k^2 成正比，即与 1/v^2 成正比
        w = 1.0 / (vel**2 + 1e-6)
    elif type == "gradient":
        # 关注速度变化剧烈的区域（界面）
        # 仅适用于 2D 图像格式的 vel
        grad_x = torch.abs(vel[:, :, 1:, :] - vel[:, :, :-1, :])
        # ... 适当 padding 并取模长
        w = torch.nn.functional.pad(grad_x, (0, 0, 1, 0)) 
    
    # 归一化，保证积分均值为 1，避免 Loss 量级爆炸
    w = w / torch.mean(w, dim=(1, 2, 3) if w.dim()==4 else 1, keepdim=True)
    return w
def get_helmholtz_spatial_weights(coords, velocity, omega, source_loc, alpha=0.1, sigma=0.5):
    """
    计算方案一：物理尺度变权重项 lambda(x)
    
    Args:
        coords (torch.Tensor): 空间坐标, 形状为 (N, dim), 如 (N, 2) 或 (N, 3)
        velocity (torch.Tensor): 对应坐标处的速度模型值, 形状为 (N, 1)
        omega (float): 角频率 (2 * pi * f)
        source_loc (torch.Tensor): 震源位置坐标, 形状为 (1, dim)
        alpha (float): 波数平衡系数，调节高波数区域的惩罚强度
        sigma (float): 震源加权的高斯标准差，控制“能量锚点”的范围
        
    Returns:
        weights (torch.Tensor): 空间权重矩阵, 形状为 (N, 1)
    """
    
    # 1. 计算局部波数 k = omega / v
    k = omega / (velocity + 1e-6)
    k_sq = k**2
    
    # 2. 归一化波数项 (防止不同量级速度模型导致权重失效)
    # 使用均值或最大值归一化，使 k_bar^2 在 1 附近波动
    k_sq_bar = k_sq / torch.mean(k_sq)
    
    # 3. 计算算子平衡项: 1 / (1 + alpha * k_bar^2)
    # 逻辑：在 k 较大的区域（低速区，波长短，震荡剧烈），降低权重以防梯度爆炸
    operator_term = 1.0 / (1.0 + alpha * k_sq_bar)
    # print('k', operator_term.shape)
    source_loc = source_loc.unsqueeze(0).expand(coords.shape[0],coords.shape[1],-1)
    # 4. 计算源项锚点（高斯加权）
    # dist_sq: 每个点到震源的欧式距离平方
    dist_sq = torch.sum((coords - source_loc)**2, dim=-1)
    source_anchor = torch.exp(-dist_sq / (2 * sigma**2))
    # print('source_anchor', source_anchor.shape)
    # 5. 组合权重
    # 我们希望：在源附近权重极大（强制收敛），在全域受算子平衡项调节
    # 这里加 1e-2 是为了保证全域仍有基本的 PDE 约束，不至于震源之外完全不更新
    weights = operator_term * (source_anchor + 1e-2)

    return weights

def build_epoch_velocity_gradient_prob(
    train_loader,
    device,
    eps=1e-8,
    use_max_mix=False,
    mean_weight=0.7,
    max_weight=0.3,
):
    """
    基于整个训练集 velocity model 构造 epoch-level 结构采样概率图。

    Args:
        train_loader: 训练数据 DataLoader，每个 batch 至少包含 vel 在 index 0
        device: 计算设备
        eps: 数值稳定项
        use_max_mix: True 时 score = mean_weight*mean + max_weight*max
        mean_weight: mean+max 混合的 mean 权重
        max_weight: mean+max 混合的 max 权重

    Returns:
        prob: [Z*X] 扁平概率分布，归一化后 sum=1
        score: [Z, X] 原始分数图（用于可视化诊断）
    """
    score_sum = None
    score_max_global = None
    count = 0

    with torch.no_grad():
        for batch_data in train_loader:
            vel_batch = batch_data[0]
            vel = vel_batch.to(device)  # [B, 1, Z, X]
            B, _, Z, X = vel.shape

            grad_z = vel[:, :, 2:, 1:-1] - vel[:, :, :-2, 1:-1]
            grad_x = vel[:, :, 1:-1, 2:] - vel[:, :, 1:-1, :-2]

            grad_mag = torch.sqrt(grad_z ** 2 + grad_x ** 2 + eps)
            grad_mag = F.pad(grad_mag, (1, 1, 1, 1), mode='replicate').squeeze(1)  # [B, Z, X]

            batch_sum = grad_mag.sum(dim=0)  # [Z, X]

            if score_sum is None:
                score_sum = torch.zeros_like(batch_sum)
                score_max_global = torch.zeros_like(batch_sum)

            score_sum += batch_sum
            count += B

            if use_max_mix:
                batch_max = grad_mag.max(dim=0).values
                score_max_global = torch.maximum(score_max_global, batch_max)

    score_mean = score_sum / max(count, 1)

    if use_max_mix:
        score = mean_weight * score_mean + max_weight * score_max_global
    else:
        score = score_mean

    score = torch.clamp(score, min=0.0)
    prob = score.reshape(-1)
    prob = prob / (prob.sum() + eps)

    return prob, score

def sample_shared_y_ran_from_epoch_prob(
    prob,
    args,
    num_pts=900,
    structure_ratio=0.60,
    surface_ratio=0.20,
    uniform_ratio=0.20,
    source_ratio=0.0,
    source_coords=None,
    surface_depth_grids=5,
    source_r_min_grids=1.5,
    source_r_max_grids=8.0,
    replacement=True,
):
    """
    从 epoch-level 概率图中采样一组共享 y_ran。
    """
    device = prob.device
    nz, nx, dh = args.nz, args.nx, args.dh

    max_z = nz * dh
    max_x = nx * dh

    if source_coords is None or source_ratio <= 0:
        uniform_ratio = uniform_ratio + source_ratio
        source_ratio = 0.0

    num_structure = int(num_pts * structure_ratio)
    num_surface = int(num_pts * surface_ratio)
    num_source = int(num_pts * source_ratio)
    num_uniform = num_pts - num_structure - num_surface - num_source

    y_parts = []

    # 1. epoch-level structure points
    if num_structure > 0:
        sampled_indices = torch.multinomial(
            prob,
            num_samples=num_structure,
            replacement=replacement,
        )

        z_idx = sampled_indices // nx
        x_idx = sampled_indices % nx

        z = z_idx.float() * dh + torch.rand(num_structure, device=device) * dh
        x = x_idx.float() * dh + torch.rand(num_structure, device=device) * dh

        z = z.clamp(0.0, max_z)
        x = x.clamp(0.0, max_x)

        y_struct = torch.stack([z, x], dim=-1)
        y_parts.append(y_struct)

    # 2. surface points
    if num_surface > 0:
        surface_depth = surface_depth_grids * dh

        z = torch.rand(num_surface, device=device) * surface_depth
        x = torch.rand(num_surface, device=device) * max_x

        z = z.clamp(0.0, max_z)
        x = x.clamp(0.0, max_x)

        y_surface = torch.stack([z, x], dim=-1)
        y_parts.append(y_surface)

    # 3. source-near points
    if num_source > 0 and source_coords is not None:
        source_coords = source_coords.to(device)

        src_id = torch.randint(
            low=0,
            high=source_coords.shape[0],
            size=(num_source,),
            device=device,
        )
        src = source_coords[src_id]

        theta = 2.0 * torch.pi * torch.rand(num_source, device=device)

        r_min = source_r_min_grids * dh
        r_max = source_r_max_grids * dh
        r = r_min + (r_max - r_min) * torch.rand(num_source, device=device)

        z = src[:, 0] + r * torch.cos(theta)
        x = src[:, 1] + r * torch.sin(theta)

        z = z.clamp(0.0, max_z)
        x = x.clamp(0.0, max_x)

        y_source = torch.stack([z, x], dim=-1)
        y_parts.append(y_source)

    # 4. uniform points
    if num_uniform > 0:
        z = torch.rand(num_uniform, device=device) * max_z
        x = torch.rand(num_uniform, device=device) * max_x

        z = z.clamp(0.0, max_z)
        x = x.clamp(0.0, max_x)

        y_uniform = torch.stack([z, x], dim=-1)
        y_parts.append(y_uniform)

    y_shared = torch.cat(y_parts, dim=0)

    if y_shared.shape[0] > num_pts:
        y_shared = y_shared[:num_pts]
    elif y_shared.shape[0] < num_pts:
        extra = num_pts - y_shared.shape[0]
        z = torch.rand(extra, device=device) * max_z
        x = torch.rand(extra, device=device) * max_x
        y_extra = torch.stack([z, x], dim=-1)
        y_shared = torch.cat([y_shared, y_extra], dim=0)

    return y_shared
