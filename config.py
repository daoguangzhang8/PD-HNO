class Args:
    # ==========================================
    # 1. 路径与文件配置 (Paths & I/O)
    # ==========================================
    load_path = '/home/sharedata/zdg'         # 数据集加载根目录
    weights_save_path = '/home/ShareData/zdg' # 模型权重保存根目录
    # 方案2新增 Trunk 输入维度，必须使用独立输出目录从头训练。
    save_doc = 'output_branch1m24_local_combined_multiscale204080_gpu3_bs32_resume_from200'
    filename = 'PI_DeepONet_pde'              # 保存的模型前缀名称

    # 训练数据文件名
    # vel_filename = 'velocity_data_70_70_n1.npy'
    # backgroundfield_filename = 'backgroundfield_data_freq5_1source_70_70_n1.npy'
    # wavefield_filename = 'wavefield_data_freq5_5sources_70_70_n1.npy'

    # 自由表面训练数据
    # vel_filename = 'freesurface_velocity_freq3to20_5sources_160_180_pml20_n1.npy'
    # backgroundfield_filename = 'freesurface_backgroundfield_freq3to20_5sources_160_180_pml20_n1.npy'
    # wavefield_filename = 'freesurface_wavefield_freq3to20_5sources_160_180_pml20_n1.npy'
    # freq_filename = 'freesurface_freq_used_5sources_160_180_pml20_n1.npy'

    # 新生成的 CurveFlatA + StyleA 多频、多随机震源训练数据。
    training_data_dir = 'openfwi_curveflat_style_cpu'
    vel_filename = f'{training_data_dir}/freesurface_full_5sources_velocity.npy'
    backgroundfield_filename = f'{training_data_dir}/freesurface_full_5sources_background.npy'
    wavefield_filename = f'{training_data_dir}/freesurface_full_5sources_wavefield.npy'
    freq_filename = f'{training_data_dir}/freesurface_full_5sources_freq_used.npy'
    source_coord_filename = f'{training_data_dir}/source_grid_coords.npy'
    require_source_coords = True                    # 方案2训练必须使用生成时保存的精确震源位置
    default_freq = 5.0                                                 # 默认频率 (Hz)，当 freq 文件不存在或 freq_batch=None 时使用

    # 外部泛化测试集配置 (支持动态扩展)
    ext_val_datasets = {
        # 'Marmousi': {'prefix': 'marmousi_', 'loc_target': 2},
        # 'BP': {'prefix': 'bp_', 'loc_target': 0},
    }

    # 标签来源配置
    use_fno_as_label = False                  # True: 使用 FNO 预测作为软标签 | False: 使用真实标签
    fno_weights_path = ''                     # FNO 预训练权重路径 (当 use_fno_as_label=True 时需要指定)

    # ==========================================
    # 2. 硬件与设备配置 (Hardware & Device)
    # ==========================================
    # 启动时通过 CUDA_VISIBLE_DEVICES=2 只暴露物理 GPU 2，因此进程内编号为 0。
    device = 0                                 # 单 GPU 模式: 进程内 GPU 编号
    use_parallel = False                      # 是否启用多 GPU 并行训练 (True: 多GPU | False: 单GPU)
    num_gpus = 2                              # 使用的 GPU 数量
    min_gpu_memory = 19 * 1024                # GPU 最小可用内存 (MB)，低于此值的 GPU 不会被使用

    # ==========================================
    # 3. 物理网格与边界条件 (Physical Grid & PML)
    # ==========================================
    dh = 20                                     # 空间网格间距 (m)，物理坐标 = 网格索引 * dh
    nx = 140                                   # 物理模型 x 方向网格数 (不含外延 PML)
    nz = 140                                   # 物理模型 z 方向网格数 (不含外延 PML)
    pml = True                                # 是否启用 PML (Perfectly Matched Layer, 完美匹配层) 吸收边界
    pml_total = 20                            # PML 吸收层的总网格厚度
    pml_crop = 15                              # 训练时裁剪/忽略的 PML 网格数
    pml_active = pml_total - pml_crop         # 剩余参与训练的 PML 网格数
    pde_generation_dh = 10.0                 # 标签生成细网格间距，用于半网格 PML 界面映射

    # 边界类型配置
    # 'full_pml': 四边 PML 吸收边界，原始数据 90×90 → 网络输入 72×72
    # 'free_surface': 顶部自由表面 + 其他三边 PML，原始数据 80×90 → 网络输入 71×72
    boundary_type = 'free_surface'            # 根据实际数据选择
    n_freq_ranges = 3                          # 合并数据来源的频段数量 (单频数据设为1)

    # ==========================================
    # 4. 数据集与采样 (Dataset & Sampling)
    # ==========================================
    # 多震源时是总训练样本数，而非“每个震源各 nvel_train 个”。
    # 当前候选空间为 2000 velocity × 3 frequency × 5 source = 30000，
    # dataloader 会在五个 source 间均分（差值最多为 1）。
    nvel_train = 7200
    source_list = [0, 1, 2, 3, 4]              # 训练数据中启用的震源编号列表

    # 空间点采样模式
    sampling_mode = 'halton'                  # 'full_grid': 全网格采样 | 'halton': Halton 准随机采样
    halton_sample_ratio = 0.5                   # Halton 采样比例（仅在 sampling_mode='halton' 时生效，如 0.2 表示采样 20% 的网格点）

    # 批处理配置
    batch_size = 1500                          # Trunk Net 坐标采样批次大小 (num_sample)
    batch_size_v = 32                          # Branch Net 速度场/背景场批次大小 (Batch_v)
    ny_train = int(nz * nx * halton_sample_ratio)  # 训练集空间采样点总数 (由网格尺寸和采样比例自动计算)
    accumulation_steps = 2                    # 梯度累加步数 (用于等效增大 batch size，节约显存)

    # 验证集
    valid_rate = 0.01                          # 验证集划分比例
    valid_batch_size = 350                    # 验证集坐标采样批次大小
    valid_batch_size_v = 6                    # 验证集速度场批次大小

    # ==========================================
    # 5. 训练超参数 (Training Hyperparameters)
    # ==========================================
    # 全新训练的总 epoch：实际运行 0--1000。
    NIter = 1000 + 1
    lr = 2 * 1e-4                             # 初始基础学习率
    weight_decay = 1e-4                       # 优化器权重衰减 (L2 正则化)系数

    # DDP 与单卡结果对齐配置
    ddp_scale_lr = False                       # True: DDP 学习率按 GPU 数线性放大 | False: 保持与单卡相同
    ddp_split_batch_size_v = True              # True: DDP 每卡 batch_size_v=batch_size_v/num_gpus，使全局 batch 接近单卡
    data_seed = 1                              # train/valid 划分及 Halton 基础点集
    ddp_seed = 20260410                        # DDP sampler 与各 rank 随机流的基础种子
    ddp_coordinate_seed = 20260412             # 各 rank 共享的坐标 batch 排列种子
    ddp_assert_coordinate_alignment = True     # 首轮运行时检查各 rank 坐标完全对齐

    # 学习率调度器
    scheduler_type = 'cosine'               # 'plateau': ReduceLROnPlateau | 'cosine': CosineAnnealingWarmRestarts
    
    use_warmup = True                        # 是否启用 warmup 预热
    warmup_epochs = 100                       # 学习率热身 (Warmup) 的 epoch 数

    # ReduceLROnPlateau 参数
    factor = 0.9                              # 学习率衰减因子
    patience = 20                             # 触发衰减的容忍 epoch 数量
    min_lr = 1e-5                             # 允许的最小学习率

    # CosineAnnealingWarmRestarts 参数
    cosine_T_0 = 901                         # 接替训练时恢复 checkpoint 中原有 scheduler 状态
    cosine_T_mult = 2                         # 后续周期倍增系数 (T_0, T_0*2, T_0*4, ...)
    cosine_eta_min = 1e-5                     # 余弦退火最低学习率

    # ==========================================
    # 6. 损失函数 (Loss Function)
    # ==========================================
    a = 1                                     # 数据拟合项 (Data Loss) 权重
    b = 1                                     # PDE 物理残差项 (PDE Loss) 权重
    c = 0                                     # 正则化项 (Regularization Loss) 权重
    d = 1                                   # 包络损失项 (Envelope Loss) 权重（MSE 形式，仅在标签点上计算）

    # 动态权重调整
    if_adjust = False                          # 是否在训练过程中动态调整 Loss 权重
    adjust_from = 2000                        # 从第几个 epoch 开始动态调整
    adjust_every = 1000                       # 每隔多少个 epoch 调整一次权重
    adjust_speed = 1.1                        # 权重衰减/增长的速度因子

    # ==========================================
    # 7. 训练控制与保存 (Training Control & Checkpoints)
    # ==========================================
    # 全新训练：不加载任何模型、优化器或调度器状态。
    if_load_model = True
    resume_checkpoint = '/home/zhangdaoguang/Code/PIDeeponet_old_from6004/output_branch1m24_local_combined_multiscale204080_gpu3_bs32_run_20260815/PI_DeepONet_pde_PI_model_200epoch_weights_145.pth'
    resume_epoch = 200                       # 从 checkpoint 的全局 epoch 200 接替训练
    resume_restore_optimizer = True
    resume_restore_scheduler = True

    # 新数据与新 Trunk 必须重新标定 loss normalizer；首个 epoch 自动估计。
    normalizer_calibration_only = False
    normalizer_reference_output_dir = ''
    normalizer_reference_epoch = 0
    fixed_data_normalizer = None             # 接替训练使用 checkpoint 内保存的 normalizer
    fixed_pde_normalizer = None
    fixed_env_normalizer = 1.0

    validate_every = 200                      # 每隔多少个 epoch 执行一次模型验证
    save_fig_every = 100                       # 每隔多少个 epoch 保存一次验证/测试可视化图片
    save_model_every = 100                    # 每隔多少个 epoch 保存一次模型权重文件

    # 验证时同步执行 Marmousi 未见频率测试
    enable_marmousi_eval = True               # True: 每次正式验证后预测 Marmousi 10/20 Hz
    marmousi_eval_frequencies = [10, 20]       # 训练频率集合中未出现的频率
    marmousi_eval_sources = [2]                # 测试震源位置
    marmousi_eval_batch_size = 1600            # 推理坐标 batch size
    marmousi_eval_data_dir = '/home/sharedata/zdg/external_test'
    marmousi_eval_output_dir = 'marmousi_unseen_frequency'

    # 5 Hz 保留震源测试：保存网格 [x,z] =
    # [50,0], [70,0], [90,0], [110,0], [130,0]，均已从训练候选中删除。
    enable_marmousi_unseen_source_eval = True
    marmousi_unseen_source_eval_frequencies = [5]
    marmousi_unseen_source_eval_sources = [0, 1, 2, 3, 4]
    marmousi_unseen_source_eval_data_dir = (
        '/home/sharedata/zdg/'
        'marmousi_unseen_sources_5hz_alpha0_160x180_x50_70_90_110_130_v3'
    )
    marmousi_unseen_source_eval_output_dir = 'marmousi_5hz_unseen_sources'

    # ==========================================
    # 8. 微调与域适应 (Fine-Tuning)
    # ==========================================
    if_finetune = False                       # 是否在外部复杂地层 (如 Marmousi) 上进行微调评估
    ft_NIter = 1000                             # 微调阶段的迭代步数
    ft_lr = 2e-5                              # 微调阶段的专属学习率
    ft_a = 0.2                                # 微调阶段的数据 Loss 权重
    ft_b = 1                                  # 微调阶段的 PDE Loss 权重
    ft_c = 1                                  # 微调阶段的正则化 Loss 权重

    # ==========================================
    # 9. Positional Encoding (位置编码)
    # ==========================================
    pe_max_scale = 12.0                          # PE 最高频率尺度 (原值=6.0; 建议扫描: 8.0, 10.0, 12.0)
    use_trunk_freq_encoding = True               # 是否将频率显式编码后拼接到 Trunk 坐标输入
    trunk_freq_embed_dim = 8                     # 频率编码输出维度；Trunk 输入维度=16+该值
    trunk_freq_num_bands = 3                     # 频率 Fourier band 数: [f, sin/cos(1f), sin/cos(2f), ...]
    trunk_freq_norm_hz = 25.0                    # 频率归一化参考值，覆盖当前 3-25Hz 训练范围
    frequency_min_hz = 3.0                       # 训练频率下界（用于 FiLM 平滑正则边界处理）
    frequency_max_hz = 25.0                      # 训练频率上界
    use_wavenumber_encoding = True               # 恢复8维绝对波数编码 cat
    wavenumber_num_bands = 2                     # 波数谐波 [1, 2]，每个 band 产生 z/x 的 sin/cos
    wavenumber_init_scale = 1.0                  # 波数编码初始增益
    use_relative_source_encoding = True          # 恢复16维 source embedding + 12维相对相位 cat
    source_radius_mode = 'legacy'                # 'squared': 实验性 r²/L 编码，无坐标开平方；改变特征语义
    relative_source_embed_dim = 16               # 相对位置 Fourier 特征经 MLP 后的维度
    relative_source_num_bands = 3                # Δz/Δx/r 的归一化 Fourier band 数
    relative_source_phase_num_bands = 2          # kΔz/kΔx/kr 的物理相位谐波数
    use_joint_source_frequency_fusion = False    # 关闭24维联合门控方案，恢复原60维 Trunk
    joint_source_embed_dim = 16
    joint_source_phase_num_bands = 2
    joint_source_init_gate = -2.0
    use_film_frequency_conditioning = True       # 显式频率条件注入所有 FiLM 层
    film_freq_hidden_dim = 32                    # 频率条件 MLP 隐层宽度
    film_freq_init_gate = -2.0                   # sigmoid 后约 0.119，避免初期压过 Branch 条件
    film_smooth_delta_hz = 1.0                   # FiLM 二阶差分的频率间隔
    film_smooth_weight = 1e-2                    # FiLM 频率平滑正则权重

    # 连续频率 PDE 约束（不增加有标签数据）
    enable_continuous_frequency_pde = True       # 总开关；False 时完全保持原训练流程
    continuous_pde_weight = 1.0                  # 连续频率 PDE loss 在总 loss 中的权重
    continuous_pde_start_epoch = 1               # epoch 0 仍用于估计原 PDE 归一化系数
    continuous_pde_batch_ratio = 0.25             # 每个 velocity batch 用于连续频率的样本比例
    continuous_pde_num_points = 256              # 每次连续频率 PDE 的空间配点数
    continuous_frequency_min_hz = 3.0            # 连续均匀采样的频率下界
    continuous_frequency_max_hz = 25.0           # 连续均匀采样的频率上界

    # 解析背景场 U0 = q * i/4 * H_0^(2)(k r)。下列参数对应
    # /home/zhangdaoguang/Code/data/modeling.py:
    #   source RHS = 2*0.25 = 0.5, h = 40/TIMES = 10 m (TIMES=4)
    # 连续 Dirac 源强度 q = RHS*h^2 = 50。
    analytic_u0_source_rhs_amplitude = 2.0 * 0.25
    analytic_u0_generation_dh = 10.0
    analytic_u0_background_velocity = 1500.0
    analytic_u0_min_radius = analytic_u0_generation_dh / 2.0
    analytic_u0_source_search_depth = 4          # 仅在存储 U0 顶部搜索震源位置
    pde_attenuation_alpha = 0.0                   # 与新生成的无耗散(alpha=0, Q=∞)标签严格一致
    # 波数 PE 的 freq 和 c_ref 由训练数据动态提供:
    #   freq → freq_batch (来自 freesurface_freq_used_*.npy)
    #   c_ref → vel.mean() (每个速度模型的平均速度)

    # ==========================================
    # 10. 网络架构 (Network Architecture)
    # ==========================================
    in_channels = 2                           # 波场相关输入通道数 (如复数波场的实部、虚部)
    in_channels_vel = 1                       # 速度模型输入通道数 (1个通道代表速度 v)
    branch2_type = 'fno'                   # Branch2 架构: 'fno'(原始FNO) | 'resnet'(残差CNN) | 'conv'(简单CNN)
    input_shape_trunk = (batch_size, in_channels, 1, 2)       # Trunk Net (评估坐标) 的输入形状占位
    input_shape_branch1 = (batch_size, in_channels_vel, nz, nx) # Branch Net 1 (速度场) 输入形状占位
    input_shape_branch2 = (batch_size, in_channels, nz, nx)     # Branch Net 2 (背景场/震源) 输入形状占位

    # ==========================================
    # 11. Sobol 采样配置 (Sobol Sampling)
    # ==========================================
    sampling_strategy = 'original'               # 'original': 双层循环+Halton | 'sobol': 单层循环+Sobol
    sobol_points_per_epoch = 10000              # Sobol 模式: 每 epoch 采样点数 (所有 velocity batch 共享)
    valid_sobol_points = 300                  # Sobol 模式: 验证集每 epoch 采样点数

    # ==========================================
    # 12. 三阶段渐进训练 (Staged Curriculum Training)
    # ==========================================
    staged_training = False                   # 总开关，False 则使用原始单阶段训练

    # 每阶段使用独立数据集，文件名通过 freq_range 替换基础文件名中的 'freq3to20' 得到
    # 例如基础文件名含 'freq3to20' → Stage 0 替换为 'freq3to11'
    stages = [
        {
            'name': 'low_freq',
            'freq_range': '3to11',               # 替换基础文件名中的 'freq3to20'
            'freq_min': 3.0, 'freq_max': 11.0,   # 信息标签，用于日志打印
            'NIter': 6001,
            'lr': 2e-4,                           # 从头训练，完整 LR
            'warmup_epochs': 100,
            'a': 1, 'b': 1, 'c': 0,
            'replay_stages': [],
            'replay_ratio': 0.2,                  # replay 数据保留比例 (1.0=全部, 0.5=随机抽取50%)
            'data_dir': '/home/sharedata/zdg/multifreq_selected/freq_3to11',
        },
        {
            'name': 'mid_freq',
            'freq_range': '12to18',
            'freq_min': 12.0, 'freq_max': 18.0,
            'NIter': 2001,
            'lr': 1e-4,                           # 课程学习，适度降低
            'warmup_epochs': 50,
            'a': 1, 'b': 1, 'c': 0,
            'replay_stages': [0],
            'replay_ratio': 0.2,
            'data_dir': '/home/sharedata/zdg/multifreq_selected/freq_12to18',
        },
        {
            'name': 'high_freq',
            'freq_range': '18to25',
            'freq_min': 18.0, 'freq_max': 25.0,
            'NIter': 1001,
            'lr': 5e-5,                           # 高频更难，进一步降低
            'warmup_epochs': 50,
            'a': 1, 'b': 1, 'c': 0,
            'replay_stages': [0, 1],
            'replay_ratio': 0.2,
            'data_dir': '/home/sharedata/zdg/multifreq_selected/freq_18to25',
        },
    ]

    # ==========================================
    # 13. y_ran Epoch-Level 共享采样 (Epoch Shared Sampling)
    # ==========================================
    use_y_ran = False                          # False: 不使用自由点 | True: 使用 y_ran 自由点参与 PDE 计算

    use_epoch_shared_y_ran = True              # True: 使用 epoch 级共享采样 | False: 使用原始 per-model 采样

    y_ran_num_pts = 300                        # y_ran 采样点总数
    y_ran_structure_ratio = 0.60               # epoch-structure 采样点比例
    y_ran_surface_ratio = 0.20                 # 表层采样点比例
    y_ran_uniform_ratio = 0.20                 # 均匀采样点比例
    y_ran_source_ratio = 0.0                   # 震源附近采样点比例 (实验A不使用)

    y_ran_surface_depth_grids = 5              # 表层深度（网格点数）
    y_ran_use_max_mix = False                  # True: score = mean_weight*mean + max_weight*max | False: 纯 mean
    y_ran_mean_weight = 0.7                    # mean+max 混合时的 mean 权重
    y_ran_max_weight = 0.3                     # mean+max 混合时的 max 权重

    # 概率图更新频率: 1=每epoch, >1=每N个epoch, 0=只计算一次并缓存
    y_ran_prob_update_every = 0
