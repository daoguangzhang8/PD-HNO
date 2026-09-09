import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Labconfig import *
from model.utils import *
from model.dataloader import *
from model.net_module import *
from model.pde_ops import pml_profiles, source_radial_coordinate

class Pi_DeepONet(nn.Module):
    """
    物理信息神经网络 (PI-DeepONet)
    结合 FNO (Branch) 与 FiLM/Fourier 编码 (Trunk)，用于求解带 PML 边界条件的 Helmholtz 方程。
    """
    def __init__(self, args):
        super().__init__()
        self.args = args  # 保存args以便其他方法使用
        self.device = args.device
        self.feat_dim = 256  # 特征维度，必须能被注意力头数整除

        # --- 超参数 ---
        input_shape_branch1 = args.input_shape_branch1
        input_shape_branch2 = args.input_shape_branch2
        self.b2 = args.batch_size
        
        # --- 编码器与特征提取 ---
        self.pos_encoder = PositionalEncoding(embed_dim=4, max_scale=args.pe_max_scale)
        self.use_wavenumber_encoding = bool(getattr(args, 'use_wavenumber_encoding', False))
        self.wavenumber_num_bands = int(getattr(args, 'wavenumber_num_bands', 2))
        self.kpe = WavenumberPE(embed_dim=self.wavenumber_num_bands)
        # kpe_alpha 仅在启用波数编码时作为可学习参数；关闭时注册为 None，避免
        # DDP(find_unused_parameters=False) 因存在未使用参数而报错。
        if self.use_wavenumber_encoding:
            self.kpe_alpha = nn.Parameter(torch.tensor(
                float(getattr(args, 'wavenumber_init_scale', 1.0))
            ))
        else:
            self.register_parameter('kpe_alpha', None)
        # self.fencoder = FourierFeatureEncoder(input_dim=2, mapping_size=self.feat_dim)  # 未使用，已注释

        # --- 网络分支 (Branch) ---
        self.branch1 = nn.Sequential(
            FNO2d(input_shape_branch1[1], self.feat_dim, modes1=24, modes2=24, width=32),
        )

        branch2_type = getattr(args, 'branch2_type', 'fno')
        if branch2_type == 'resnet':
            self.branch2 = ResNetBranch2d(input_shape_branch2[1], self.feat_dim, base_width=64)
        elif branch2_type == 'conv':
            self.branch2 = ConvBranch2d(input_shape_branch2[1], self.feat_dim)
        else:
            self.branch2 = nn.Sequential(
                FNO2d(input_shape_branch2[1], self.feat_dim, modes1=32, modes2=32, width=32),
            )
        
        # --- 注意力与特征融合 ---
        self.channel_attention1 = ChannelAttention(self.feat_dim, reduction=8)
        self.channel_attention2 = ChannelAttention(self.feat_dim, reduction=8)
        self.combinedlayer1 = GaussianWeightedLayer(self.feat_dim, dh=args.dh)
        self.combinedlayer2 = GaussianWeightedLayer(self.feat_dim, dh=args.dh)
        self.attengate = AttenGate(use_softmax=True)

        self.smooth_feature_encoder = MultiScaleSmoothBlockEncoder(
            self.feat_dim, self.feat_dim, grid_sizes=(20, 40, 80)
        )

        # --- 主干网络 (Trunk) 与输出层 ---
        self.use_trunk_freq_encoding = bool(getattr(args, 'use_trunk_freq_encoding', False))
        self.trunk_freq_embed_dim = int(getattr(args, 'trunk_freq_embed_dim', 8))
        self.trunk_freq_num_bands = int(getattr(args, 'trunk_freq_num_bands', 3))
        self.use_film_frequency_conditioning = bool(
            getattr(args, 'use_film_frequency_conditioning', False)
        )
        self.use_relative_source_encoding = bool(
            getattr(args, 'use_relative_source_encoding', False)
        )
        self.use_joint_source_frequency_fusion = bool(
            getattr(args, 'use_joint_source_frequency_fusion', False)
        )
        self.joint_source_embed_dim = int(
            getattr(args, 'joint_source_embed_dim', 16)
        )
        self.joint_source_phase_num_bands = int(
            getattr(args, 'joint_source_phase_num_bands', 2)
        )
        self.relative_source_embed_dim = int(
            getattr(args, 'relative_source_embed_dim', 16)
        )
        self.relative_source_num_bands = int(
            getattr(args, 'relative_source_num_bands', 3)
        )
        self.relative_source_phase_num_bands = int(
            getattr(args, 'relative_source_phase_num_bands', 2)
        )
        if self.use_trunk_freq_encoding:
            freq_feature_dim = 1 + 2 * self.trunk_freq_num_bands
            self.trunk_freq_encoder = nn.Sequential(
                nn.Linear(freq_feature_dim, 32),
                nn.GELU(),
                nn.Linear(32, self.trunk_freq_embed_dim),
            )
            trunk_input_dim = 16 + self.trunk_freq_embed_dim
        else:
            self.trunk_freq_encoder = None
            trunk_input_dim = 16

        if self.use_joint_source_frequency_fusion:
            if not self.use_trunk_freq_encoding:
                raise ValueError('联合震源-频率融合要求 use_trunk_freq_encoding=True')
            if self.trunk_freq_embed_dim != 8 or self.joint_source_embed_dim != 16:
                raise ValueError(
                    '当前24维方案要求 position=16、frequency=8、source=16'
                )
            joint_input_dim = 3 + 6 * self.joint_source_phase_num_bands
            self.joint_source_encoder = nn.Sequential(
                nn.Linear(joint_input_dim, 32),
                nn.GELU(),
                nn.Linear(32, self.joint_source_embed_dim),
                nn.LayerNorm(self.joint_source_embed_dim),
            )
            self.joint_frequency_modulator = nn.Linear(
                self.trunk_freq_embed_dim, self.joint_source_embed_dim
            )
            self.joint_position_norm = nn.LayerNorm(16)
            self.joint_source_gate = nn.Parameter(torch.tensor(
                float(getattr(args, 'joint_source_init_gate', -2.0))
            ))
        else:
            self.joint_source_encoder = None
            self.joint_frequency_modulator = None
            self.joint_position_norm = None
            self.register_parameter('joint_source_gate', None)

        if self.use_wavenumber_encoding:
            trunk_input_dim += 4 * self.wavenumber_num_bands

        if self.use_relative_source_encoding:
            relative_feature_dim = 3 * (1 + 2 * self.relative_source_num_bands)
            self.relative_source_encoder = nn.Sequential(
                nn.Linear(relative_feature_dim, 32),
                nn.GELU(),
                nn.Linear(32, self.relative_source_embed_dim),
                nn.LayerNorm(self.relative_source_embed_dim),
            )
            # 每个 band: dz/dx/r 各自 sin+cos，共 6 维。
            trunk_input_dim += (
                self.relative_source_embed_dim
                + 6 * self.relative_source_phase_num_bands
            )
        else:
            self.relative_source_encoder = None

        if self.use_film_frequency_conditioning:
            film_freq_hidden_dim = int(getattr(args, 'film_freq_hidden_dim', 32))
            self.film_freq_encoder = nn.Sequential(
                nn.Linear(1, film_freq_hidden_dim),
                nn.GELU(),
                nn.Linear(film_freq_hidden_dim, self.feat_dim),
                nn.LayerNorm(self.feat_dim),
            )
            self.film_freq_gate = nn.Parameter(torch.tensor(
                float(getattr(args, 'film_freq_init_gate', -2.0))
            ))
        else:
            self.film_freq_encoder = None
            self.register_parameter('film_freq_gate', None)

        self.trunk = FiLMTrunk(input_dim=trunk_input_dim, width=self.feat_dim, branch_feat_dim=self.feat_dim)
        self.final_layer = nn.Linear(self.feat_dim, 2)  # 输出实部和虚部
        
        # --- 损失函数组件 ---
        self.loss_function = nn.MSELoss(reduction='mean')
        # self.loss_function_point = nn.MSELoss(reduction='none')  # 未使用，已注释

        self._init_weights()

    def _prepare_frequency(self, freq_batch, batch_size, device, dtype):
        """返回形状为 [B] 的频率及其归一化值。"""
        if freq_batch is None:
            freq_batch = torch.full(
                (batch_size,),
                float(getattr(self.args, 'default_freq', 5.0)),
                device=device,
                dtype=dtype,
            )
        else:
            freq_batch = freq_batch.to(device=device, dtype=dtype).reshape(-1)
            if freq_batch.numel() == 1 and batch_size > 1:
                freq_batch = freq_batch.expand(batch_size)
            elif freq_batch.numel() != batch_size:
                raise ValueError(
                    f'freq_batch 元素数 {freq_batch.numel()} 与 batch_size={batch_size} 不一致'
                )

        norm_hz = max(float(getattr(self.args, 'trunk_freq_norm_hz', 25.0)), 1e-8)
        return freq_batch, freq_batch / norm_hz

    def _encode_trunk_frequency(self, freq_batch, batch_size, num_points, device, dtype):
        """编码每个样本的频率，并广播到所有 Trunk 查询点。"""
        if self.trunk_freq_encoder is None:
            return None

        _, freq_norm = self._prepare_frequency(
            freq_batch, batch_size, device, dtype
        )
        features = [freq_norm.unsqueeze(-1)]
        for band in range(self.trunk_freq_num_bands):
            phase = (2.0 ** band) * 2.0 * np.pi * freq_norm
            features.extend([torch.sin(phase).unsqueeze(-1), torch.cos(phase).unsqueeze(-1)])

        freq_features = torch.cat(features, dim=-1)
        freq_encoded = self.trunk_freq_encoder(freq_features)
        return freq_encoded.unsqueeze(1).expand(-1, num_points, -1)

    def _encode_film_frequency(self, freq_batch, batch_size, device, dtype):
        """生成显式频率 FiLM 条件；相位快速变化由波数 PE 表达。"""
        if self.film_freq_encoder is None:
            return None
        _, freq_norm = self._prepare_frequency(
            freq_batch, batch_size, device, dtype
        )
        context = self.film_freq_encoder(freq_norm.unsqueeze(-1))
        return torch.sigmoid(self.film_freq_gate) * context

    def _prepare_source_coordinates(self, source_coord_batch, UU0, batch_size, device, dtype):
        """返回网络物理坐标系中的震源位置 ``[B,2]``，顺序为 ``[z,x]``。"""
        if source_coord_batch is None:
            source_coord_batch = self._infer_source_coordinates(UU0)
        source_coord_batch = torch.as_tensor(
            source_coord_batch, device=device, dtype=dtype
        )
        if source_coord_batch.ndim == 1:
            source_coord_batch = source_coord_batch.unsqueeze(0)
        if source_coord_batch.shape[-1] != 2:
            raise ValueError(
                f'source_coord_batch 最后一维必须是 [z,x] 两个坐标，实际为 '
                f'{tuple(source_coord_batch.shape)}'
            )
        source_coord_batch = source_coord_batch.reshape(-1, 2)
        if source_coord_batch.shape[0] == 1 and batch_size > 1:
            source_coord_batch = source_coord_batch.expand(batch_size, -1)
        elif source_coord_batch.shape[0] != batch_size:
            raise ValueError(
                f'source_coord_batch batch={source_coord_batch.shape[0]} 与模型 batch={batch_size} 不一致'
            )
        if not torch.isfinite(source_coord_batch).all():
            raise ValueError('source_coord_batch 包含 NaN/Inf')
        return source_coord_batch

    def _encode_relative_source(self, y, source_coord_batch, freq_batch, vel):
        """编码相对坐标；squared 模式用 r²/L 替代 r（径向相位也随之改变）。"""
        if self.relative_source_encoder is None:
            return None
        batch_size = vel.shape[0]
        source_coord_batch = self._prepare_source_coordinates(
            source_coord_batch, None, batch_size, y.device, y.dtype
        )
        delta = y - source_coord_batch.unsqueeze(1)
        dz, dx = delta[..., 0:1], delta[..., 1:2]
        z_scale = max(float(self.args.dh) * max(vel.shape[-2] - 1, 1), 1e-8)
        x_scale = max(float(self.args.dh) * max(vel.shape[-1] - 1, 1), 1e-8)
        r_scale = float(np.hypot(z_scale, x_scale))
        radius = source_radial_coordinate(
            dz, dx, r_scale, getattr(self.args, 'source_radius_mode', 'legacy'))
        relative_base = torch.cat(
            [dz / z_scale, dx / x_scale, radius / r_scale], dim=-1
        )
        relative_features = [relative_base]
        for band in range(self.relative_source_num_bands):
            phase = (2.0 ** band) * 2.0 * np.pi * relative_base
            relative_features.extend([torch.sin(phase), torch.cos(phase)])
        relative_embedding = self.relative_source_encoder(
            torch.cat(relative_features, dim=-1)
        )

        freq_hz, _ = self._prepare_frequency(
            freq_batch, batch_size, y.device, y.dtype
        )
        c_ref = vel.mean(dim=(2, 3)).squeeze(1).to(dtype=y.dtype).clamp_min(1e-6)
        k_ref = ((2.0 * np.pi * freq_hz * 1e-3) / c_ref).view(-1, 1, 1)
        physical_phases = [k_ref * dz, k_ref * dx, k_ref * radius]
        phase_features = []
        for band in range(self.relative_source_phase_num_bands):
            scale = 2.0 ** band
            for phase in physical_phases:
                phase_features.extend([
                    torch.sin(scale * phase), torch.cos(scale * phase)
                ])
        return torch.cat([relative_embedding, *phase_features], dim=-1)

    def _fuse_joint_source_frequency(self, position_encoded, freq_encoded, y,
                                     source_coord_batch, freq_batch, vel):
        """将震源相对位置融入16维位置编码，不增加 Trunk 输入维度。"""
        if self.joint_source_encoder is None:
            return position_encoded
        batch_size = vel.shape[0]
        source_coord_batch = self._prepare_source_coordinates(
            source_coord_batch, None, batch_size, y.device, y.dtype
        )
        delta = y - source_coord_batch.unsqueeze(1)
        dz, dx = delta[..., 0:1], delta[..., 1:2]
        z_scale = max(float(self.args.dh) * max(vel.shape[-2] - 1, 1), 1e-8)
        x_scale = max(float(self.args.dh) * max(vel.shape[-1] - 1, 1), 1e-8)
        radius = source_radial_coordinate(
            dz, dx, float(np.hypot(z_scale, x_scale)),
            getattr(self.args, 'source_radius_mode', 'legacy'))
        relative_base = torch.cat(
            [dz / z_scale, dx / x_scale, radius / float(np.hypot(z_scale, x_scale))],
            dim=-1,
        )

        freq_hz, _ = self._prepare_frequency(
            freq_batch, batch_size, y.device, y.dtype
        )
        c_ref = vel.mean(dim=(2, 3)).squeeze(1).to(dtype=y.dtype).clamp_min(1e-6)
        k_ref = ((2.0 * np.pi * freq_hz * 1e-3) / c_ref).view(-1, 1, 1)
        physical_phases = [k_ref * dz, k_ref * dx, k_ref * radius]
        joint_features = [relative_base]
        for band in range(self.joint_source_phase_num_bands):
            harmonic = 2.0 ** band
            for phase in physical_phases:
                joint_features.extend([
                    torch.sin(harmonic * phase),
                    torch.cos(harmonic * phase),
                ])
        source_feature = self.joint_source_encoder(
            torch.cat(joint_features, dim=-1)
        )
        frequency_modulation = 1.0 + torch.tanh(
            self.joint_frequency_modulator(freq_encoded)
        )
        gated_source = (
            torch.sigmoid(self.joint_source_gate)
            * source_feature
            * frequency_modulation
        )
        return self.joint_position_norm(position_encoded + gated_source)

    def film_frequency_smoothness_loss(self, freq_batch, batch_size, device, dtype):
        """约束频率 FiLM 条件在相邻频率间近似线性，不平滑最终复波场。"""
        if self.film_freq_encoder is None:
            return torch.zeros((), device=device, dtype=dtype)

        freq_hz, _ = self._prepare_frequency(
            freq_batch, batch_size, device, dtype
        )
        delta_hz = max(float(getattr(self.args, 'film_smooth_delta_hz', 1.0)), 1e-6)
        freq_min = float(getattr(self.args, 'frequency_min_hz', 3.0))
        freq_max = float(getattr(self.args, 'frequency_max_hz', 25.0))
        if freq_max - freq_min < 2.0 * delta_hz:
            return torch.zeros((), device=device, dtype=dtype)

        # 边界频率改用其最近的内部中心，始终保持对称的 +/- delta。
        center_hz = freq_hz.clamp(freq_min + delta_hz, freq_max - delta_hz)
        norm_hz = max(float(getattr(self.args, 'trunk_freq_norm_hz', 25.0)), 1e-8)
        center = center_hz.unsqueeze(-1) / norm_hz
        offset = delta_hz / norm_hz

        context_minus = self.film_freq_encoder(center - offset)
        context_center = self.film_freq_encoder(center)
        context_plus = self.film_freq_encoder(center + offset)
        second_difference = context_plus - 2.0 * context_center + context_minus

        # 相对尺度避免编码器整体幅值改变时正则强度漂移。
        reference_energy = context_center.detach().square().mean().clamp_min(1e-6)
        return second_difference.square().mean() / reference_energy

    def _init_weights(self):
        """初始化网络权重，Sin 激活层使用 SIREN 初始化 (Sitzmann et al., 2020)"""
        # 收集 Sin 激活前的 Linear 层
        # SIREN: W ~ U(-sqrt(6/fan_in), sqrt(6/fan_in))，保持 sin 激活各层方差稳定
        sin_linears = set()
        for module in self.modules():
            if isinstance(module, FiLMTrunk):
                # fc1/fc2/fc3 后接 Sin，fc4 是输出层无激活
                sin_linears.update([module.fc1, module.fc2, module.fc3])
            elif isinstance(module, GaussianWeightedLayer):
                # compress: [Linear→Sin, Linear→Sin, Linear(输出)]
                sin_linears.update([module.compress[0], module.compress[2]])

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m in sin_linears:
                    bound = np.sqrt(6.0 / m.in_features)
                    nn.init.uniform_(m.weight, -bound, bound)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)

    def _forward_impl(self, vel, y, UU0, freq_batch=None, source_coord_batch=None,
                      trunk_coordinate_scale=None):
        """
        前向传播
        Args:
            vel: 速度场模型 [B_v, C, Z, X]
            y: 查询坐标点 [B_v, B_pts, 2]
            UU0: 背景波场 [B_v, 2, Z, X]
            freq_batch: 每个样本的频率 [B_v] (Hz), 来自数据文件
            trunk_coordinate_scale: 可选的每样本坐标缩放 ``α``。仅作用于
                Trunk 的位置/波数编码；Branch 对速度和背景场仍使用原物理网格。
                这是 Huang & Alkhalifah 单参考频率方法在固定网格 DeepONet
                中的安全实现：不把坐标缩放错误地用于局部介质特征采样。
        Returns:
            outputs: 预测波场残差 (实部和虚部) [B_v, B_pts, 2]
        """
        # --- 1. 坐标预处理与 Trunk 特征提取 (Query) ---
        Z_dim = vel.shape[-2]
        X_dim = vel.shape[-1]

        z_normalized = 2 * y[:, :, 0:1] / (self.args.dh * (Z_dim - 1)) - 1
        x_normalized = 2 * y[:, :, 1:2] / (self.args.dh * (X_dim - 1)) - 1
        # ``y_normalized`` 与 Branch 特征图处于同一物理网格，不能缩放。
        y_normalized = torch.cat([z_normalized, x_normalized], dim=2)  # [B_v, B_pts, 2]

        if trunk_coordinate_scale is None:
            coordinate_scale = torch.ones(
                vel.shape[0], 1, 1, device=y.device, dtype=y.dtype
            )
        else:
            coordinate_scale = torch.as_tensor(
                trunk_coordinate_scale, device=y.device, dtype=y.dtype
            ).reshape(-1)
            if coordinate_scale.numel() == 1 and vel.shape[0] > 1:
                coordinate_scale = coordinate_scale.expand(vel.shape[0])
            if coordinate_scale.numel() != vel.shape[0]:
                raise ValueError(
                    'trunk_coordinate_scale 元素数必须为 1 或与 velocity batch 一致'
                )
            if torch.any(coordinate_scale <= 0):
                raise ValueError('trunk_coordinate_scale 必须为正数')
            coordinate_scale = coordinate_scale.view(-1, 1, 1)

        y_trunk = y * coordinate_scale
        z_trunk_normalized = (
            2 * y_trunk[:, :, 0:1] / (self.args.dh * (Z_dim - 1)) - 1
        )
        x_trunk_normalized = (
            2 * y_trunk[:, :, 1:2] / (self.args.dh * (X_dim - 1)) - 1
        )

        z_encoded = self.pos_encoder(z_trunk_normalized)
        x_encoded = self.pos_encoder(x_trunk_normalized)
        y_encoded = torch.cat([z_encoded, x_encoded], dim=2)  # [B_v, B_pts, 16]

        if self.use_wavenumber_encoding:
            freq_hz, _ = self._prepare_frequency(
                freq_batch, vel.shape[0], y.device, y.dtype
            )
            # vel 已在 dataloader 中转换为 km/s (= m/ms)，omega 使用 rad/ms，故 k 为 1/m。
            c_ref = vel.mean(dim=(2, 3)).squeeze(1).to(dtype=y.dtype).clamp_min(1e-6)
            omega = 2.0 * np.pi * freq_hz * 1e-3
            # y_trunk=αy 时须配套 k/α，因而相位仍为物理的 k·y；
            # 对 Huang 参考频率模式而言，它等价于以 f_ref 表示波数。
            k_ref = (omega / c_ref).view(-1, 1, 1) / coordinate_scale
            k_encoded = self.kpe(y_trunk, k_ref)
            y_encoded = torch.cat([y_encoded, self.kpe_alpha * k_encoded], dim=2)

        freq_encoded = self._encode_trunk_frequency(
            freq_batch,
            batch_size=vel.shape[0],
            num_points=y.shape[1],
            device=y.device,
            dtype=y.dtype,
        )
        if self.use_joint_source_frequency_fusion:
            source_coord_batch = self._prepare_source_coordinates(
                source_coord_batch, UU0, vel.shape[0], y.device, y.dtype
            )
            y_encoded = self._fuse_joint_source_frequency(
                y_encoded, freq_encoded, y, source_coord_batch, freq_batch, vel
            )

        if freq_encoded is not None:
            y_encoded = torch.cat([y_encoded, freq_encoded], dim=2)

        if self.use_relative_source_encoding:
            source_coord_batch = self._prepare_source_coordinates(
                source_coord_batch, UU0, vel.shape[0], y.device, y.dtype
            )
            relative_source_encoded = self._encode_relative_source(
                y, source_coord_batch, freq_batch, vel
            )
            y_encoded = torch.cat([y_encoded, relative_source_encoded], dim=2)
        
        # --- 2. Branch 特征提取与 Tokenization (Memory/Key-Value) ---
        B1_raw = self.branch1(vel)
        B2_raw = self.branch2(UU0)
        
        B1_raw = self.channel_attention1(B1_raw)
        B2_raw = self.channel_attention2(B2_raw)
        
        B1_feat = self.combinedlayer1(vel, y, B1_raw)
        B2_feat = self.combinedlayer2(vel, y, B2_raw, False)
        
        # 注意力门控与特征平滑融合
        B = self.attengate(B1_feat, B2_feat)
        B_encoded = self.smooth_feature_encoder(B1_raw + B2_raw, y_normalized)
        film_freq_context = self._encode_film_frequency(
            freq_batch,
            batch_size=vel.shape[0],
            device=y.device,
            dtype=y.dtype,
        )
        if film_freq_context is not None:
            B_encoded = B_encoded + film_freq_context.unsqueeze(1)
        
        # --- 3. Trunk 与 Branch 融合输出 ---
        T_raw = self.trunk(y_encoded, B_encoded)
        outputs = self.final_layer(B * T_raw)
        
        return outputs

    def forward(self, vel, y, UU0, freq_batch=None, source_coord_batch=None,
                continuous_inputs=None,
                trunk_coordinate_scale=None):
        """执行常规前向，并可在同一次 DDP forward 内执行连续频率前向。

        continuous_inputs 为 ``(vel, y, analytic_UU0, freq, source_coord)``。传入时返回
        ``(outputs, continuous_outputs)``；默认不传时接口与旧版完全相同。
        """
        outputs = self._forward_impl(
            vel, y, UU0, freq_batch=freq_batch,
            source_coord_batch=source_coord_batch,
            trunk_coordinate_scale=trunk_coordinate_scale,
        )
        if continuous_inputs is None:
            return outputs

        vel_cont, y_cont, UU0_cont, freq_cont, source_coord_cont = continuous_inputs
        outputs_cont = self._forward_impl(
            vel_cont, y_cont, UU0_cont, freq_batch=freq_cont,
            source_coord_batch=source_coord_cont,
        )
        return outputs, outputs_cont

    def _infer_source_coordinates(self, reference_UU0):
        """从已知频率 U0 的顶部能量峰值只提取震源坐标。"""
        with torch.no_grad():
            search_depth = int(getattr(self.args, 'analytic_u0_source_search_depth', 4))
            search_depth = max(1, min(search_depth, reference_UU0.shape[-2]))
            energy = reference_UU0[:, :, :search_depth, :].square().sum(dim=1)
            flat_index = energy.flatten(1).argmax(dim=1)
            source_z = torch.div(
                flat_index, reference_UU0.shape[-1], rounding_mode='floor'
            )
            source_x = flat_index.remainder(reference_UU0.shape[-1])
            source_coord = torch.stack([source_z, source_x], dim=-1)
            return source_coord.to(dtype=reference_UU0.dtype) * float(self.args.dh)

    def analytic_background_field(self, frequencies, source_coordinates, nz, nx):
        """
        用二维均匀介质 Green 函数生成任意连续频率的背景场。

        ``U0 = q * i/4 * H_0^(2)(k r)``，所以两个实通道分别是
        ``q*Y0(kr)/4`` 和 ``q*J0(kr)/4``。离散 RHS 需乘以生成
        网格的 ``h^2`` 才是连续 Dirac 源强度。
        """
        device, dtype = frequencies.device, frequencies.dtype
        dh = float(self.args.dh)
        z = torch.arange(nz, device=device, dtype=dtype) * dh
        x = torch.arange(nx, device=device, dtype=dtype) * dh
        zz, xx = torch.meshgrid(z, x, indexing='ij')

        dz = zz.unsqueeze(0) - source_coordinates[:, 0, None, None]
        dx = xx.unsqueeze(0) - source_coordinates[:, 1, None, None]
        min_radius = max(float(getattr(self.args, 'analytic_u0_min_radius', 5.0)), 1e-6)
        radius = torch.sqrt(dz.square() + dx.square()).clamp_min(min_radius)

        c0 = max(float(getattr(self.args, 'analytic_u0_background_velocity', 1500.0)), 1e-6)
        kr = (2.0 * torch.pi * frequencies[:, None, None] / c0) * radius
        source_rhs = float(getattr(self.args, 'analytic_u0_source_rhs_amplitude', 0.5))
        generation_dh = float(getattr(self.args, 'analytic_u0_generation_dh', 10.0))
        source_strength = source_rhs * generation_dh ** 2

        u0_real = 0.25 * source_strength * torch.special.bessel_y0(kr)
        u0_imag = 0.25 * source_strength * torch.special.bessel_j0(kr)
        return torch.stack([u0_real, u0_imag], dim=1)

    def prepare_continuous_frequency_batch(self, vel, reference_UU0,
                                           source_coord_batch=None):
        """从当前 batch 构造无标签的连续频率 PDE 子 batch。"""
        batch_size = vel.shape[0]
        ratio = float(getattr(self.args, 'continuous_pde_batch_ratio', 0.25))
        continuous_batch_size = min(batch_size, max(1, int(round(batch_size * ratio))))
        sample_index = torch.randperm(batch_size, device=vel.device)[:continuous_batch_size]
        vel_cont = vel.index_select(0, sample_index)
        reference_cont = reference_UU0.index_select(0, sample_index)

        freq_min = float(getattr(
            self.args, 'continuous_frequency_min_hz',
            getattr(self.args, 'frequency_min_hz', 3.0)
        ))
        freq_max = float(getattr(
            self.args, 'continuous_frequency_max_hz',
            getattr(self.args, 'frequency_max_hz', 25.0)
        ))
        if freq_max <= freq_min:
            raise ValueError('continuous_frequency_max_hz 必须大于 continuous_frequency_min_hz')
        freq_cont = torch.empty(
            continuous_batch_size, device=vel.device, dtype=vel.dtype
        ).uniform_(freq_min, freq_max)

        if source_coord_batch is None:
            source_coordinates = self._infer_source_coordinates(reference_cont)
        else:
            all_source_coordinates = self._prepare_source_coordinates(
                source_coord_batch, reference_UU0, batch_size,
                vel.device, vel.dtype,
            )
            source_coordinates = all_source_coordinates.index_select(0, sample_index)
        UU0_cont = self.analytic_background_field(
            freq_cont, source_coordinates, vel.shape[-2], vel.shape[-1]
        )

        num_points = max(1, int(getattr(self.args, 'continuous_pde_num_points', 256)))
        pml_active = int(getattr(self.args, 'pml_active', 0)) if self.args.pml else 0
        if getattr(self.args, 'boundary_type', 'full_pml') == 'free_surface':
            z_min_index = 0
        else:
            z_min_index = pml_active
        z_max_index = max(z_min_index, vel.shape[-2] - pml_active - 1)
        x_min_index = pml_active
        x_max_index = max(x_min_index, vel.shape[-1] - pml_active - 1)
        point_min = torch.tensor(
            [z_min_index * self.args.dh, x_min_index * self.args.dh],
            device=vel.device,
            dtype=vel.dtype,
        )
        point_max = torch.tensor(
            [z_max_index * self.args.dh, x_max_index * self.args.dh],
            device=vel.device,
            dtype=vel.dtype,
        )
        # 保持共享配点策略；局部特征采样现在也支持每个样本独立坐标。
        y_cont = torch.rand(1, num_points, 2, device=vel.device, dtype=vel.dtype)
        y_cont = (point_min + y_cont * (point_max - point_min))
        y_cont = y_cont.expand(continuous_batch_size, -1, -1).clone()
        y_cont.requires_grad_(True)
        return vel_cont, y_cont, UU0_cont, freq_cont, source_coordinates
                        
    def loss_BC(self, vel, y, UU0, labels, freq_batch=None, source_coord_batch=None):
        """计算数据拟合损失 (Data/BC Loss)"""
        pred = self.forward(
            vel, y, UU0, freq_batch=freq_batch,
            source_coord_batch=source_coord_batch,
        )
        loss_u = self.loss_function(pred, labels)
        return loss_u

    def dynamic_barrier_loss(self, error, r0=8, lambda_aux=1.0):
        """
        带动态自适应系数的流形屏障惩罚函数。
        在安全区 (r0) 内部，牵引力系数连续衰减，在圆心处严格为 0。
        """
        x = torch.clamp(error / (r0 + 1e-8), min=0.0, max=1.0)
        dynamic_coeff = lambda_aux * (x ** 2)
        return dynamic_coeff * error
        
    def loss_PDE_Scatter_pml(self, vel, y, UU0, freq_batch=None,
                             source_coord_batch=None,
                             return_pointwise=False):
        """
        计算包含 PML 吸收边界条件的散射场 Helmholtz 方程物理残差损失。
        （向后兼容接口，内部调用 _compute_pde_residual）

        Args:
            freq_batch: 每个样本对应的频率值 [B_v]。若为 None 则使用默认值 5 Hz。
        """
        y.requires_grad_(True)
        Delta_U = self.forward(
            vel, y, UU0, freq_batch=freq_batch,
            source_coord_batch=source_coord_batch,
        )
        return self._compute_pde_residual(
            vel, y, UU0, Delta_U, freq_batch=freq_batch,
            return_pointwise=return_pointwise,
        )

    def _compute_pde_residual(self, vel, y, UU0, Delta_U, freq_batch=None,
                              return_pointwise=False):
        """
        计算包含 PML 吸收边界条件的散射场 Helmholtz 方程物理残差。
        接受已计算好的 Delta_U（forward 输出），避免重复前向传播。

        Args:
            vel: 速度场 [B_v, 1, Z, X]
            y: 坐标点 [B_v, N, 2]，需要 requires_grad=True
            UU0: 背景波场 [B_v, 2, Z, X]
            Delta_U: forward 输出 [B_v, N, 2]
            freq_batch: 每个样本对应的频率值 [B_v]
        """
        batch_size_v = vel.shape[0]
        batch_size_pts = y.shape[1]
        y_sample = y.expand(batch_size_v, -1, -1)

        Z_dim = vel.shape[2]
        X_dim = vel.shape[3]
        SPATIAL_SCALE = float(self.args.dh)

        # --- 1. 坐标归一化与 Grid 构造 ---
        z_pixel = y_sample[:, :, 0] / SPATIAL_SCALE
        x_pixel = y_sample[:, :, 1] / SPATIAL_SCALE
        z_norm = 2 * (z_pixel / (Z_dim - 1)) - 1
        x_norm = 2 * (x_pixel / (X_dim - 1)) - 1

        grid = torch.stack([x_norm, z_norm], dim=-1).unsqueeze(1)  # [B_v, 1, B_pts, 2]

        # --- 2. 可微双线性插值采样 ---
        c_sampled = F.grid_sample(vel[:, :1, :, :], grid, mode='bilinear', padding_mode='border', align_corners=True)
        c = c_sampled.view(batch_size_v, batch_size_pts).detach()

        U0_sampled = F.grid_sample(UU0, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(2)
        U0_real = U0_sampled[:, 0, :].detach()
        U0_imag = U0_sampled[:, 1, :].detach()

        # --- 3. 物理常数与衰减因子准备 ---
        c0 = torch.ones_like(c) * 1.5
        f0 = 10
        if freq_batch is not None:
            f = freq_batch.unsqueeze(1).expand(batch_size_v, batch_size_pts)
        else:
            f = y.new_full((batch_size_v, batch_size_pts), float(self.args.default_freq))
        omega = 2 * torch.pi * f * 1e-3
        k = (1 / c) ** 2
        k0 = (1 / c0) ** 2

        Q = 75
        # 外部 FDFD 数据生成可选择无耗散 (alpha=0)；由调用方配置以避免
        # 标签与 PDE 物理系数不一致。未配置时保持旧模型的 1/Q 行为。
        alpha = float(getattr(self.args, 'pde_attenuation_alpha', 1 / Q))
        rhot = (1 - alpha / torch.pi * torch.log(f / 50) - 1j * alpha / 2) ** 2

        kr, ki = k * torch.real(rhot), k * torch.imag(rhot)
        k0r, k0i = k0 * torch.real(rhot), k0 * torch.imag(rhot)

        a0 = 1.79
        C = 2 * torch.pi * a0 * f0 / f

        # --- 4. 一阶与二阶导数计算 (Autograd) ---
        Delta_U_real, Delta_U_imag = Delta_U[:, :, 0], Delta_U[:, :, 1]

        lx, lz = pml_profiles(self.args, y, Z_dim, X_dim)
        pml_tmp1 = C ** 2 * lx ** 2 * lz ** 2
        pml_tmp2 = C ** 2 * lx ** 4
        pml_tmp3 = C ** 2 * lz ** 4
        pml_tmp4 = C * (lz ** 2 - lx ** 2)
        pml_tmp5 = C * (lx ** 2 + lz ** 2)

        # 计算一阶导数
        Delta_U_grad_real = torch.autograd.grad(Delta_U_real, y, grad_outputs=torch.ones_like(Delta_U_real), create_graph=True, retain_graph=True, only_inputs=True)[0]
        Delta_U_grad_imag = torch.autograd.grad(Delta_U_imag, y, grad_outputs=torch.ones_like(Delta_U_imag), create_graph=True, retain_graph=True, only_inputs=True)[0]

        Delta_Uz_real, Delta_Ux_real = Delta_U_grad_real[:, :, 0], Delta_U_grad_real[:, :, 1]
        Delta_Uz_imag, Delta_Ux_imag = Delta_U_grad_imag[:, :, 0], Delta_U_grad_imag[:, :, 1]

        # 修正的一阶导数 (带 PML)
        eu_zr = (1 + pml_tmp1) / (1 + pml_tmp3) * Delta_Uz_real - pml_tmp4 / (1 + pml_tmp3) * Delta_Uz_imag
        eu_xr = (1 + pml_tmp1) / (1 + pml_tmp2) * Delta_Ux_real + pml_tmp4 / (1 + pml_tmp2) * Delta_Ux_imag
        eu_zi = pml_tmp4 / (1 + pml_tmp3) * Delta_Uz_real + (1 + pml_tmp1) / (1 + pml_tmp3) * Delta_Uz_imag
        eu_xi = -pml_tmp4 / (1 + pml_tmp2) * Delta_Ux_real + (1 + pml_tmp1) / (1 + pml_tmp2) * Delta_Ux_imag

        # 计算二阶导数
        Delta_Uzz_real = torch.autograd.grad(eu_zr, y, grad_outputs=torch.ones_like(eu_zr), create_graph=True, retain_graph=True, only_inputs=True)[0][:, :, 0]
        Delta_Uxx_real = torch.autograd.grad(eu_xr, y, grad_outputs=torch.ones_like(eu_xr), create_graph=True, retain_graph=True, only_inputs=True)[0][:, :, 1]
        Delta_Uzz_imag = torch.autograd.grad(eu_zi, y, grad_outputs=torch.ones_like(eu_zi), create_graph=True, retain_graph=True, only_inputs=True)[0][:, :, 0]
        Delta_Uxx_imag = torch.autograd.grad(eu_xi, y, grad_outputs=torch.ones_like(eu_xi), create_graph=True, retain_graph=True, only_inputs=True)[0][:, :, 1]

        # --- 5. 组合 PDE 残差 ---
        ur_r = (1 - pml_tmp1) * omega ** 2 * (kr * (Delta_U_real + U0_real) - ki * (Delta_U_imag + U0_imag))
        ui_r = pml_tmp5 * omega ** 2 * (kr * (Delta_U_imag + U0_imag) + ki * (Delta_U_real + U0_real))
        u0r_r = (1 - pml_tmp1) * omega ** 2 * (-k0r * U0_real + k0i * U0_imag)
        u0i_r = pml_tmp5 * omega ** 2 * (-k0r * U0_imag - k0i * U0_real)

        ur_i = (-pml_tmp5) * omega ** 2 * (kr * (Delta_U_real + U0_real) - ki * (Delta_U_imag + U0_imag))
        ui_i = (1 - pml_tmp1) * omega ** 2 * (kr * (Delta_U_imag + U0_imag) + ki * (Delta_U_real + U0_real))
        u0r_i = (-pml_tmp5) * omega ** 2 * (-k0r * U0_real + k0i * U0_imag)
        u0i_i = (1 - pml_tmp1) * omega ** 2 * (-k0r * U0_imag - k0i * U0_real)

        residual_real = Delta_Uzz_real + Delta_Uxx_real + ur_r + ui_r + u0r_r + u0i_r
        residual_imag = Delta_Uzz_imag + Delta_Uxx_imag + ur_i + ui_i + u0r_i + u0i_i

        pointwise_loss = residual_real ** 2 + residual_imag ** 2
        if return_pointwise:
            return pointwise_loss
        return torch.mean(pointwise_loss)
    
    def loss_Reg(self, vel, y, UU0, source_coord, freq_batch=None):
        """震源区域正则化损失"""
        z_coord, x_coord = y[:, 0], y[:, 1]
        source_z, source_x = source_coord[:, 0], source_coord[:, 1]
        
        inside_distance = 100 - torch.sqrt((z_coord - source_z) ** 2 + (x_coord - source_x) ** 2)
        coe = F.relu(inside_distance) / (inside_distance + 1e-15)

        pred = self.forward(
            vel, y, UU0, freq_batch=freq_batch,
            source_coord_batch=source_coord,
        )
        N_reg = torch.clamp(torch.count_nonzero(coe), min=1.0).to(vel.device)

        return torch.sum(coe * (pred[:, 0] ** 2 + pred[:, 1] ** 2)) / N_reg

    def loss_op(self, model0, vel, y, UU0, freq_batch=None, source_coord_batch=None):
        """模型间操作损失 (如知识蒸馏或微调约束)"""
        with torch.no_grad():
            pred0 = model0(
                vel, y, UU0, freq_batch=freq_batch,
                source_coord_batch=source_coord_batch,
            )
        pred_ft = self.forward(
            vel, y, UU0, freq_batch=freq_batch,
            source_coord_batch=source_coord_batch,
        )
        return torch.sum((pred0 - pred_ft) ** 2)
        
    def get_ortho_loss(self, T, weight):
        """
        计算基底正交性损失。
        通过归一化 Gram 矩阵，使 Trunk 输出在序列维度上互相正交。
        """
        B_v, N, p = T.shape
        gram = torch.bmm(T.transpose(-2, -1), T)
        
        diag = torch.diagonal(gram, dim1=-2, dim2=-1).unsqueeze(-1) + 1e-8
        gram_normalized = gram / torch.sqrt(diag @ diag.transpose(-2, -1))
        
        gram_matrix = torch.bmm(T.transpose(1, 2), T) / N
        eye = torch.eye(p, device=T.device).unsqueeze(0).expand(B_v, -1, -1)
        
        loss = torch.mean((gram_matrix - eye) ** 2)
        return loss * weight
    
    def get_trunk_output(self, vel, y):
        """独立提取 Trunk 网络的基底输出"""
        # 修复：使用实际网格尺寸而不是硬编码 72
        Z_dim = vel.shape[2]
        X_dim = vel.shape[3]
        y_norm = 2 * (y - 0) / (self.args.dh * X_dim) - 1  # 使用实际x维度
        z_enc = self.pos_encoder(y_norm[:, :, 0:1])
        x_enc = self.pos_encoder(y_norm[:, :, 1:2])

        y_encoded = torch.cat([z_enc, x_enc], dim=-1)
        physical_context = get_local_physical_features(vel, y, eps=1e-3)
        y_encoded = torch.cat([y_encoded, physical_context], dim=-1)

        return self.trunk(y_encoded)

    def generate_structure_aware_y_ran(self, vel, num_pts=20000, max_z=None, max_x=None):
        """
        结构感知自适应采样点生成。
        根据速度场空间梯度的高低，自适应分配采样点（50% 结构点，50% 表层点）。

        Args:
            vel: 速度场 [B, C, Z, X]
            num_pts: 采样点数量
            max_z: z方向最大坐标（默认使用vel的实际尺寸）
            max_x: x方向最大坐标（默认使用vel的实际尺寸）

        Returns:
            y_ran: 采样点坐标 [B, num_pts, 2]，requires_grad=True
        """
        # 修复：如果未提供max_z和max_x，使用vel的实际尺寸
        if max_z is None:
            max_z = float(vel.shape[2]) * self.args.dh  # 转换为实际坐标
        if max_x is None:
            max_x = float(vel.shape[3]) * self.args.dh

        B_v = vel.shape[0]
        device = vel.device

        # 计算网格步长
        dz = max_z / vel.shape[2]  # z方向网格步长
        dx = max_x / vel.shape[3]  # x方向网格步长

        with torch.no_grad():
            # 计算空间梯度幅度
            grad_z = vel[:, :, 2:, 1:-1] - vel[:, :, :-2, 1:-1]
            grad_x = vel[:, :, 1:-1, 2:] - vel[:, :, 1:-1, :-2]
            vel_grad_mag = torch.sqrt(grad_z**2 + grad_x**2 + 1e-8)
            vel_grad_mag = F.pad(vel_grad_mag, (1, 1, 1, 1), mode='replicate').squeeze(1)

            y_ran_list = []
            for b in range(B_v):
                prob_dist = vel_grad_mag[b].view(-1)
                prob_dist = prob_dist / (prob_dist.sum() + 1e-8)

                # 修改采样策略：50% 结构点，50% 表层点
                num_structure = int(num_pts * 0.5)
                num_surface = num_pts - num_structure

                # --- 1. 抽取结构边界点（50%）---
                if num_structure > 0:
                    sampled_indices = torch.multinomial(prob_dist, num_samples=num_structure, replacement=True)
                    z_idx = sampled_indices // vel.shape[3]
                    x_idx = sampled_indices % vel.shape[3]

                    z_coords = z_idx.float() * dz + (torch.rand(num_structure, device=device) * dz)
                    x_coords = x_idx.float() * dx + (torch.rand(num_structure, device=device) * dx)
                    y_struct = torch.stack([z_coords, x_coords], dim=1)
                else:
                    y_struct = torch.empty((0, 2), device=device)

                # --- 2. 抽取表层点（50%）---
                # 表层定义：z < 2 个网格点的深度范围
                if num_surface > 0:
                    # 表层深度范围：[0, 2*dz]
                    surface_depth = 2.0 * dz

                    # 在表层范围内随机采样 z 坐标
                    z_surf = torch.rand(num_surface, device=device) * surface_depth

                    # x 坐标在整个范围内均匀采样
                    x_surf = torch.rand(num_surface, device=device) * max_x

                    y_surf = torch.stack([z_surf, x_surf], dim=1)
                else:
                    y_surf = torch.empty((0, 2), device=device)

                # 合并结构点和表层点
                y_ran_list.append(torch.cat([y_struct, y_surf], dim=0))

        y_ran = torch.stack(y_ran_list, dim=0)
        return y_ran.requires_grad_(True)

    def envelope_barrier_loss(self, vel, y, UU0, u_fno, lambda_env=1.0,
                              freq_batch=None, source_coord_batch=None):
        """计算波场包络的流形屏障惩罚损失，消除高频相位错位的影响"""
        u_pred = self.forward(
            vel, y, UU0, freq_batch=freq_batch,
            source_coord_batch=source_coord_batch,
        )
        
        env_pred = torch.sqrt(u_pred[..., 0]**2 + u_pred[..., 1]**2 + 1e-8)
        env_fno = torch.sqrt(u_fno[..., 0]**2 + u_fno[..., 1]**2 + 1e-8)
        
        loss_env = torch.abs(env_pred - env_fno)
        return torch.mean(loss_env)

        
    def loss(self, vel, y, UU0, labels, a, b, c, d=0., data_norm_coe=1.,
             pde_norm_coe=1., env_norm_coe=1., freq_batch=None, y_ran=None,
             use_continuous_pde=False, source_coord_batch=None):
        """
        核心损失函数计算接口。
        优化：只做一次 forward pass，同时服务于 BC loss、PDE loss 和 Envelope loss。

        Args:
            d: envelope loss 权重
            env_norm_coe: envelope loss 归一化系数
            freq_batch: 每个样本对应的频率值 [B_v]。若为 None 则使用默认值。
            y_ran: 预计算的自适应采样点 [B_v, N_ran, 2]。若提供则拼接到 y 后；若为 None 则不生成。
            use_continuous_pde: 是否在本次训练调用中加入连续频率 PDE。
        """
        batch_size_v = vel.shape[0]
        nz, nx = vel.shape[2], vel.shape[3]

        # 1. 提取标签坐标 (根据给定的 y)
        batch_idx = torch.arange(batch_size_v, device=labels.device)[:, None]
        z_coord = (y[:, :, 0] / self.args.dh).long().clamp(0, nz - 1)
        x_coord = (y[:, :, 1] / self.args.dh).long().clamp(0, nx - 1)
        labels = labels[batch_idx, :, z_coord, x_coord]  # [B_v, 2, B_pts] -> [B_v, B_pts, 2]

        # 2. 拼接自适应采样点 (仅当显式提供 y_ran 时)
        n_y = y.shape[1]
        if y_ran is not None and y_ran.shape[1] > 0:
            y_combined = torch.cat([y, y_ran], dim=1)
        else:
            y_combined = y

        y_combined.requires_grad_(True)

        continuous_inputs = None
        continuous_enabled = (
            use_continuous_pde
            and bool(getattr(self.args, 'enable_continuous_frequency_pde', False))
        )
        if continuous_enabled:
            continuous_inputs = self.prepare_continuous_frequency_batch(
                vel, UU0, source_coord_batch=source_coord_batch
            )

        # 3. 只做一次 model forward；开启时在其内额外计算连续频率子 batch。
        forward_result = self.forward(
            vel, y_combined, UU0, freq_batch=freq_batch,
            source_coord_batch=source_coord_batch,
            continuous_inputs=continuous_inputs,
        )
        if continuous_inputs is None:
            Delta_U = forward_result
            Delta_U_cont = None
        else:
            Delta_U, Delta_U_cont = forward_result

        # 4. BC loss: 使用前 n_y 个点的预测
        pred_y = Delta_U[:, :n_y, :]
        loss_u = self.loss_function(pred_y, labels) / data_norm_coe

        # 5. Envelope loss (已禁用)
        # env_pred = torch.sqrt(pred_y[..., 0]**2 + pred_y[..., 1]**2 + 1e-8)
        # env_label = torch.sqrt(labels[..., 0]**2 + labels[..., 1]**2 + 1e-8)
        # loss_env = self.loss_function(env_pred, env_label) / env_norm_coe
        loss_env = torch.tensor(0.0, device=vel.device)

        # 6. PDE loss: 使用完整输出计算物理残差
        loss_f_combined = self._compute_pde_residual(vel, y_combined, UU0, Delta_U, freq_batch=freq_batch) / pde_norm_coe

        # Disabling the auxiliary loss must also skip its computation graph.
        # The FiLM conditioning network remains active in the main forward.
        loss_r = (
            self.film_frequency_smoothness_loss(
                freq_batch, batch_size_v, vel.device, vel.dtype
            )
            if float(getattr(self.args, 'film_smooth_weight', 0.0)) != 0.0
            else vel.new_zeros(())
        )

        loss_continuous_pde = torch.zeros((), device=vel.device, dtype=vel.dtype)
        if continuous_inputs is not None:
            vel_cont, y_cont, UU0_cont, freq_cont, source_coord_cont = continuous_inputs
            loss_continuous_pde = self._compute_pde_residual(
                vel_cont, y_cont, UU0_cont, Delta_U_cont, freq_batch=freq_cont
            ) / pde_norm_coe

        # 7. 根据权重加权求和
        film_smooth_weight = float(getattr(self.args, 'film_smooth_weight', 0.0))
        continuous_pde_weight = float(getattr(self.args, 'continuous_pde_weight', 1.0))
        loss_val = (
            a * loss_u
            + b * loss_f_combined
            + film_smooth_weight * loss_r
            + continuous_pde_weight * loss_continuous_pde
        )

        return loss_val, loss_f_combined, loss_u, loss_r, loss_env, loss_continuous_pde

    def compute_loss(self, Delta_U, vel, y, UU0, labels, y_combined,
                     a, b, c, d=0., data_norm_coe=1., pde_norm_coe=1.,
                     env_norm_coe=1., freq_batch=None, continuous_inputs=None,
                     Delta_U_cont=None, source_coord_batch=None):
        """
        在 DDP forward 之后计算损失 (不包含 forward 调用)。
        供 DDP 训练使用：先通过 DDP wrapper 调用 forward()，再调用此方法计算 loss。

        Args:
            Delta_U: model.forward() 的输出 [B_v, B_pts, 2]
            vel: 速度场 [B_v, 1, Z, X]
            y: 数据坐标点 [B_v, B_data_pts, 2]
            UU0: 背景波场 [B_v, 2, Z, X]
            labels: 标签波场 [B_v, 2, Z, X]
            y_combined: 拼接后的坐标 [B_v, B_data_pts + B_ran_pts, 2]，requires_grad=True
            a, b, c, d: 损失权重
            data_norm_coe: 数据损失归一化系数
            pde_norm_coe: PDE 损失归一化系数
            env_norm_coe: envelope 损失归一化系数
            freq_batch: 频率值 [B_v]
            continuous_inputs: 连续频率 ``(vel, y, analytic_UU0, freq, source_coord)``。
            Delta_U_cont: DDP forward 产生的连续频率预测。
        Returns:
            (total_loss, loss_f, loss_u, loss_r, loss_env, loss_continuous_pde)
        """
        batch_size_v = vel.shape[0]
        nz, nx = vel.shape[2], vel.shape[3]
        n_y = y.shape[1]

        # 1. 提取标签值
        batch_idx = torch.arange(batch_size_v, device=labels.device)[:, None]
        z_coord = (y[:, :, 0] / self.args.dh).long().clamp(0, nz - 1)
        x_coord = (y[:, :, 1] / self.args.dh).long().clamp(0, nx - 1)
        labels_extracted = labels[batch_idx, :, z_coord, x_coord]

        # 2. 数据拟合损失
        pred_y = Delta_U[:, :n_y, :]
        loss_u = self.loss_function(pred_y, labels_extracted) / data_norm_coe

        # 3. Envelope loss (已禁用)
        # env_pred = torch.sqrt(pred_y[..., 0]**2 + pred_y[..., 1]**2 + 1e-8)
        # env_label = torch.sqrt(labels_extracted[..., 0]**2 + labels_extracted[..., 1]**2 + 1e-8)
        # loss_env = self.loss_function(env_pred, env_label) / env_norm_coe
        loss_env = torch.tensor(0.0, device=vel.device)

        # 4. PDE 物理残差损失
        loss_f = self._compute_pde_residual(vel, y_combined, UU0, Delta_U, freq_batch=freq_batch) / pde_norm_coe

        loss_r = (
            self.film_frequency_smoothness_loss(
                freq_batch, batch_size_v, vel.device, vel.dtype
            )
            if float(getattr(self.args, 'film_smooth_weight', 0.0)) != 0.0
            else vel.new_zeros(())
        )

        loss_continuous_pde = torch.zeros((), device=vel.device, dtype=vel.dtype)
        if continuous_inputs is not None:
            if Delta_U_cont is None:
                raise ValueError('continuous_inputs 已提供，但 Delta_U_cont 为 None')
            vel_cont, y_cont, UU0_cont, freq_cont, source_coord_cont = continuous_inputs
            loss_continuous_pde = self._compute_pde_residual(
                vel_cont, y_cont, UU0_cont, Delta_U_cont, freq_batch=freq_cont
            ) / pde_norm_coe

        # 5. 加权求和
        film_smooth_weight = float(getattr(self.args, 'film_smooth_weight', 0.0))
        continuous_pde_weight = float(getattr(self.args, 'continuous_pde_weight', 1.0))
        loss_val = (
            a * loss_u
            + b * loss_f
            + film_smooth_weight * loss_r
            + continuous_pde_weight * loss_continuous_pde
        )

        return loss_val, loss_f, loss_u, loss_r, loss_env, loss_continuous_pde
