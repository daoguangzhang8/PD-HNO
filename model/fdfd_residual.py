"""可微的 9 点 FDFD 散射场残差。

矩阵系数直接复用 ``Code/data/getA9_PML.py`` 的离散格式；矩阵在每个
速度/频率组合初始化一次，训练时仅对网络预测场进行稀疏矩阵乘法，因此不读取
真实散射波场标签，也不会把 SciPy 操作放进反向传播图。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


_DATA_GENERATION_DIR = Path(__file__).resolve().parents[2] / 'data'
if str(_DATA_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_GENERATION_DIR))

from getA9_PML import getA9_PML  # noqa: E402
from optimal_Parameters import optimal_Parameters  # noqa: E402


def _to_torch_sparse(matrix, component: str, device: torch.device) -> torch.Tensor:
    """把 SciPy 复稀疏矩阵的一个实/虚部转为可反传 dense-vector 的 torch 矩阵。"""
    coo = matrix.tocoo()
    values = np.real(coo.data) if component == 'real' else np.imag(coo.data)
    keep = np.abs(values) > 0
    indices = torch.from_numpy(
        np.stack([coo.row[keep], coo.col[keep]], axis=0).astype(np.int64, copy=False)
    ).to(device)
    tensor_values = torch.from_numpy(values[keep].astype(np.float32, copy=False)).to(device)
    return torch.sparse_coo_tensor(
        indices, tensor_values, size=coo.shape, device=device, dtype=torch.float32
    ).coalesce()


class CroppedFDFDScatteredResidual:
    """裁剪网络域上的 FDFD 散射残差。

    原始网格的底部、左右各裁去 ``pml_crop`` 层。只取裁剪域中一层边界以内的
    残差行，确保 9 点 stencil 的所有未知量都由网络预测，不对缺失的外侧 PML
    作任何人为填充假设。
    """

    def __init__(
        self,
        velocity_km_s: torch.Tensor,
        frequency_hz: float,
        *,
        dh: float,
        pml_total: int,
        pml_crop: int,
        attenuation_alpha: float,
        device: torch.device,
    ):
        if velocity_km_s.ndim != 2:
            raise ValueError('velocity_km_s 必须是 [nz, nx]')
        self.nz, self.nx = map(int, velocity_km_s.shape)
        if self.nz < 3 or self.nx < 3:
            raise ValueError('FDFD 残差至少需要 3×3 网格')
        self.device = device

        # external_test 的 free-surface 裁剪：顶部不裁，底部裁 pml_crop，
        # 左右各裁 pml_crop。原始网格仍有 total 层 PML。
        full_nz = self.nz + int(pml_crop)
        full_nx = self.nx + 2 * int(pml_crop)
        z_offset = 0
        x_offset = int(pml_crop)

        velocity = velocity_km_s.detach().cpu().numpy().astype(np.float64, copy=False)
        full_velocity = np.empty((full_nz, full_nx), dtype=np.float64)
        full_velocity[:self.nz, x_offset:x_offset + self.nx] = velocity
        # 被裁掉的外侧 PML 不参与选中的 residual 行；边缘复制仅使矩阵构造完整。
        full_velocity[self.nz:, x_offset:x_offset + self.nx] = velocity[-1:, :]
        full_velocity[:, :x_offset] = full_velocity[:, x_offset:x_offset + 1]
        full_velocity[:, x_offset + self.nx:] = (
            full_velocity[:, x_offset + self.nx - 1:x_offset + self.nx]
        )

        frequency_hz = float(frequency_hz)
        alpha = float(attenuation_alpha)
        rhot = (1 - alpha / np.pi * np.log(frequency_hz / 50.0) - 1j * alpha / 2) ** 2
        # 训练/测试张量中的速度已是 km/s（m/ms），与 Helmholtz 的 omega
        # 单位相匹配；这里绝不能再次 /1000，否则 m 会被错误放大 10^6 倍。
        m = rhot / np.maximum(full_velocity, 1e-6) ** 2
        m0 = np.full_like(m, rhot / 1.5 ** 2, dtype=np.complex128)

        velocity_m_s = full_velocity * 1000.0
        gmin = float(velocity_m_s.min() / (float(dh) * frequency_hz))
        gmax = float(velocity_m_s.max() / (float(dh) * frequency_hz))
        b, c, d, e = optimal_Parameters(gmin, gmax)

        # [[top, left], [bottom, right]]：顶部自由表面，其他三侧 PML。
        n_pml = np.array([[0, pml_total], [pml_total, pml_total]], dtype=np.int64)
        a, _, _, _ = getA9_PML(
            n_pml, full_nz, full_nx, frequency_hz, 10.0, float(dh),
            m, float(b), float(c), float(d), float(e),
        )
        a0, _, _, _ = getA9_PML(
            n_pml, full_nz, full_nx, frequency_hz, 10.0, float(dh),
            m0, float(b), float(c), float(d), float(e),
        )

        z_index, x_index = np.meshgrid(
            np.arange(self.nz), np.arange(self.nx), indexing='ij'
        )
        crop_columns = (
            z_index + z_offset + full_nz * (x_index + x_offset)
        ).reshape(-1, order='F')
        inner_z, inner_x = np.meshgrid(
            np.arange(1, self.nz - 1), np.arange(1, self.nx - 1), indexing='ij'
        )
        inner_rows = (
            inner_z + z_offset + full_nz * (inner_x + x_offset)
        ).reshape(-1, order='F')

        # 行只保留完全包含于 crop 的 stencil。若此断言失败，绝不能静默把缺失项置零。
        a_crop = a.tocsr()[inner_rows][:, crop_columns].tocsr()
        a0_crop = a0.tocsr()[inner_rows][:, crop_columns].tocsr()
        if a.tocsr()[inner_rows].nnz != a_crop.nnz:
            raise RuntimeError('选中的 FDFD stencil 触及被裁掉的 PML 区域')
        if a0.tocsr()[inner_rows].nnz != a0_crop.nnz:
            raise RuntimeError('选中的背景 FDFD stencil 触及被裁掉的 PML 区域')

        self.a_real = _to_torch_sparse(a_crop, 'real', device)
        self.a_imag = _to_torch_sparse(a_crop, 'imag', device)
        self.a0_real = _to_torch_sparse(a0_crop, 'real', device)
        self.a0_imag = _to_torch_sparse(a0_crop, 'imag', device)
        self.n_inner = (self.nz - 2) * (self.nx - 2)

    @staticmethod
    def _fortran_flatten(field: torch.Tensor) -> torch.Tensor:
        """[B,H,W] → [B,H*W]，与 SciPy ``reshape(order='F')`` 保持一致。"""
        return field.transpose(-2, -1).reshape(field.shape[0], -1)

    def residual(self, delta_u: torch.Tensor, background_u: torch.Tensor):
        """返回复散射残差的实、虚部，shape 均为 [B, n_inner]。"""
        if delta_u.shape[-2:] != (self.nz, self.nx):
            raise ValueError('delta_u 空间维度与构造 FDFD 矩阵时不一致')
        dr = self._fortran_flatten(delta_u[:, 0])
        di = self._fortran_flatten(delta_u[:, 1])
        u0r = self._fortran_flatten(background_u[:, 0])
        u0i = self._fortran_flatten(background_u[:, 1])
        real_parts, imag_parts = [], []
        for batch_index in range(delta_u.shape[0]):
            def mm(matrix, vector):
                return torch.sparse.mm(matrix, vector.unsqueeze(1)).squeeze(1)

            # A_m ΔU + (A_m - A_0) U_0 = 0
            real = (
                mm(self.a_real, dr[batch_index]) - mm(self.a_imag, di[batch_index])
                + mm(self.a_real - self.a0_real, u0r[batch_index])
                - mm(self.a_imag - self.a0_imag, u0i[batch_index])
            )
            imag = (
                mm(self.a_real, di[batch_index]) + mm(self.a_imag, dr[batch_index])
                + mm(self.a_real - self.a0_real, u0i[batch_index])
                + mm(self.a_imag - self.a0_imag, u0r[batch_index])
            )
            real_parts.append(real)
            imag_parts.append(imag)
        return torch.stack(real_parts), torch.stack(imag_parts)

    def pointwise_loss(self, delta_u: torch.Tensor, background_u: torch.Tensor):
        """返回 [B,H,W] 每点 |r|² 图；最外一层无完整 stencil，填 NaN。"""
        real, imag = self.residual(delta_u, background_u)
        pointwise = torch.full(
            (delta_u.shape[0], self.nz, self.nx),
            float('nan'), device=delta_u.device, dtype=delta_u.dtype,
        )
        pointwise[:, 1:-1, 1:-1] = (
            real.square() + imag.square()
        ).reshape(delta_u.shape[0], self.nx - 2, self.nz - 2).transpose(1, 2)
        return pointwise

    def mean_loss(self, delta_u: torch.Tensor, background_u: torch.Tensor):
        pointwise = self.pointwise_loss(delta_u, background_u)
        return torch.nanmean(pointwise)


class PhysicalWindowFDFDScatteredResidual:
    """物理域 ``[x, z]`` 窗口上的精确 9 点 FDFD 散射残差。

    ``gen_external_test.py`` 的全场存储布局是 ``[x, z]``，而 FDFD 矩阵
    的内部布局为 ``[z, x]``、Fortran order。该类接收已经去除 PML 的物理域
    280×280 场，但仍以生成器的完整 PML 域构建矩阵；只保留物理域最外一层
    以内的 residual 行。因此每个被使用的 9 点 stencil 都完全由网络输出
    提供，不会以零值或复制值伪造未知 PML 波场。

    这使得网络可以严格预测 280×280 物理波场，同时训练 loss 与标签生成器
    的离散算子保持一致（仅不对物理域边界的 stencil 施加残差）。
    """

    def __init__(
        self,
        velocity_km_s: torch.Tensor,
        frequency_hz: float,
        *,
        dh: float,
        n_pml: np.ndarray,
        attenuation_alpha: float,
        device: torch.device,
    ):
        if velocity_km_s.ndim != 2:
            raise ValueError('velocity_km_s 必须是存储布局 [nx, nz] 的二维数组')
        self.nx, self.nz = map(int, velocity_km_s.shape)
        if self.nz < 3 or self.nx < 3:
            raise ValueError('FDFD 残差至少需要 3×3 网格')
        self.device = device

        n_pml = np.asarray(n_pml, dtype=np.int64)
        if n_pml.shape != (2, 2) or np.any(n_pml < 0):
            raise ValueError('n_pml 必须为非负的 [[top,left],[bottom,right]] 2×2 数组')
        self.n_pml = n_pml.copy()
        top, left = map(int, n_pml[0])
        bottom, right = map(int, n_pml[1])
        full_nz = self.nz + top + bottom
        full_nx = self.nx + left + right

        # 输入/输出的存储布局为 [x,z]；矩阵构造时转回 FDFD 的 [z,x]。
        physical_velocity_zx = (
            velocity_km_s.detach().cpu().numpy().astype(np.float64, copy=False).T
        )
        full_velocity = np.pad(
            physical_velocity_zx, ((top, bottom), (left, right)), mode='edge'
        )
        if full_velocity.shape != (full_nz, full_nx):
            raise RuntimeError('物理窗口与 PML 扩展后的尺寸不一致')

        frequency_hz = float(frequency_hz)
        alpha = float(attenuation_alpha)
        rhot = (1 - alpha / np.pi * np.log(frequency_hz / 50.0) - 1j * alpha / 2) ** 2
        # ``velocity_km_s`` 已是 km/s，见 CroppedFDFDScatteredResidual 同处说明。
        m = rhot / np.maximum(full_velocity, 1e-6) ** 2
        m0 = np.full_like(m, rhot / 1.5 ** 2, dtype=np.complex128)

        # 与生成器一致：优化的 9 点系数仅由未扩展的物理速度范围决定。
        velocity_m_s = physical_velocity_zx * 1000.0
        gmin = float(velocity_m_s.min() / (float(dh) * frequency_hz))
        gmax = float(velocity_m_s.max() / (float(dh) * frequency_hz))
        b, c, d, e = optimal_Parameters(gmin, gmax)
        a, _, _, _ = getA9_PML(
            n_pml, full_nz, full_nx, frequency_hz, 10.0, float(dh),
            m, float(b), float(c), float(d), float(e),
        )
        a0, _, _, _ = getA9_PML(
            n_pml, full_nz, full_nx, frequency_hz, 10.0, float(dh),
            m0, float(b), float(c), float(d), float(e),
        )

        z_index, x_index = np.meshgrid(
            np.arange(self.nz), np.arange(self.nx), indexing='ij'
        )
        crop_columns = (
            z_index + top + full_nz * (x_index + left)
        ).reshape(-1, order='F')
        inner_z, inner_x = np.meshgrid(
            np.arange(1, self.nz - 1), np.arange(1, self.nx - 1), indexing='ij'
        )
        inner_rows = (
            inner_z + top + full_nz * (inner_x + left)
        ).reshape(-1, order='F')

        a_crop = a.tocsr()[inner_rows][:, crop_columns].tocsr()
        a0_crop = a0.tocsr()[inner_rows][:, crop_columns].tocsr()
        if a.tocsr()[inner_rows].nnz != a_crop.nnz:
            raise RuntimeError('选中的物理域 FDFD stencil 触及未预测的 PML 区域')
        if a0.tocsr()[inner_rows].nnz != a0_crop.nnz:
            raise RuntimeError('选中的背景 FDFD stencil 触及未预测的 PML 区域')

        self.a_real = _to_torch_sparse(a_crop, 'real', device)
        self.a_imag = _to_torch_sparse(a_crop, 'imag', device)
        self.a0_real = _to_torch_sparse(a0_crop, 'real', device)
        self.a0_imag = _to_torch_sparse(a0_crop, 'imag', device)
        self.n_inner = (self.nz - 2) * (self.nx - 2)

    @staticmethod
    def _storage_flatten(field: torch.Tensor) -> torch.Tensor:
        """[B,x,z] 的 C-order 正是 FDFD ``[z,x]`` 的 Fortran order。"""
        return field.reshape(field.shape[0], -1)

    def residual(self, delta_u: torch.Tensor, background_u: torch.Tensor):
        if delta_u.shape[-2:] != (self.nx, self.nz):
            raise ValueError('delta_u 空间维度与构造 FDFD 矩阵时不一致')
        dr = self._storage_flatten(delta_u[:, 0])
        di = self._storage_flatten(delta_u[:, 1])
        u0r = self._storage_flatten(background_u[:, 0])
        u0i = self._storage_flatten(background_u[:, 1])
        real_parts, imag_parts = [], []
        for batch_index in range(delta_u.shape[0]):
            def mm(matrix, vector):
                return torch.sparse.mm(matrix, vector.unsqueeze(1)).squeeze(1)

            real = (
                mm(self.a_real, dr[batch_index]) - mm(self.a_imag, di[batch_index])
                + mm(self.a_real - self.a0_real, u0r[batch_index])
                - mm(self.a_imag - self.a0_imag, u0i[batch_index])
            )
            imag = (
                mm(self.a_real, di[batch_index]) + mm(self.a_imag, dr[batch_index])
                + mm(self.a_real - self.a0_real, u0i[batch_index])
                + mm(self.a_imag - self.a0_imag, u0r[batch_index])
            )
            real_parts.append(real)
            imag_parts.append(imag)
        return torch.stack(real_parts), torch.stack(imag_parts)

    def pointwise_loss(self, delta_u: torch.Tensor, background_u: torch.Tensor):
        """返回存储布局 [B,x,z] 的 |r|² 图；物理域外圈填 NaN。"""
        real, imag = self.residual(delta_u, background_u)
        pointwise = torch.full(
            (delta_u.shape[0], self.nx, self.nz),
            float('nan'), device=delta_u.device, dtype=delta_u.dtype,
        )
        pointwise[:, 1:-1, 1:-1] = (
            real.square() + imag.square()
        ).reshape(delta_u.shape[0], self.nx - 2, self.nz - 2)
        return pointwise

    def mean_loss(self, delta_u: torch.Tensor, background_u: torch.Tensor):
        return torch.nanmean(self.pointwise_loss(delta_u, background_u))
