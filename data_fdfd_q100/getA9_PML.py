import numpy as np
from scipy.sparse import spdiags
from scipy import sparse
import math as mm
import numpy.matlib
from I_matrix import I_matrix
from scipy.sparse import csr_matrix



def getA9_PML(n_pml, nz, nx, freq, modelf0, modelh, mv, b, c, d, e):
    """
    Rotated 9-point finite difference scheme with PML in 2D

    参数:
        n_pml: PML层厚度矩阵，形状为 (2, 2)
               [[顶部PML, 左侧PML],
                [底部PML, 右侧PML]]
        nz, nx: 扩展后的模型尺寸
        freq: 频率
        modelf0: 参考频率
        modelh: 网格间距
        mv: 慢度模型
        b, c, d, e: 9点差分系数
    """

    # 如果 n_pml 是标量，转换为统一矩阵（向后兼容）
    if np.isscalar(n_pml):
        n_pml = np.array([[n_pml, n_pml], [n_pml, n_pml]])

    # Lossless acoustic modeling: alpha=0 is equivalent to Q=infinity.
    # Keep the expression below so the operator remains structurally identical
    # to the attenuating implementation without applying material attenuation.
    alpha = 0.0
    rhot = (1 - alpha / np.pi * np.log(freq / 50) - 1j * alpha / 2) ** 2

    mv = mv * rhot
    I = I_matrix(nx, nz, c, d, e)
    h = modelh
    f0 = modelf0
    omega = 1e-3 * 2 * mm.pi * freq
    beta = 2 * mm.pi * 1.79 * f0 / freq

    # b = 0.7926
    # d = 0.3768
    # e = -0.0064
    # c = 1 - d - e

    # PML 边界厚度
    # top_pml = n_pml[0, 1]    # 顶部 PML (z方向起始)
    # bottom_pml = n_pml[1, 1]  # 底部 PML (z方向结束)
    # left_pml = n_pml[0, 0]    # 左侧 PML (x方向起始)
    # right_pml = n_pml[1, 0]   # 右侧 PML (x方向结束)
    top_pml = n_pml[0, 0]    # 顶部 PML (z方向起始)
    bottom_pml = n_pml[1, 0]  # 底部 PML (z方向结束)
    left_pml = n_pml[0, 1]    # 左侧 PML (x方向起始)
    right_pml = n_pml[1, 1]   # 右侧 PML (x方向结束)
    # 归一化因子 (避免除零)
    max_pml_z = max(n_pml[:, 0])
    max_pml_x = max(n_pml[:, 1])

    xc = np.arange(1, nx + 1)
    zc = np.arange(1, nz + 1)

    # PML 衰减函数 (与 MATLAB 版本一致)
    # pmlz: z方向 PML - 顶部和底部
    # pmlx: x方向 PML - 左侧和右侧
    def pmlz(x):
        # 顶部 PML: x <= top_pml
        top_term = 1 - 1j * beta * ((top_pml - x + 1/2) / max(max_pml_z, 1))**2 * (top_pml - x + 1/2 > 0)
        # 底部 PML: x > nz - bottom_pml
        bottom_term = 1 - 1j * beta * ((x - 1/2 - (nz - bottom_pml)) / max(max_pml_z, 1))**2 * (x - 1/2 - (nz - bottom_pml) > 0)
        return top_term + bottom_term - 1  # 减去1因为两项都包含1

    def pmlx(x):
        # 左侧 PML: x <= left_pml
        left_term = 1 - 1j * beta * ((left_pml - x + 1/2) / max(max_pml_x, 1))**2 * (left_pml - x + 1/2 > 0)
        # 右侧 PML: x > nx - right_pml
        right_term = 1 - 1j * beta * ((x - 1/2 - (nx - right_pml)) / max(max_pml_x, 1))**2 * (x - 1/2 - (nx - right_pml) > 0)
        return left_term + right_term - 1  # 减去1因为两项都包含1

    # 离散算子 L
    def L(pmlN, pmlP, n, h):
        return csr_matrix.multiply(
            spdiags([np.roll(pmlN, -1), -pmlN - pmlP, np.roll(pmlP, 1)], [-1, 0, 1], len(pmlN), n),
            1 / h**2
        )

    Dxx = L(1 / pmlx(xc - 1/2), 1 / pmlx(xc + 1/2), nx, h)
    Dzz = L(1 / pmlz(zc - 1/2), 1 / pmlz(zc + 1/2), nz, h)

    # DEL 算子
    Del = csr_matrix.multiply(
        sparse.kron(Dxx, spdiags(np.roll(pmlz(zc), 0), 0, nz, nz)) +
        sparse.kron(spdiags(np.roll(pmlx(xc), 0), 0, nx, nx), Dzz), b
    ) + csr_matrix.multiply(
        sparse.kron(Dxx, spdiags(np.roll(pmlz(zc + 1), 1), +1, nz, nz)) +
        sparse.kron(spdiags(np.roll(pmlx(xc + 1), 1), +1, nx, nx), Dzz), ((1 - b) / 2)
    ) + csr_matrix.multiply(
        sparse.kron(Dxx, spdiags(np.roll(pmlz(zc - 1), -1), -1, nz, nz)) +
        sparse.kron(spdiags(np.roll(pmlx(xc - 1), -1), -1, nx, nx), Dzz), ((1 - b) / 2)
    )

    vec = lambda x: np.reshape(x, (1, nx * nz), order='F')
    V = spdiags(vec(mv), [0], nx * nz, nx * nz)

    gridX, gridZ = numpy.meshgrid(xc, zc)
    C0 = np.multiply(pmlz(gridZ), pmlx(gridX))
    CC = spdiags(vec(C0), [0], nx * nz, nx * nz)

    ex = np.ones((1, nx))
    ez = np.ones((1, nz))
    D = lambda xs, zs: sparse.kron(spdiags(ex, xs, nx, nx), spdiags(ez, zs, nz, nz))

    I0 = csr_matrix.multiply(D(-1, 0) + D(1, 0) + D(0, -1) + D(0, 1), 0.25 * d)
    I45 = csr_matrix.multiply(D(-1, 1) + D(1, -1) + D(-1, -1) + D(1, 1), 0.25 * e)
    I = spdiags(np.ones(nx * nz) * c, 0, nz * nx, nx * nz)
    B = I + I0 + I45

    A = Del + csr_matrix.multiply(CC * (V * B), omega**2)

    return A, Del, B, CC
