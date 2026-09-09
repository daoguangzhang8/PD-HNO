from scipy import sparse
import numpy as np


def getFPML(n_pml, n):
    """
    创建从模型域到PML扩展域的扩展矩阵

    参数:
        n_pml: PML层厚度矩阵，形状为 (2, 2) 或 (2, 3)
               [[顶部PML, 左侧PML],
                [底部PML, 右侧PML]]
               对于3D: [[顶部, 左侧, 前面],
                       [底部, 右侧, 后面]]
        n: 模型尺寸，形状为 (2,) 或 (3,)
           [nz, nx] 或 [nz, nx, ny]

    返回:
        FPML: 扩展矩阵（稀疏矩阵）
    """

    # 检查是2D还是3D
    is_3d = (len(n) == 3)

    # X方向（水平方向）
    # MATLAB: PML1=sparse(model.n_pml(1,2),model.n(2));
    #         PML2=sparse(model.n_pml(2,2),model.n(2));
    PML1_x = sparse.csc_matrix((n_pml[0, 0], n[1]))  # 顶部PML
    PML2_x = sparse.csc_matrix((n_pml[1, 0], n[1]))  # 底部PML

    # MATLAB: PML1(:,1)=1; PML2(:,end)=1;
    if n_pml[0, 0] > 0:
        PML1_x_col = sparse.csc_matrix((n_pml[0, 0], n[1]))
        PML1_x_col[:, 0] = 1
        PML1_x = PML1_x_col

    if n_pml[1, 0] > 0:
        PML2_x_col = sparse.csc_matrix((n_pml[1, 0], n[1]))
        PML2_x_col[:, -1] = 1
        PML2_x = PML2_x_col

    # MATLAB: DUM1=[PML1;speye(model.n(2));PML2];
    DUM1 = sparse.vstack([PML1_x, sparse.eye(n[1]), PML2_x])

    # Z方向（垂直方向）
    # MATLAB: PML1=sparse(model.n_pml(1,1),model.n(1));
    #         PML2=sparse(model.n_pml(2,1),model.n(1));
    PML1_z = sparse.csc_matrix((n_pml[0, 1], n[0]))  # 左侧PML
    PML2_z = sparse.csc_matrix((n_pml[1, 1], n[0]))  # 右侧PML

    # MATLAB: PML1(:,1)=1; PML2(:,end)=1;
    if n_pml[0, 1] > 0:
        PML1_z_col = sparse.csc_matrix((n_pml[0, 1], n[0]))
        PML1_z_col[:, 0] = 1
        PML1_z = PML1_z_col

    if n_pml[1, 1] > 0:
        PML2_z_col = sparse.csc_matrix((n_pml[1, 1], n[0]))
        PML2_z_col[:, -1] = 1
        PML2_z = PML2_z_col

    # MATLAB: DUM2=[PML1;speye(model.n(1));PML2];
    DUM2 = sparse.vstack([PML1_z, sparse.eye(n[0]), PML2_z])

    if is_3d:
        # Y方向（第三维度）
        PML1_y = sparse.csc_matrix((n_pml[0, 2], n[2]))
        PML2_y = sparse.csc_matrix((n_pml[1, 2], n[2]))

        if n_pml[0, 2] > 0:
            PML1_y_col = sparse.csc_matrix((n_pml[0, 2], n[2]))
            PML1_y_col[:, 0] = 1
            PML1_y = PML1_y_col

        if n_pml[1, 2] > 0:
            PML2_y_col = sparse.csc_matrix((n_pml[1, 2], n[2]))
            PML2_y_col[:, -1] = 1
            PML2_y = PML2_y_col

        DUM0 = sparse.vstack([PML1_y, sparse.eye(n[2]), PML2_y])

        # MATLAB: FPML=kron(DUM0,kron(DUM1,DUM2));
        FPML = sparse.kron(DUM0, sparse.kron(DUM1, DUM2))
    else:
        # MATLAB: FPML=kron(DUM1,DUM2);
        FPML = sparse.kron(DUM1, DUM2)

    return FPML


# 保留旧版本的接口以保持兼容性
def getFPML_legacy(Lpml, nx, nz):
    """
    旧版本接口（保持向后兼容）
    假设四个边界的PML厚度相同
    """
    n_pml = np.array([[Lpml, Lpml], [Lpml, Lpml]])
    n = np.array([nz, nx])
    return getFPML(n_pml, n)
