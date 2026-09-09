from config import *
from Labconfig import *
from model.utils import *

def plot_sinlge(model, args, times, vel_pred, UU0_pred, labels_pred, freq=None,
                source_coord=None):
    # 使用 pml_active 作为画图时的裁切力度（训练数据保留了 pml_active 个 PML 网格）
    L = args.pml_active
    L1 = args.pml_active * times
    # 使用标签的实际尺寸来确定网格大小
    actual_nz = labels_pred.shape[2]  # 实际的 z 维度
    actual_nx = labels_pred.shape[3]  # 实际的 x 维度
    Nz = actual_nz * times
    Nx = actual_nx * times

    nx = actual_nx
    nz = actual_nz
    spatial_step = float(args.dh)
    device = args.device

    # 根据边界类型计算裁切后的有效网格数（物理区域）
    if args.boundary_type == 'free_surface':
        # free_surface: 顶部不裁切，底部裁 L1，左右各裁 L1
        tag_nz = actual_nz * times - L1      # z 方向：只裁底部
        tag_nx = actual_nx * times - 2 * L1  # x 方向：左右都裁
    else:  # 'full_pml'
        # full_pml: 四边都裁切
        tag_nz = actual_nz * times - 2 * L1  # z 方向：上下都裁
        tag_nx = actual_nx * times - 2 * L1  # x 方向：左右都裁


    x_coords = torch.arange(0, nx * times)
    z_coords = torch.arange(0, nz * times)
    NN = x_coords.shape[0] * z_coords.shape[0]

    grid_z, grid_x = torch.meshgrid(z_coords, x_coords, indexing='ij')
    points = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=1)
    y_pred = points.float() * spatial_step / times

    dataset_test = TensorDataset(y_pred)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    model.eval()
    freq_batch_plot = freq.to(device) if freq is not None else None
    u_test = []
    with torch.no_grad():
        for batch in dataloader_test:
            y_pred_batch = batch[0].to(device)
            y_batch = y_pred_batch.unsqueeze(0)
            source_coord_plot = source_coord.to(device) if source_coord is not None else None
            u_pred_batch = model(
                vel_pred.to(device), y_batch, UU0_pred.to(device),
                freq_batch=freq_batch_plot,
                source_coord_batch=source_coord_plot,
            ).squeeze(0)
            u_test.append(u_pred_batch.detach().cpu().numpy())

    # 保存测试结果（修复：使用实际计算出的 Nz, Nx）
    U_pred_test = np.vstack(u_test)
    U_pred_test = U_pred_test.reshape(Nz, Nx, 2)

    # 根据边界类型确定切片范围
    if args.boundary_type == 'free_surface':
        z_slice_pred = slice(0, -L1)    # 顶部不切
    else:
        z_slice_pred = slice(L1, -L1)   # 上下都切
    x_slice_pred = slice(L1, -L1)       # 左右都切

    U_pred_real_test = U_pred_test[z_slice_pred, x_slice_pred, 0]
    U_pred_imag_test = U_pred_test[z_slice_pred, x_slice_pred, 1]

    y_pred_np = y_pred.detach().cpu().numpy()
    labels_pred_np = labels_pred.detach().cpu().numpy()
    U_test = labels_pred_np[0,:,:,:]

    # 标签数据使用原始网格的裁切（不乘 times）
    if args.boundary_type == 'free_surface':
        z_slice_label = slice(0, -L)    # 顶部不切
    else:
        z_slice_label = slice(L, -L)    # 上下都切
    x_slice_label = slice(L, -L)        # 左右都切

    U_test_real = U_test[0, z_slice_label, x_slice_label]
    U_test_imag = U_test[1, z_slice_label, x_slice_label]

    # 计算物理区域的实际距离范围（使用实际网格尺寸）
    physical_size_z = args.nz * args.dh
    physical_size_x = args.nx * args.dh

    # 为参考数据（裁切后的标签）生成横坐标
    x_test = np.linspace(0, physical_size_x, num=U_test_real.shape[1], endpoint=True)

    # 为预测数据生成横坐标
    x_pred = np.linspace(0, physical_size_x, num=U_pred_real_test.shape[1], endpoint=True)

    # ===================== 绘图逻辑修改 =====================
    figure1, ax1 = plt.subplots(figsize=(10, 6))
    # 绘制参考数据
    mid_x_idx = U_test_real.shape[1] // 2  # 取中间位置
    ax1.plot(x_test, U_test_real[:, mid_x_idx], label='Reference Real', linewidth=2, color='#ff7f0e')
    # 绘制预测数据
    mid_x_idx_pred = U_pred_real_test.shape[1] // 2
    ax1.plot(x_pred, U_pred_real_test[:, mid_x_idx_pred], label='Predicted Real', linewidth=1.5, color='#1f77b4')
    
    # 图表美化
    freq_str = f'{freq.item():.1f}Hz' if freq is not None else ''
    ax1.set_title(f'Real Part Comparison at Mid X-Line ({args.boundary_type}) {freq_str}', fontsize=12)
    ax1.set_xlabel('Distance (m)', fontsize=10)
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_xlim(0, physical_size_x)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # 保存图片
    plt.tight_layout()
    plt.savefig(args.save_doc + '/singleline.png', bbox_inches='tight')
    plt.close(figure1)



def plot_loss(epoch, save_doc, loss_log, loss_data_log, loss_pde_log, valid_u_loss, valid_f_loss, suffix=''):
    """
    绘制并保存训练和验证的损失曲线
    """
    train_loss_lenth = len(loss_data_log)
    valid_loss_lenth = len(valid_u_loss)
    
    # 构建 X 轴，确保与当前 epoch 对应
    train_x_axis = np.linspace(0, epoch, num=train_loss_lenth, endpoint=True)
    valid_x_axis = np.linspace(0, epoch, num=valid_loss_lenth, endpoint=True)
    
    # 确保保存路径存在 (可选防护)
    os.makedirs(save_doc, exist_ok=True)

    # 1. 绘制 Data Loss 曲线
    plt.figure()
    plt.plot(train_x_axis, loss_data_log, label="Training Data Loss")
    plt.plot(valid_x_axis, valid_u_loss, label="Valid Data Loss")
    plt.yscale('log')
    plt.legend()
    plt.title(f'epoch {epoch} Data Loss')
    plt.savefig(f"{save_doc}/Dataloss_curve{suffix}.png")
    plt.close()

    # 2. 绘制 PDE Loss 曲线
    plt.figure()
    plt.plot(train_x_axis, loss_pde_log, label="Training PDE Loss")
    plt.plot(valid_x_axis, valid_f_loss, label="Valid PDE Loss")
    plt.yscale('log')
    plt.legend()
    plt.title(f'epoch {epoch} PDE Loss')
    plt.savefig(f"{save_doc}/PDEloss_curve{suffix}.png")
    plt.close()

    # 3. 绘制 Total Loss 曲线
    plt.figure()
    plt.plot(loss_log, label="Total Loss")
    plt.yscale('log')
    plt.legend()
    plt.title(f'epoch {epoch} Total Loss')
    plt.savefig(f"{save_doc}/loss_curve{suffix}.png")
    plt.close()



def test_plot(args, model, fno, i, dataloader_y, vel, UU0, labels, filename,
              if_fine_tune, loc=2, freq=None, source_coord=None):
    if if_fine_tune:
        model = fine_tuning(args, model, fno, dataloader_y, vel, UU0, labels, freq=freq)
    model.eval()
    device = args.device
    L = args.pml_active
    freq_batch_plot = freq.to(device) if freq is not None else None
    source_coord_plot = source_coord.to(device) if source_coord is not None else None
    u_pred = []

    with torch.no_grad():
        for batch in dataloader_y:
            y_batch = batch[0].to(device)
            y_batch = y_batch.unsqueeze(0)
            u_batch = model(
                vel.to(device), y_batch, UU0.to(device),
                freq_batch=freq_batch_plot,
                source_coord_batch=source_coord_plot,
            ).squeeze(0)
            u_pred.append(u_batch.detach().cpu().numpy())

    # 处理预测结果（修复：使用实际数据尺寸而非配置参数）
    U_pred = np.vstack(u_pred)
    # 使用标签的实际尺寸来确定 reshape 参数
    actual_nz = labels.shape[2]  # 实际的 z 维度
    actual_nx = labels.shape[3]  # 实际的 x 维度
    U_pred = U_pred.reshape(actual_nz, actual_nx, 2)

    # 根据边界类型确定切片范围
    if args.boundary_type == 'free_surface':
        z_slice = slice(0, -L)    # 顶部不切
    else:
        z_slice = slice(L, -L)    # 上下都切
    x_slice = slice(L, -L)        # 左右都切

    U_pred_real = U_pred[z_slice, x_slice, 0]
    U_pred_imag = U_pred[z_slice, x_slice, 1]

    # 处理标签数据（与原逻辑一致）
    labels_np = labels.detach().cpu().numpy()
    U_ref = labels_np[0,:,:,:]
    U_ref_real = U_ref[0, z_slice, x_slice]
    U_ref_imag = U_ref[1, z_slice, x_slice]

    # 计算误差（与原逻辑一致）
    Umaxr, Uminr = np.max(U_ref_real), np.min(U_ref_real)
    Umaxi, Umini = np.max(U_ref_imag), np.min(U_ref_imag)
    Rr = np.maximum(np.abs(Umaxr), np.abs(Uminr))
    Ri = np.maximum(np.abs(Umaxi), np.abs(Umini))
    err_abs_real = U_ref_real - U_pred_real
    err_abs_imag = U_ref_imag - U_pred_imag
    eRr = np.max(np.abs(err_abs_real))
    eRi = np.max(np.abs(err_abs_imag))
    metrics_real = calculate_regression_metrics(U_pred_real, U_ref_real)

    # 2. 虚部指标
    metrics_imag = calculate_regression_metrics(U_pred_imag, U_ref_imag)
    print("="*60)
    print(f"模型预测性能指标汇总：{filename}")
    print("="*60)
    # 实部
    print(f"\n【实部】")
    print(f"MSE: {metrics_real['mse']:.6f}")
    print(f"MAE: {metrics_real['mae']:.6f}")
    print(f"R²:  {metrics_real['r2']:.6f}")
    # 虚部
    print(f"\n【虚部】")
    print(f"MSE: {metrics_imag['mse']:.6f}")
    print(f"MAE: {metrics_imag['mae']:.6f}")
    print(f"R²:  {metrics_imag['r2']:.6f}")
    # 绘制预测结果（统一 colorbar 范围）
    real_max = max(np.abs(U_ref_real).max(), np.abs(U_pred_real).max())
    imag_max = max(np.abs(U_ref_imag).max(), np.abs(U_pred_imag).max())

    freq_str = f'{freq.item():.1f}Hz' if freq is not None else ''

    fig1 = plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(U_ref_real, aspect='auto', cmap='seismic', vmin=-real_max, vmax=real_max)
    plt.title(f'ref real {freq_str}')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.subplot(2, 2, 2)
    plt.imshow(U_pred_real, aspect='auto', cmap='seismic', vmin=-real_max, vmax=real_max)
    plt.title(f'pred real epoch {i} {freq_str}')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.subplot(2, 2, 3)
    plt.imshow(U_ref_imag, aspect='auto', cmap='seismic', vmin=-imag_max, vmax=imag_max)
    plt.title(f'ref imag {freq_str}')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.subplot(2, 2, 4)
    plt.imshow(U_pred_imag, aspect='auto', cmap='seismic', vmin=-imag_max, vmax=imag_max)
    plt.title(f'pred imag epoch {i} {freq_str}')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.savefig(args.save_doc + '/epoch_plot_' + f'{filename}.png')
    plt.close()

    # 绘制误差图（与原逻辑一致）
    fig_e = plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(err_abs_real, aspect='auto', cmap='bwr')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title(f'epoch {i} {filename} real error')
    plt.colorbar()
    plt.clim(-eRr, eRr)

    plt.subplot(1, 2, 2)
    plt.imshow(err_abs_imag, aspect='auto', cmap='bwr')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title(f'epoch {i} {filename} imag error')
    plt.colorbar()
    plt.clim(-eRi, eRi)
    plt.savefig(args.save_doc + '/error_' + f'{filename}.png')
    plt.close()
    
def fine_tuning(args, model0, fno, dataloader_y, vel, UU0, labels, freq=None):
    device = args.device
    model_ft = copy.deepcopy(model0).to(device)
    NIter = args.ft_NIter
    lr = args.ft_lr
    a = args.ft_a
    b = args.ft_b
    c = args.ft_c
    nz = args.nz
    nx = args.nx
    pml_crop = args.pml_crop
    
    first_flag = True
    pde_norm_coe = 1.
    data_norm_coe = 1.
    
    optimizer = optim.Adam(model_ft.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.factor, patience=50, min_lr=1e-6
        )
    loss_f_log = []
    loss_u_log = []
    loss_r_log = []
    loss_op_batch = []
    
    # --- 用于追踪最佳模型的变量 ---
    best_loss = float('inf')
    best_model_state = None
    # -----------------------------------

    # --- 新增：记录微调开始时间 ---
    start_time = time.time()
    print(f" 开始在目标速度场上执行域适应微调 (总迭代次数: {NIter})...")
    # ------------------------------

    model_ft.train()
    with torch.no_grad():
        labels_fno = fno(vel.to(device), UU0.to(device)).to(device)
    optimizer.zero_grad()
    for i in range(NIter):

        batch_loss = []
        for batch in dataloader_y:
            y_batch = batch[0].to(device)
            y_batch = y_batch.unsqueeze(0).expand(vel.shape[0], -1, -1)
            
            # 计算损失
            freq_dev = freq.to(device) if freq is not None else None
            loss, loss_f, loss_u, loss_r, _, _ = model_ft.loss(
                vel.to(device), y_batch, UU0.to(device), labels.to(device),
                a, b, c, 0., data_norm_coe, pde_norm_coe, 1., freq_batch=freq_dev
            )
            loss_op = c * model_ft.loss_op(model0, vel.to(device), y_batch, UU0.to(device), freq_batch=freq_dev)
            loss = loss + loss_op
            
            # 修复：将除以累加步数后的结果重新赋值给 loss，否则梯度会按原比例回传
            loss = loss 
            
            # 反向传播
            loss.backward()
            
            # if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad() # 更新完后清空梯度
            scheduler.step(loss) # 传入当前损失以调整学习率

            batch_loss.append(loss.item())
            loss_u_log.append(loss_u.item())
            loss_f_log.append(loss_f.item())
            loss_op_batch.append(loss_op.item())
            
        # 动态权重更新
        if first_flag:
            data_norm_coe = np.mean(loss_u_log)
            pde_norm_coe = np.mean(loss_f_log)
            first_flag = False
            
        # 计算当前 Epoch 的平均总损失
        current_epoch_loss = np.mean(batch_loss)
        
        # 评估并保存最佳模型
        if i > 0 and current_epoch_loss < best_loss:
            best_loss = current_epoch_loss
            # 仅保存 state_dict，避免显存泄漏和深拷贝带来的开销
            best_model_state = copy.deepcopy(model_ft.state_dict())
            print(f"微调 {i}/{NIter} --> 发现最佳模型! 当前总损失: {best_loss:.6f}")

        print(f"微调 {i}/{NIter}, PDE损失: {np.mean(loss_f_log):.6f}, 数据损失: {np.mean(loss_u_log):.6f}, 锚定损失:{np.mean(loss_op_batch):.6f}")
        
        loss_f_log = []
        loss_u_log = []
        loss_r_log = []
        loss_op_batch = []

    # --- 新增：计算并格式化总耗时 ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    # ------------------------------

    if best_model_state is not None:
        print(f"\n 微调结束。总耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        print(f"正在加载历史最佳模型权重 (最低损失: {best_loss:.6f}) 用于后续测试与绘图...")
        model_ft.load_state_dict(best_model_state)
    else:
        print("\n 警告：未找到最佳模型（可能是迭代次数过少），将返回最终模型。")
        print(f"总耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    
    return model_ft
