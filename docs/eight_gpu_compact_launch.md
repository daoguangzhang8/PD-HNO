# 单机八卡：1000轮错峰输出

本机没有启动训练。将更新后的项目复制到目标八卡服务器后执行。

## 参数

- 全局模型条件样本BS64，每卡8；y BS1500，坐标梯度累计2。
- 完整PML（crop=0），Halton比例0.5，radius=squared，完整FiLM空间导数。
- 仅Data+PDE；学习率2e-4、Adam weight_decay=1e-4，warmup100轮，余弦参数沿用配置。
- 编号1..1000，共1000轮；不额外训练epoch0。
- NCCL集合通信超时10分钟，不是每步延迟。

## 周期核对

| 事件 | 轮次 | 文件行为 |
|---|---|---|
| 日志 | 每轮 | run.log中一行稳定的epoch摘要，另有运行诊断 |
| validation | 50,100,...,1000 | 只计算/追加valid loss，不立即写文件 |
| 常规图和两类Marmousi图 | 1（首次更新后）,51,102,...,969 | 更新各自最新图，不另存同轮多个版本 |
| 完整loss NPY | 同绘图轮次 | loss_history.npy覆盖更新；第1000轮再刷新最终完整历史 |
| checkpoint | 100,200,...,900；最终1000 | 周期保存，最后仅保存checkpoint_final.pth；首次出图不存checkpoint |

首次图片使用完成第1轮更新后的模型，不使用未更新的epoch0模型。51与100最早在5100相遇，当前1000轮内绘图不会与checkpoint重叠。50与100相遇不造成额外验证文件写入。两类Marmousi与常规绘图处在同一输出轮。

loss_history.npy是无需allow_pickle的结构化数组，字段：epoch、total、data、pde、raw_data、raw_pde、validation_data、validation_pde、lr、train_seconds。
非验证轮的validation字段为NaN，不能当作0；绘图使用真实epoch位置。

最终主要输出：runtime_config.json、run.log、loss_history.npy、loss_curve.png、wavefields.png、10个checkpoint，以及两个Marmousi子目录（各wavefields.png与metrics_latest.json）。
图片合并实部/虚部/真值/预测/误差，覆盖最新版本；完整loss历史不丢弃。

## 执行命令

下面路径必须替换为目标服务器真实路径。输出应使用本地高速盘的新目录，脚本拒绝覆盖已有目录。

```bash
tmux new-session -d -s pideeponet_bs64 \
  bash /path/to/PIDeeponet_old_from6004/scripts/launch_eight_gpu_compact.sh \
  /local_nvme/pideeponet_bs64_run1 \
  /path/to/pytorch/bin/python \
  /datasets/openfwi_curveflat_style_cpu \
  /datasets/external_test \
  /datasets/marmousi_unseen_sources_5hz_alpha0_160x180_x50_70_90_110_130_v3
```

五个位置参数依次为：输出目录、Python程序、五个训练NPY所在目录、Marmousi未见频率数据目录、Marmousi未见震源数据目录。
脚本内部使用CUDA 0..7、端口29501；不是torchrun，不要再包一层torchrun。输出和日志均由rank0/主启动器写到输出目录。

## 跳变与验证边界

- DDP按点累计、复数FNO同步、全局平均沿用已测试实现，不重复除以卡数或累计次数。
- validation固定成员和顺序且不丢尾批，避免每次随机遗漏一个验证样本。
- epoch日志总loss与data+pde使用相同归一化口径，原始值同时输出；不平滑、不隐藏异常值。
- baseline仍按首个训练epoch的逐点加权均值确定，然后固定。这不是训练前冻结权重的epoch0 baseline；新日志显式说明。
- 无法承诺真实loss不会震荡，也不能保证任意存储阻塞都不会超过10分钟。目标机器完整模型八卡NCCL吞吐/存储仍需上线短测。

本次不改变单卡旧入口的默认输出格式；新的compact启动方式限定从头开始、非分阶段DDP。旧输出目录和正在运行的任务不受影响。

## 异常快照

八卡入口rank0捕获可处理的异常、KeyboardInterrupt或SIGTERM时，尽力写出checkpoint_interrupted.pth，不在异常处理期间调用集合通信。
包含当前模型、Adam、调度器、warmup状态、已完成epoch、异常所在epoch、normalizer、已有loss历史和错误信息。
未完成的坐标累计梯度不保存，重启未完成epoch会重复部分样本，因此不是逐微批精确恢复。
若optimizer_step_in_progress=True，错误可能发生在部分参数更新过程中，不应直接视为安全续训点，应优先用上一份正常checkpoint。
SIGKILL、断电、底层NCCL直接终止进程、失效CUDA上下文、磁盘写入失败或父进程强制清理都可能阻止异常保存；周期checkpoint仍是可靠恢复的基础。

## Hugging Face 评估数据

两个完整目录已上传到daoguangzhang/openfwi的openfwi_curveflat_style_cpu/下：external_test/和marmousi_unseen_sources_5hz_alpha0_160x180_x50_70_90_110_130_v3/。
18个文件，共292727174字节，逐文件大小与SHA256校验通过。提交ae812bc4e0a08f299d697f0090d34fa7054a72a7。
