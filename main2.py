"""
Physics-Informed DeepONet 训练入口

使用方法:
    python main2.py  (根据 config.py 中的 use_parallel 自动选择单卡/多卡)
"""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
import torch

from Labconfig import *
from config import *
from model.PI_DeepOnet import *
from model.plotting import *
from model.utils import get_available_gpus


def main(args=None):
    # 参数设置
    args = Args() if args is None else args

    # ==========================================
    # 检查训练模式
    # ==========================================
    use_parallel = getattr(args, 'use_parallel', False)
    use_staged = getattr(args, 'staged_training', False)

    if use_parallel:
        # 多 GPU 并行模式
        print("=" * 60)
        print("多 GPU 并行训练模式")
        print("=" * 60)

        # 检测可用 GPU
        num_gpus = getattr(args, 'num_gpus', 2)
        min_gpu_memory = getattr(args, 'min_gpu_memory', 10240)  # MB

        available_gpus = get_available_gpus(min_memory_mb=min_gpu_memory, require_count=num_gpus)

        requested = getattr(args, 'device_ids', None)
        device_ids = list(requested) if requested is not None else available_gpus[:num_gpus]
        if (len(device_ids) != num_gpus or len(set(device_ids)) != num_gpus
                or any(i not in available_gpus for i in device_ids)):
            raise RuntimeError(
                f'DDP 需要 {num_gpus} 张满足显存要求的可见 GPU，请求={device_ids}，'
                f'可用={available_gpus}；拒绝自动回退或替换设备。'
            )
        else:
            # DDP rank 是通信编号，不一定等于物理 GPU 编号。显式传递
            # CUDA_VISIBLE_DEVICES 映射后的进程内设备编号。
            args.device_ids = device_ids
            print(f"✅ 检测到 {len(available_gpus)} 个可用 GPU: {device_ids}")
            print("=" * 60)

            if use_staged:
                from model.train_distributed import train_distributed_staged
                train_distributed_staged(args, device_ids=device_ids)
            else:
                from model.train_distributed import train_distributed
                train_distributed(args, device_ids=device_ids)
            return

    # 单 GPU 模式
    print("=" * 60)
    print("单 GPU 训练模式")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(args.device)}")
        gpu_memory = torch.cuda.get_device_properties(args.device).total_memory / (1024 ** 3)
        print(f"GPU 内存: {gpu_memory:.1f} GB")
    else:
        print("⚠️ CUDA 不可用，将使用 CPU 训练")

    print("=" * 60)

    # 导入并调用单卡训练函数
    from model.train import train
    train(args)


if __name__ == "__main__":
    print('*******************************************')
    print('           START TRAINING Pi_DeepONet      ')
    print('*******************************************')
    main()
