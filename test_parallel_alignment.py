"""单卡与 DDP 数据/梯度语义对齐的回归测试。"""

import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from model.dataloader import set_dataloader_epoch_seed
from model.train_distributed import (
    _assert_coordinate_batches_aligned,
    _ddp_batch_size_v,
    _ddp_continuous_batch_spec,
    _flush_ddp_accumulation,
    _manual_average_gradients,
    _validate_ddp_alignment_options,
)
from model.utils import Halton_Sample


def _build_tiny_physics_model():
    torch.manual_seed(1234)
    return nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
    )


def _tiny_physics_loss(model, inputs, labels):
    inputs = inputs.clone().requires_grad_(True)
    prediction = model(inputs)
    first_derivative = torch.autograd.grad(
        prediction,
        inputs,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True,
    )[0]
    second_derivative = torch.autograd.grad(
        first_derivative[:, :1],
        inputs,
        grad_outputs=torch.ones_like(first_derivative[:, :1]),
        create_graph=True,
    )[0][:, :1]
    return (
        torch.nn.functional.mse_loss(prediction, labels)
        + second_derivative.square().mean()
    )


def _distributed_gradient_worker(rank, world_size, init_path, output_dir):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _build_tiny_physics_model()
        ddp_model = torch.nn.parallel.DistributedDataParallel(model)

        rank_x = (
            torch.tensor([[1.0, 2.0], [2.0, 1.0]])
            if rank == 0
            else torch.tensor([[-1.0, 3.0], [4.0, -2.0]])
        )
        rank_y = (
            torch.tensor([[0.5], [1.0]])
            if rank == 0
            else torch.tensor([[-0.5], [2.0]])
        )
        with ddp_model.no_sync():
            _tiny_physics_loss(ddp_model, rank_x, rank_y).backward()
        _manual_average_gradients(model, world_size, bucket_mb=1)

        # 模拟 epoch 结尾只累积了 1/2 个窗口；flush 后应等价于一次完整
        # 的全局平均梯度更新，且不把梯度带入下一 epoch。
        flush_model = _build_tiny_physics_model()
        flush_ddp = torch.nn.parallel.DistributedDataParallel(flush_model)
        flush_optimizer = torch.optim.SGD(flush_model.parameters(), lr=0.01)
        with flush_ddp.no_sync():
            (_tiny_physics_loss(flush_ddp, rank_x, rank_y) / 2.0).backward()
        _flush_ddp_accumulation(
            flush_ddp,
            flush_optimizer,
            pending_steps=1,
            accumulation_steps=2,
            world_size=world_size,
        )
        flushed_parameters = [
            parameter.detach().clone() for parameter in flush_model.parameters()
        ]
        first_gradients = [
            parameter.grad.detach().clone() for parameter in model.parameters()
        ]

        # 连续两个 no_sync/manual-allreduce iteration，覆盖 DDP reducer 状态切换。
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.step()
        optimizer.zero_grad()
        with ddp_model.no_sync():
            _tiny_physics_loss(ddp_model, rank_x, rank_y).backward()
        _manual_average_gradients(model, world_size, bucket_mb=1)

        aligned_batches = [
            (torch.tensor([[0.0, 10.0], [20.0, 30.0]]),),
            (torch.tensor([[40.0, 50.0]]),),
        ]
        _assert_coordinate_batches_aligned(
            aligned_batches, torch.device("cpu"), rank, world_size
        )

        mismatched_batches = [
            (torch.tensor([[float(rank), 10.0], [20.0, 30.0]]),),
        ]
        mismatch_detected = False
        try:
            _assert_coordinate_batches_aligned(
                mismatched_batches, torch.device("cpu"), rank, world_size
            )
        except RuntimeError:
            mismatch_detected = True

        torch.save(
            {
                "gradients": first_gradients,
                "flushed_parameters": flushed_parameters,
                "mismatch_detected": mismatch_detected,
            },
            os.path.join(output_dir, f"rank_{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


class ParallelAlignmentTests(unittest.TestCase):
    def test_halton_seed_is_reproducible(self):
        first = np.asarray(Halton_Sample((17, 19), 100, seed=1234))
        second = np.asarray(Halton_Sample((17, 19), 100, seed=1234))
        other = np.asarray(Halton_Sample((17, 19), 100, seed=1235))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))

    def test_single_and_ddp_global_velocity_batches_match(self):
        seed = 20260410
        dataset = TensorDataset(torch.arange(73))
        single_loader = DataLoader(
            dataset,
            batch_size=36,
            sampler=RandomSampler(
                dataset, generator=torch.Generator().manual_seed(seed)
            ),
            drop_last=True,
            generator=torch.Generator().manual_seed(seed + 1_000_000),
        )
        single_batches = [batch[0].tolist() for batch in single_loader]

        rank_batches = []
        for rank in range(2):
            sampler = DistributedSampler(
                dataset,
                num_replicas=2,
                rank=rank,
                shuffle=True,
                seed=seed,
                drop_last=True,
            )
            sampler.set_epoch(0)
            loader = DataLoader(
                dataset, batch_size=18, sampler=sampler, drop_last=True
            )
            rank_batches.append([batch[0].tolist() for batch in loader])

        merged_batches = [
            [value for pair in zip(left, right) for value in pair]
            for left, right in zip(*rank_batches)
        ]
        self.assertEqual(single_batches, merged_batches)

    def test_coordinate_epoch_seed_can_be_replayed(self):
        seed = 20260412
        dataset = TensorDataset(torch.arange(25))
        loader = DataLoader(
            dataset,
            batch_size=6,
            sampler=RandomSampler(
                dataset, generator=torch.Generator().manual_seed(seed)
            ),
            generator=torch.Generator().manual_seed(seed + 1_000_000),
        )
        set_dataloader_epoch_seed(loader, seed, 9)
        first = [batch[0].tolist() for batch in loader]
        list(loader)
        set_dataloader_epoch_seed(loader, seed, 9)
        replay = [batch[0].tolist() for batch in loader]
        self.assertEqual(first, replay)

    def test_batch_and_continuous_pde_partition(self):
        args = SimpleNamespace(batch_size_v=36, ddp_split_batch_size_v=True)
        self.assertEqual(_ddp_batch_size_v(args, 2), 18)
        args.batch_size_v = 35
        with self.assertRaises(ValueError):
            _ddp_batch_size_v(args, 2)

        rank0_count, rank0_scale = _ddp_continuous_batch_spec(18, 2, 0, 0.25)
        rank1_count, rank1_scale = _ddp_continuous_batch_spec(18, 2, 1, 0.25)
        self.assertEqual((rank0_count, rank1_count), (5, 4))
        self.assertAlmostEqual((rank0_count + rank1_count), 9)
        self.assertAlmostEqual(rank0_scale, 10.0 / 9.0)
        self.assertAlmostEqual(rank1_scale, 8.0 / 9.0)

    def test_unsupported_ddp_paths_fail_fast(self):
        valid = SimpleNamespace(
            sampling_strategy="original", use_y_ran=False, branch2_type="fno"
        )
        _validate_ddp_alignment_options(valid)
        for override in (
            {"sampling_strategy": "sobol"},
            {"use_y_ran": True},
            {"branch2_type": "resnet"},
        ):
            invalid = SimpleNamespace(**vars(valid))
            for key, value in override.items():
                setattr(invalid, key, value)
            with self.assertRaises(NotImplementedError):
                _validate_ddp_alignment_options(invalid)

    @unittest.skipUnless(
        os.environ.get("RUN_DISTRIBUTED_TESTS") == "1",
        "设置 RUN_DISTRIBUTED_TESTS=1 后运行 CPU/Gloo 双进程测试",
    )
    def test_manual_gradient_average_and_coordinate_guard(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            init_path = os.path.join(temp_dir, "gloo_init")
            mp.spawn(
                _distributed_gradient_worker,
                args=(2, init_path, temp_dir),
                nprocs=2,
                join=True,
            )

            combined_x = torch.tensor([
                [1.0, 2.0], [2.0, 1.0], [-1.0, 3.0], [4.0, -2.0]
            ])
            combined_y = torch.tensor([[0.5], [1.0], [-0.5], [2.0]])
            reference = _build_tiny_physics_model()
            _tiny_physics_loss(reference, combined_x, combined_y).backward()

            flushed_reference = _build_tiny_physics_model()
            flushed_optimizer = torch.optim.SGD(
                flushed_reference.parameters(), lr=0.01
            )
            _tiny_physics_loss(
                flushed_reference, combined_x, combined_y
            ).backward()
            flushed_optimizer.step()

            for rank in range(2):
                result = torch.load(
                    os.path.join(temp_dir, f"rank_{rank}.pt"),
                    weights_only=True,
                )
                for distributed_gradient, parameter in zip(
                    result["gradients"], reference.parameters()
                ):
                    torch.testing.assert_close(
                        distributed_gradient, parameter.grad
                    )
                for distributed_parameter, reference_parameter in zip(
                    result["flushed_parameters"],
                    flushed_reference.parameters(),
                ):
                    torch.testing.assert_close(
                        distributed_parameter, reference_parameter
                    )
                self.assertTrue(result["mismatch_detected"])


if __name__ == "__main__":
    unittest.main()
