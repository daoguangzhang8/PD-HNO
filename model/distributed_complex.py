"""DDP compatibility for native complex FNO parameters (PyTorch 2.6).

Real parameters use native DDP. Complex parameters retain their original
storage/state_dict and synchronize real/imaginary gradient views at group end.
"""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@torch.no_grad()
def broadcast_module_state(module):
    for tensor in list(module.parameters()) + list(module.buffers()):
        value = tensor.detach()
        dist.broadcast(torch.view_as_real(value) if value.is_complex() else value, src=0)


def wrap_complex_safe_ddp(module, device_id=None):
    ignored = [name for name, value in list(module.named_parameters()) + list(module.named_buffers())
               if value.is_complex()]
    DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(module, ignored)
    broadcast_module_state(module)
    return DistributedDataParallel(
        module, device_ids=None if device_id is None else [device_id],
        find_unused_parameters=False,
    )


@torch.no_grad()
def average_complex_gradients(module, bucket_bytes=32 * 1024 * 1024):
    """Call on EVERY rank after the synchronized backward, before optimizer.step.

    Independent real/imaginary averaging equals complex averaging. No extra
    division by accumulation_steps; that weighting was applied to each loss.
    """
    parameters = [p for p in module.parameters() if p.requires_grad and p.is_complex()]
    if not parameters:
        return
    present = torch.tensor([p.grad is not None for p in parameters],
                           device=parameters[0].device, dtype=torch.int32)
    dist.all_reduce(present)
    world_size = dist.get_world_size()
    if not bool((present == world_size).all()):
        raise RuntimeError('Missing complex FNO gradients on one or more ranks')
    bucket, size = [], 0

    def flush():
        if not bucket:
            return
        flat = torch.cat([g.reshape(-1) for g in bucket])
        dist.all_reduce(flat)
        flat.div_(world_size)
        offset = 0
        for grad in bucket:
            grad.copy_(flat[offset:offset + grad.numel()].view_as(grad))
            offset += grad.numel()

    for parameter in parameters:
        grad = torch.view_as_real(parameter.grad)
        grad_size = grad.numel() * grad.element_size()
        if bucket and size + grad_size > bucket_bytes:
            flush()
            bucket, size = [], 0
        bucket.append(grad)
        size += grad_size
    flush()
