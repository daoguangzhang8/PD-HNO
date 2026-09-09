# PD-HNO — Physics-informed operator learning

Single-machine eight-GPU training with full spatial derivatives through the
FiLM-conditioned PI-DeepONet and complex-valued FNO branches.

## Eight-GPU configuration

- Global model-condition batch 64: 8 per GPU.
- Coordinate microbatch 1500; accumulate 2 coordinate batches per update.
- Full PML (`pml_crop=0`); 50% Halton grid sampling; squared source-radius encoding.
- Data + PDE losses only; FiLM spatial derivatives remain enabled.
- 1000 training epochs, numbered 1–1000. Adam LR 2e-4, warmup 100 epochs.
- Validation every 50 epochs, without a separate file write.
- First plots after epoch 1, then every 51 epochs, including Marmousi unseen
  frequencies and unseen sources. Plots overwrite their latest versions.
- Checkpoints at 100–900 in increments of 100; final weights/state in
  `checkpoint_final.pth`. Catchable failures attempt `checkpoint_interrupted.pth`.
- NCCL collective timeout: 10 minutes. This is not a per-step delay.

## Setup and launch

Use a Linux machine with eight visible CUDA GPUs, a compatible driver, and tmux.
The tested PyTorch environment is 2.6.0+cu124. Install the matching CUDA-enabled
PyTorch build for your machine, plus numpy, scipy, matplotlib, h5py and tqdm.
`requirements.txt` is the original project's environment note, not a lockfile.

The datasets are hosted separately:
[OpenFWI and external evaluation datasets](https://huggingface.co/datasets/daoguangzhang/openfwi/tree/main/openfwi_curveflat_style_cpu).
Download the five training NPY files and both evaluation subdirectories. No
training data, model checkpoints, access tokens or run logs are committed here.

Replace all paths below with paths on the target server. Use a NEW output
directory on a fast local disk; the launcher refuses to overwrite an existing one.

```bash
tmux new-session -d -s pideeponet_bs64 \
  bash /path/to/PD-HNO/scripts/launch_eight_gpu_compact.sh \
  /local_nvme/pideeponet_bs64_run1 \
  /path/to/pytorch/bin/python \
  /datasets/openfwi_curveflat_style_cpu \
  /datasets/openfwi_curveflat_style_cpu/external_test \
  /datasets/openfwi_curveflat_style_cpu/marmousi_unseen_sources_5hz_alpha0_160x180_x50_70_90_110_130_v3
```

The launcher selects CUDA devices 0–7, port 29501, writes all console output to
`OUTPUT/run.log`, and uses `mp.spawn` internally. **Do not wrap it in torchrun.**
This launcher is for fresh single-machine training, not multi-node 4+4 training.
See [launch and output details](docs/eight_gpu_compact_launch.md).

## Loss history and recovery

`loss_history.npy` contains a complete structured history: actual epoch,
normalized total/data/PDE losses, raw data/PDE losses, validation losses, LR and
training time. Load it with `numpy.load(path, allow_pickle=False)`; validation
fields are NaN on epochs without validation.

The normalizers are point-weighted means from the first training epoch, then
fixed. They are not a frozen pre-training epoch-0 evaluation. Values are not
smoothed to conceal oscillations.

Interrupted checkpoints include current model, optimizer, scheduler, warmup,
normalizers and epoch metadata. They do not support exact microbatch replay;
pending accumulated gradients are discarded. An interruption inside an
optimizer update is flagged and may require the previous regular checkpoint.
SIGKILL, power loss, fatal CUDA/NCCL aborts or storage failures can prevent the
emergency checkpoint from being saved.

## Verification and limits

```bash
python -m unittest test_compact_output test_point_averaging \
  test_validation_graph_cleanup test_ddp_launch test_main_losses_only test_pde_derivatives
RUN_DISTRIBUTED_TESTS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest test_eight_rank_training test_parallel_alignment
```

CPU/Gloo eight-rank testing covers production PDE loss, complex FNO parameters,
coordinate tails and Adam updates against a same-data single-process BS64
reference. A complete PI-DeepONet also passed small CPU/Gloo update tests.
This does not replace a full-model, eight-physical-GPU NCCL smoke test on the
target hardware. Mathematical batch averaging equivalence does not guarantee
identical trajectories with different initialization, sampling or normalizers.

See [alignment audit](docs/eight_gpu_alignment_20260909.md) and
[PDE corrections](PDE_CORRECTIONS.md) for implementation details and limitations.
