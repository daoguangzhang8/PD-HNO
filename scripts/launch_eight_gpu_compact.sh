#!/usr/bin/env bash
# Run on the TARGET eight-GPU server, not automatically by the assistant.
# Arguments: output_directory python_executable training_data_dir marmousi_freq_dir marmousi_source_dir
set -euo pipefail
if [[ "$#" -ne 5 ]]; then
    echo 'Usage: bash launch_eight_gpu_compact.sh OUTPUT PYTHON DATA_DIR MARMOUSI_FREQ_DIR MARMOUSI_SOURCE_DIR' >&2
    exit 2
fi
TRAIN_OUTPUT="$1"
TRAIN_PYTHON="$2"
TRAIN_DATA="$3"
TRAIN_MARMOUSI_FREQ="$4"
TRAIN_MARMOUSI_SOURCE="$5"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ -e "$TRAIN_OUTPUT" ]]; then
    echo "Refusing to overwrite existing output: $TRAIN_OUTPUT" >&2
    exit 2
fi
for TRAIN_DIRECTORY in "$TRAIN_DATA" "$TRAIN_MARMOUSI_FREQ" "$TRAIN_MARMOUSI_SOURCE"; do
    [[ -d "$TRAIN_DIRECTORY" ]] || { echo "Missing dataset directory: $TRAIN_DIRECTORY" >&2; exit 2; }
done
for TRAIN_FILENAME in freesurface_full_5sources_velocity.npy freesurface_full_5sources_background.npy freesurface_full_5sources_wavefield.npy freesurface_full_5sources_freq_used.npy source_grid_coords.npy; do
    [[ -f "$TRAIN_DATA/$TRAIN_FILENAME" ]] || { echo "Missing training file: $TRAIN_DATA/$TRAIN_FILENAME" >&2; exit 2; }
done
mkdir -p -- "$TRAIN_OUTPUT"
exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MKL_THREADING_LAYER=GNU MPLBACKEND=Agg \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
    "$TRAIN_PYTHON" -u run_main_with_overrides.py \
    --parallel --num-gpus 8 --gpu-ids 0 1 2 3 4 5 6 7 \
    --batch-size-v 64 --batch-size-y 1500 --accumulation-steps 2 \
    --from-scratch --main-losses-only --pml-crop 0 --source-radius-mode squared \
    --compact-output --epochs 1000 --nccl-timeout-minutes 10 \
    --validate-every 50 --save-fig-every 51 --save-model-every 100 \
    --master-port 29501 \
    --data-dir "$TRAIN_DATA" \
    --marmousi-freq-dir "$TRAIN_MARMOUSI_FREQ" \
    --marmousi-source-dir "$TRAIN_MARMOUSI_SOURCE" \
    --save-doc "$TRAIN_OUTPUT" > "$TRAIN_OUTPUT/run.log" 2>&1
