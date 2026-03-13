# Inference & Evaluation

This directory contains inference and evaluation code.

## Workflow

### Preprocess

The script will:

- Walk every `.mp4` file under `DATA_ROOT`, preserving the `<class_name>/<video>` hierarchy.
- Decode each video at `FRAME_RATE` fps and extract a `CLIP_LEN`-frame clip.
- Apply the standard `r2plus1d_18` spatial preprocessing (resize to 128×171, centre-crop to 112×112). **Note:** mean/std normalisation is intentionally omitted — it is baked into the exported model.
- Select a clip from the video using dynamic frame selection.
- Save each clip as a float32 `.npy` tensor of shape `(1, 3, T, 112, 112)` under `OUT_ROOT`.
- Write a `manifest.jsonl` to `OUT_ROOT` that records the video path, class label, tensor path, shape, and dtype for every sample — this file is required by `evaluate.py`.

```bash
python inference/preprocess_and_save.py
```

### Compile Model

`compile_model.py` is the **primary and recommended** tool for compiling the model on Qualcomm AI Hub for the [LPCVC Track 2](https://github.com/lpcvai/26LPCVC_Track2_Sample_Solution/tree/097404e5845dd265099b2fb19f253d8759799cca) and downloading the resulting `.bin` (QNN Context Binary) to the local machine. It handles model tracing, ONNX conversion, AI Hub submission, profiling, and download all in one go.

By default, the script will:

1. Trace the patched PyTorch model to TorchScript.
2. Submit a **compile job** to AI Hub targeting `QNN_CONTEXT_BINARY`.
3. Download the compiled `.bin` model to `./export_assets/` when the job completes.

Profiling and on-Hub inferencing are **skipped by default** (`skip_profiling=True`, `skip_inferencing=True`) to save time.

**Note:** Make sure to update the checkpoint before compiling the model

```bash
python inference/compile_model.py \
    --device "Dragonwing IQ-9075 EVK" \
    --target-runtime QNN_CONTEXT_BINARY \
    --precision w8a8 \
    --num-calibration-samples 128 \
    --do-profiling \
    --do-inferencing \
    --do-downloading
```

### On Device Inference

Once the compiled `.bin` from `example_export.py`  has been downloaded, use `run_inference.py` to run the full dataset through the model on AI Hub and collect the output logits.

**Configure the script:**

```python
# --- Model source: choose one ---
# Option A (recommended): pre-compiled binary from example_export.py
DLC_PATH  = "./export_assets/resnet_2plus1d.bin"
ONNX_DIR  = ""   # leave empty when using DLC_PATH

# Option B: compile from ONNX on-the-fly
DLC_PATH  = ""   # leave empty
ONNX_DIR  = "/path/to/onnx_dir"

# --- Data ---
data_path = "/path/to/preprocessed_tensors"   # OUT_ROOT from preprocess_and_save.py
OUTPUT_H5 = "dataset-export.h5"               # where results are written

# --- Input shape (must match your compiled model) ---
BATCH, C, T, H, W = 1, 3, 16, 112, 112

# --- Channel layout ---
# Set True only if the .bin was compiled with channel-last (NTHWC) input.
# The .bin from example_export.py uses channel-first (NCTHW), so keep False.
IS_DLC_CHANNEL_LAST = False

# --- Quick debug mode ---
USE_SINGLE_TENSOR    = False   # True to run only one sample
SINGLE_TENSOR_INDEX  = 0      # which tensor to pick from data_path
```

**Run:**

```bash
python inference/run_inference.py
```

The script:

1. Loads all `.npy` tensors from `data_path`, sorted in the same order as the `manifest.jsonl`.
2. Uploads or compiles the model on AI Hub.
3. Sends the dataset to AI Hub in chunks of 538 samples (to stay under the 2 GB flatbuffer limit).
4. Waits for all inference jobs to complete and collects the output logits.
5. Writes all logits to `OUTPUT_H5` (default `dataset-export.h5`) in HDF5 format for consumption by `evaluate.py`.

> :warning: Make sure `T` (frame count) and `IS_DLC_CHANNEL_LAST` match the model you compiled. Mismatches will cause silent accuracy degradation or shape errors.

### Evaluate results

`evaluate.py` loads the HDF5 output from `run_inference.py`, matches each prediction to the ground-truth label from `manifest.jsonl`, and reports Top-1 and Top-5 accuracy.

**Prerequisites:**

- `dataset-export.h5` — produced by `run_inference.py`
- `manifest.jsonl` — produced by `preprocess_and_save.py` (inside `OUT_ROOT`)
- `class_map.json` — maps class folder names to integer indices (should be in the repo root)

**Run:**

```bash
python inference/evaluate.py \
    --h5 dataset-export.h5 \
    --manifest /path/to/preprocessed_tensors/manifest.jsonl \
    --class_map class_map.json
```

For a quick sanity check on a single sample, first set `USE_SINGLE_TENSOR = True` in `run_inference.py`, run inference, then run `evaluate.py --verbose`.

### Share Model with LPCVC

Once, you are happy with the accuracy and inference metrics, make sure to share the compiled job with LPCVC before submitting the details.

```bash
python inference/share_job_lpcvc.py 
```
