# Edge-Active Experiment Journal

Track all training runs, model experiments, and optimization attempts for LPCVC Track 2.

---

## ⚠️ Important Note

**Previous experiments (baseline_run1, baseline_192x192, baseline_192x192_3clips) were moved to `INVALID_EXPERIMENTS.md`**

These experiments were trained on incorrectly cached validation data instead of training data, and used 8 frames instead of the required 16 frames. All experiments below use the correct dataset and 16 frames per clip.

---

## Journal Entry Template

```markdown
### Run ID: [run_name]
**Date:** YYYY-MM-DD  
**Checkpoint:** `./checkpoints/[run_name]/checkpoint_best.pth`  
**Status:** ✅ Complete | 🔄 Running | ❌ Failed

#### Hypothesis / Motivation
Why are you running this experiment? What do you expect to improve?

#### Configuration Changes
- **From baseline:** [What changed from the previous best model]
- **Key parameters:**
  - Resolution: 
  - Clips per video:
  - Model architecture:
  - Batch size:
  - Learning rate:
  - Epochs:

#### Results
| Metric | Value | vs Baseline | vs Previous |
|--------|-------|-------------|-------------|
| Video Acc@1 | X.XX% | +X.XX% | +X.XX% |
| Clip Acc@1 | X.XX% | +X.XX% | +X.XX% |
| Inference Time | X.X ms | +X.X ms | +X.X ms |
| Model Size | XXX MB | +XX MB | +XX MB |
| Training Time | X.X hrs | | |

#### Analysis
- What worked well?
- What didn't work?
- Unexpected observations?

#### Next Steps
- [ ] Action item 1
- [ ] Action item 2

---
```

## Experiments Log

### Run ID: `r2plus1d_16frames_112x112_10epochs` 🔄

**Date:** 2025-03-02  
**Checkpoint:** `./checkpoints/r2plus1d_16frames_112x112_10epochs/checkpoint_best.pth`  
**Status:** ❌ Failed (Epoch 4/10 complete)

#### Hypothesis / Motivation

First valid experiment with correct dataset and frame count:

1. Establish true baseline performance on full training set (190K videos)
2. Use 16 frames per clip (leaderboard requirement)
3. Verify model can achieve competition-level accuracy
4. Get on leaderboard to validate entire pipeline

#### Configuration Changes

- **First valid experiment** - trained on correct data
- **Key parameters:**
  - Resolution: 112×112
  - Clips per video: 1 (training), 3 (validation)
  - Model architecture: r2plus1d_18
  - Batch size: 20
  - Learning rate: 0.01
  - Epochs: 10
  - **Frames per clip: 16** ✅ (correct for leaderboard)
  - Frame rate: 4 fps
  - Training data: 190,254 videos (correct!)

#### Results (Epoch 1/10)

| Metric | Value | vs Leaderboard Baseline |
|--------|-------|-------------------------|
| Video Acc@1 | 92.33% | +10.37% |
| Clip Acc@1 | 90.74% | +8.78% |
| Video Acc@5 | 99.45% | N/A |
| Training Time (epoch) | ~6 hours | N/A |
| Estimated Total | ~60 hours | N/A |

#### Analysis (After Epoch 1)

**What's working well:**

- Though the accuracy calculated when running the scripts provided locally was `92.33`, the accuracy provided by the leaderboard was only `81.4%` which is lower than the baseline accuracy
- The inference time was `21.707` when the inference time of the baseline was `21.5`
- The accuracy did not improvide after the first epoch and it dropped quite a bit with epoch 4, so the run was cancelled.

**Observations:**

- Training on 190K videos takes ~8hrs/epoch (vs ~40min on wrong 11K dataset)
- Accuracy will likely improve to 96-97% by epoch 10
- Current model (epoch 1) is already submission-worthy

**Next steps during training:**

- [x] Export epoch 1 model and submit to leaderboard
- [ ] Test export/submission pipeline
- [ ] Wait for epoch 10 completion
- [ ] Submit improved epoch 10 model

---

### Run ID: `r2plus1d_vGPU_A5000_batch64_unfrozen`

**Status:** 🔄 Ready to Launch
**Hardware:** 1x A5000 Pro (24GB VRAM)

#### Strategy

- **Optimizer:** SGD, LR=0.001 (lower for full unfreeze).
- **Scheduler:** CosineAnnealing (iteration-based) + 2 Epoch Warmup.
- **Batch Size:** 64 (utilizing 24GB VRAM).
- **Goal:** Establish a "True Backbone" baseline without the Epoch 4 crash.

---

## Quick Reference - Best Models

| Run ID | Video Acc@1 | Inference Time | Model Size | Status | Notes |
|--------|-------------|----------------|------------|--------|-------|
| r2plus1d_16frames_112x112 (epoch 1) | 92.33% | TBD | 120 MB | 🔄 Training | First valid model, already beats baseline |

**Leaderboard Baseline:** 81.96%

---

## Lessons Learned

### Export Issues

- **Problem:** Dynamo exporter creates separate .onnx.data file
- **Solution:** Use `dynamo=False` for single self-contained ONNX file
- **Note:** Remove `dynamic_axes` for AI Hub compatibility

### Training Optimization

- **Layer freezing** speeds up training significantly when fine-tuning
- **Dataset caching** is essential (33 min → 0.3s load time)
- **Batch size** needs reduction when increasing resolution

### Qualcomm AI Hub

- **INT8 quantization** is extremely effective on Qualcomm NPU
- **Profiling** doesn't need input data, only measures latency
- **Status checking** needs `.code` attribute, not string comparison

---

## Ideas to Try

- [ ] Cosine annealing learning rate schedule
- [ ] Mixup/CutMix augmentation
- [ ] Global Uniform Sampling
- [ ] Ensemble multiple models at test time
- [ ] Test-time augmentation (multiple crops/clips per video)
- [ ] Knowledge distillation from larger model
- [ ] Optical flow as additional input channel
- [ ] Pre-training on larger video dataset (Kinetics-600, SSv2)
- [ ] Mixed precision training (FP16)
- [ ] Gradient accumulation for effectively larger batch sizes
- [ ] Different optimizers (AdamW, Lion)
- [ ] Label smoothing
