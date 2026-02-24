# Edge-Active Experiment Journal

Track all training runs, model experiments, and optimization attempts for LPCVC Track 2.

---

## Journal Entry Template

```markdown
### Run ID: [run_name]
**Date:** YYYY-MM-DD  
**Checkpoint:** `./checkpoints/[run_name]/checkpoint_best.pth`  
**Status:** ✅ Complete | 📋 Planned | 🔄 Running | ❌ Failed

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

### Run ID: `baseline_run1`

- **Date:** 2025-02-17  
- **Checkpoint:** `./checkpoints/baseline_run1/checkpoint_best.pth`  
- **Status:** ✅ Complete
- **Compile Job ID:** jg94xe1w5 (Quantized)
- **Profile Job ID:** jpevymdo5 (Quantized)

#### Hypothesis / Motivation

Establish baseline by fine-tuning official checkpoint on QEVD dataset to understand:

1. What accuracy can we achieve with minimal changes
2. How well the official checkpoint transfers to QEVD
3. Baseline inference time on Qualcomm hardware

#### Configuration Changes

- **From official checkpoint:** Fine-tuning with frozen early layers
- **Key parameters:**
  - Resolution: 112×112
  - Clips per video: 1
  - Model architecture: r2plus1d_18
  - Batch size: 24
  - Learning rate: 0.01
  - Epochs: 15
  - Frames per clip: 8
  - Frame rate: 4 fps

#### Results

| Metric | Value | vs Official Checkpoint |
| -------- | ------- | ------------------------ |
| Video Acc@1 | 96.84% | +13.68% |
| Clip Acc@1 | 96.84% | +13.68% |
| Inference Time | 2.3 ms | N/A |
| Model Size | 120 MB (33 MB INT8) | 0 MB |
| Training Time | 6.3 hrs | N/A |

#### Analysis

**What worked well:**

- Official checkpoint provided excellent starting point
- Layer freezing (only training layer4 + fc) sped up training significantly
- Dataset caching reduced data loading time from 33 min → 0.3s
- INT8 quantization on Qualcomm NPU is incredibly effective (2.3ms!)
- Accuracy jumped from 83.16% → 96.84% just from fine-tuning

**What didn't work:**

- Initial ONNX export issues with dynamo exporter splitting weights into separate file

**Unexpected observations:**

- 2.3ms inference is 43x faster than 100ms requirement - massive headroom for accuracy improvements
- Training took longer than expected (~6.3 hours on MPS) despite layer freezing
- Memory usage on device is tiny (1-4 MB) - can afford much larger models

#### Next Steps

- [x] Export model to ONNX successfully: <https://workbench.aihub.qualcomm.com/jobs/jg94xe1w5/>
- [x] Profile on Qualcomm AI Hub: <https://workbench.aihub.qualcomm.com/jobs/jpevymdo5/>
- [x] Increase resolution to 192×192 to improve spatial detail
- [ ] Try 3 clips per video for better temporal coverage
- [ ] Experiment with larger models (r2plus1d_34, mvit, etc.)

---

### Run ID: `baseline_192x192_load_official`

- **Date:** 2025-02-21  
- **Checkpoint:** `./checkpoints/baseline_192x192/checkpoint_best.pth`  
- **Status:** ✅ Complete
- **Compile Job ID:** jp87e2zo5 (Not Quantized)
- **Profile Job ID:** jp3mznmng (Not Quantized)
- **Compile Job ID:** jgjlrdlep (Quantized)
- **Profile Job ID:** jpev7ovv5 (Quantized)

#### Hypothesis / Motivation

Current 112×112 resolution may be losing important spatial details for activity recognition. Increasing to 192×192 should:

1. Capture finer-grained motion patterns
2. Better distinguish between similar activities
3. Still remain well under 100ms latency budget (estimate ~7-10ms)

#### Configuration Changes

- **From baseline_run1:** Resolution 112×112 → 192×192
- **Key parameters:**
  - Resolution: 192×192 (was 112×112)
  - Train resize: (220, 250) (was 128, 171)
  - Train crop: (192, 192) (was 112, 112)
  - Batch size: 16 (reduced from 24 due to memory)
  - Clips per video: 1
  - Model architecture: r2plus1d_18
  - Learning rate: 0.01
  - Epochs: 20

#### Results

| Metric | Value | vs Official Checkpoint |
| -------- | ------- | ------------------------ |
| Video Acc@1 | 98.59% | +13.68% |
| Clip Acc@1 | 98.59% | +13.68% |
| Inference Time | 1.9 ms (11ms FP32) | N/A |
| Model Size | 120 MB (91 MB FP32) (33 MB INT8) | 0 MB |
| Training Time | 20.3 hrs | N/A |

#### Analysis

*To be filled after training*

#### Next Steps

*To be determined based on results*

---

### Run ID: `baseline_192x192_3clips`

**Date:** 2025-02-22  
**Checkpoint:** `./checkpoints/baseline_192x192_3clips/checkpoint_best.pth`  
**Status:** ✅ Complete

#### Hypothesis / Motivation

Using only 1 clip per video during training may miss important temporal variations. Using 3 clips should:

1. Provide better temporal coverage of each activity
2. Make video-level aggregation more robust
3. Improve generalization to test set

#### Configuration Changes

- **From baseline_192x192:** Clips per video 1 → 3
- **Key parameters:**
  - Resolution: 192×192
  - Clips per video: 3 (was 1)
  - Model architecture: r2plus1d_18
  - Batch size: 16
  - Learning rate: 0.01
  - Epochs: 15
  - Training time: ~30 hours

#### Results

| Metric | Value | vs baseline_run1 | vs baseline_192x192 |
| -------- | ------- | ------------------ | --------------------- |
| Video Acc@1 | 99.28% | +2.44% | +0.69% |
| Clip Acc@1 | 98.01% | +1.17% | -0.58% |
| Video Acc@5 | 100.0% | +3.16% | +0.11% |
| Inference Time | TBD (est. ~5.7ms) | TBD | TBD |
| Training Time | ~30 hrs | +8.7 hrs | N/A |

#### Analysis

**What worked exceptionally well:**

- **3-clip aggregation** significantly improved video-level accuracy (+0.69%)
- **Top-5 accuracy reached 100%** from epoch 7 onwards - perfect on hard examples
- **Training remained stable** despite increased complexity
- **Minimal overfitting** - train/val gap remained small throughout

**Trade-offs:**

- Training took ~3x longer due to 3 clips per video
- Slightly lower clip-level accuracy (-0.58%) but much better video-level accuracy
- Inference will require 3 forward passes + aggregation (~5.7ms total, still excellent)

**Key insight:**
Multi-clip aggregation is **highly effective** for this dataset. The model learns complementary information from different temporal segments, and voting/averaging reduces errors.

#### Next Steps

- [x] Train with 192×192 resolution and 3 clips
- [ ] Profile inference time on AI Hub (expect ~5.7ms for 3 clips)
- [ ] Try different aggregation strategies (max, weighted average)
- [ ] Experiment with larger models (r2plus1d_34) now that we have latency headroom
- [ ] Try 5 clips per video to see if accuracy improves further
- [ ] Test-time augmentation with multiple crops

---

## Model Architecture Experiments (Future)

### Run ID: `r2plus1d_34_192x192`

**Status:** 📋 Planned

#### Hypothesis

r2plus1d_34 (34 layers) should provide better feature extraction than r2plus1d_18, improving accuracy with minimal latency increase.

---

### Run ID: `mvit_v2_s_192x192`

**Status:** 📋 Planned

#### Hypothesis

Modern vision transformer architecture (MViT v2) may outperform CNN-based approaches for video understanding.

---

## Quick Reference - Best Models

| Run ID | Video Acc@1 | Inference Time | Model Size | Notes |
| -------- | ------------- | ---------------- | ------------ | ------- |
| baseline_run1 | 96.84% | 2.3 ms | 120 MB | Official checkpoint fine-tuned |
| baseline_192x192 | 98.59% | TBD | 120 MB | Higher resolution |
| baseline_192x192_3clips | 99.28% | TBD | 120 MB | Multi-clip |

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

- [ ] Ensemble multiple models at test time
- [ ] Test-time augmentation (multiple crops/clips per video)
- [ ] Knowledge distillation from larger model
- [ ] Optical flow as additional input channel
- [ ] Pre-training on larger video dataset (Kinetics-600, SSv2)
- [ ] Mixed precision training (FP16)
- [ ] Gradient accumulation for effectively larger batch sizes
- [ ] Different optimizers (AdamW, Lion)
- [ ] Cosine annealing learning rate schedule
- [ ] Label smoothing
- [ ] Mixup/CutMix augmentation

---

## Competition Notes

- **Submission deadline:** March 1st, 2025
- **Latency requirement:** <100ms per video on Dragonwing IQ-9075
- **Current headroom:** ~97.7ms (2.3ms used)
- **Organizer email:** <lowpowervision@gmail.com>
- **Leaderboard metric:** Video Acc@1 (primary), with latency constraint

---

## Resources

- **Dataset:** QEVD-Fit-300k (92 classes, 11,452 videos)
- **Device:** Qualcomm Dragonwing IQ-9075 EVK
- **AI Hub:** <https://aihub.qualcomm.com>
- **Competition:** LPCVC Track 2
- **Baseline repo:** <https://github.com/lpcvai/26LPCVC_Track2_Sample_Solution>
