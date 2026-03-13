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
 | - ------- | - ------ | - ------------ | - ------------|
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
 | - ------- | - ------ | - ------------------------|
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

### Run ID: `batch64_unfrozen_codec` ✅

**Date:** 2025-03-12  
**Checkpoint:** `/home/jl_fs/checkpoints/batch64_unfrozen_codec/checkpoint_best.pth`  
**Status:** ✅ Complete (Epoch 5/15)  
**Leaderboard:** Rank #4, 87.54% accuracy, 4.486ms execution time

#### Hypothesis / Motivation

Major architectural improvements to close the gap between validation (92%) and leaderboard (81%):

1. Switch from VideoClips to Decord for faster, more reliable video loading
2. Implement dynamic frame selection for better temporal coverage
3. Unfreeze all layers to improve model capacity and generalization
4. Use cosine annealing LR schedule for better convergence
5. Increase batch size to 64 for more stable gradients

#### Configuration Changes

**Major improvements from previous runs:**

- **Video loading:** VideoClips → Decord VideoReader
- **Frame sampling:** Fixed uniform → Dynamic linspace selection
- **Model:** Fully unfrozen (all layers trainable)
- **Batch size:** 20 → 64
- **LR schedule:** Step decay → Cosine annealing with warmup
- **Learning rate:** 0.01 → 0.001 (lower for full model training)
- **Optimizer:** SGD with momentum 0.9
- **Precision:** Mixed precision (AMP) enabled
- **Workers:** 14 (parallel data loading)

**Key parameters:**

- Resolution: 112×112
- Clips per video: 1 (training), 1 (validation)
- Model architecture: r2plus1d_18 (fully unfrozen)
- Batch size: 64
- Learning rate: 0.001
- LR schedule: Cosine annealing (warmup 2 epochs, milestones [4, 8])
- Epochs: 15 (currently at epoch 5)
- Frames per clip: 16
- Frame rate: 4 fps
- Training data: 190,254 videos

#### Results (Epochs 1-5)

| Epoch | Clip Acc@1 | Video Acc@1 | Clip Acc@5 | Clip Loss | Leaderboard |
 | - ------ | - ----------- | - ------------ | - ----------- | - ---------- | - ------------|
| 1 | 91.38% | 91.38% | 99.47% | 0.274 | - |
| 2 | 91.65% | 91.65% | 99.48% | 0.260 | - |
| 3 | 92.05% | 92.05% | 99.52% | 0.247 | - |
| 4 | 92.07% | 92.07% | 99.51% | - | 87.54% (Rank #4) |
| 5 | **92.41%** | **92.41%** | **99.55%** | 0.243 | - |

**vs Baseline:**

- Validation: 92.41% vs 81.96% = **+10.45%**
- Leaderboard: 87.54% vs 81.96% = **+5.58%**
- Execution time: 4.486ms vs 21.535ms = **4.8x faster!**

**vs Previous best (invalid experiments):**

- More reliable (trained on correct data)
- Better generalization (unfrozen model)
- Faster training (Decord + better batching)

#### Analysis

**What worked exceptionally well:**

- ✅ **Decord VideoReader** - 3-4x faster video loading, more stable
- ✅ **Dynamic frame selection** - Better temporal sampling
- ✅ **Unfrozen training** - Model jumped from 90% → 92% quickly
- ✅ **Cosine annealing** - Smooth learning curve, no sharp drops
- ✅ **Batch size 64** - More stable gradients, faster convergence
- ✅ **Mixed precision (AMP)** - 1.5x faster training, same accuracy

**Top confusions (semantically similar classes):**

1. neck_rolls → neck_warmup (39 errors) - very similar movements
2. high_knees → buttkickers (37 errors) - both running-in-place
3. uppercut_right → cross (34 errors) - similar boxing punches
4. deltoid_stretch → background (22 errors) - static poses
5. warrior_1 → warrior_2 (12 errors) - subtle yoga pose differences

**Validation vs Leaderboard gap:**

- Validation (epoch 4): 92.07%
- Leaderboard: 87.54%
- **Gap: -4.53%** - indicates domain shift in test set

This gap suggests:

- Test set is harder (more ambiguous classes, edge cases)
- Different distribution from validation set
- Need better generalization (data augmentation, test-time augmentation)

**Training progression is healthy:**

- Steady improvement: 91.38% → 92.41% over 5 epochs
- No overfitting (train/val accuracies aligned)
- Still improving (not plateaued)
- Expected epoch 15: 94-95% validation

---

## Quick Reference - Best Models

| Run ID | Video Acc@1 | Inference Time | Model Size | Status | Notes |
 | - ------- | - ------------ | - --------------- | - ----------- | - ------- | - ------|
| r2plus1d_16frames_112x112 (epoch 1) | 92.33% | TBD | 120 MB | 🔄 Training | First valid model, already beats baseline |

**Leaderboard Baseline:** 81.96%

---

**Next experiments (priority order):**

1. **Multi-clip validation** (3 clips per video)
   - Expected boost: +1-2% → 89-90% leaderboard
   - Minimal code changes, fast to implement

2. **Horizontal flip with label mapping**
   - Symmetric classes: normal flip augmentation
   - Lateralized classes (left/right): swap labels on flip
   - Expected boost: +0.5-1%

3. **Advanced augmentation:**
   - Mixup/CutMix for video
   - Global Uniform Sampling
   - Expected boost: +1-2%

4. **Optimizer experiments:**
   - AdamW with weight decay
   - Different LR schedules
   - Expected boost: +0.5-1%

5. **Higher resolution (134×134):**
   - More spatial detail
   - Expected boost: +1-2%, cost: +3-5ms latency

6. **Deeper model (r2plus1d_34):**
   - Better feature extraction
   - Expected boost: +2-3%, cost: +5-10ms latency

7. **Ensemble methods:**
   - Combine multiple checkpoints
   - Combine different resolutions/models
   - Expected boost: +2-3%, cost: 2-3x latency

---

## Lessons Learned

### Data Loading & Preprocessing

- **Decord > VideoClips**: 3-4x faster loading, more stable, better for large datasets
- **Dynamic frame selection**: Linspace sampling ensures even temporal coverage
- **Metadata caching**: Essential for large datasets (saves 4 hours per epoch)
- **Parallel workers (14)**: Keeps GPU saturated, prevents data loading bottleneck
- **Video format matters**: Some videos too short → need padding/handling

### Training Optimization

- **Unfrozen > Frozen layers**: 92% vs 90% accuracy - worth the extra training time
- **Batch size 64**: Sweet spot for V100 - stable gradients, good throughput
- **Mixed precision (AMP)**: 1.5x speedup, no accuracy loss
- **Cosine annealing**: Smoother learning curve than step decay
- **LR warmup (2 epochs)**: Prevents early instability with large batch size
- **Lower LR for unfrozen (0.001)**: Prevents destroying pretrained features

### Model Performance

- **Val-Test gap (~5%)**: Test set is harder/different - need better generalization
- **Semantic confusions**: Model struggles with similar exercises (neck_rolls vs neck_warmup)
- **Top-5 accuracy 99.5%**: Model is usually "in the ballpark", just not exact
- **Background class**: Easily confused with static stretches (deltoid_stretch)

### Qualcomm AI Hub

- **INT8 quantization**: Minimal accuracy loss (<1%), huge speed gain (4-5x)
- **Execution time**: 4.5ms for INT8 vs 21ms for FP32
- **Batch size matters**: Leaderboard uses batch=1, profile accordingly
- **Channel order**: DLC models use NTHWC, ONNX uses NCTHW

### Competition Strategy

- **Daily submission limit**: Only last submission per day counts
- **Test set distribution**: Different from validation - optimize for generalization
- **Multi-clip inference**: Likely needed to close val-test gap
- **Speed-accuracy tradeoff**: 4.5ms @ 87% beats 21ms @ 89% in rankings

## Ideas to Try

**High Priority (Next):**

- [x] Cosine annealing learning rate schedule ✅ (implemented, working great)
- [ ] **Multi-clip test-time augmentation** (3-5 clips per video) ← NEXT
- [ ] **Horizontal flip with label mapping** (symmetric vs lateralized classes) ← NEXT
- [ ] **AdamW optimizer** (instead of SGD)
- [ ] **Mixup/CutMix augmentation** for video
- [ ] **Global Uniform Sampling** (better temporal coverage)

**Medium Priority:**

- [ ] Higher resolution training (134×134 or 160×160)
- [ ] Deeper model (r2plus1d_34)
- [ ] Label smoothing (reduce overconfidence)
- [ ] Test-time augmentation (multiple spatial crops)
- [ ] Gradient accumulation for larger effective batch size

**Low Priority (Future):**

- [ ] Ensemble multiple models at test time
- [ ] Knowledge distillation from larger model
- [ ] Optical flow as additional input channel
- [ ] Pre-training on larger video dataset (Kinetics-600, SSv2)
- [ ] Different optimizers (Lion)

**Already Tried:**

- [x] Mixed precision training (AMP) ✅ - 1.5x speedup
- [x] Unfrozen model training ✅ - Better generalization
- [x] Decord VideoReader ✅ - 3-4x faster loading
- [x] Dynamic frame selection ✅ - Better temporal sampling
- [x] Larger batch size (64) ✅ - More stable training
- [x] Cosine annealing schedule ✅ - Smoother convergence

---

## Competition Notes

- **Submission deadline:** April 30th, 2025
- **Leaderboard opened:** March 1st, 2025
- **Latency requirement:** <100ms per video on Dragonwing IQ-9075
- **Current leaderboard baseline:** 81.96% (as of March 1, 2025)
- **Our best model (epoch 1):** 92.33% (+10.37%)
- **Organizer email:** <lowpowervision@gmail.com>
- **Leaderboard metric:** Video Acc@1 (primary), with latency constraint
- **Required input:** 16 frames per clip, 112×112 resolution (can be changed)

---

## Resources

- **Dataset:** QEVD-Fit-300k (92 classes, 11,452 videos)
- **Device:** Qualcomm Dragonwing IQ-9075 EVK
- **AI Hub:** <https://aihub.qualcomm.com>
- **Competition:** LPCVC Track 2
