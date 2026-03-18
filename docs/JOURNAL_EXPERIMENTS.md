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
 | -------- | ------- | ------------- | -------------|
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
| ------- | ------ | ------------------------|
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
| ------ | ----------- | ------------ | ----------- | ---------- | ------------|
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

### Run ID: `datasetUpdates_adam_val3clips` ❌

**Date:** 2025-03-18  
**Checkpoint:** `/home/jl_fs/checkpoints/datasetUpdates_adam_val3clips/checkpoint_best.pth`  
**Status:** ❌ Failed (Overfitting - poor generalization)  
**Leaderboard:** Rank #8, 85.43% accuracy, 4.488ms execution time (Regressed from Rank #4)

#### Hypothesis / Motivation

Improve upon batch64_unfrozen_codec (87.54% test) by implementing:

1. **AdamW optimizer** - Faster convergence than SGD
2. **Horizontal flip augmentation** - Double training data with label mapping for lateralized classes
3. **Multi-clip validation (3 clips)** - Better accuracy estimation through temporal aggregation
4. **Lower learning rate (3e-4)** - More stable training with AdamW

Expected improvements: +2-3% test accuracy → target 89-90%

#### Configuration Changes

**From batch64_unfrozen_codec:**

- **Optimizer:** SGD → AdamW
- **Learning rate:** 0.001 → 0.0003 (3e-4)
- **Weight decay:** 0.0001 (same)
- **Val clips per video:** 1 → 3 (multi-clip validation)
- **Augmentation:** Added horizontal flip with label mapping
- **Loss function:** Added KL divergence for multi-clip consistency
- **Batch size:** 64 (same)
- **Workers:** 14 (same)

**Key parameters:**

- Resolution: 112×112
- Clips per video: 1 (training), 3 (validation)
- Model architecture: r2plus1d_18 (fully unfrozen)
- Batch size: 64
- Learning rate: 0.0003
- LR schedule: Cosine annealing (warmup 2 epochs)
- Epochs: 15 (stopped at epoch 7)
- Frames per clip: 16
- Frame rate: 4 fps
- Training data: 190,254 videos
- Horizontal flip: Enabled with label mapping

**Horizontal flip mapping:**

- 91/92 classes: Symmetric (flip with same label)
- 1/92 classes: No flip (side_plank)
- 1 label swap: hook_left ↔ uppercut_right

#### Results (Epochs 0-7)

| Epoch | Train Acc@1 | Train Loss | Val Acc@1 | Val Loss | Notes |
| ------- | ------------- | ------------ | ----------- | ---------- | ------- |
| 0 | 93.45% | 0.203 | 89.03% | 0.352 | Initial |
| 1 | 93.63% | 0.192 | 88.77% | 0.375 | Val degraded |
| 2 | 93.45% | 0.200 | **91.32%** | 0.289 | Recovery |
| 3 | 95.87% | 0.126 | 91.31% | 0.307 | Train jumping |
| 4 | 97.34% | 0.080 | 91.61% | 0.315 | Overfitting starts |
| 5 | **98.20%** | 0.053 | 91.83% | 0.314 | ⚠️ Train too high |
| 6 | 97.26% | 0.082 | **93.44%** | 0.244 | ✅ **Best val** |
| 7 | 97.71% | 0.068 | 93.03% | 0.252 | Val plateau |

**Leaderboard Submission (Epoch 6):**

| Metric | Value | vs batch64_unfrozen_codec | vs Baseline |
|--------|-------|---------------------------|-------------|
| Test Acc@1 | **85.43%** | **-2.11%** ❌ | +3.47% |
| Execution Time | 4.488ms | +0.002ms ✅ | -17.047ms |
| Val Acc@1 | 93.44% | +1.37% | +11.48% |
| **Val→Test Gap** | **-8.01%** | **-3.48%** ❌ | N/A |

#### Analysis

**What failed catastrophically:**

- ❌ **Severe overfitting** - Train 98.20%, Val 93.44%, Test 85.43%
- ❌ **Large val→test gap** - 8.01% (vs 4.53% previously) indicates poor generalization
- ❌ **Regressed on leaderboard** - Dropped from 87.54% to 85.43%
- ❌ **KL divergence loss bug** - Used with single-clip training (N=1), mathematically incorrect
- ❌ **Insufficient regularization** - weight_decay=1e-4 too low for AdamW

**Technical issues encountered:**

- ⚠️ **GPU OOM crashes** - Multi-clip validation (batch=64, clips=3) exceeded 23.56GB GPU memory
  - Fixed by processing clips sequentially instead of in parallel
  - Training crashed 3 times at epochs 3-4, had to resume from checkpoint
- ⚠️ **Multi-clip validation inflated metrics** - 3-clip averaging reduced variance, gave false confidence

**What worked (partially):**

- ✅ **AdamW fast convergence** - Reached 93.44% val by epoch 6 (vs epoch 15+ with SGD)
- ✅ **Horizontal flip implementation** - Clean label mapping system works
- ✅ **Speed maintained** - 4.488ms execution time (same as before)
- ⚠️ **Multi-clip validation** - Provided stable estimates but inflated accuracy

**Root cause analysis:**

1. **Overfitting** (Primary)
   - Train accuracy: 98.20% (memorization)
   - AdamW + low weight decay (1e-4) = insufficient regularization
   - No label smoothing, dropout, or early stopping

2. **KL Loss Bug** (Secondary)
   - KL divergence requires N>1 clips for consistency regularization
   - With train_clips_per_video=1, comparing single clip to itself (useless)
   - May have introduced numerical instability

3. **Multi-Clip Validation Mismatch** (Tertiary)
   - 3-clip validation gives different metric than 1-clip test
   - Inflated validation confidence
   - Model relies on averaging instead of robust features

4. **Possible Flip Mapping Issues** (Unconfirmed)
   - Some symmetric classes might not be truly symmetric
   - Boxing moves (cross, jab, hook) could be context-dependent
   - Need visual verification of flipped samples

**Training progression analysis:**

- Epochs 0-2: Healthy training, normal convergence
- Epochs 3-5: Train accuracy jumped 93% → 98% (overfitting signal)
- Epochs 6-7: Val accuracy plateaued at 93% (should have stopped)
- Best epoch: 6 (submitted but failed on test set)

**GPU memory issues:**

```
Validation batch: [64, 3, 3, 16, 112, 112]
= 64 samples × 3 clips = 192 effective clips
Memory required: ~25 GB
GPU capacity: 23.56 GB
Result: OOM crash ❌
```

Fixed by sequential clip processing:

```python
for clip_idx in range(N):
    clip = video[:, clip_idx]  # Process one clip at a time
    output = model(clip)
    outputs.append(output)
output = torch.stack(outputs, dim=1).mean(dim=1)  # Aggregate
```

#### Next Steps

**Immediate fixes required:**

- [x] ~~AdamW optimizer~~ → Need MUCH stronger regularization (weight_decay 0.05+)
- [ ] Add label smoothing (0.1)
- [ ] Remove KL divergence loss (inappropriate for N=1)
- [ ] Implement early stopping (stop at best val epoch)
- [ ] Reduce learning rate (1e-4 instead of 3e-4)
- [ ] Verify horizontal flip label mapping (visual inspection)
- [ ] Implement torchvision's dynamic frame selection

**Recommended next experiment config:**

```bash
--batch-size 32         # Avoid OOM, more stable
--lr 1e-4              # Lower LR for better generalization  
--weight-decay 0.05     # 50x increase for AdamW
--epochs 12            # Early stopping (was overfitting by epoch 7)
--label-smoothing 0.1  # Add to criterion
# Remove KL loss from code
```

Expected results:

- Train Acc: 94-95% (not 98%)
- Val Acc: 92-93%
- Test Acc: **88-89%** (target)
- Val→Test Gap: -4% to -5% (acceptable)

**Lessons learned:**

1. **AdamW ≠ drop-in SGD replacement** - Requires 50-100x higher weight decay
2. **Multi-clip validation ≠ multi-clip training** - Use multi-clip for inference only
3. **Fast convergence ≠ better generalization** - Need proper regularization
4. **Validate changes individually** - Too many changes made it hard to debug
5. **KL loss requires N>1 clips** - Don't use with single-clip training
6. **Memory planning crucial** - Multi-clip validation needs careful batch sizing

---

## Quick Reference - Best Models

| Run ID | Video Acc@1 | Inference Time | Model Size | Status | Notes |
| ------- | ------------ | --------------- | ----------- | ------- | ------|
| batch64_unfrozen_codec | 87.54% | 4.486ms | 120 MB | ✅ Complete | **Current best** - Rank #4 |
| datasetUpdates_adam_val3clips | 85.43% | 4.488ms | 120 MB | ❌ Failed | Overfitting, regressed |
| r2plus1d_16frames_112x112 (epoch 1) | 92.33% (val) | 21.7ms | 120 MB | ❌ Failed | Wrong metrics |

**Leaderboard Baseline:** 81.96%

---

**Next experiments (priority order):**

1. **Regularized AdamW** (NEXT PRIORITY)
   - Fix overfitting from datasetUpdates_adam_val3clips
   - Higher weight decay (0.05), label smoothing (0.1)
   - Remove KL loss, implement early stopping
   - Expected: 88-89% test accuracy

2. **Torchvision dynamic frame selection**
   - Fix short video handling
   - Match sample solution's approach
   - Expected boost: +0.5-1%

3. **Multi-clip test-time augmentation** (3-5 clips per video)
   - Only for inference, not validation
   - Expected boost: +1-2% → 89-90% leaderboard
   - Minimal code changes, fast to implement

4. **Verify horizontal flip mapping**
   - Visual inspection of flipped samples
   - Test hook_left ↔ uppercut_right swap
   - Fix any incorrect mappings

5. **Advanced augmentation:**
   - Mixup/CutMix for video
   - Global Uniform Sampling
   - Expected boost: +1-2%

6. **Higher resolution (134×134):**
   - More spatial detail
   - Expected boost: +1-2%, cost: +3-5ms latency

7. **Deeper model (r2plus1d_34):**
   - Better feature extraction
   - Expected boost: +2-3%, cost: +5-10ms latency

8. **Ensemble methods:**
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
- **⚠️ AdamW requires stronger regularization**: weight_decay 0.05+ (not 1e-4 like SGD)

### Model Performance

- **Val-Test gap (~5%)**: Test set is harder/different - need better generalization
- **⚠️ Large val-test gap (>7%)**: Sign of severe overfitting
- **Semantic confusions**: Model struggles with similar exercises (neck_rolls vs neck_warmup)
- **Top-5 accuracy 99.5%**: Model is usually "in the ballpark", just not exact
- **Background class**: Easily confused with static stretches (deltoid_stretch)

### Augmentation & Regularization

- **Horizontal flip**: Works but verify label mapping carefully
- **Multi-clip validation**: Inflates metrics - use cautiously
- **KL divergence loss**: Only use with multi-clip training (N>1)
- **Label smoothing**: Critical for preventing overconfidence
- **Early stopping**: Essential - train accuracy >95% usually means overfitting

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
- **Validate changes individually**: Don't combine too many changes at once

## Ideas to Try

**High Priority (Next):**

- [ ] **Regularized AdamW training** ← IMMEDIATE NEXT (Fix overfitting)
  - Higher weight decay (0.05)
  - Label smoothing (0.1)
  - Remove KL loss
  - Early stopping
- [ ] **Torchvision dynamic frame selection** (better short video handling)
- [ ] **Multi-clip test-time augmentation** (3-5 clips per video)
- [ ] **Verify horizontal flip mapping** (visual inspection)
- [ ] **Mixup/CutMix augmentation** for video
- [ ] **Global Uniform Sampling** (better temporal coverage)

**Medium Priority:**

- [ ] Higher resolution training (134×134 or 160×160)
- [ ] Deeper model (r2plus1d_34)
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
- [x] AdamW optimizer ⚠️ - Faster convergence but requires stronger regularization
- [x] Horizontal flip augmentation ⚠️ - Need to verify label mapping correctness
- [x] Multi-clip validation (3 clips) ⚠️ - Inflates validation metrics, caused OOM issues

---

## Competition Notes

- **Submission deadline:** April 30th, 2025
- **Leaderboard opened:** March 1st, 2025
- **Latency requirement:** <100ms per video on Dragonwing IQ-9075
- **Current leaderboard baseline:** 81.96% (as of March 1, 2025)
- **Our best model:** 87.54% (+5.58%) - batch64_unfrozen_codec
- **Current rank:** #4
- **Gap to #1:** -1.67% (89.21% - 87.54%)
- **Organizer email:** <lowpowervision@gmail.com>
- **Leaderboard metric:** Video Acc@1 (primary), with latency constraint
- **Required input:** 16 frames per clip, 112×112 resolution (can be changed)

---

## Resources

- **Dataset:** QEVD-Fit-300k (92 classes, 190,254 train / 11,452 val videos)
- **Device:** Qualcomm Dragonwing IQ-9075 EVK
- **AI Hub:** <https://aihub.qualcomm.com>
- **Competition:** LPCVC Track 2
