# Invalid Experiments - Archived

**⚠️ WARNING: All experiments in this file were trained on INCORRECTLY CACHED DATA**

## Issue Summary

**Problem Discovered:** 2025-03-02

A critical bug was found in the dataset caching mechanism:

- The training split was mistakenly using validation data (11,452 videos) instead of training data (190,254 videos)
- Additionally, models were trained with 8 frames per clip instead of the required 16 frames for LPCVC leaderboard
- This resulted in artificially inflated accuracy numbers that don't represent true model performance

**Impact:**

- All results below are **invalid** for competition submission
- Training was much faster than expected (suspicious sign)
- Accuracy numbers were unrealistically high
- Models cannot be submitted to leaderboard (wrong frame count)

**Resolution:**

- Dataset cache was deleted and regenerated with correct training split
- All future experiments use 16 frames per clip
- See main EXPERIMENT_JOURNAL.md for valid experiments

---

## Archived Experiments

### Run ID: `baseline_run1` ❌ INVALID

**Date:** 2025-02-17  
**Checkpoint:** `./checkpoints/baseline_run1/checkpoint_best.pth`  
**Status:** ❌ Invalid - Trained on validation data, 8 frames

#### Configuration

- Resolution: 112×112
- Clips per video: 1
- Model architecture: r2plus1d_18
- Batch size: 24
- Learning rate: 0.01
- Epochs: 15
- **Frames per clip: 8** ❌ (should be 16)
- Frame rate: 4 fps

#### Results (INVALID)

| Metric | Value |
|--------|-------|
| Video Acc@1 | 96.84% |
| Training Time | 6.3 hrs |

**Why Invalid:**

- Trained on only 11,452 validation videos (should be 190,254 training videos)
- Used 8 frames (leaderboard requires 16 frames)
- Cannot be submitted to competition

---

### Run ID: `baseline_192x192` ❌ INVALID

**Date:** 2025-02-XX  
**Checkpoint:** `./checkpoints/baseline_192x192/checkpoint_best.pth`  
**Status:** ❌ Invalid - Trained on validation data, 8 frames

#### Configuration

- Resolution: 192×192
- Clips per video: 1
- Model architecture: r2plus1d_18
- Batch size: 16
- Learning rate: 0.01
- Epochs: 15
- **Frames per clip: 8** ❌ (should be 16)

#### Results (INVALID)

| Metric | Value |
|--------|-------|
| Video Acc@1 | 98.59% |

**Why Invalid:**

- Same caching issue as baseline_run1
- Used 8 frames instead of 16
- Cannot be submitted to competition

---

### Run ID: `baseline_192x192_3clips` ❌ INVALID

**Date:** 2025-02-24  
**Checkpoint:** `./checkpoints/baseline_192x192_3clips/checkpoint_best.pth`  
**Status:** ❌ Invalid - Trained on validation data, 8 frames

#### Configuration

- Resolution: 192×192
- Clips per video: 3 (at test time)
- Model architecture: r2plus1d_18
- Batch size: 16
- Learning rate: 0.01
- Epochs: 15
- **Frames per clip: 8** ❌ (should be 16)

#### Results (INVALID)

| Metric | Value |
|--------|-------|
| Video Acc@1 | 99.28% |
| Clip Acc@1 | 98.01% |
| Video Acc@5 | 100.0% |

**Why Invalid:**

- Same caching issue as previous runs
- Used 8 frames instead of 16
- The 99.28% accuracy was suspiciously high - now we know why
- Cannot be submitted to competition

---

## Lessons Learned

### Data Integrity is Critical

1. **Always verify training data** - don't assume cached data is correct
2. **Check sample counts** - 11K videos for training was a red flag we missed
3. **Dataset size matters** - training time was suspiciously fast (should have been 3-4x longer)
4. **Verify competition requirements** - 16 frames was clearly specified but we missed it

### Signs of Invalid Data

- Training completed much faster than expected
- Accuracy numbers were unrealistically high (99.28%)
- Validation accuracy higher than typical for video classification
- Small number of training samples (11,452 vs expected 190,254)

### How to Prevent This

1. Print dataset statistics at start of training
2. Manually verify cache files match expected data
3. Delete caches when changing major configurations
4. Check sample paths to ensure correct split is being used
5. Read competition requirements carefully (frames per clip!)

---

## Moving Forward

All future experiments will:

- ✅ Use correct training split (190,254 videos)
- ✅ Use 16 frames per clip (leaderboard requirement)
- ✅ Verify dataset statistics before training
- ✅ Check cache freshness when changing configurations

See **JOURNAL_EXPERIMENTS.md** for all valid experiments going forward.
