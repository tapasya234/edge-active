# Source Code (Training)

This directory contains the core training code.

## Files

- **train.py** - Main training script
- **dataset.py** - Decord-based dataset implementation
- **transforms.py** - Video augmentation transforms
- **presets.py** - Training/validation preset configurations
- **utils.py** - Training utilities (MetricLogger, etc.)

## Usage

```bash
# Basic training
python src/train.py \
    --data-path /path/to/QEVD_organised \
    --clip-len 16 \
    --batch-size 64 \
    --epochs 15

# With all options
python src/train.py \
    --load-official-checkpoint \
    --cache-dataset \
    --clip-len 16 \
    --frame-rate 4 \
    --epochs 15 \
    --batch-size 64 \
    --lr 0.001 \
    --output-dir ./checkpoints/my_experiment
```

## Adding New Models

1. Add model definition to `src/models/`
2. Import in `train.py`
3. Update argparse to include new model option
