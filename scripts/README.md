# Utility Scripts

Standalone scripts for data processing and analysis.

## Files

- **data_metadata_cache.py** - Create metadata cache for faster dataset loading
- **inspect_checkpoints.py** - Debug and inspect model checkpoints
- **analyse_errors.py** - Generate confusion matrix and error analysis

## Usage

```bash
# Create metadata cache (run once per dataset)
python scripts/data_metadata_cache.py
```

```bash
# Inspect a checkpoint
python scripts/inspect_checkpoints.py --checkpoint /path/to/model.pth
```

```bash
# Analyze model errors
python scripts/analyse_errors.py --checkpoint /path/to/model.pth
```
