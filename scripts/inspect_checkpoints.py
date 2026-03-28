import sys
import types
from pathlib import Path
import json

# --- START HACK ---
# 1. Create a dummy module object
m = types.ModuleType("early_stopper")
# 2. Add it to sys.modules so 'import early_stopper' would work
sys.modules["early_stopper"] = m
# 3. Define a dummy class inside that module to match the pickled reference
# We use type() to create a class dynamically that does nothing
m.EarlyStopper = type("EarlyStopper", (), {})
# --- END HACK ---

import torch  # Now torch can load the checkpoint safely


def inspect_checkpoint(checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Checkpoint path recieved is not a file: {checkpoint_path}")

    # weights_only=False is required because EarlyStopper is a custom Python object
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    inspection_path = checkpoint_path.parent / "inspection.json"

    # Extract data
    output = {
        "Checkpoint Name": checkpoint_path.name,
        "Epoch": checkpoint.get("epoch", "N/A"),
    }

    # Extract early stopper state if it exists (handling both spellings just in case)
    stopper = checkpoint.get("early_stopper") or checkpoint.get("earky_stopper")
    if stopper:
        output["Early Stopper State"] = (
            vars(stopper) if hasattr(stopper, "__dict__") else stopper
        )

    if "args" in checkpoint:
        args_obj = checkpoint["args"]
        output["Training Configuration"] = (
            vars(args_obj) if hasattr(args_obj, "__dict__") else args_obj
        )

    if "metrics" in checkpoint:
        output["Final Metrics"] = {
            k: float(v) for k, v in checkpoint["metrics"].items()
        }

    with open(inspection_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True, default=str)

    print(f"Inspection data saved to {inspection_path}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <checkpoint_path>")
        sys.exit(1)

    inspect_checkpoint(sys.argv[1])
