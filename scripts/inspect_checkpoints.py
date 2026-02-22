import torch
import json
from pathlib import Path


def inspect_checkpoint(checkpoint_path: str):
    """
    Will save the 'args' and 'metrics' saved as a part of the checkpoints in a seperare json file.

    :param checkpoint_path: Path to the checkpoint that will be inspected.
    :type checkpoint_path: Path
    """

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Checkpoint path recieved is not a file: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    inspection_path = checkpoint_path.parent / "inspection.json"
    output = {
        "Checkpoint Name": checkpoint_path.name,
        "Epoch": checkpoint.get("epoch", "N/A"),
    }

    if "args" in checkpoint:
        args_obj = checkpoint["args"]
        if hasattr(args_obj, "__dict__"):
            args_obj = vars(args_obj)
        output["Training Configuration"] = args_obj

    if "metrics" in checkpoint:
        output["Final Metrics"] = {
            k: float(v) for k, v in checkpoint["metrics"].items()
        }

    with open(inspection_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)

    print(f"Inspection data saved to {inspection_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <checkpoint_path>")
        sys.exit(1)

    inspect_checkpoint(sys.argv[1])
