import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import json
from presets import VideoClassificationPresetEval

# Import your existing components
from decord_dataset import QEVDDecordDataset
from train import load_model, get_args_parser
import utils

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


def analyse():
    # 1. Setup Environment
    parser = get_args_parser()
    args = parser.parse_args([])  # Using defaults
    utils.init_distributed_mode(args)
    print(f"Args: {args}")
    device = torch.device("cuda")

    # 2. Load Dataset & Model
    class_map_path = Path(ROOT / "class_map.json")
    with open(class_map_path, "r") as f:
        class_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_idx.items()}

    # Load validation metadata (ensure this path is correct)
    metadata = torch.load(
        Path("/home/jl_fs/kinetics_cache/val/torchcodec_metadata.pt"),
        weights_only=False,
    )
    transform = VideoClassificationPresetEval(
        crop_size=tuple(args.val_crop_size),
        resize_size=tuple(args.val_resize_size),
    )
    dataset = QEVDDecordDataset(metadata, class_map_path, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        collate_fn=default_collate,
    )

    model = load_model(args, device, num_classes=92)
    checkpoint_path = Path(
        "/home/jl_fs/checkpoints/batch64_unfrozen_codec/checkpoint_best.pth"
    )
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 3. Collect Predictions
    all_preds = []
    all_targets = []

    print("Gathering predictions for Confusion Matrix...")
    with torch.inference_mode():
        for video, target, _ in tqdm(loader):
            output = model(video.to(device))
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())

    # 4. Calculate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)

    # 5. Identify Top Confusions
    # Zero out the diagonal (correct predictions) to find errors
    np.fill_diagonal(cm, 0)
    top_errors = np.unravel_index(np.argsort(cm, axis=None)[-10:], cm.shape)

    print("\n--- TOP 10 CONFUSIONS ---")
    for i in range(len(top_errors[0]) - 1, -1, -1):
        true_idx = top_errors[0][i]
        pred_idx = top_errors[1][i]
        count = cm[true_idx, pred_idx]
        print(
            f"True: {idx_to_class[true_idx]} | Predicted: {idx_to_class[pred_idx]} | Count: {count}"
        )

    # 6. Plot Mini Confusion Matrix (only for top problematic classes)
    # Get indices of classes that actually have the most errors
    row_sums = cm.sum(axis=1)
    # Find the top 20 indices that have the highest error counts
    # We use min() to ensure we don't request more classes than we actually have
    num_to_plot = min(20, len(cm))
    top_error_indices = np.argsort(row_sums)[-num_to_plot:]

    plt.figure(figsize=(14, 12))
    # Slice the matrix for both rows and columns using the top error indices
    sub_cm = cm[np.ix_(top_error_indices, top_error_indices)]

    sns.heatmap(
        sub_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[idx_to_class[i] for i in top_error_indices],
        yticklabels=[idx_to_class[i] for i in top_error_indices],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: Top {num_to_plot} Most Confused Classes")

    path = checkpoint_path.parent / "top_confusions.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"\nPlot saved to {path}")


if __name__ == "__main__":
    analyse()
