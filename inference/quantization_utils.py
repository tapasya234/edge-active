import os
import numpy as np
import random


def load_calibration_samples(
    calib_dir="/Users/tapasyagutta/workspace/edge-active/output_tensors",
    num_samples=184,
    seed=15,
):
    random.seed(seed)
    files = []

    if not os.path.isdir(calib_dir):
        raise ValueError(f"Calibration data not found: {calib_dir}")

    for cls in sorted(os.listdir(calib_dir)):
        cls_dir = os.path.join(calib_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        candidates = [
            os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(".npy")
        ]

        if candidates:
            chosen = random.sample(candidates, min(2, len(candidates)))
            files.extend(chosen)

    print(f"Collected {len(files)} candidate calibration tensors")

    # if we still need more to reach num_samples
    if len(files) < num_samples:
        all_files = []
        for cls in sorted(os.listdir(calib_dir)):
            cls_dir = os.path.join(calib_dir, cls)
            if os.path.isdir(cls_dir):
                for f in os.listdir(cls_dir):
                    if f.endswith(".npy"):
                        all_files.append(os.path.join(cls_dir, f))

        extra = random.sample(all_files, num_samples - len(files))
        files.extend(extra)

    random.shuffle(files)
    files = files[:num_samples]
    tensors = []

    for f in files:
        arr = np.load(f).astype(np.float32)

        # ensure batch dimension exists
        if arr.shape != (1, 3, 16, 112, 112):
            raise ValueError(f"Calibration tensor shape mismatch: {f} -> {arr.shape}")

        tensors.append(arr)

    print(f"Using {len(tensors)} calibration samples")

    calibration_dataset = {"video": tensors}

    return calibration_dataset


if __name__ == "__main__":
    calib = load_calibration_samples()

    print(type(calib))
    print(calib.keys())
    print(len(calib["video"]))
    print(calib["video"][0].shape)
