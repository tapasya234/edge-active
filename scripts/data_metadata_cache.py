import os
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from torchvision.io import read_video_timestamps
from tqdm import tqdm
from decord import VideoReader, cpu


def get_single_video_metadata(path):
    try:
        # Initializing the decoder performs the necessary scan
        vr = VideoReader(path, ctx=cpu(0))

        return path, int(len(vr)), float(vr.get_avg_fps())
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return path, None, None


def run_manual_scan(split="train"):
    root_dir = f"/home/data/QEVD_organised/{split}"
    print(f"Indexing files in {root_dir}...")
    video_paths = [str(p) for p in sorted(Path(root_dir).rglob("*.mp4"))]
    print(f"Found {len(video_paths)} total videos. Starting manual parallel scan...")

    results = []
    with ProcessPoolExecutor(max_workers=14) as executor:
        for res in tqdm(
            executor.map(get_single_video_metadata, video_paths),
            total=len(video_paths),
            desc=f"Scanning {split}",
        ):
            if res[1] is not None:
                results.append(res)

    print(f"Successfully processed {len(results)}/{len(video_paths)} videos")

    metadata = {
        "video_paths": [r[0] for r in results],
        "video_num_frames": torch.tensor([r[1] for r in results], dtype=torch.int32),
        "video_fps": torch.tensor([r[2] for r in results], dtype=torch.float32),
    }

    cache_path = f"/home/jl_fs/kinetics_cache/{split}/torchcodec_metadata.pt"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(metadata, cache_path)
    print(f"Saved metadata to {cache_path} (Videos: {len(results)})")


if __name__ == "__main__":
    # Create metadata for both splits
    run_manual_scan("train")
    run_manual_scan("val")
