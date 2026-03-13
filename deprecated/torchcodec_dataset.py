import json
from pathlib import Path

import torch

from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder


class QEVDTorchCodecDataset(Dataset):
    def __init__(
        self,
        metadata,
        class_map_path,
        frames_per_clip=16,
        transform=None,
        output_format="NCHW",
    ):
        self.video_paths = metadata["video_paths"]
        self.num_frames_list = metadata["video_num_frames"]
        self.fps_list = metadata["video_fps"]
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.output_format = output_format

        # 1. Load the official mapping
        with open(class_map_path, "r") as f:
            self.class_to_idx = json.load(f)

        # 2. Reverse it to get class names if needed (e.g. for printing)
        self.classes = sorted(
            self.class_to_idx.keys(), key=lambda x: self.class_to_idx[x]
        )

        # 3. Pre-map samples to ensure every folder name matches the JSON
        self.samples = []
        for p in self.video_paths:
            # Assumes folder structure: .../split/class_name/video.mp4
            folder_name = Path(p).parent.name

            if folder_name in self.class_to_idx:
                self.samples.append((p, self.class_to_idx[folder_name]))
            else:
                # This helps catch typos like 'push-ups' vs 'pushups'
                print(
                    f"Warning: Folder '{folder_name}' not found in class_map. Skipping."
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        T_total = self.num_frames_list[idx].item()

        try:
            decoder = VideoDecoder(path, dimension_order=self.output_format)
            indices = torch.linspace(0, T_total - 1, self.frames_per_clip).long()

            frame_batch = decoder.get_frames_at_indices(indices.tolist())
            video = frame_batch.data  # [16, C, H, W]

        except Exception as e:
            # Fallback for corrupted files
            video = torch.zeros((self.frames_per_clip, 3, 112, 112), dtype=torch.uint8)

        if self.transform:
            video = self.transform(video)

        return video, label, idx
