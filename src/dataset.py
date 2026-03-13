import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu


class QEVDDecordDataset(Dataset):
    def __init__(
        self,
        metadata,
        class_map_path,
        frames_per_clip=16,
        transform=None,
    ):
        self.video_paths = metadata["video_paths"]
        self.num_frames_list = metadata["video_num_frames"]
        self.frames_per_clip = frames_per_clip
        self.transform = transform

        with open(class_map_path, "r") as f:
            self.class_to_idx = json.load(f)

        self.samples = []
        for p in self.video_paths:
            folder_name = Path(p).parent.name
            if folder_name in self.class_to_idx:
                self.samples.append((p, self.class_to_idx[folder_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            # ctx=cpu(0) ensures we use CPU for decoding to keep GPU free for training
            vr = VideoReader(path, ctx=cpu(0))
            T_total = self.num_frames_list[idx].item()

            # Use your "Dynamic Selection" logic
            indices = (
                torch.linspace(0, T_total - 1, self.frames_per_clip).long().tolist()
            )

            # Get frames: returns [T, H, W, C] uint8 numpy array
            video_data = vr.get_batch(indices).asnumpy()
            video = torch.from_numpy(video_data)  # [16, H, W, 3]

            # Permute to [T, C, H, W] for the spatial transforms (Resize/Crop)
            video = video.permute(0, 3, 1, 2)

        except Exception as e:
            # Fallback for corrupted clips
            video = torch.zeros((self.frames_per_clip, 3, 112, 112), dtype=torch.uint8)

        # Apply transforms (Presets will convert to float and flip to [C, T, H, W])
        if self.transform:
            video = self.transform(video)

        return video, label, idx
