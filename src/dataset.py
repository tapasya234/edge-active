import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu


class QEVDDecordDataset(Dataset):
    def __init__(
        self,
        metadata,
        class_map_path,
        class_flip_map_path,
        frames_per_clip=16,
        transform=None,
        is_train_dataset=False,
        num_clips_per_video=1,
    ):
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.is_train_dataset = is_train_dataset
        self.num_clips_per_video = num_clips_per_video

        if not self.is_train_dataset:
            self.num_clips_per_video = 1

        video_paths = metadata["video_paths"]
        num_frames_list = metadata["video_num_frames"]
        with open(class_map_path, "r") as f:
            class_to_idx = json.load(f)

        with open(class_flip_map_path, "r") as f:
            class_to_flipped_idx = json.load(f)

        self.samples = []
        for i, p in enumerate(video_paths):
            folder_name = Path(p).parent.name
            if folder_name in class_to_idx:
                self.samples.append(
                    (
                        p,
                        class_to_idx[folder_name],
                        class_to_flipped_idx[folder_name],
                        int(num_frames_list[i]),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx, flipped_idx, T_total = self.samples[idx]

        clips = []
        try:
            vr = VideoReader(path, ctx=cpu(0))

            indices = []
            for _ in range(self.num_clips_per_video):
                if self.is_train_dataset:
                    indices = self._get_sample_train_indices(T_total)
                else:
                    indices = self._get_sample_validation_indices(T_total)

                clip = vr.get_batch(indices).asnumpy()
                clip = torch.from_numpy(clip).to(torch.uint8)

                clip = clip.permute(0, 3, 1, 2).contiguous()

                if self.transform:
                    clip, did_flip = self.transform(clip, flipped_idx != -1)
                    if self.is_train_dataset and did_flip:
                        label_idx = flipped_idx

                clips.append(clip)

            video_tensor = torch.stack(clips)

        except Exception as e:
            # Fallback for corrupted clips
            video_tensor = torch.zeros(
                (self.num_clips_per_video, 3, self.frames_per_clip, 112, 112)
            )
        finally:
            del vr

        return video_tensor, label_idx, idx

    def _get_sample_train_indices(self, T_total):
        seg_size = T_total / self.frames_per_clip

        indices = []
        for i in range(self.frames_per_clip):
            start = int(i * seg_size)
            end = int((i + 1) * seg_size)

            if end > start:
                indices.append(random.randint(start, end - 1))
            else:
                indices.append(start)

        return [idx % T_total for idx in indices]

    def _get_sample_validation_indices(self, T_total):
        segment_size = T_total / self.frames_per_clip

        indices = [int((i + 0.5) * segment_size) for i in range(self.frames_per_clip)]
        indices = [min(i, T_total - 1) for i in indices]

        return indices
