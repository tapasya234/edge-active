import json
from pathlib import Path
import random
import math

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

        video_paths = metadata["video_paths"]
        num_frames_list = metadata["video_num_frames"]
        fps_list = metadata["video_fps"]

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
                        int(fps_list[i]),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx, flipped_idx, T_total, fps = self.samples[idx]

        clips = []
        try:
            vr = VideoReader(path, ctx=cpu(0))

            for _ in range(self.num_clips_per_video):
                if self.is_train_dataset:
                    indices = self._get_sample_train_indices(T_total, fps)
                else:
                    indices = self._get_sample_validation_indices(T_total, fps)

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
            print(f"Warning: Corrupted video at index {idx}, using zero tensor: {e}")
        finally:
            del vr

        return video_tensor, label_idx, idx

    def _compute_adjusted_frame_rate(self, T_total, fps=30.0):
        if T_total >= self.frames_per_clip:
            return fps

        video_duration = T_total / fps
        return math.ceil(self.frames_per_clip / video_duration)

    def _resample_indices(self, T_total, fps=30.0, target_frame_rate=None):
        if target_frame_rate is None:
            return list(range(T_total))

        if target_frame_rate == fps:
            return list(range(T_total))

        total_frames_after_resample = int(T_total * target_frame_rate / fps)

        step = fps / target_frame_rate
        indices = [int(i * step) for i in range(total_frames_after_resample)]
        return [min(idx, T_total - 1) for idx in indices]

    def _get_sample_train_indices(self, T_total, fps=30.0):
        # Adjust frame rate for short videos
        frame_rate = self._compute_adjusted_frame_rate(T_total, fps)

        # Resample to the adjusted frame rate
        resampled_indices = self._resample_indices(T_total, fps, frame_rate)

        # Sanity Check
        if len(resampled_indices) < self.frames_per_clip:
            indices = torch.linspace(
                0, len(resampled_indices) - 1, self.frames_per_clip
            ).long()
            return [resampled_indices[i] for i in indices]

        seg_size = len(resampled_indices) / self.frames_per_clip
        indices = []

        for i in range(self.frames_per_clip):
            start = int(i * seg_size)
            end = int((i + 1) * seg_size)

            if end > start:
                indices.append(random.randint(start, end - 1))
            else:
                indices.append(start)

        return [resampled_indices[idx] for idx in indices]

    def _get_sample_validation_indices(self, T_total, fps=30.0):
        # Adjust frame rate for short videos
        frame_rate = self._compute_adjusted_frame_rate(T_total, fps)

        # Resample to this frame rate
        resampled_indices = self._resample_indices(T_total, fps, frame_rate)

        # Sanity Check
        if len(resampled_indices) < self.frames_per_clip:
            indices = torch.linspace(
                0, len(resampled_indices) - 1, self.frames_per_clip
            ).long()
            return [resampled_indices[i] for i in indices]

        # Sample from center of each segment
        seg_size = len(resampled_indices) / self.frames_per_clip
        indices = [int((i + 0.5) * seg_size) for i in range(self.frames_per_clip)]
        return [min(i, len(resampled_indices) - 1) for i in indices]
