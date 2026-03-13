from typing import Tuple

import torchvision
from torch import Tensor


class KineticsWithVideoId(torchvision.datasets.Kinetics):
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label, video_idx


class KineticsWithDynamicFrames(KineticsWithVideoId):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        video, audio, label, video_idx = super().__getitem__(idx)

        # Apply your dynamic frame selection logic here
        # video shape: (T, C, H, W) after transform
        selected_frames = self.select_frames_dynamically(video)

        return selected_frames, audio, label, video_idx

    def select_frames_dynamically(self, video):
        # Your dynamic frame selection logic from before
        # This is where you'd implement motion-based sampling,
        # optical flow analysis, or whatever strategy you choose
        pass
