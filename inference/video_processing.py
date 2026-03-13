import torch
import torchvision.transforms as transforms
from torchvision.io import read_video


def process_video(
    video_path: str,
    clip_len: int = 16,
    clip_strategy: str = "uniform",
    device=None,
    output_dtype=torch.float32,
):
    device = device or torch.device("cpu")

    # Load video
    video, _, _ = read_video(video_path, pts_unit="sec")
    video = video.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

    total_frames = video.shape[0]

    spatial = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),  # uint8 → [0, 1] float
            transforms.Resize((128, 171), antialias=False),
            transforms.CenterCrop((112, 112)),
        ]
    )

    # Dynamic frame selection
    if clip_strategy == "uniform":
        indices = torch.linspace(0, total_frames - 1, clip_len).long()
        indices = indices.clamp(max=total_frames - 1)
    else:
        indices = torch.arange(clip_len).clamp(max=total_frames - 1)

    clip = video[indices]  # (T, C, H, W)
    clip = spatial(clip)  # (T, C, 112, 112)
    clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)

    # Add batch dimension
    return clip.unsqueeze(0).to(device=device, dtype=output_dtype)
