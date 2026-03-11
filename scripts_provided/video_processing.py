import torch
import torchvision.transforms as transforms
from torchvision.datasets.video_utils import VideoClips


def process_video(
    video_path: str,
    batch_size: int = 1,
    clip_len: int = 16,
    frame_rate: int = None,  # Set to None to use all available frames
    clip_strategy: str = "uniform",
    device: torch.device | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Load a video and return a batch of preprocessed clips for r2plus1d_18.

    Output shape: (B, 3, T, 112, 112), dtype `output_dtype`.

    Preprocessing pipeline:
      1. Decode frames at `frame_rate` fps.
      2. Convert uint8 pixel values to float32 in [0, 1].
      3. Resize the shorter side to 128×171, then centre-crop to 112×112.

    Note: mean/std normalisation (mean=(0.43216, 0.394666, 0.37645),
    std=(0.22803, 0.22145, 0.216989)) is intentionally NOT applied here —
    it is baked into the exported ONNX graph (inside NormalizedVideoModel).
    Applying it twice would produce incorrect results.

    Args:
        video_path:     Path to the input video file.
        batch_size:     Number of clips to sample from the video.
        clip_len:       Number of frames per clip. Must match the model input.
        frame_rate:     Target frames per second used when decoding.
        clip_strategy:  How to pick clips from the video:
                          "uniform" — evenly spaced clips (recommended),
                          "first"   — always use the first clip.
        device:         Torch device to move the output tensor to.
        output_dtype:   Output tensor dtype (default: float32).

    Returns:
        Tensor of shape (batch_size, 3, clip_len, 112, 112).
    """
    device = device or torch.device("cpu")

    # Use native FPS to avoid '0 clips found' errors on short videos
    video_clips = VideoClips(
        [video_path],
        clip_length_in_frames=clip_len,
        frames_between_clips=1,
        frame_rate=None,  # Changed from 4 to None
        output_format="TCHW",
    )

    # Calculate total frames available in the video
    # video_clips.video_pts[0] contains the timestamps of all frames
    total_frames = len(video_clips.video_pts[0])

    spatial = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),  # uint8 → [0, 1] float
            transforms.Resize((128, 171), antialias=False),
            transforms.CenterCrop((112, 112)),
        ]
    )

    # COMPETITION ALIGNMENT: Dynamic Frame Selection
    # Instead of picking a 'clip' from a list, we pick 16 indices
    # spread across the ENTIRE duration of the video.
    if clip_strategy == "uniform":
        # This maps 0-15 to the full range of the video
        indices = torch.linspace(0, total_frames - 1, steps=clip_len).long()
    else:
        indices = torch.arange(0, clip_len).long()

    # Load the specific frames
    video, audio, info, video_idx = video_clips.get_clip(0)  # Get base video

    # Select our 16 dynamic frames
    # video is (Total_T, C, H, W)
    clip = video[indices]

    # Apply spatial transforms: (16, 3, H, W) -> (16, 3, 112, 112)
    clip = spatial(clip)

    # Permute to (C, T, H, W) for R(2+1)D: (3, 16, 112, 112)
    clip = clip.permute(1, 0, 2, 3)

    # Add batch dimension: (1, 3, 16, 112, 112)
    return clip.unsqueeze(0).to(device=device, dtype=output_dtype)
