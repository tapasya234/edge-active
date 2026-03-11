import json
from pathlib import Path

import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.video_utils import VideoClips


class QEVDVideoDataset(VisionDataset):
    """Video dataset for QEVD without Kinetics hardcoding"""

    def __init__(
        self,
        root,
        frames_per_clip,
        split="train",
        step_between_clips=1,
        transform=None,
        frame_rate=None,
        extensions=("mp4", "avi"),
        output_format="TCHW",
        _precomputed_metadata=None,
    ):
        super().__init__(root)
        self.clip_len = frames_per_clip
        self.split = split
        self.transform = transform
        self.extensions = extensions

        # Get canonical class list
        self.classes, self.class_to_idx = self._load_or_create_canonical_classes(root)
        self.num_classes = len(self.classes)

        # Get video paths from metadata or scan
        if _precomputed_metadata:
            video_paths = _precomputed_metadata["video_paths"]
        else:
            video_paths = self._scan_videos(Path(root) / split)

        # Build samples with correct labels
        self.samples = []
        valid_paths = []
        for path in video_paths:
            class_name = Path(path).parent.name
            if class_name in self.class_to_idx:
                label = self.class_to_idx[class_name]
                self.samples.append((path, label))
                valid_paths.append(path)

        # Create VideoClips with the SAME path ordering as samples
        if _precomputed_metadata:
            # Filter metadata to match valid_paths
            filtered_metadata = self._filter_metadata(
                _precomputed_metadata, valid_paths
            )
            self.video_clips = VideoClips(
                video_paths=valid_paths,
                clip_length_in_frames=frames_per_clip,
                frames_between_clips=step_between_clips,
                # frame_rate=frame_rate,
                frame_rate=None,
                _precomputed_metadata=filtered_metadata,
                output_format=output_format,
                num_workers=14,
                pts_unit="sec",
            )
        else:
            self.video_clips = VideoClips(
                video_paths=valid_paths,
                clip_length_in_frames=frames_per_clip,
                frames_between_clips=step_between_clips,
                # frame_rate=frame_rate,
                frame_rate=None,
                output_format=output_format,
                num_workers=14,
                pts_unit="sec",
            )

    def _load_or_create_canonical_classes(self, root):
        """Load or create canonical class list"""
        print("Loading or creating canonical classes for ", root)
        class_file = Path(root) / "canonical_classes.json"

        if class_file.exists():
            with open(class_file) as f:
                data = json.load(f)
            return data["classes"], data["class_to_idx"]

        # Create canonical list from both splits
        all_classes = set()
        for split in ["train", "val"]:
            split_dir = Path(root) / split
            if split_dir.exists():
                all_classes.update([d.name for d in split_dir.iterdir() if d.is_dir()])

        classes = sorted(list(all_classes))
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        # Save for future use
        with open(class_file, "w") as f:
            json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, indent=2)

        print(f"Created canonical class list: {len(classes)} classes")
        return classes, class_to_idx

    def _scan_videos(self, split_dir):
        """Scan directory for videos"""
        paths = []
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                for ext in self.extensions:
                    paths.extend([str(p) for p in sorted(class_dir.glob(f"*.{ext}"))])
        return paths

    def _filter_metadata(self, metadata, valid_paths):
        """Filter metadata to match valid_paths ordering"""
        path_to_idx = {p: i for i, p in enumerate(metadata["video_paths"])}

        filtered = {"video_paths": [], "video_pts": [], "video_fps": []}

        for path in valid_paths:
            if path in path_to_idx:
                idx = path_to_idx[path]
                filtered["video_paths"].append(path)
                filtered["video_pts"].append(metadata["video_pts"][idx])
                filtered["video_fps"].append(metadata["video_fps"][idx])

        return filtered

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        T = video.shape[0]

        # Implement "Dynamic Selection": Generate exactly 'clip_len' indices
        # spread across the available 'T' frames.
        # This handles both short videos (upsampling) and long videos (downsampling)
        if T != self.clip_len:
            indices = torch.linspace(0, T - 1, self.clip_len).long()
            video = video[indices]

        if self.transform is not None:
            video = self.transform(video)

        _, label = self.samples[video_idx]

        return video, audio, label, video_idx
