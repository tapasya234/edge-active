import torch
from torchvision.transforms import transforms
from transforms import ConvertBCHWtoCBHW

MEAN = (0.43216, 0.394666, 0.37645)
STD_DEV = (0.22803, 0.22145, 0.216989)


class VideoClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        resize_size,
        mean=MEAN,
        std=STD_DEV,
    ):
        self.prep = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                # We hard-code antialias=False to preserve results after we changed
                # its default from None to True (see
                # https://github.com/pytorch/vision/pull/7160)
                # TODO: we could re-train the video models with antialias=True?
                transforms.Resize(resize_size, antialias=False),
            ]
        )

        self.normalize = transforms.Normalize(mean=mean, std=std)

        self.crop = transforms.RandomCrop(crop_size)
        self.to_tensor = ConvertBCHWtoCBHW()

    def __call__(self, x, can_flip=False):
        x = self.prep(x)
        did_flip = False
        if can_flip and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])
            did_flip = True

        x = self.normalize(x)
        x = self.crop(x)
        x = self.to_tensor(x)

        return x, did_flip


class VideoClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size,
        mean=MEAN,
        std=STD_DEV,
    ):
        self.transforms = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                # We hard-code antialias=False to preserve results after we changed
                # its default from None to True (see
                # https://github.com/pytorch/vision/pull/7160)
                # TODO: we could re-train the video models with antialias=True?
                transforms.Resize(resize_size, antialias=False),
                transforms.Normalize(mean=mean, std=std),
                transforms.CenterCrop(crop_size),
                ConvertBCHWtoCBHW(),
            ]
        )

    def __call__(self, x, can_flip=False):
        # Returns None to match the did_flipped bool value returned by VideoClassificationPresetTrain
        return self.transforms(x), None
