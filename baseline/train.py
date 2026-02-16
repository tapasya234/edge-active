import datetime
import os
import time
import warnings

from pathlib import Path

import random
import numpy as np


import presets
import torch
import torch.utils.data
import torchvision
import torchvision.datasets.video_utils
import utils
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.samplers import (
    DistributedSampler,
    RandomClipSampler,
    UniformClipSampler,
)
from kinetics_dataset import KineticsWithVideoId


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


def system_config(
    seed_value: int = 42, should_use_deterministic_algorithms: bool = False
) -> str:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.enabled = True

        if should_use_deterministic_algorithms:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.benchmark = True

        return "cuda"

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        torch.mps.manual_seed(seed_value)

        if should_use_deterministic_algorithms:
            torch.use_deterministic_algorithms(True)

        return "mps"

    if should_use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    return "cpu"


def train_one_epoch(
    model,
    criterion,
    optimizer,
    lr_scheduler,
    data_loader,
    device,
    epoch,
    print_freq,
    scaler=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter(
        "clips/s", utils.SmoothedValue(window_size=10, fmt="{value:.3f}")
    )

    header = f"Epoch: [{epoch}]"
    for video, _, target, _ in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        video, target = video.to(device), target.to(device)
        with torch.amp.autocast(device.type, enabled=scaler is not None):
            output = model(video)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = video.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["clips/s"].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()

    # Return average metrics
    return {
        "loss": metric_logger.loss.global_avg,
        "acc1": metric_logger.acc1.global_avg,
        "acc5": metric_logger.acc5.global_avg,
    }


def evaluate(model, criterion, data_loader, num_classes, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0
    # Group and aggregate output of a video
    num_videos = len(data_loader.dataset.samples)
    print(f"Evaluating {num_videos} videos\n")

    agg_preds = torch.zeros(
        (num_videos, num_classes), dtype=torch.float32, device=device
    )
    agg_targets = torch.zeros((num_videos), dtype=torch.int32, device=device)
    with torch.inference_mode():
        for video, _, target, video_idx in metric_logger.log_every(
            data_loader, 100, header
        ):
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss = criterion(output, target)

            # Use softmax to convert output into prediction probability
            preds = torch.softmax(output, dim=1)
            for b in range(video.size(0)):
                idx = video_idx[b].item()
                agg_preds[idx] += preds[b].detach()
                agg_targets[idx] = target[b].detach().item()

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = video.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if isinstance(data_loader.sampler, DistributedSampler):
        # Get the len of UniformClipSampler inside DistributedSampler
        num_data_from_sampler = len(data_loader.sampler.dataset)
    else:
        num_data_from_sampler = len(data_loader.sampler)

    if (
        hasattr(data_loader.dataset, "__len__")
        and num_data_from_sampler != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the sampler has {num_data_from_sampler} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(
        " * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5
        )
    )
    # Reduce the agg_preds and agg_targets from all gpu and show result
    agg_preds = utils.reduce_across_processes(agg_preds)
    agg_targets = utils.reduce_across_processes(
        agg_targets, op=torch.distributed.ReduceOp.MAX
    )
    agg_acc1, agg_acc5 = utils.accuracy(agg_preds, agg_targets, topk=(1, 5))
    print(
        " * Video Acc@1 {acc1:.3f} Video Acc@5 {acc5:.3f}".format(
            acc1=agg_acc1, acc5=agg_acc5
        )
    )

    # Return both clip-level and video-level metrics
    return {
        "clip_loss": metric_logger.loss.global_avg,
        "clip_acc1": metric_logger.acc1.global_avg,
        "clip_acc5": metric_logger.acc5.global_avg,
        "video_acc1": agg_acc1.item(),
        "video_acc5": agg_acc5.item(),
    }


def _get_cache_path(filepath, args):
    import hashlib

    value = f"{filepath}-{args.clip_len}-{args.kinetics_version}-{args.frame_rate}"
    h = hashlib.sha1(value.encode()).hexdigest()

    return (
        Path.home() / ".torch" / "vision" / "datasets" / "kinetics" / (h[:10] + ".pt")
    )


def collate_fn(batch):
    # remove audio from the batch
    return default_collate(batch)


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")
    val_resize_size = tuple(args.val_resize_size)
    val_crop_size = tuple(args.val_crop_size)
    train_resize_size = tuple(args.train_resize_size)
    train_crop_size = tuple(args.train_crop_size)

    data_path = Path(args.data_path)
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(train_dir, args)
    transform_train = presets.VideoClassificationPresetTrain(
        crop_size=train_crop_size, resize_size=train_resize_size
    )

    if args.cache_dataset and cache_path.exists():
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path, weights_only=False)
        dataset.transform = transform_train
    else:
        if args.distributed:
            print(
                "It is recommended to pre-compute the dataset cache on a single-gpu first, as it will be faster"
            )
        dataset = KineticsWithVideoId(
            args.data_path,
            frames_per_clip=args.clip_len,
            num_classes=args.kinetics_version,
            split="val",  # train
            step_between_clips=1,
            transform=transform_train,
            frame_rate=args.frame_rate,
            extensions=(
                "avi",
                "mp4",
            ),
            output_format="TCHW",
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            utils.save_on_master((dataset, train_dir), cache_path)

    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(val_dir, args)

    if args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        transform_test = weights.transforms()
    else:
        transform_test = presets.VideoClassificationPresetEval(
            crop_size=val_crop_size, resize_size=val_resize_size
        )

    if args.cache_dataset and cache_path.exists():
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path, weights_only=False)
        dataset_test.transform = transform_test
    else:
        if args.distributed:
            print(
                "It is recommended to pre-compute the dataset cache on a single-gpu first, as it will be faster"
            )
        dataset_test = KineticsWithVideoId(
            data_path,
            frames_per_clip=args.clip_len,
            num_classes=args.kinetics_version,
            split="val",  # "test",#
            step_between_clips=1,
            transform=transform_test,
            frame_rate=args.frame_rate,
            extensions=(
                "avi",
                "mp4",
            ),
            output_format="TCHW",
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            utils.save_on_master((dataset_test, val_dir), cache_path)

    print("Creating data loaders")
    print("Training samples: ", len(dataset))
    print("Validation samples: ", len(dataset_test))

    train_sampler = RandomClipSampler(dataset.video_clips, args.clips_per_video)
    test_sampler = UniformClipSampler(dataset_test.video_clips, args.clips_per_video)
    if args.distributed:
        train_sampler = DistributedSampler(train_sampler)
        test_sampler = DistributedSampler(test_sampler, shuffle=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    num_classes = len(dataset.classes)
    print("Creating model, num_classes: ", num_classes)

    model = torchvision.models.get_model(args.model, weights=args.weights)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # optional layer freezing for faster training
    for name, param in model.named_parameters():
        if not name.startswith("layer4") and not name.startswith("fc"):
            param.requires_grad = False

    # optional model loading
    if args.load_official_checkpoint:
        official_checkpoint = (
            ROOT / "checkpoints" / "official" / "official_checkpoint.pth"
        )
        if not official_checkpoint.exists():
            raise FileNotFoundError(
                f"Official checkpoint not found at {official_checkpoint}"
            )
        model_ckpt = torch.load(
            official_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(model_ckpt["model"])
        print(f"Loaded official checkpoint from {official_checkpoint}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(device) if args.amp else None

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    iters_per_epoch = len(data_loader)
    lr_milestones = [
        iters_per_epoch * (m - args.lr_warmup_epochs) for m in args.lr_milestones
    ]
    main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma
    )

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[warmup_iters],
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # We set weights_only to False because True gave error on cached dataset
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        start_time = time.time()
        history = {
            "val_clip_loss": [],
            "val_clip_acc1": [],
            "val_clip_acc5": [],
            "val_video_acc1": [],
            "val_video_acc5": [],
        }

        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        val_metrics = evaluate(
            model, criterion, data_loader_test, num_classes, device=device
        )

        history["val_clip_loss"].append(val_metrics["clip_loss"])
        history["val_clip_acc1"].append(val_metrics["clip_acc1"])
        history["val_clip_acc5"].append(val_metrics["clip_acc5"])
        history["val_video_acc1"].append(val_metrics["video_acc1"])
        history["val_video_acc5"].append(val_metrics["video_acc5"])

        # Save training history
        if output_dir:
            import json

            history_file = output_dir / "testing_history.json"
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Testing time {total_time_str}")
        return

    print("Start training")
    best_acc = 0.0  # Track best validation accuracy

    history = {
        "train_loss": [],
        "train_acc1": [],
        "train_acc5": [],
        "val_clip_loss": [],
        "val_clip_acc1": [],
        "val_clip_acc5": [],
        "val_video_acc1": [],
        "val_video_acc5": [],
    }

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model,
            criterion,
            optimizer,
            lr_scheduler,
            data_loader,
            device,
            epoch,
            args.print_freq,
            scaler,
        )

        val_metrics = evaluate(
            model, criterion, data_loader_test, len(dataset.classes), device=device
        )

        # Log metrics
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc1"].append(train_metrics["acc1"])
        history["train_acc5"].append(train_metrics["acc5"])
        history["val_clip_loss"].append(val_metrics["clip_loss"])
        history["val_clip_acc1"].append(val_metrics["clip_acc1"])
        history["val_clip_acc5"].append(val_metrics["clip_acc5"])
        history["val_video_acc1"].append(val_metrics["video_acc1"])
        history["val_video_acc5"].append(val_metrics["video_acc5"])

        print(f"\nEpoch {epoch} Summary:")
        print(
            f"  Train - Loss: {train_metrics['loss']:.4f}, Acc@1: {train_metrics['acc1']:.2f}%"
        )
        print(
            f"  Val   - Video Acc@1: {val_metrics['video_acc1']:.2f}%, Clip Acc@1: {val_metrics['clip_acc1']:.2f}%"
        )

        if output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "metrics": val_metrics,
                "history": history,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()

            latest_checkpoint = output_dir / "checkpoint_latest.pth"
            utils.save_on_master(checkpoint, latest_checkpoint)

            # Use video-level accuracy for best model
            if val_metrics["video_acc1"] > best_acc:
                best_acc = val_metrics["video_acc1"]
                best_checkpoint = output_dir / "checkpoint_best.pth"
                utils.save_on_master(checkpoint, best_checkpoint)
                print(f"✓ New best model! Video Acc@1: {best_acc:.2f}%")

    # Save training history
    if output_dir:
        import json

        history_file = output_dir / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Baseline Training", add_help=add_help)

    # Get the path to the dataset
    default_data_root = ROOT / "data" / "QEVD_organised"
    parser.add_argument(
        "--data-path", default=str(default_data_root), type=str, help="dataset path"
    )

    # Get the kinetics version to use as weights on the pre-trained model
    parser.add_argument(
        "--kinetics-version",
        default="400",
        type=str,
        choices=["400", "600"],
        help="Select kinetics version",
    )

    # Get the model to use when training the model
    parser.add_argument("--model", default="r2plus1d_18", type=str, help="model name")

    # Get the device to use when training the model
    # By default, it'll check if GPU is available.
    # If it isn't, then it'll check MPS and will fallback to CPU.
    device = system_config()
    parser.add_argument(
        "--device",
        default=device,
        type=str,
        help="device (Use cuda, mps or cpu Default: dependent on what is available)",
    )

    # The clip length, i.e., number of frames per clip, to use from each clip/video
    parser.add_argument(
        "--clip-len", default=8, type=int, metavar="N", help="number of frames per clip"
    )

    # The frame rate of the video used when training the model
    parser.add_argument(
        "--frame-rate", default=4, type=int, metavar="N", help="the frame rate"
    )

    parser.add_argument(
        "--clips-per-video",
        default=1,
        type=int,
        metavar="N",
        help="maximum number of clips per video to consider",
    )

    # The batch size to use when training the model
    parser.add_argument(
        "-b",
        "--batch-size",
        default=24,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )

    parser.add_argument(
        "--epochs",
        default=15,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=10,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 10)",
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-milestones",
        nargs="+",
        default=[20, 30, 40],
        type=int,
        help="decrease lr on milestones",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=10,
        type=int,
        help="the number of epochs to warmup (default: 10)",
    )
    parser.add_argument(
        "--lr-warmup-method",
        default="linear",
        type=str,
        help="the warmup method (default: linear)",
    )
    parser.add_argument(
        "--lr-warmup-decay", default=0.001, type=float, help="the decay for lr"
    )
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")

    checkpoints_dir = ROOT / "checkpoints"
    parser.add_argument(
        "--output-dir",
        default=str(checkpoints_dir),
        type=str,
        help="path to save outputs",
    )
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only.",
    )

    # distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )

    parser.add_argument(
        "--val-resize-size",
        default=(128, 171),
        nargs="+",
        type=int,
        help="the resize size used for validation (default: (128, 171))",
    )
    parser.add_argument(
        "--val-crop-size",
        default=(112, 112),
        nargs="+",
        type=int,
        help="the central crop size used for validation (default: (112, 112))",
    )
    parser.add_argument(
        "--train-resize-size",
        default=(128, 171),
        nargs="+",
        type=int,
        help="the resize size used for training (default: (128, 171))",
    )
    parser.add_argument(
        "--train-crop-size",
        default=(112, 112),
        nargs="+",
        type=int,
        help="the random crop size used for training (default: (112, 112))",
    )
    parser.add_argument(
        "--weights", default=None, type=str, help="the weights enum name to load"
    )

    # Mixed precision training parameters
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision training (CUDA only, limited MPS support)",
    )

    parser.add_argument(
        "--load-official-checkpoint",
        action="store_true",
        help="Load the official pretrained checkpoint before training",
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
