import datetime
import time
import json
import random
import numpy as np
from pathlib import Path

import presets
import torch
import torch.utils.data
import torchvision
import utils
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.samplers import DistributedSampler


from dataset import QEVDDecordDataset
from save_configs import save_config

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"


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
        import os

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

    if device.type == "mps":
        torch.mps.empty_cache()

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter(
        "clips/s", utils.SmoothedValue(window_size=10, fmt="{value:.3f}")
    )

    header = f"Epoch: [{epoch}]"
    for video, target, _ in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        video, target = video.to(device), target.to(device)
        with torch.amp.autocast(
            device.type, enabled=(scaler is not None and device.type == "cuda")
        ):
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

    return {
        "loss": metric_logger.loss.global_avg,
        "acc1": metric_logger.acc1.global_avg,
        "acc5": metric_logger.acc5.global_avg,
    }


def evaluate(model, criterion, data_loader, num_classes, device):
    if device.type == "mps":
        torch.mps.empty_cache()

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    num_videos = len(data_loader.dataset.samples)

    agg_preds = torch.zeros((num_videos, num_classes), dtype=torch.float32)
    agg_targets = torch.zeros((num_videos,), dtype=torch.long)
    clip_counts = torch.zeros((num_videos,), dtype=torch.float32)

    with torch.inference_mode():
        for video, target, video_idx in metric_logger.log_every(
            data_loader, 100, header
        ):
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(video)
            loss = criterion(output, target)

            idx = video_idx.cpu()
            tgt = target.cpu()

            agg_preds.index_add_(0, idx, output.cpu())
            # clip_counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
            clip_counts.index_add_(0, idx, torch.ones(idx.shape[0]))
            agg_targets.index_copy_(0, idx, tgt)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = video.shape[0]

            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()

    clip_counts = clip_counts.clamp(min=1.0).unsqueeze(1)
    agg_preds = agg_preds / clip_counts

    print(
        f" * Clip Acc@1 {metric_logger.acc1.global_avg:.3f} "
        f"Clip Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )

    agg_acc1, agg_acc5 = utils.accuracy(agg_preds, agg_targets, topk=(1, 5))

    print(f" * Video Acc@1 {agg_acc1.item():.3f} Video Acc@5 {agg_acc5.item():.3f}")

    return {
        "clip_loss": metric_logger.loss.global_avg,
        "clip_acc1": metric_logger.acc1.global_avg,
        "clip_acc5": metric_logger.acc5.global_avg,
        "video_acc1": agg_acc1.item(),
        "video_acc5": agg_acc5.item(),
    }


def _get_precomputed_metadata_path(args, split="train"):
    if args.device == "cuda":
        cache_root = Path(f"/home/jl_fs/kinetics_cache/{split}")
        cache_root.mkdir(parents=True, exist_ok=True)

        return cache_root / "torchcodec_metadata.pt"

    return (
        Path.home()
        / ".torch"
        / "vision"
        / "datasets"
        / "kinetics"
        / split
        / "torchcodec_metadata.pt"
    )


def load_train_dataset(
    args, crop_size: tuple, resize_size: tuple, class_map_path: Path
):
    print("Loading training data")
    st = time.time()

    transform_train = presets.VideoClassificationPresetTrain(
        crop_size=crop_size, resize_size=resize_size
    )

    metadata = torch.load(
        _get_precomputed_metadata_path(args, "train"),
        weights_only=False,
    )

    dataset = QEVDDecordDataset(
        frames_per_clip=args.clip_len,
        transform=transform_train,
        metadata=metadata,
        class_map_path=class_map_path,
    )

    print(f"Training: {len(dataset.samples)} videos")
    print(f"Time: {time.time() - st:.2f}s")
    return dataset


def load_valid_dataset(
    args, crop_size: tuple, resize_size: tuple, class_map_path: Path
):
    print("Loading validation data")
    st = time.time()

    if args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        transform_test = weights.transforms()
    else:
        transform_test = presets.VideoClassificationPresetEval(
            crop_size=crop_size, resize_size=resize_size
        )

    metadata = torch.load(
        _get_precomputed_metadata_path(args, "val"),
        weights_only=False,
    )

    dataset = QEVDDecordDataset(
        frames_per_clip=args.clip_len,
        transform=transform_test,
        metadata=metadata,
        class_map_path=class_map_path,
    )

    print(f"Validation: {len(dataset.samples)} videos")
    print(f"Time: {time.time() - st:.2f}s")
    return dataset


def get_dataloader(args, dataset, sampler):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        # shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        pin_memory_device="cuda",
        collate_fn=default_collate,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=4 if args.workers > 0 else None,
    )


def test_model(
    model,
    criterion: nn.CrossEntropyLoss,
    test_data_loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
    output_dir: Path,
):
    print("Start testing")
    start_time = time.time()

    # Ensures bit-wise reproducible results for evaluation
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Run the evaluation logic
    val_metrics = evaluate(
        model, criterion, test_data_loader, num_classes, device=device
    )

    # Prepare history for JSON serialization
    history = {
        "val_clip_loss": [val_metrics["clip_loss"]],
        "val_clip_acc1": [val_metrics["clip_acc1"]],
        "val_clip_acc5": [val_metrics["clip_acc5"]],
        "val_video_acc1": [val_metrics["video_acc1"]],
        "val_video_acc5": [val_metrics["video_acc5"]],
    }

    save_history(history, output_dir, "testing_history.json")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Testing time {total_time_str}")
    return


def load_model(args, device: torch.device, num_classes: int):
    # 1. Initialize the standard model (defaults to 400 classes)
    model = torchvision.models.get_model(args.model, weights=args.weights)

    # 2. SWAP THE HEAD TO 92 FIRST
    # The checkpoint has shape [92, 512], so the model must too.
    print(f"Adjusting model head to {num_classes} BEFORE loading checkpoint")
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 3. NOW LOAD THE CHECKPOINT
    if args.load_official_checkpoint:
        official_checkpoint = (
            ROOT / "checkpoints" / "official" / "official_checkpoint.pth"
        )
        print(f"Loading weights from {official_checkpoint}")
        model_ckpt = torch.load(
            official_checkpoint, map_location="cpu", weights_only=False
        )

        # This will now succeed because both are 92
        model.load_state_dict(model_ckpt["model"], strict=True)

    # 4. Sync and Move to Device
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    for param in model.parameters():
        param.requires_grad = True

    return model.to(device)


def get_learning_rate_scheduler(args, train_dataloader, optimizer):
    iters_per_epoch = len(train_dataloader)

    total_cosine_iters = iters_per_epoch * (args.epochs - args.lr_warmup_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_cosine_iters, eta_min=1e-6
    )

    warmup_iters = iters_per_epoch * args.lr_warmup_epochs
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_lr_scheduler, scheduler],
        milestones=[warmup_iters],
    )


def save_history(history, output_dir, filename="training_history.json"):
    if output_dir and utils.is_main_process():
        with open(output_dir / filename, "w") as f:
            json.dump(history, f, indent=2)
        print(f"History saved to {filename}")


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True)

    args.data_path = Path(args.data_path)

    # Save the args used when running the command
    save_config(args)

    class_map_path = Path(ROOT / "class_map.json")
    valid_dataset = load_valid_dataset(
        args,
        tuple(args.val_crop_size),
        tuple(args.val_resize_size),
        class_map_path,
    )

    if args.distributed:
        test_sampler = DistributedSampler(valid_dataset, shuffle=False)
    else:
        test_sampler = torch.utils.data.SequentialSampler(valid_dataset)

    valid_dataloader = get_dataloader(args, valid_dataset, test_sampler)

    print("Validation Dataset")
    print("Total videos:", len(valid_dataset.samples))
    model = load_model(args, device=device, num_classes=92)
    criterion = nn.CrossEntropyLoss()

    if args.test_only:
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            model_without_ddp = model
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu]
                )
                model_without_ddp = model.module
            model_without_ddp.load_state_dict(checkpoint["model"])
            if args.amp and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
        test_model(
            model=model,
            criterion=criterion,
            test_data_loader=valid_dataloader,
            num_classes=92,
            device=device,
            output_dir=args.output_dir,
        )
        return

    train_dataset = load_train_dataset(
        args,
        tuple(args.train_crop_size),
        tuple(args.train_resize_size),
        class_map_path,
    )
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_dataloader = get_dataloader(args, train_dataset, train_sampler)

    print("Training Dataset")
    print("Total videos:", len(train_dataset.samples))

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scaler = (
        torch.amp.GradScaler(device) if args.amp and device.type == "cuda" else None
    )
    lr_scheduler = get_learning_rate_scheduler(args, train_dataloader, optimizer)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start training")
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

    best_acc = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

        if "history" in checkpoint:
            history = checkpoint["history"]
            print(f"Resuming history from epoch {checkpoint['epoch']}")

            if "val_video_acc1" in history and len(history["val_video_acc1"]) > 0:
                best_acc = max(history["val_video_acc1"])

    start_time = time.time()

    try:
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                model,
                criterion,
                optimizer,
                lr_scheduler,
                train_dataloader,
                device,
                epoch,
                args.print_freq,
                scaler,
            )
            val_metrics = evaluate(model, criterion, valid_dataloader, 92, device)

            # Log metrics
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc1"].append(train_metrics["acc1"])
            history["train_acc5"].append(train_metrics["acc5"])
            history["val_clip_loss"].append(val_metrics["clip_loss"])
            history["val_clip_acc1"].append(val_metrics["clip_acc1"])
            history["val_clip_acc5"].append(val_metrics["clip_acc5"])
            history["val_video_acc1"].append(val_metrics["video_acc1"])
            history["val_video_acc5"].append(val_metrics["video_acc5"])

            if args.output_dir and utils.is_main_process():
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
                utils.save_on_master(
                    checkpoint, args.output_dir / "checkpoint_latest.pth"
                )
                if val_metrics["video_acc1"] > best_acc:
                    best_acc = val_metrics["video_acc1"]
                    utils.save_on_master(
                        checkpoint, args.output_dir / "checkpoint_best.pth"
                    )

    finally:
        save_history(history, args.output_dir, "training_history.json")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")

        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

        # Force cleanup of MPS cache on Mac
        if device.type == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()

        # Explicitly exit to prevent hanging
        import sys

        sys.exit(0)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Baseline Training", add_help=add_help)
    parser.add_argument(
        "--data-path", default=str(ROOT / "data" / "QEVD_organised"), type=str
    )
    parser.add_argument(
        "--kinetics-version",
        default="400",
        type=str,
        choices=["400", "600"],
        help="Select kinetics version",
    )
    parser.add_argument("--model", default="r2plus1d_18", type=str)
    parser.add_argument(
        "--device",
        default=system_config(),
        type=str,
        help="device (Use cuda, mps or cpu Default: dependent on what is available)",
    )
    parser.add_argument(
        "--clip-len", default=16, type=int, help="number of frames per clip"
    )
    parser.add_argument("--frame-rate", default=4, type=int)
    parser.add_argument(
        "--train-clips-per-video",
        default=1,
        type=int,
        help="maximum number of clips per video to consider during training",
    )
    parser.add_argument(
        "--val-clips-per-video",
        default=1,
        type=int,
        help="maximum number of clips per video to consider during evaluation",
    )
    parser.add_argument("-b", "--batch-size", default=24, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("-j", "--workers", default=10, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--lr-milestones", nargs="+", default=[20, 30, 40], type=int)
    parser.add_argument("--lr-gamma", default=0.1, type=float)
    parser.add_argument("--lr-warmup-epochs", default=10, type=int)
    parser.add_argument("--lr-warmup-method", default="linear", type=str)
    parser.add_argument("--lr-warmup-decay", default=0.001, type=float)
    parser.add_argument("--print-freq", default=500, type=int)
    parser.add_argument("--output-dir", default=str(ROOT / "checkpoints"), type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--sync-bn", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--load-official-checkpoint", action="store_true")
    parser.add_argument("--val-resize-size", default=(128, 171), nargs="+", type=int)
    parser.add_argument("--val-crop-size", default=(112, 112), nargs="+", type=int)
    parser.add_argument("--train-resize-size", default=(128, 171), nargs="+", type=int)
    parser.add_argument("--train-crop-size", default=(112, 112), nargs="+", type=int)
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--dist-url", default="env://", type=str)
    return parser


if __name__ == "__main__":
    main(get_args_parser().parse_args())
