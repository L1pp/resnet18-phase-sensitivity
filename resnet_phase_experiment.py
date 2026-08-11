"""ResNet18 stride/downsampling phase-sensitivity experiment.

This script deliberately keeps data generation and training separate.  Run
``python resnet_phase_experiment.py check`` first, then ``prepare`` to create
the compressed offline dataset.  Training is never started implicitly; the
default stage is ``check``.

The experiment compares three backbones:

* ``baseline``: torchvision ``resnet18(weights=None)`` with only ``fc``
  replaced by a one-dimensional coordinate head.
* ``antialias``: the same random ResNet18, but every stride-2 operation is
  changed to stride 1 followed by a fixed 3x3 binomial depthwise blur pool.
* ``no_gap``: the original ResNet18 with the final 7x7 feature map flattened
  into a small MLP, instead of the original adaptive average pool.

No external data are downloaded.  All configuration that affects generated
data or model construction is near the top of this file, and is included in
the cache fingerprint so stale data cannot accidentally be used for training.
Results use one fixed seed by default and should therefore be treated as an
exploratory diagnostic; repeat seeds are advisable before making a broad claim.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Experiment configuration.  Edit this block rather than scattering values
# through the implementation.
# ---------------------------------------------------------------------------

SEED = 20260810
IMAGE_SIZE = 224
# A 21-pixel segment leaves generous interior margin even for the 32..191
# dense sweep, while still being short enough to expose feature-map phase.
LINE_LENGTH = 21.0
LINE_THICKNESS = 3.0
LINE_Y = IMAGE_SIZE * 0.50
EDGE_MARGIN = 20.0

# Centre x is in pixels; target is x / (IMAGE_SIZE - 1).  Every integer bin
# has several different sub-pixel examples in train and an independent set of
# offsets in validation.  Neither split contains an exact integer centre,
# which leaves the dense integer test genuinely independent.
TRAIN_OFFSETS = (0.10, 0.30, 0.50, 0.70, 0.90)
VAL_OFFSETS = (0.05, 0.25, 0.45, 0.65, 0.85)
DENSE_TEST_START = 32
DENSE_TEST_COUNT = 160  # 160 = 5 * 32, convenient for residue/FFT inspection.

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0  # Windows-safe and intentionally offline/reproducible.
AMP = True
GRAD_CLIP_NORM = 5.0

NO_GAP_HIDDEN = 256
BLUR_KERNEL = ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0))
VARIANTS = ("baseline", "antialias", "no_gap")
PERIODS = (2, 4, 8, 16, 32)
EQUIV_SHIFTS = (1, 2, 4, 8, 16, 32)

# Outputs live beside this script.  This keeps the implementation self
# contained in the reviewable work directory requested by the parent agent.
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
EQUIV_DIR = OUTPUT_DIR / "equivariance"


def _jsonable_config() -> Dict[str, object]:
    """Return the complete data/model configuration used in fingerprints."""

    x_min = LINE_LENGTH / 2.0 + EDGE_MARGIN
    x_max = IMAGE_SIZE - LINE_LENGTH / 2.0 - EDGE_MARGIN
    return {
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "line_length": LINE_LENGTH,
        "line_thickness": LINE_THICKNESS,
        "line_y": LINE_Y,
        "edge_margin": EDGE_MARGIN,
        "x_min": x_min,
        "x_max": x_max,
        "train_offsets": list(TRAIN_OFFSETS),
        "val_offsets": list(VAL_OFFSETS),
        "dense_test_start": DENSE_TEST_START,
        "dense_test_count": DENSE_TEST_COUNT,
        "no_gap_hidden": NO_GAP_HIDDEN,
        "blur_kernel": BLUR_KERNEL,
        "variants": list(VARIANTS),
    }


def config_fingerprint() -> str:
    payload = json.dumps(_jsonable_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch, including CUDA when present."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms are useful for this controlled experiment.  A
    # warning (rather than a hard error) is appropriate on unusual backends.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_package_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:  # pragma: no cover - environment-specific
        return f"unavailable: {exc}"


def write_run_metadata(stage: str, device: torch.device | None = None, extra: Mapping[str, object] | None = None) -> None:
    """Persist reproducibility/environment metadata without requiring extra packages.

    ``run_metadata.json`` is updated at each pipeline stage and retains a short
    stage history, so a later training/analyze run does not erase the original
    environment and cache provenance.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUTPUT_DIR / "run_metadata.json"
    previous: Dict[str, object] = {}
    if metadata_path.exists():
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    if device is None:
        device = choose_device()
    now = _datetime.datetime.now(_datetime.timezone.utc).isoformat()
    stage_entry: Dict[str, object] = {
        "stage": stage,
        "timestamp_utc": now,
        "device": str(device),
        "data_fingerprint": config_fingerprint(),
    }
    if extra:
        stage_entry.update(dict(extra))
    history = list(previous.get("stage_history", []))
    history.append(stage_entry)
    payload: Dict[str, object] = {
        "last_stage": stage,
        "last_updated_utc": now,
        "stage_history": history,
        "seed": SEED,
        "data_fingerprint": config_fingerprint(),
        "config": _jsonable_config(),
        "training_config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_workers": NUM_WORKERS,
            "amp": AMP,
            "grad_clip_norm": GRAD_CLIP_NORM,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": _safe_package_version("torch"),
            "torchvision": _safe_package_version("torchvision"),
            "numpy": _safe_package_version("numpy"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "device": str(device),
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Offline synthetic dataset
# ---------------------------------------------------------------------------


def allowed_x_range() -> Tuple[float, float]:
    return LINE_LENGTH / 2.0 + EDGE_MARGIN, IMAGE_SIZE - LINE_LENGTH / 2.0 - EDGE_MARGIN


def render_line(cx: float) -> np.ndarray:
    """Render one antialiased horizontal white line as uint8 NCHW.

    Pixel-square overlap gives a deterministic fractional-translation signal
    without PIL/OpenCV.  The line has fixed y, length, direction and width;
    only its centre x varies.
    """

    h = w = IMAGE_SIZE
    xs = np.arange(w, dtype=np.float32) + 0.5
    ys = np.arange(h, dtype=np.float32) + 0.5
    x_left = float(cx) - LINE_LENGTH / 2.0
    x_right = float(cx) + LINE_LENGTH / 2.0
    y_top = LINE_Y - LINE_THICKNESS / 2.0
    y_bottom = LINE_Y + LINE_THICKNESS / 2.0

    # Width/height of intersection between each pixel square and rectangle.
    x_cov = np.clip(np.minimum(xs + 0.5, x_right) - np.maximum(xs - 0.5, x_left), 0.0, 1.0)
    y_cov = np.clip(np.minimum(ys + 0.5, y_bottom) - np.maximum(ys - 0.5, y_top), 0.0, 1.0)
    image = np.outer(y_cov, x_cov) * 255.0
    image = np.rint(image).astype(np.uint8)
    return np.repeat(image[None, ...], 3, axis=0)


def _centres_for_offsets(offsets: Sequence[float]) -> np.ndarray:
    x_min, x_max = allowed_x_range()
    # Integer bins cover the full allowable range; offsets are deliberately
    # below one, so the final centre remains <= x_max.
    first = int(math.ceil(x_min))
    last = int(math.floor(x_max - max(offsets)))
    bins = np.arange(first, last + 1, dtype=np.float32)
    centres = np.concatenate([bins + float(offset) for offset in offsets])
    return centres.astype(np.float32, copy=False)


def _dense_centres() -> np.ndarray:
    x_min, x_max = allowed_x_range()
    end = DENSE_TEST_START + DENSE_TEST_COUNT - 1
    if DENSE_TEST_START < math.ceil(x_min) or end > math.floor(x_max):
        raise ValueError(
            f"dense test [{DENSE_TEST_START}, {end}] is outside allowed centre range "
            f"[{x_min}, {x_max}]"
        )
    return np.arange(DENSE_TEST_START, end + 1, dtype=np.float32)


def make_split(centres: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centres = np.asarray(centres, dtype=np.float32)
    images = np.stack([render_line(float(cx)) for cx in centres], axis=0)
    targets = (centres / float(IMAGE_SIZE - 1)).astype(np.float32)
    return images, targets, centres


def _write_npz(path: Path, images: np.ndarray, targets: np.ndarray, centres: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=images,
        targets=targets,
        x_px=centres,
        metadata_json=np.array(json.dumps({"fingerprint": config_fingerprint()})),
    )


def prepare_data(force: bool = False) -> None:
    """Generate train/validation/dense-test arrays and a fingerprinted manifest."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "manifest.json"
    if manifest_path.exists() and not force:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cache_files_present = all(
                (DATA_DIR / name).exists() for name in ("train.npz", "val.npz", "dense_test.npz")
            )
            if manifest.get("fingerprint") == config_fingerprint() and cache_files_present:
                print(f"[prepare] cache already matches fingerprint {config_fingerprint()}; nothing to do")
                write_run_metadata("prepare", choose_device(), {"cache_action": "reused"})
                return
            raise RuntimeError(
                "existing data cache is missing files or has a different fingerprint; "
                "refusing to overwrite without --force"
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"cannot parse existing cache manifest {manifest_path}; refusing overwrite without --force"
            ) from exc
    elif not manifest_path.exists() and not force:
        stale_files = [name for name in ("train.npz", "val.npz", "dense_test.npz") if (DATA_DIR / name).exists()]
        if stale_files:
            raise RuntimeError(
                f"found dataset files without a manifest ({stale_files}); refusing overwrite without --force"
            )

    train_centres = _centres_for_offsets(TRAIN_OFFSETS)
    val_centres = _centres_for_offsets(VAL_OFFSETS)
    dense_centres = _dense_centres()
    # Deterministic but different order in each split.  The arrays themselves
    # are generated before training and remain untouched thereafter.
    rng = np.random.default_rng(SEED)
    train_centres = train_centres[rng.permutation(len(train_centres))]
    val_centres = val_centres[rng.permutation(len(val_centres))]

    for split, centres in (
        ("train", train_centres),
        ("val", val_centres),
        ("dense_test", dense_centres),
    ):
        images, targets, x_px = make_split(centres)
        _write_npz(DATA_DIR / f"{split}.npz", images, targets, x_px)
        print(f"[prepare] {split:10s}: {len(centres):5d} samples -> {DATA_DIR / (split + '.npz')}")

    metadata = {
        "fingerprint": config_fingerprint(),
        "config": _jsonable_config(),
        "counts": {"train": len(train_centres), "val": len(val_centres), "dense_test": len(dense_centres)},
        "dtype": "uint8 images, float32 targets/x_px",
        "target_definition": "x_px / (image_size - 1)",
        "exploratory_note": "single fixed seed; repeat seeds before general claims",
    }
    manifest_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_run_metadata("prepare", choose_device(), {"cache_action": "generated", "sample_counts": metadata["counts"]})
    print(f"[prepare] metadata fingerprint: {metadata['fingerprint']}")


def _load_npz_split(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = DATA_DIR / f"{name}.npz"
    manifest_path = DATA_DIR / "manifest.json"
    if not expected.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"missing offline dataset cache for {name}; run `python {Path(__file__).name} prepare` first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = manifest.get("fingerprint")
    wanted = config_fingerprint()
    if actual != wanted:
        raise RuntimeError(
            f"dataset fingerprint mismatch ({actual!r} != {wanted!r}); rerun prepare with --force"
        )
    with np.load(expected, allow_pickle=False) as data:
        keys = set(data.files)
        required = {"images", "targets", "x_px", "metadata_json"}
        if not required.issubset(keys):
            raise RuntimeError(f"{expected} is missing keys {sorted(required - keys)}")
        metadata_raw = data["metadata_json"].item()
        metadata = json.loads(str(metadata_raw))
        if metadata.get("fingerprint") != wanted:
            raise RuntimeError(f"embedded metadata fingerprint mismatch in {expected}")
        images = np.asarray(data["images"])
        targets = np.asarray(data["targets"], dtype=np.float32)
        centres = np.asarray(data["x_px"], dtype=np.float32)
    if images.ndim != 4 or images.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"unexpected image shape in {expected}: {images.shape}")
    if len(images) != len(targets) or len(images) != len(centres):
        raise RuntimeError(f"length mismatch in {expected}")
    return images, targets, centres


class OfflineLineDataset(Dataset):
    """Dataset backed exclusively by a pre-generated npz array."""

    def __init__(self, images: np.ndarray, targets: np.ndarray):
        self.images = images
        self.targets = targets

    def __len__(self) -> int:
        return int(len(self.targets))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index]).to(dtype=torch.float32).div_(255.0)
        target = torch.tensor([float(self.targets[index])], dtype=torch.float32)
        return image, target


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _torchvision_resnet18() -> nn.Module:
    try:
        import torchvision.models as models
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(f"torchvision import failed: {exc}") from exc
    try:
        # Explicitly use the requested original architecture and no weights.
        return models.resnet18(weights=None)
    except TypeError:  # old torchvision compatibility, still no pretrained weights
        return models.resnet18(pretrained=False)


class BlurPool(nn.Module):
    """Fixed 3x3 binomial low-pass filter followed by decimation."""

    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        kernel = torch.tensor(BLUR_KERNEL, dtype=torch.float32)
        kernel = kernel / kernel.sum()
        self.channels = int(channels)
        self.stride = int(stride)
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3).repeat(self.channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise RuntimeError(f"BlurPool expected {self.channels} channels, got {x.shape[1]}")
        return F.conv2d(x, self.kernel.to(dtype=x.dtype), stride=self.stride, padding=1, groups=self.channels)


class ConvWithOptionalBlur(nn.Module):
    """Run an existing convolution at stride 1, then optionally blur/downsample."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        stride = conv.stride
        stride_value = stride[0] if isinstance(stride, tuple) else int(stride)
        self.conv = conv
        if stride_value == 2:
            self.conv.stride = (1, 1)
            self.blur = BlurPool(conv.out_channels, stride=2)
        elif stride_value == 1:
            self.blur = nn.Identity()
        else:
            raise ValueError(f"anti-alias conversion only supports stride 1/2, got {stride}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blur(self.conv(x))


class AntiAliasBasicBlock(nn.Module):
    """BasicBlock equivalent with blur before every original stride-2 sample."""

    expansion = 1

    def __init__(self, old_block: nn.Module):
        super().__init__()
        self.conv1 = ConvWithOptionalBlur(old_block.conv1)
        self.bn1 = old_block.bn1
        self.relu = old_block.relu
        self.conv2 = old_block.conv2
        self.bn2 = old_block.bn2
        self.downsample = None
        if old_block.downsample is not None:
            old_conv = old_block.downsample[0]
            old_bn = old_block.downsample[1]
            self.downsample = nn.Sequential(ConvWithOptionalBlur(old_conv), old_bn)
        self.stride = old_block.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def _anti_alias_backbone() -> nn.Module:
    backbone = _torchvision_resnet18()
    backbone.conv1 = ConvWithOptionalBlur(backbone.conv1)
    # MaxPool's window is kept; it no longer decimates directly.  BlurPool is
    # the only stride-2 operation after the stride-1 max pooling.
    old_pool = backbone.maxpool
    old_pool.stride = 1
    backbone.maxpool = nn.Sequential(old_pool, BlurPool(64, stride=2))
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        old_layer = getattr(backbone, layer_name)
        new_layer = nn.Sequential(*(AntiAliasBasicBlock(block) for block in old_layer))
        setattr(backbone, layer_name, new_layer)
    return backbone


class CoordinateResNet(nn.Module):
    """ResNet backbone plus a coordinate regression head."""

    def __init__(self, backbone: nn.Module, variant: str):
        super().__init__()
        self.backbone = backbone
        self.variant = variant

    def _forward_features(self, x: torch.Tensor, collect: bool = False):
        b = self.backbone
        stages: Dict[str, torch.Tensor] = {}
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        if collect:
            stages["stem_conv"] = x
        x = b.maxpool(x)
        if collect:
            stages["stem_pool"] = x
        x = b.layer1(x)
        if collect:
            stages["layer1"] = x
        x = b.layer2(x)
        if collect:
            stages["layer2"] = x
        x = b.layer3(x)
        if collect:
            stages["layer3"] = x
        x = b.layer4(x)
        if collect:
            stages["layer4"] = x
            return stages
        return x

    def forward_features(self, x: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self._forward_features(x, collect=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x, collect=False)
        if self.variant == "no_gap":
            x = torch.flatten(x, 1)
        else:
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        return self.backbone.fc(x)


def _infer_backbone_feature_shape(backbone: nn.Module) -> Tuple[int, int, int]:
    """Infer ``(C,H,W)`` with a CPU dummy, without touching BN statistics."""

    was_training = backbone.training
    backbone.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
            x = backbone.conv1(dummy)
            x = backbone.bn1(x)
            x = backbone.relu(x)
            x = backbone.maxpool(x)
            x = backbone.layer1(x)
            x = backbone.layer2(x)
            x = backbone.layer3(x)
            x = backbone.layer4(x)
            shape = tuple(int(value) for value in x.shape[1:])
    finally:
        backbone.train(was_training)
    if len(shape) != 3:
        raise RuntimeError(f"unexpected inferred layer4 feature shape: {shape}")
    return shape  # type: ignore[return-value]


def build_model(variant: str) -> CoordinateResNet:
    if variant not in VARIANTS:
        raise ValueError(f"unknown model variant {variant!r}; choose from {VARIANTS}")
    backbone = _anti_alias_backbone() if variant == "antialias" else _torchvision_resnet18()
    if variant == "no_gap":
        feature_channels, feature_height, feature_width = _infer_backbone_feature_shape(backbone)
        in_features = feature_channels * feature_height * feature_width
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, NO_GAP_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(NO_GAP_HIDDEN, 1),
        )
    else:
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
    return CoordinateResNet(backbone, variant)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _autocast_context(device: torch.device, enabled: bool):
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> Mapping[str, object]:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("data_fingerprint") != config_fingerprint():
        raise RuntimeError(f"checkpoint {path} was made with a different data fingerprint")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint


def train_variant(
    variant: str,
    train_images: np.ndarray,
    train_targets: np.ndarray,
    val_images: np.ndarray,
    val_targets: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    # Reset before both model construction and DataLoader creation.  Every
    # variant therefore gets the same seed, initialization stream and batch
    # order (architecture-specific extra heads remain the only difference).
    seed_everything(SEED)
    train_set = OfflineLineDataset(train_images, train_targets)
    val_set = OfflineLineDataset(val_images, val_targets)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    model = build_model(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    criterion = nn.MSELoss()
    amp_enabled = bool(AMP and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_val = float("inf")
    best_epoch = 0
    history: List[Dict[str, float]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{variant}_best.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, amp_enabled):
                predictions = model(images)
                loss = criterion(predictions, targets)
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            train_sum += float(loss.detach().item()) * len(images)
            train_count += len(images)

        model.eval()
        val_sum = 0.0
        val_abs_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with _autocast_context(device, amp_enabled):
                    predictions = model(images)
                    loss = criterion(predictions, targets)
                val_sum += float(loss.item()) * len(images)
                val_abs_sum += float((predictions - targets).abs().sum().item())
                val_count += len(images)
        train_mse = train_sum / max(1, train_count)
        val_mse = val_sum / max(1, val_count)
        val_mae = val_abs_sum / max(1, val_count)
        scheduler.step(val_mse)
        row = {
            "epoch": float(epoch),
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(f"[{variant}] epoch {epoch:03d}/{EPOCHS}: train_mse={train_mse:.6g} val_mae={val_mae:.6g}")
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            torch.save(
                {
                    "variant": variant,
                    "epoch": epoch,
                    "best_val_mse": best_val,
                    "data_fingerprint": config_fingerprint(),
                    "model_state": model.state_dict(),
                    "config": _jsonable_config(),
                },
                checkpoint_path,
            )

    history_path = CHECKPOINT_DIR / f"{variant}_history.csv"
    _write_rows(history_path, history)
    print(f"[{variant}] best epoch={best_epoch}, val_mse={best_val:.6g}; checkpoint={checkpoint_path}")
    return {"variant": variant, "best_epoch": float(best_epoch), "best_val_mse": best_val}


def train(variants: Sequence[str]) -> None:
    train_images, train_targets, _ = _load_npz_split("train")
    val_images, val_targets, _ = _load_npz_split("val")
    device = choose_device()
    write_run_metadata(
        "train",
        device,
        {
            "variants": list(variants),
            "data_counts": {"train": int(len(train_targets)), "val": int(len(val_targets))},
        },
    )
    print(
        f"[train] device={device}, AMP={AMP and device.type == 'cuda'}, "
        f"train={len(train_targets)}, val={len(val_targets)}"
    )
    summaries = [
        train_variant(variant, train_images, train_targets, val_images, val_targets, device)
        for variant in variants
    ]
    _write_rows(ANALYSIS_DIR / "training_summary.csv", summaries)


# ---------------------------------------------------------------------------
# Analysis: dense predictions, residue buckets and FFT
# ---------------------------------------------------------------------------


def _predict(model: nn.Module, images: np.ndarray, device: torch.device) -> np.ndarray:
    dataset = OfflineLineDataset(images, np.zeros(len(images), dtype=np.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    outputs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch, _ in loader:
            outputs.append(model(batch.to(device)).detach().cpu().numpy().reshape(-1))
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)


def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "matplotlib is required for PNG analysis; please install it in the default environment "
            f"(import error: {exc})"
        ) from exc


def _write_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _linear_detrend(values: np.ndarray, x_px: np.ndarray) -> np.ndarray:
    """Remove a least-squares line against the *physical* x coordinates."""

    values = np.asarray(values, dtype=np.float64)
    x_px = np.asarray(x_px, dtype=np.float64)
    if len(values) < 2 or np.allclose(x_px, x_px[0]):
        return values - values.mean() if len(values) else values.copy()
    slope, intercept = np.polyfit(x_px, values, deg=1)
    return values - (slope * x_px + intercept)


def _residue_rows(
    x_px: np.ndarray, errors: np.ndarray, period: int, signal: str = "raw"
) -> List[Dict[str, object]]:
    """Summarize a residue bucket using the real x_px values, not row indices."""

    if signal == "detrended":
        signal_values = _linear_detrend(errors, x_px)
    elif signal == "raw":
        signal_values = np.asarray(errors, dtype=np.float64)
    else:
        raise ValueError(f"unknown residue signal {signal!r}")
    rows = []
    # Dense test x positions are exact integer pixels.  Modulo is intentionally
    # computed from those physical coordinates (not the sample index).
    residues = np.mod(np.asarray(x_px, dtype=np.int64), period)
    for residue in range(period):
        values = signal_values[residues == residue]
        values_px = values * float(IMAGE_SIZE - 1)
        if len(values) == 0:
            std = float("nan")
            std_px = float("nan")
        elif len(values) == 1:
            # Sample std (ddof=1) is undefined for one value; write 0 and make
            # the convention explicit in std_definition below.
            std = 0.0
            std_px = 0.0
        else:
            std = float(values.std(ddof=1))
            std_px = float(values_px.std(ddof=1))
        rows.append(
            {
                "period": period,
                "residue": residue,
                "signal": signal,
                "mean_error": float(values.mean()) if len(values) else float("nan"),
                "mae": float(np.abs(values).mean()) if len(values) else float("nan"),
                "std": std,
                "mean_error_px": float(values_px.mean()) if len(values_px) else float("nan"),
                "mae_px": float(np.abs(values_px).mean()) if len(values_px) else float("nan"),
                "std_px": std_px,
                "count": int(len(values)),
                "std_definition": "sample_ddof1; count=1 written as 0; empty=nan",
            }
        )
    return rows


def _fft_rows(
    errors: np.ndarray, x_px: np.ndarray
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    n = len(errors)
    if n == 0:
        return [], []
    # The requested ordering is explicit: remove a linear trend in physical x,
    # then apply a Hann window before the FFT.  This avoids a smooth regression
    # drift masquerading as a stride-related low-frequency peak.
    detrended = _linear_detrend(errors, x_px)
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(detrended * window))
    power = spectrum**2
    frequencies = np.fft.rfftfreq(n, d=1.0)
    rows = [
        {
            "bin": int(i),
            "frequency_cycles_per_pixel": float(freq),
            "amplitude": float(amp),
            "preprocessing": "linear_detrend_then_hann",
        }
        for i, (freq, amp) in enumerate(zip(frequencies, spectrum))
    ]
    total_non_dc_energy = float(power[1:].sum()) if len(power) > 1 else 0.0
    targets = []
    for period in PERIODS:
        target_frequency = 1.0 / period
        index = int(np.argmin(np.abs(frequencies - target_frequency)))
        peak_index = int(np.argmax(spectrum[1:]) + 1) if len(spectrum) > 1 else 0
        neighborhood = np.arange(max(1, index - 2), min(len(spectrum), index + 3), dtype=np.int64)
        local_peak_index = int(neighborhood[np.argmax(spectrum[neighborhood])]) if len(neighborhood) else index
        noise_bins = np.setdiff1d(np.arange(1, len(spectrum), dtype=np.int64), neighborhood)
        noise_floor = float(np.median(spectrum[noise_bins])) if len(noise_bins) else float("nan")
        local_peak_amplitude = float(spectrum[local_peak_index])
        peak_noise_ratio = (
            local_peak_amplitude / max(noise_floor, 1e-12) if np.isfinite(noise_floor) else float("nan")
        )
        peak_energy_share = (
            float(power[local_peak_index] / total_non_dc_energy) if total_non_dc_energy > 0 else float("nan")
        )
        targets.append(
            {
                "period": period,
                "target_frequency": target_frequency,
                "nearest_bin": index,
                "amplitude": float(spectrum[index]),
                "local_peak_bin": local_peak_index,
                "local_peak_frequency": float(frequencies[local_peak_index]),
                "local_peak_amplitude": local_peak_amplitude,
                "target_neighborhood_bins": f"{int(neighborhood[0])}-{int(neighborhood[-1])}" if len(neighborhood) else "",
                "noise_floor_median_amplitude": noise_floor,
                "peak_noise_ratio": peak_noise_ratio,
                "peak_energy_share_non_dc": peak_energy_share,
                "global_nonzero_peak_frequency": float(frequencies[peak_index]),
                "global_nonzero_peak_amplitude": float(spectrum[peak_index]),
                "preprocessing": "linear_detrend_then_hann",
            }
        )
    return rows, targets


def analyze(variants: Sequence[str]) -> None:
    seed_everything()
    images, targets, x_px = _load_npz_split("dense_test")
    device = choose_device()
    write_run_metadata("analyze", device, {"variants": list(variants), "dense_count": int(len(x_px))})
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    plt = _import_matplotlib()
    all_summary: List[Dict[str, object]] = []
    prediction_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for variant in variants:
        checkpoint_path = CHECKPOINT_DIR / f"{variant}_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing {checkpoint_path}; run train before analyze")
        model = build_model(variant).to(device)
        _load_checkpoint(model, checkpoint_path, device)
        pred = _predict(model, images, device)
        true_x = targets
        errors = pred - true_x
        scale_px = float(IMAGE_SIZE - 1)
        true_x_px = true_x * scale_px
        pred_x_px = pred * scale_px
        errors_px = errors * scale_px
        adjacent_steps_px = np.diff(pred_x_px)
        adjacent_step_mean_px = float(adjacent_steps_px.mean()) if len(adjacent_steps_px) else float("nan")
        adjacent_step_mae_px = (
            float(np.abs(adjacent_steps_px - 1.0).mean()) if len(adjacent_steps_px) else float("nan")
        )
        nonmonotonic_fraction = (
            float(np.mean(adjacent_steps_px <= 0.0)) if len(adjacent_steps_px) else float("nan")
        )
        prediction_cache[variant] = (pred, errors)
        prediction_rows = [
            {
                "index": int(i),
                "x_px": float(x_px[i]),
                "true_x": float(true_x[i]),
                "pred_x": float(pred[i]),
                "error": float(errors[i]),
                "true_x_px": float(true_x_px[i]),
                "pred_x_px": float(pred_x_px[i]),
                "error_px": float(errors_px[i]),
            }
            for i in range(len(x_px))
        ]
        _write_rows(ANALYSIS_DIR / f"predictions_{variant}.csv", prediction_rows)

        residue_rows: List[Dict[str, object]] = []
        for period in PERIODS:
            residue_rows.extend(_residue_rows(x_px, errors, period, signal="raw"))
            residue_rows.extend(_residue_rows(x_px, errors, period, signal="detrended"))
        _write_rows(ANALYSIS_DIR / f"residue_buckets_{variant}.csv", residue_rows)
        fft_rows, fft_target_rows = _fft_rows(errors, x_px)
        _write_rows(ANALYSIS_DIR / f"fft_{variant}.csv", fft_rows)
        _write_rows(ANALYSIS_DIR / f"fft_targets_{variant}.csv", fft_target_rows)

        target_amp = {int(row["period"]): float(row["amplitude"]) for row in fft_target_rows}
        target_ratio = {int(row["period"]): float(row["peak_noise_ratio"]) for row in fft_target_rows}
        all_summary.append(
            {
                "variant": variant,
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                "head_note": (
                    "no_gap has a 512*7*7 flatten+MLP head (much higher capacity)"
                    if variant == "no_gap"
                    else "GAP coordinate head"
                ),
                "mae": float(np.abs(errors).mean()),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "max_abs_error": float(np.abs(errors).max()),
                "mae_px": float(np.abs(errors_px).mean()),
                "rmse_px": float(np.sqrt(np.mean(errors_px**2))),
                "max_abs_error_px": float(np.abs(errors_px).max()),
                "adjacent_step_mean_px": adjacent_step_mean_px,
                "adjacent_step_mae_vs1_px": adjacent_step_mae_px,
                "nonmonotonic_fraction": nonmonotonic_fraction,
                "fft_amp_p2": target_amp.get(2, float("nan")),
                "fft_amp_p4": target_amp.get(4, float("nan")),
                "fft_amp_p8": target_amp.get(8, float("nan")),
                "fft_amp_p16": target_amp.get(16, float("nan")),
                "fft_amp_p32": target_amp.get(32, float("nan")),
                "fft_ratio_p2": target_ratio.get(2, float("nan")),
                "fft_ratio_p4": target_ratio.get(4, float("nan")),
                "fft_ratio_p8": target_ratio.get(8, float("nan")),
                "fft_ratio_p16": target_ratio.get(16, float("nan")),
                "fft_ratio_p32": target_ratio.get(32, float("nan")),
            }
        )

    # Three requested plots (plus one FFT view) compare all selected models.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_px, targets, "k-", label="true_x")
    for variant, (pred, _) in prediction_cache.items():
        ax.plot(x_px, pred, ".-", ms=3, label=variant)
    ax.set(xlabel="x (pixel)", ylabel="normalized x", title="true_x vs pred_x")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "true_vs_pred.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant, (_, errors) in prediction_cache.items():
        ax.plot(x_px, errors, ".-", ms=3, label=variant)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(
        xlabel="x (pixel)",
        ylabel="prediction error (normalized x)",
        title="prediction error vs x (raw normalized error)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "error_vs_x.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant, (_, errors) in prediction_cache.items():
        ax.plot(
            np.mod(np.asarray(x_px, dtype=np.int64), PERIODS[-1]),
            errors,
            ".",
            ms=5,
            label=variant,
            alpha=0.8,
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.set(
        xlabel=f"x mod {PERIODS[-1]} (pixel)",
        ylabel="prediction error (normalized x)",
        title=f"error by x mod {PERIODS[-1]}",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "error_x_mod32.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(prediction_cache), 1, figsize=(9, 3 * len(prediction_cache)), squeeze=False)
    frequencies = np.fft.rfftfreq(len(x_px), d=1.0)
    for row_index, (variant, (_, errors)) in enumerate(prediction_cache.items()):
        fft_values = np.abs(np.fft.rfft(_linear_detrend(errors, x_px) * np.hanning(len(errors))))
        ax = axes[row_index, 0]
        ax.plot(frequencies, fft_values, lw=1.2)
        for period in PERIODS:
            ax.axvline(1.0 / period, ls="--", lw=0.8, label=f"period {period}" if row_index == 0 else None)
        ax.set_ylabel(f"{variant}\namplitude (normalized error)")
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("frequency (cycles / pixel)")
    if len(prediction_cache):
        axes[0, 0].legend(ncol=5, fontsize=8)
    fig.suptitle("FFT of linear detrended, Hann-windowed error")
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "error_fft.png", dpi=150)
    plt.close(fig)

    _write_rows(ANALYSIS_DIR / "model_comparison.csv", all_summary)
    print("\n[analyze] summary (primary metrics in pixels; FFT ratios are peak/noise-floor)")
    print(
        "variant      params       MAE_px     RMSE_px    max|err|px step_mean  step_MAE  nonmono   "
        "FFT ratio p2 p4 p8 p16 p32"
    )
    for row in all_summary:
        print(
            f"{row['variant']:<12s} {row['parameter_count']:>10d}  {row['mae_px']:.6g}  "
            f"{row['rmse_px']:.6g}  {row['max_abs_error_px']:.6g}  "
            f"{row['adjacent_step_mean_px']:.6g}  {row['adjacent_step_mae_vs1_px']:.6g}  "
            f"{row['nonmonotonic_fraction']:.5g}  {row['fft_ratio_p2']:.5g}  {row['fft_ratio_p4']:.5g}  "
            f"{row['fft_ratio_p8']:.5g}  {row['fft_ratio_p16']:.5g}  {row['fft_ratio_p32']:.5g}"
        )
    if any(row["variant"] == "no_gap" for row in all_summary):
        print("[analyze] note: no_gap has substantially more head capacity; compare periodicity with this caveat.")


# ---------------------------------------------------------------------------
# Direct feature-map translation equivariance
# ---------------------------------------------------------------------------


def _shift_right(image: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels < 0:
        raise ValueError("only non-negative shifts are supported")
    if pixels == 0:
        return image.clone()
    shifted = torch.zeros_like(image)
    shifted[..., pixels:] = image[..., :-pixels]
    return shifted


def _relative_difference(reference: torch.Tensor, shifted: torch.Tensor) -> Tuple[float, float, int]:
    """Normalized L1/L2 on an explicitly selected overlap (never a circular roll)."""

    if reference.numel() == 0 or shifted.numel() == 0:
        return float("nan"), float("nan"), 0
    diff = shifted - reference
    denom_l1 = shifted.abs().mean().clamp_min(1e-8)
    denom_l2 = shifted.square().mean().sqrt().clamp_min(1e-8)
    l1 = float((diff.abs().mean() / denom_l1).item())
    l2 = float((diff.square().mean().sqrt() / denom_l2).item())
    return l1, l2, int(shifted.shape[-1])


def _raw_unaligned_difference(reference: torch.Tensor, shifted: torch.Tensor) -> Tuple[float, float, int]:
    """Same-coordinate comparison, exposing the unaligned phase error."""

    # Both inputs retain the same feature map dimensions.  We compare every
    # column; there is intentionally no roll and no spatial alignment here.
    return _relative_difference(reference, shifted)


def _fractional_aligned_difference(
    reference: torch.Tensor, shifted: torch.Tensor, shift_px: int, cumulative_stride: int
) -> Tuple[float, float, int]:
    """Align the reference by delta=shift_px/stride using bilinear sampling."""

    delta = float(shift_px) / float(cumulative_stride)
    _, _, h, w = reference.shape
    cols = torch.arange(w, device=reference.device, dtype=reference.dtype)
    rows = torch.arange(h, device=reference.device, dtype=reference.dtype)
    src_x = cols - delta
    valid = (src_x >= 0.0) & (src_x <= float(w - 1))
    # align_corners=True makes these coordinates exact in feature-pixel units.
    gx = src_x / max(1.0, float(w - 1)) * 2.0 - 1.0
    gy = rows / max(1.0, float(h - 1)) * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    sampled = F.grid_sample(reference, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return _relative_difference(sampled[..., :, valid], shifted[..., :, valid])


def _exact_integer_difference(
    reference: torch.Tensor, shifted: torch.Tensor, shift_px: int, cumulative_stride: int
) -> Tuple[float, float, int]:
    """Exact integer feature shift, emitted only when shift_px is stride-divisible."""

    delta = shift_px // cumulative_stride
    if delta == 0:
        ref_valid, shifted_valid = reference, shifted
    else:
        # Direct overlap slices avoid torch.roll's circular wraparound.
        ref_valid = reference[..., :, :-delta]
        shifted_valid = shifted[..., :, delta:]
    return _relative_difference(ref_valid, shifted_valid)


def equivariance(variants: Sequence[str], source_mode: str = "auto") -> None:
    EQUIV_DIR.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    if source_mode not in ("auto", "random", "trained"):
        raise ValueError(f"unknown equivariance source mode {source_mode!r}")
    write_run_metadata("equivariance", device, {"variants": list(variants), "source_mode": source_mode})
    # A fixed, interior, fractional-centre line keeps the comparison focused on
    # translation rather than random image content or clipping boundaries.
    image = torch.from_numpy(render_line(112.3)).float().div_(255.0).unsqueeze(0).to(device)
    cumulative = {
        "stem_conv": 2,
        "stem_pool": 4,
        "layer1": 4,
        "layer2": 8,
        "layer3": 16,
        "layer4": 32,
    }
    for variant in variants:
        # Random-initialized comparisons are independent but reproducible for
        # each variant; trained checkpoints are loaded after the same reset.
        seed_everything(SEED)
        model = build_model(variant).to(device)
        checkpoint_path = CHECKPOINT_DIR / f"{variant}_best.pt"
        source = "random_initialization"
        if source_mode == "trained" and not checkpoint_path.exists():
            raise FileNotFoundError(f"requested trained equivariance but checkpoint is missing: {checkpoint_path}")
        if source_mode in ("auto", "trained") and checkpoint_path.exists():
            _load_checkpoint(model, checkpoint_path, device)
            source = "trained_checkpoint"
        model.eval()
        with torch.no_grad():
            reference = model.forward_features(image)
        rows: List[Dict[str, object]] = []
        for shift_px in EQUIV_SHIFTS:
            shifted_image = _shift_right(image, shift_px)
            with torch.no_grad():
                shifted_features = model.forward_features(shifted_image)
            for stage, stride in cumulative.items():
                reference_stage = reference[stage]
                shifted_stage = shifted_features[stage]
                feature_height = int(reference_stage.shape[-2])
                feature_width = int(reference_stage.shape[-1])
                raw_l1, raw_l2, raw_overlap = _raw_unaligned_difference(reference_stage, shifted_stage)
                frac_l1, frac_l2, frac_overlap = _fractional_aligned_difference(
                    reference_stage, shifted_stage, shift_px, stride
                )
                rows.append(
                    {
                        "variant": variant,
                        "source": source,
                        "stage": stage,
                        "cumulative_stride": stride,
                        "shift_px": shift_px,
                        "comparison": "raw_unaligned",
                        "alignment": "none",
                        "feature_height": feature_height,
                        "feature_width": feature_width,
                        "overlap_height": feature_height,
                        "overlap_feature_columns": raw_overlap,
                        "normalized_l1": raw_l1,
                        "normalized_l2": raw_l2,
                    }
                )
                rows.append(
                    {
                        "variant": variant,
                        "source": source,
                        "stage": stage,
                        "cumulative_stride": stride,
                        "shift_px": shift_px,
                        "comparison": "fractional_aligned",
                        "alignment": "bilinear_fractional",
                        "feature_height": feature_height,
                        "feature_width": feature_width,
                        "overlap_height": feature_height,
                        "overlap_feature_columns": frac_overlap,
                        "normalized_l1": frac_l1,
                        "normalized_l2": frac_l2,
                    }
                )
                if shift_px % stride == 0:
                    exact_l1, exact_l2, exact_overlap = _exact_integer_difference(
                        reference_stage, shifted_stage, shift_px, stride
                    )
                    rows.append(
                        {
                            "variant": variant,
                            "source": source,
                            "stage": stage,
                            "cumulative_stride": stride,
                            "shift_px": shift_px,
                            "comparison": "exact_integer",
                            "alignment": "exact_integer",
                            "feature_height": feature_height,
                            "feature_width": feature_width,
                            "overlap_height": feature_height,
                            "overlap_feature_columns": exact_overlap,
                            "normalized_l1": exact_l1,
                            "normalized_l2": exact_l2,
                        }
                    )
        _write_rows(EQUIV_DIR / f"equivariance_{variant}.csv", rows)
        print(f"[equivariance] {variant}: {source} -> {EQUIV_DIR / (f'equivariance_{variant}.csv')}")

    # Compact visual: one panel per model, one curve per stage.
    plt = _import_matplotlib()
    fig, axes = plt.subplots(len(variants), 2, figsize=(13, 3.5 * len(variants)), squeeze=False)
    for i, variant in enumerate(variants):
        path = EQUIV_DIR / f"equivariance_{variant}.csv"
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for metric, col in (("normalized_l1", 0), ("normalized_l2", 1)):
            ax = axes[i, col]
            for stage in ("stem_conv", "stem_pool", "layer1", "layer2", "layer3", "layer4"):
                # Plot the consistently available fractional-aligned series;
                # raw and exact values remain in CSV for direct inspection.
                subset = [
                    r for r in rows if r["stage"] == stage and r["comparison"] == "fractional_aligned"
                ]
                ax.plot(
                    [int(r["shift_px"]) for r in subset],
                    [float(r[metric]) for r in subset],
                    ".-",
                    label=stage,
                )
            ax.set_title(f"{variant}: {metric}")
            ax.set_xlabel("input shift (px)")
            ax.set_ylabel("normalized difference")
            ax.grid(alpha=0.25)
            if i == 0 and col == 0:
                ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(EQUIV_DIR / "equivariance_summary.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Environment check and command-line entry point
# ---------------------------------------------------------------------------


def check() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"torchvision: ", end="")
    try:
        import torchvision

        print(torchvision.__version__)
    except Exception as exc:
        print(f"IMPORT ERROR ({exc})")
        return 1
    try:
        import matplotlib

        print(f"matplotlib: {matplotlib.__version__}")
    except Exception as exc:
        print(f"matplotlib: IMPORT ERROR ({exc})")
    print(f"image size: {IMAGE_SIZE}x{IMAGE_SIZE}, allowed x: {allowed_x_range()}")
    print(f"data fingerprint: {config_fingerprint()}")
    print("note: single fixed seed is exploratory; repeat seeds for general claims")
    print(f"output directory: {OUTPUT_DIR}")
    print(f"cache present: {(DATA_DIR / 'manifest.json').exists()}")

    # Lightweight architecture smoke test: no data generation cache and no
    # optimization/training are performed by check.
    try:
        seed_everything()
        sample = torch.from_numpy(render_line(112.3)).float().div_(255.0).unsqueeze(0)
        for variant in VARIANTS:
            model = build_model(variant).eval()
            with torch.no_grad():
                output = model(sample)
                stages = model.forward_features(sample)
            shapes = {name: tuple(value.shape[-2:]) for name, value in stages.items()}
            print(f"{variant:10s}: output={tuple(output.shape)}, stages={shapes}")
    except Exception as exc:
        print(f"architecture smoke test failed: {exc}")
        return 1
    write_run_metadata(
        "check",
        choose_device(),
        {"cache_present": (DATA_DIR / "manifest.json").exists(), "architecture_smoke_test": "passed"},
    )
    print("check completed (training was not started)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        choices=("check", "prepare", "train", "analyze", "equivariance", "all"),
        default="check",
        help="pipeline stage (default: check; does not train)",
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--force", action="store_true", help="overwrite a mismatched/existing dataset cache")
    parser.add_argument(
        "--equiv-source",
        choices=("auto", "trained", "random"),
        default="auto",
        help="feature equivariance weights: trained checkpoint, deterministic random init, or auto",
    )
    args = parser.parse_args(argv)

    if args.stage == "check":
        return check()
    if args.stage == "prepare":
        seed_everything()
        prepare_data(force=args.force)
        return 0
    if args.stage == "train":
        train(args.variants)
        return 0
    if args.stage == "analyze":
        analyze(args.variants)
        return 0
    if args.stage == "equivariance":
        equivariance(args.variants, source_mode=args.equiv_source)
        return 0
    if args.stage == "all":
        seed_everything()
        prepare_data(force=args.force)
        train(args.variants)
        analyze(args.variants)
        equivariance(args.variants, source_mode=args.equiv_source)
        return 0
    raise AssertionError(f"unhandled stage {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
