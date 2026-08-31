"""Extract GAP features Z[s,t] and the frozen linear head."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import torch

from .common import (
    ANALYSIS_GATE_MAE_PX,
    BATCH_SIZE,
    dump_json,
    fingerprint,
    gap_features,
    images_to_tensor,
    profile_spec,
    run_dirs,
    train_run_spec,
    write_run_metadata,
)
from .generate_data import load_split
from .train import load_run_model


def extract(profile: str = "factorial") -> Dict[str, Any]:
    raise RuntimeError(
        "legacy extract writes to factorial wreckage; "
        "use: python phase1_gap_rep/run.py extract --run adamw_l1_scratch"
    )


def extract_run(run_name: str) -> Dict[str, Any]:
    run = train_run_spec(run_name)
    spec = profile_spec(run.data_profile)
    d = run_dirs(run.name)
    metrics_path = d["tables"] / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing {metrics_path}; eval the run before extract")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_mae = float(metrics["test"]["mae_px"])
    if test_mae > ANALYSIS_GATE_MAE_PX:
        raise RuntimeError(
            f"[extract:{run.name}] test MAE={test_mae:.3f}px > {ANALYSIS_GATE_MAE_PX}px；拒绝提取。"
        )
    model = load_run_model(run.name, "best")
    dev = next(model.parameters()).device
    z = np.zeros((spec.n_shapes, spec.n_translations, 512), dtype=np.float32)
    filled = np.zeros((spec.n_shapes, spec.n_translations), dtype=bool)
    model.eval()
    with torch.no_grad():
        for split in ("train", "val", "test"):
            data = load_split(spec, split)
            images = data["images"]
            sid = data["shape_id"].astype(np.int64)
            tid = data["translation_id"].astype(np.int64)
            feats: List[np.ndarray] = []
            for start in range(0, len(images), BATCH_SIZE):
                batch = images_to_tensor(images[start : start + BATCH_SIZE]).to(dev)
                feats.append(gap_features(model, batch).float().cpu().numpy())
            array = np.concatenate(feats, axis=0)
            z[sid, tid] = array
            filled[sid, tid] = True
            print(f"[extract:{run.name}] {split} n={len(images)}")
    if not bool(np.all(filled)):
        missing = int(np.size(filled) - np.count_nonzero(filled))
        raise RuntimeError(f"incomplete factorial features: missing {missing} cells")
    weight = model.fc.weight.detach().float().cpu().numpy()
    bias = model.fc.bias.detach().float().cpu().numpy()
    d["features"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d["features"] / "Z.npz", Z=z, fingerprint=np.asarray(fingerprint(spec)))
    np.savez_compressed(d["features"] / "head.npz", W=weight, b=bias, fingerprint=np.asarray(fingerprint(spec)))
    info = {
        "run": run.name,
        "data_profile": spec.name,
        "Z_shape": list(z.shape),
        "W_shape": list(weight.shape),
        "b_shape": list(bias.shape),
        "fingerprint": fingerprint(spec),
        "test_mae_px": test_mae,
        "analysis_gate_mae_px": ANALYSIS_GATE_MAE_PX,
        "loss": f"MSE + {run.l1_weight} * L1",
    }
    dump_json(d["features"] / "extract_info.json", info)
    write_run_metadata(run, "extract", info)
    print(f"[extract:{run.name}] saved Z={tuple(z.shape)} W={tuple(weight.shape)} test_mae={test_mae:.3f}px")
    return info
