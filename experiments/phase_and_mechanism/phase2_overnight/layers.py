"""Layer-wise position field for R18 / R50 / ConvNeXt Full64."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from phase1_gap_rep.common import dump_json

from .ckpt import load_arch_checkpoint
from .features import extract_stage_gaps
from .geometry import additive_energies, h_of_t, local_jh, metric_geometry, svd_ranks
from .protocol import PHASE2_ROOT, SEED_ATLAS, load_protocol
from .render_dense import load_dense_cache
from .train_regime import model_dir

OUT = PHASE2_ROOT / "layers"


def run_layers() -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    proto = load_protocol()
    cache = load_dense_cache()
    keep = [proto["eval_shape_ids"].index(i) for i in proto["traj_shape_ids"] if i in proto["eval_shape_ids"]]
    gy = list(range(0, 41, 2))
    idx = [tx * 41 + ty for tx in gy for ty in gy]
    imgs = np.asarray(cache["images"])[keep][:, idx].reshape(-1, 224, 224)
    t_flat = cache["t"].reshape(cache["images"].shape[0], -1, 2)[0, idx]
    rows: List[Dict[str, Any]] = []
    for arch in ("resnet18", "resnet50", "convnext_tiny"):
        path = model_dir(arch, "G64", SEED_ATLAS) / "checkpoints" / "best_slim.pt"
        if not path.exists():
            rows.append({"arch": arch, "missing": str(path)})
            continue
        try:
            model = load_arch_checkpoint(path, arch)
            feats = extract_stage_gaps(model, imgs)
        except Exception as exc:
            rows.append({"arch": arch, "error": str(exc)})
            continue
        layer_info = {}
        for name, zflat in feats.items():
            if zflat.size == 0 or zflat.shape[1] == 0:
                continue
            z = zflat.reshape(len(keep), len(idx), -1)
            h = h_of_t(z)
            layer_info[name] = {
                "energy": additive_energies(z),
                "svd": svd_ranks(h),
                "metric": metric_geometry(h, t_flat),
                "median_sigma_ratio": float(np.median(local_jh(h)["sigma2_over_sigma1"])),
            }
        rows.append({"arch": arch, "layers": layer_info})
    dump_json(OUT / "layerwise.json", rows)
    return {"rows": rows}
