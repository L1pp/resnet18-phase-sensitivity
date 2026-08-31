"""Part C: R18 trajectory on G64, seed 20260813."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from phase1_gap_rep.common import dump_json, setup_matplotlib_chinese
from phase1_gap_rep.train import _predict

from .ckpt import load_arch_checkpoint
from .features import extract_stage_gaps
from .geometry import additive_energies, control_metrics, h_of_t, local_jh, metric_geometry, svd_ranks, tangent_rotation_summary
from .probe import position_readout
from .protocol import PHASE2_ROOT, SEED_TRAJECTORY, TRAJECTORY_EPOCHS, load_protocol
from .render_dense import load_dense_cache
from .train_regime import model_dir, train_regime


def run_part_c() -> Dict[str, Any]:
    proto = load_protocol()
    summary = train_regime("resnet18", "G64", SEED_TRAJECTORY, trajectory=True)
    out = model_dir("resnet18", "G64", SEED_TRAJECTORY)
    cache = load_dense_cache()
    eval_ids = proto["traj_shape_ids"]
    all_eval = proto["eval_shape_ids"]
    keep = [all_eval.index(i) for i in eval_ids if i in all_eval]
    # 21x21 = every other of 41
    gy = list(range(0, 41, 2))
    idx = []
    for tx in gy:
        for ty in gy:
            idx.append(tx * 41 + ty)
    n_s = cache["images"].shape[0]
    imgs = np.asarray(cache["images"])[keep][:, idx]
    flat = imgs.reshape(-1, 224, 224)
    P = cache["P"][keep][:, idx].reshape(-1, 6)
    ckpts = ["init.pt"] + [f"epoch_{e:03d}.pt" for e in TRAJECTORY_EPOCHS] + ["best_slim.pt"]
    rows = []
    t_flat = cache["t"].reshape(n_s, -1, 2)[0, idx]
    for name in ckpts:
        path = out / "checkpoints" / name
        if not path.exists():
            continue
        epoch = 0 if name.startswith("init") else (100 if "best" in name else int(name.split("_")[1].split(".")[0]))
        try:
            model = load_arch_checkpoint(path, "resnet18")
        except Exception as exc:
            rows.append({"ckpt": name, "error": str(exc)})
            continue
        pred = _predict(model, flat)
        beh, _, _, _ = control_metrics(pred, P)
        feats = extract_stage_gaps(model, flat, stages=["gap"])
        z = feats["gap"].reshape(len(keep), len(idx), -1)
        en = additive_energies(z)
        h = h_of_t(z)
        even = slice(None, None, 2)
        odd = slice(1, None, 2)
        z_tr = z[:, even, :].reshape(-1, z.shape[-1])
        z_ev = z[:, odd, :].reshape(-1, z.shape[-1])
        t_grid = t_flat.reshape(len(idx), 2)
        t_tr = np.repeat(t_grid[even][None, :, :], len(keep), axis=0).reshape(-1, 2)
        t_ev = np.repeat(t_grid[odd][None, :, :], len(keep), axis=0).reshape(-1, 2)
        ridge = position_readout(z_tr, t_tr, z_ev, t_ev)
        row = {
            "ckpt": name,
            "epoch": epoch,
            "behavioral": beh,
            "energy": en,
            "svd": svd_ranks(h),
            "tangent": tangent_rotation_summary(local_jh(h)),
            "metric": metric_geometry(h, t_flat),
            "ridge": ridge,
        }
        rows.append(row)
        print(f"[phase2 C] {name} t_mae={beh['t_mae_px']:.3f} pos_energy={en['position_energy']:.4f}")
    dump_json(out / "tables" / "trajectory.json", rows)
    plt = setup_matplotlib_chinese()
    xs = [r["epoch"] for r in rows if "energy" in r]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, [r["energy"]["position_frac"] for r in rows if "energy" in r], label="position frac")
    ax.plot(xs, [r["energy"]["interaction_frac"] for r in rows if "energy" in r], label="interaction frac")
    ax.set_xlabel("epoch")
    ax.set_title("Part C GAP energy fractions")
    ax.legend()
    fig.savefig(PHASE2_ROOT / "part_c_energy.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return {"train": summary, "trajectory": rows}
