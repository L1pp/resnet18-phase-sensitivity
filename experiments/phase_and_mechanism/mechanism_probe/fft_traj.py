"""S1: 2D FFT of affine-removed residual fields already on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from error_extension.field_metrics import load_field_npz
from today_shortcycle.fields import fft_energy

from .const import DRIFT_ROOT, FFT_STEPS_2D, FFT_STEPS_6D, RESULTS_ROOT, STAGES
from .blob_io import cosine, ensure_results, write_json


def _power2d(u: np.ndarray) -> np.ndarray:
    arr = np.asarray(u, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr[None, ...]
    if arr.ndim != 4 or arr.shape[-1] != 2:
        raise ValueError(f"expected [n,h,w,2], got {arr.shape}")
    spec_x = np.fft.fft2(arr[..., 0], axes=(-2, -1))
    spec_y = np.fft.fft2(arr[..., 1], axes=(-2, -1))
    power = (np.abs(spec_x) ** 2 + np.abs(spec_y) ** 2).mean(axis=0)
    return np.fft.fftshift(power)


def _radial(shifted: np.ndarray) -> Dict[str, Any]:
    ny, nx = shifted.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.ogrid[:ny, :nx]
    radius = np.hypot(yy - cy, xx - cx)
    rmax = int(np.floor(radius.max()))
    profile = []
    for r in range(rmax + 1):
        m = (radius >= r) & (radius < r + 1)
        if not np.any(m):
            profile.append(0.0)
        else:
            profile.append(float(shifted[m].mean()))
    total = float(shifted.sum())
    low = float(shifted[radius <= 2.0].sum())
    return {
        "profile": profile,
        "low_frac": (low / total) if total > 0 else float("nan"),
        "dc_frac": float(shifted[cy, cx] / total) if total > 0 else float("nan"),
    }


def _topk_modes(shifted: np.ndarray, k: int = 12, *, skip_dc: bool = True) -> List[Dict[str, Any]]:
    ny, nx = shifted.shape
    cy, cx = ny // 2, nx // 2
    work = shifted.copy()
    if skip_dc:
        work[cy, cx] = 0.0
    total = float(shifted.sum())
    flat = work.reshape(-1)
    order = np.argsort(flat)[::-1][:k]
    rows = []
    for idx in order:
        iy, ix = divmod(int(idx), nx)
        ky, kx = int(iy - cy), int(ix - cx)
        val = float(shifted[iy, ix])
        rows.append(
            {
                "ij": [int(iy), int(ix)],
                "kxy": [ky, kx],
                "power": val,
                "frac": (val / total) if total > 0 else float("nan"),
            }
        )
    return rows


def analyze_u(u: np.ndarray) -> Dict[str, Any]:
    power = _power2d(u)
    summary = fft_energy(u)
    radial = _radial(power)
    return {
        "fft": summary,
        "radial": radial,
        "top12": _topk_modes(power, 12),
        "log_power": np.log10(np.maximum(power, 1e-18)),
        "power": power,
    }


def _load_u(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    blob = load_field_npz(path)
    u = blob.get("u")
    if u is None:
        return None
    return np.asarray(u, dtype=np.float64)


def _step_dir(root: Path, step: int) -> Path:
    return root / "traj" / f"step_{int(step):04d}" / "field.npz"


def _pair_spec(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    pa = np.asarray(a["power"], dtype=np.float64).reshape(-1)
    pb = np.asarray(b["power"], dtype=np.float64).reshape(-1)
    return {
        "cosine_power": cosine(pa, pb),
        "cosine_log_power": cosine(np.log10(np.maximum(pa, 1e-18)), np.log10(np.maximum(pb, 1e-18))),
    }


def _track_modes(step1: Dict[str, Any], other: Dict[str, Any]) -> List[Dict[str, Any]]:
    p = np.asarray(other["power"], dtype=np.float64)
    total = float(p.sum())
    rows = []
    for rec in step1["top12"][:8]:
        iy, ix = rec["ij"]
        val = float(p[iy, ix])
        rows.append(
            {
                "kxy": rec["kxy"],
                "step1_frac": rec["frac"],
                "other_frac": (val / total) if total > 0 else float("nan"),
                "ratio": (val / rec["power"]) if rec["power"] > 0 else float("nan"),
            }
        )
    return rows


def _run_family(
    name: str,
    stage_roots: Dict[str, Path],
    steps: Sequence[int],
    *,
    delta_from_zero: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": name, "stages": {}, "comparisons": {}}
    u0: Dict[str, np.ndarray] = {}
    analyzed: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for stage, root in stage_roots.items():
        stage_blob: Dict[str, Any] = {"root": str(root), "steps": {}}
        for step in steps:
            path = _step_dir(root, step)
            u = _load_u(path)
            if u is None:
                stage_blob["steps"][str(step)] = {"missing": str(path)}
                continue
            rec = analyze_u(u)
            rec["path"] = str(path)
            rec["shape"] = list(u.shape)
            if step == 0:
                u0[stage] = u
            if delta_from_zero and step != 0 and stage in u0:
                drec = analyze_u(u - u0[stage])
                rec["delta_from_step0"] = {
                    "fft": drec["fft"],
                    "radial": drec["radial"],
                    "top12": drec["top12"],
                }
                rec["_delta_power"] = drec["power"]
                rec["_delta_log_power"] = drec["log_power"]
            analyzed[(stage, int(step))] = rec
            slim = {
                "path": rec["path"],
                "shape": rec["shape"],
                "fft": rec["fft"],
                "radial": rec["radial"],
                "top12": rec["top12"],
            }
            if "delta_from_step0" in rec:
                slim["delta_from_step0"] = rec["delta_from_step0"]
            stage_blob["steps"][str(step)] = slim
        out["stages"][stage] = stage_blob

    pairs = [("head", "l4"), ("head", "full"), ("l4", "full")]
    for step in steps:
        step_cmp: Dict[str, Any] = {}
        for a, b in pairs:
            if (a, int(step)) not in analyzed or (b, int(step)) not in analyzed:
                continue
            rec = _pair_spec(analyzed[(a, int(step))], analyzed[(b, int(step))])
            da = analyzed[(a, int(step))].get("_delta_power")
            db = analyzed[(b, int(step))].get("_delta_power")
            if da is not None and db is not None:
                rec["cosine_delta_power"] = cosine(da, db)
            step_cmp[f"{a}_vs_{b}"] = rec
        out["comparisons"][str(step)] = step_cmp

    final_step = int(steps[-1])
    persist: Dict[str, Any] = {}
    for stage in stage_roots:
        s1 = analyzed.get((stage, 1))
        sf = analyzed.get((stage, final_step))
        if s1 is None or sf is None:
            continue
        src = s1.get("delta_from_step0", s1)
        persist[stage] = _track_modes(
            {"top12": src["top12"], "power": np.asarray(s1.get("_delta_power", s1["power"]))},
            {"power": np.asarray(sf.get("_delta_power", sf["power"]))}
            if "delta_from_step0" in sf
            else sf,
        )
    out["mode_persistence_step1_to_final"] = persist
    out["_plot"] = {
        (f"{st}_{sp}"): analyzed[(st, int(sp))]
        for st, sp in analyzed
    }
    return out


def run() -> Dict[str, Any]:
    ensure_results()
    amd_roots = {s: DRIFT_ROOT / "amd" / s for s in STAGES}
    a10_roots = {s: DRIFT_ROOT / "a10" / "unfreeze_g64_corners4" / s for s in STAGES}
    mlp_roots = {
        "linear_full": DRIFT_ROOT / "local" / "mlp" / "linear__full",
        "mlp_s_full": DRIFT_ROOT / "local" / "mlp" / "mlp_s__full",
        "mlp_w_full": DRIFT_ROOT / "local" / "mlp" / "mlp_w__full",
        "mlp_w_last": DRIFT_ROOT / "local" / "mlp" / "mlp_w__last",
    }
    payload = {
        "amd_2d": _run_family("amd_2d", amd_roots, FFT_STEPS_2D),
        "a10_6d": _run_family("a10_6d", a10_roots, FFT_STEPS_6D),
        "mlp": _run_family("mlp", mlp_roots, FFT_STEPS_2D),
    }
    slim = {}
    for key, fam in payload.items():
        slim[key] = {
            "name": fam["name"],
            "comparisons": fam["comparisons"],
            "mode_persistence_step1_to_final": fam["mode_persistence_step1_to_final"],
            "stages": {
                st: {"steps": rec["steps"]}
                for st, rec in fam["stages"].items()
            },
        }
    write_json(RESULTS_ROOT / "s1_fft.json", slim)
    from .plots import plot_fft

    plot_fft(payload)
    print("[S1] wrote", RESULTS_ROOT / "s1_fft.json", flush=True)
    return slim
