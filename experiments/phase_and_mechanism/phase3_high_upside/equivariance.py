"""Translation equivariance defect before GAP, and GAP invariance defect."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from phase1_gap_rep.common import images_to_tensor

from .controlled_cnn import ControlledCNN

INTEGER_SHIFTS: Sequence[int] = (1, 2, 4, 8)


def shift_nchw(x: torch.Tensor, dx: int, dy: int, mode: str) -> torch.Tensor:
    """Shift last two dims. dx>0 moves content right; dy>0 moves content down."""
    if dx == 0 and dy == 0:
        return x
    if mode == "circular":
        return torch.roll(x, shifts=(int(dy), int(dx)), dims=(-2, -1))
    if mode != "zeros":
        raise ValueError(mode)
    shifted = torch.zeros_like(x)
    h, w = x.shape[-2:]
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return shifted
    shifted[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]
    return shifted


def downsample_shift(dx: int, dy: int, feat_h: int, img_h: int) -> tuple[int, int, bool]:
    """Map image-pixel shift onto a feature map. Exact only when divisible."""
    if feat_h <= 0 or img_h <= 0:
        return 0, 0, False
    stride = img_h / float(feat_h)
    if abs(stride - round(stride)) > 1e-6:
        return 0, 0, False
    stride_i = int(round(stride))
    if stride_i <= 1:
        return int(dx), int(dy), True
    if dx % stride_i != 0 or dy % stride_i != 0:
        return int(round(dx / stride_i)), int(round(dy / stride_i)), False
    return dx // stride_i, dy // stride_i, True


@torch.no_grad()
def measure_defects(
    model: ControlledCNN,
    images: np.ndarray,
    shifts: Sequence[int] = INTEGER_SHIFTS,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """D_eq = |F(T x) - T F(x)| ; D_GAP = |GAP(F(T x)) - GAP(F(x))|."""
    model.eval()
    device = next(model.parameters()).device
    mode = "circular" if model.circular else "zeros"
    rows: List[Dict[str, Any]] = []
    n = len(images)
    for shift in shifts:
        eq_num = 0.0
        eq_den = 0.0
        gap_num = 0.0
        gap_den = 0.0
        n_exact = 0
        n_used = 0
        for start in range(0, n, batch_size):
            batch = images_to_tensor(images[start : start + batch_size]).to(device)
            feat = model.forward_features(batch)
            gap0 = torch.flatten(F.adaptive_avg_pool2d(feat, 1), 1)
            for dx, dy in ((shift, 0), (0, shift), (-shift, 0), (0, -shift)):
                shifted = shift_nchw(batch, dx, dy, mode)
                feat_t = model.forward_features(shifted)
                gap_t = torch.flatten(F.adaptive_avg_pool2d(feat_t, 1), 1)
                fdx, fdy, exact = downsample_shift(dx, dy, feat.shape[-2], batch.shape[-2])
                feat_shift = shift_nchw(feat, fdx, fdy, mode)
                diff = feat_t - feat_shift
                eq_num += float(diff.pow(2).mean().cpu())
                eq_den += float(feat_t.pow(2).mean().cpu())
                gdiff = gap_t - gap0
                gap_num += float(gdiff.pow(2).mean().cpu())
                gap_den += float(gap0.pow(2).mean().cpu())
                n_used += 1
                n_exact += int(exact)
        rows.append(
            {
                "shift_px": int(shift),
                "pad_mode": mode,
                "d_eq": eq_num / max(n_used, 1),
                "d_eq_rel": (eq_num / eq_den) if eq_den > 1e-12 else None,
                "d_gap": gap_num / max(n_used, 1),
                "d_gap_rel": (gap_num / gap_den) if gap_den > 1e-12 else None,
                "n_pairs": n_used,
                "n_exact_feature_shift": n_exact,
            }
        )
    return {
        "variant": model.variant,
        "circular": model.circular,
        "use_stride": model.use_stride,
        "rows": rows,
        "mean_d_eq": float(np.mean([r["d_eq"] for r in rows])) if rows else None,
        "mean_d_gap": float(np.mean([r["d_gap"] for r in rows])) if rows else None,
    }
