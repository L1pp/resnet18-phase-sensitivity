"""本机 GTX1060 墙钟测算与非持久训练速度 profile。

这里的 profile 只做少量 warmup 和计时步骤：模型、优化器和显存张量
随后丢弃，不写 checkpoint，也不作为科学训练结果。
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .config import CONFIG, MODEL_SPECS, OUTPUT_ROOT, TRAIN_ROOT, config_fingerprint, ensure_output_dirs
from .models import build_models_with_shared_state, seed_everything
from .pipeline import (
    _dataset_for_model,
    _evaluate_model_mae,
    _load_dataset,
    check_environment,
    prepare_datasets,
    probe_features,
    random_features,
    write_json,
    write_report,
)
from .visuals import chinese_font_path


RECEIPT_PATH = OUTPUT_ROOT / "wallclock_receipt.json"
MARKDOWN_PATH = OUTPUT_ROOT / "墙钟估算.md"
PROFILE_MODELS = ("a0_standard", "a1_zero_s1", "a2_torus_s32", "a3_torus_s1")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 1:
        return f"{seconds * 1000.0:.0f} 毫秒"
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f} 分钟"
    return f"{minutes / 60.0:.2f} 小时"


def _timed(label: str, function: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    start = time.perf_counter()
    value = function()
    elapsed = time.perf_counter() - start
    return value, {"label": label, "elapsed_seconds": elapsed}


def profile_training_speed(
    device_name: str | None = None,
    *,
    warmup_steps: int = 2,
    timed_steps: int = 5,
) -> dict[str, Any]:
    """Measure a few forward+backward+optimizer steps and one full eval per model."""

    if device_name is None:
        device_name = "cuda"
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("墙钟 profile 需要本机 CUDA/GTX1060；这不是正式训练")
    seed_everything()
    models, _state, shared_hash = build_models_with_shared_state(device=device)
    records: dict[str, Any] = {
        "kind": "track_a_toy_nonpersistent_training_speed_profile",
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device),
        "warmup_steps": int(warmup_steps),
        "timed_steps": int(timed_steps),
        "batch_size": int(CONFIG["train_batch_size"]),
        "shared_state_hash": shared_hash,
        "models": {},
        "scientific_training": False,
        "checkpoint_written": False,
        "note": "少量计时步骤；模型、优化器和张量随后丢弃，不产生训练结论",
    }
    for name in PROFILE_MODELS:
        model = models[name]
        data = _load_dataset(_dataset_for_model(name))
        images = torch.from_numpy(data["images"]).to(device=device, dtype=torch.float32).div_(255.0)
        targets = torch.from_numpy(data["points"].astype(np.float32)).to(device=device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(CONFIG["optimizer"]["lr"]),
            weight_decay=float(CONFIG["optimizer"]["weight_decay"]),
        )
        generator = torch.Generator(device=device).manual_seed(int(CONFIG["seed"]))
        batch_size = int(CONFIG["train_batch_size"])

        def one_step() -> float:
            indices = torch.randint(0, len(images), (batch_size,), generator=generator, device=device)
            prediction = model(images[indices])
            loss = torch.nn.functional.mse_loss(prediction, targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            return float(loss.detach().cpu())

        model.train()
        _sync(device)
        for _ in range(int(warmup_steps)):
            one_step()
        _sync(device)
        step_seconds: list[float] = []
        losses: list[float] = []
        for _ in range(int(timed_steps)):
            _sync(device)
            start = time.perf_counter()
            losses.append(one_step())
            _sync(device)
            step_seconds.append(time.perf_counter() - start)

        model.eval()
        _sync(device)
        eval_start = time.perf_counter()
        full_mae = _evaluate_model_mae(
            model,
            images,
            targets,
            device=device,
            batch_size=batch_size,
        )
        _sync(device)
        eval_seconds = time.perf_counter() - eval_start
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        records["models"][name] = {
            "dataset": _dataset_for_model(name),
            "sample_count": int(len(data["points"])),
            "warmup_steps": int(warmup_steps),
            "timed_steps": int(timed_steps),
            "step_seconds": step_seconds,
            "step_seconds_mean": float(np.mean(step_seconds)),
            "step_seconds_median": float(np.median(step_seconds)),
            "step_seconds_max": float(np.max(step_seconds)),
            "eval_seconds_full_256": float(eval_seconds),
            "eval_mae_observed_not_scientific": float(full_mae),
            "last_profile_losses": losses,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
        }
        del optimizer, images, targets, model
        models[name] = None  # type: ignore[assignment]
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    del models
    gc.collect()
    torch.cuda.empty_cache()
    return records


def evaluation_count(max_steps: int, interval: int) -> int:
    return 1 + int(max_steps) // int(interval)


def estimate_training_wallclock(profile: dict[str, Any]) -> dict[str, Any]:
    """Convert short profile timings to transparent median/conservative estimates."""

    interval = int(CONFIG["train_eval_interval"])
    model_estimates: dict[str, Any] = {}
    total_typical = 0.0
    total_conservative = 0.0
    for name in PROFILE_MODELS:
        row = profile["models"][name]
        max_steps = int(CONFIG["train_max_steps"][name])
        step_median = float(row["step_seconds_median"])
        step_mean = float(row["step_seconds_mean"])
        step_max = float(row["step_seconds_max"])
        eval_seconds = float(row["eval_seconds_full_256"])
        # The upper estimate is explicit: at least the observed maximum and
        # 20% above the observed mean, with the same rule for full eval.
        step_conservative = max(step_max, step_mean * 1.20)
        eval_conservative = eval_seconds * 1.20
        evals = evaluation_count(max_steps, interval)
        typical = max_steps * step_median + evals * eval_seconds
        conservative = max_steps * step_conservative + evals * eval_conservative
        total_typical += typical
        total_conservative += conservative
        cap = max_steps
        candidates = [100, 300, 500, 800, 1000] if cap >= 1000 else [100, 200, 300]
        candidates = [steps for steps in candidates if steps <= cap]
        early = []
        for steps in candidates:
            early_evals = evaluation_count(steps, interval)
            early_typical = steps * step_median + early_evals * eval_seconds
            early_conservative = steps * step_conservative + early_evals * eval_conservative
            early.append(
                {
                    "steps": steps,
                    "evaluations_including_step1": early_evals,
                    "typical_seconds": early_typical,
                    "conservative_seconds": early_conservative,
                }
            )
        model_estimates[name] = {
            "max_steps": max_steps,
            "evaluations_including_step1": evals,
            "step_seconds_typical_median": step_median,
            "step_seconds_conservative": step_conservative,
            "eval_seconds_typical": eval_seconds,
            "eval_seconds_conservative": eval_conservative,
            "full_max_typical_seconds": typical,
            "full_max_conservative_seconds": conservative,
            "reasonable_early_stop_scenarios": early,
        }
    return {
        "formula": "steps*step_seconds + (1 + floor(steps/eval_interval))*full_eval_seconds; conservative=max(observed_max,1.2*mean), eval=1.2*observed",
        "eval_interval": interval,
        "models": model_estimates,
        "training_only_typical_seconds": total_typical,
        "training_only_conservative_seconds": total_conservative,
    }


def _assemble_receipt(
    *,
    phases: dict[str, Any],
    profile: dict[str, Any],
    training_estimate: dict[str, Any],
    profile_elapsed_seconds: float,
    report_elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    phase_total = sum(float(row["elapsed_seconds"]) for row in phases.values())
    if report_elapsed_seconds is not None:
        phase_total += float(report_elapsed_seconds)
    nontraining_total = phase_total
    training_typical = float(training_estimate["training_only_typical_seconds"])
    training_conservative = float(training_estimate["training_only_conservative_seconds"])
    return {
        "schema_version": 1,
        "kind": "track_a_toy_wallclock_receipt",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "全过程 = check + prepare + random-features + probe + A0-A3短训练 + report + tests；不含人工开机/上传等待，也不含复核返工",
        "device_scope": "本机 GTX1060；profile 是非持久少量计时，不是正式科学训练",
        "config_fingerprint": config_fingerprint(),
        "font_path": str(chinese_font_path()),
        "phases": phases,
        "profile_elapsed_seconds_excluded_from_total": float(profile_elapsed_seconds),
        "profile": profile,
        "training_estimate": training_estimate,
        "observed_nontraining_seconds": nontraining_total,
        "estimated_full_process_typical_seconds": nontraining_total + training_typical,
        "estimated_full_process_conservative_seconds": nontraining_total + training_conservative,
        "report_elapsed_seconds": report_elapsed_seconds,
        "formal_training_started": False,
        "checkpoint_written": False,
    }


def write_wallclock_markdown(receipt: dict[str, Any]) -> str:
    estimate = receipt["training_estimate"]
    training_summary_path = TRAIN_ROOT / "summary.json"
    training = json.loads(training_summary_path.read_text(encoding="utf-8")) if training_summary_path.exists() else None
    actual_training_seconds = (
        float(training["training_elapsed_seconds"]) if training is not None else None
    )
    actual_nontraining_seconds = float(receipt["observed_nontraining_seconds"])
    actual_clean_seconds = (
        actual_nontraining_seconds + actual_training_seconds
        if actual_training_seconds is not None
        else None
    )
    actual_lines: list[str] = []
    if actual_clean_seconds is not None:
        actual_lines = [
            "## 结论（本次实际）",
            "",
            f"本次纯训练实际耗时：**{_duration(actual_training_seconds)}**（{actual_training_seconds:.3f} 秒）。",
            f"已完成非训练阶段实测：**{_duration(actual_nontraining_seconds)}**（{actual_nontraining_seconds:.3f} 秒）。",
            f"因此，本次干净实验流程实测：**{_duration(actual_clean_seconds)}**（{actual_clean_seconds:.3f} 秒）。",
            "今后本机操作建议预留 **50–60 分钟**；若包括复核或重画图，建议预留 **60–90 分钟**。",
            "",
        ]
    lines = [
        "# Track A-Toy 全过程墙钟估算",
        "",
        "## 口径",
        "",
        "这里的“全过程”定义为：check、prepare、random-features、probe、A0–A3 短训练、report 和 tests。",
        (
            "不包括人工开机、上传等待，也不包括复核返工。正式短训练已另行执行并在文末回填；本页的 profile 数字仍是训练前的非持久外推。"
            if training is not None
            else "不包括人工开机、上传等待，也不包括复核返工。短训练部分本轮没有正式执行；这里只用 GTX1060 做了少量 forward+backward+optimizer 计时 profile，模型和优化器随后丢弃，不写 checkpoint。"
        ),
        "",
        *actual_lines,
        "## 已完成阶段的实测时间",
        "",
        "| 阶段 | 实测墙钟 | 说明 |",
        "|---|---:|---|",
    ]
    for label, row in receipt["phases"].items():
        lines.append(f"| {label} | {_duration(row['elapsed_seconds'])} | 实测；同一台本机、同一工作区 |" )
    lines += [
        "",
        f"已完成非训练阶段合计：**{_duration(receipt['observed_nontraining_seconds'])}**。",
        "",
        "## 被实跑推翻的低估对照：GTX1060 非持久速度 profile",
        "",
        "每个模型使用 batch16，先 warmup 2 步，再只计时 5 个完整的 forward+backward+optimizer 步；另测一次全 256 输入 eval。模型和优化器随后丢弃，profile 不计入正式科学结果。",
        "这只是正式训练前的短测，不能代表持续运行；A1/A3 的正式长跑明显更慢，因此下面的旧外推已被本次实际训练推翻。",
        "",
        "| 模型 | 单步中位数 | 单步保守值 | 全256 eval | 上限步数 | eval次数 | 上限保守时间 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in estimate["models"].items():
        lines.append(
            f"| {name} | {_duration(row['step_seconds_typical_median'])} | {_duration(row['step_seconds_conservative'])} | "
            f"{_duration(row['eval_seconds_typical'])} | {row['max_steps']} | {row['evaluations_including_step1']} | {_duration(row['full_max_conservative_seconds'])} |"
        )
    lines += [
        "",
        "保守公式：`步数 × max(实测单步最大值, 1.2 × 实测单步均值) + eval次数 × 1.2 × 实测全域eval时间`。每 50 步评估一次，并计入第 1 步的初始评估。",
        "",
        f"旧 profile 外推（仅作历史对照）：训练部分典型 **{_duration(estimate['training_only_typical_seconds'])}**，旧的保守值 **{_duration(estimate['training_only_conservative_seconds'])}**；加上当时非训练阶段曾得到 **{_duration(receipt['estimated_full_process_typical_seconds'])}** 和 **{_duration(receipt['estimated_full_process_conservative_seconds'])}**。",
        "上述约 21.7 分钟和 25.6 分钟的旧数字已被本次实际训练推翻，不能作为当前可信全过程、今后操作预算或上限。",
        "",
        "## 合理早停情景（规划，不是已经发生的结果）",
        "",
        "只有在全域 MAE<2 像素连续两次评估时才算满足早停条件；下表只是把可能停在不同步数的墙钟换算出来，不能当作本轮训练已经达到该指标。",
        "",
        "| 模型 | 停止步数 | 典型时间 | 保守时间 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in estimate["models"].items():
        for scenario in row["reasonable_early_stop_scenarios"]:
            lines.append(f"| {name} | {scenario['steps']} | {_duration(scenario['typical_seconds'])} | {_duration(scenario['conservative_seconds'])} |")
    lines += [
        "",
        "## 限制与可追溯性",
        "",
        "- 阶段表中的非训练时间来自本机实测；profile 表只保留被实跑推翻的短测对照，不是当前全过程承诺，也不是云 GPU 承诺。",
        "- profile 仅 5 个计时步，未覆盖持续训练的实际吞吐、显存碎片、系统争用或早停是否达到。",
        "- receipt 保存了每个阶段的 `elapsed_seconds`、每个模型的单步序列、eval 时间、显存峰值和计算公式。",
        f"- 机器可读 receipt：`{RECEIPT_PATH.name}`。",
        "",
    ]
    if training is not None:
        lines += [
            "",
            "## 正式短训练实测回填",
            "",
            f"正式 A0–A3 训练实际耗时：**{_duration(training['training_elapsed_seconds'])}**；`training/summary.json` 为实际记录，前面的 profile 外推不能替代这次实测。",
            "",
            "| 模型 | 实际步数 | 最佳 MAE | 最佳步 | 停止原因 |",
            "|---|---:|---:|---:|---|",
        ]
        for name, row in training["models"].items():
            reason = "连续两次 MAE<2 像素" if row["stopped_early"] else "达到步数上限"
            lines.append(
                f"| {name} | {row['steps_completed']} | {row['best_mae_px']:.3f} px | {row['best_step']} | {reason} |"
            )
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    return str(MARKDOWN_PATH)


def run_wallclock(device_name: str | None = None) -> dict[str, Any]:
    """Run the measured non-training workflow plus isolated speed profile."""

    ensure_output_dirs()
    phases: dict[str, Any] = {}
    check_value, phases["check"] = _timed("check", lambda: check_environment(device_name or "cuda"))
    _ = check_value
    _, phases["prepare"] = _timed("prepare", prepare_datasets)
    _, phases["random-features"] = _timed("random-features", lambda: random_features(device_name or "cuda"))
    _, phases["probe"] = _timed("probe", probe_features)

    profile_start = time.perf_counter()
    profile = profile_training_speed(device_name or "cuda")
    profile_elapsed = time.perf_counter() - profile_start
    training_estimate = estimate_training_wallclock(profile)

    tests_start = time.perf_counter()
    test_process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "track_a_toy/tests", "-p", "test_*.py", "-v"],
        cwd=str(Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
    )
    tests_elapsed = time.perf_counter() - tests_start
    if test_process.returncode != 0:
        raise RuntimeError(f"unit tests failed during wallclock receipt:\n{test_process.stdout}\n{test_process.stderr}")
    phases["tests"] = {
        "label": "tests",
        "elapsed_seconds": tests_elapsed,
        "returncode": test_process.returncode,
        "test_summary_tail": (test_process.stdout + test_process.stderr)[-3000:],
    }

    provisional = _assemble_receipt(
        phases=phases,
        profile=profile,
        training_estimate=training_estimate,
        profile_elapsed_seconds=profile_elapsed,
    )
    write_json(RECEIPT_PATH, provisional)
    _, report_phase = _timed("report", write_report)
    receipt = _assemble_receipt(
        phases=phases,
        profile=profile,
        training_estimate=training_estimate,
        profile_elapsed_seconds=profile_elapsed,
        report_elapsed_seconds=report_phase["elapsed_seconds"],
    )
    receipt["phases"]["report"] = report_phase
    write_json(RECEIPT_PATH, receipt)
    # Refresh the report once after the final receipt exists so the report and
    # receipt agree; this second write is intentionally not counted twice.
    write_report()
    write_wallclock_markdown(receipt)
    return receipt


def finalize_wallclock_receipt() -> dict[str, Any]:
    """Finish a receipt if the measured workflow stopped before report writing."""

    if not RECEIPT_PATH.exists():
        raise FileNotFoundError(RECEIPT_PATH)
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    if receipt.get("report_elapsed_seconds") is not None:
        write_report()
        write_wallclock_markdown(receipt)
        return receipt
    report_start = time.perf_counter()
    write_report()
    report_elapsed = time.perf_counter() - report_start
    phases = receipt["phases"]
    phases["report"] = {"label": "report", "elapsed_seconds": report_elapsed}
    observed = sum(float(row["elapsed_seconds"]) for row in phases.values())
    receipt["observed_nontraining_seconds"] = observed
    receipt["report_elapsed_seconds"] = report_elapsed
    receipt["estimated_full_process_typical_seconds"] = observed + float(receipt["training_estimate"]["training_only_typical_seconds"])
    receipt["estimated_full_process_conservative_seconds"] = observed + float(receipt["training_estimate"]["training_only_conservative_seconds"])
    write_json(RECEIPT_PATH, receipt)
    write_report()
    write_wallclock_markdown(receipt)
    return receipt


if __name__ == "__main__":
    result = run_wallclock("cuda")
    print(json.dumps({"receipt": str(RECEIPT_PATH), "markdown": str(MARKDOWN_PATH), "estimated_full_process_conservative_seconds": result["estimated_full_process_conservative_seconds"]}, ensure_ascii=False, indent=2))
