#!/usr/bin/env python
"""Profile GPU power and memory while the latency benchmark runs.

The live path samples ``nvidia-smi`` during an optional benchmark command,
combines those readings with the latency JSON and the recognizer JSON, then writes
raw machine-readable results under ``results/resource-cost/`` and prints a
paper-ready Markdown table.

Usage:
    python probes/profile_resources.py --idle-seconds 60 \
      --benchmark-command "python probes/bench_latency.py --runs 3"
    python probes/profile_resources.py --idle-seconds 0 --latency-json results/latency/latest.json
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_LATENCY_DIR = ROOT / "results"
DEFAULT_ASR_DIRS = (ROOT / "results",)
# Optional: an OpenAI-compatible server exposing Prometheus metrics.
DEFAULT_LLM_METRICS_URL = "http://127.0.0.1:8000/metrics"
NVIDIA_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,memory.used,memory.total,power.draw,utilization.gpu",
    "--format=csv,noheader,nounits",
]


@dataclass(frozen=True)
class GpuSample:
    timestamp: float
    gpu_index: int
    gpu_name: str
    memory_used_mib: float | None
    memory_total_mib: float | None
    power_w: float | None
    utilization_pct: float | None


@dataclass(frozen=True)
class SampleSummary:
    samples: int
    gpu_name: str | None
    memory_total_mib: float | None
    memory_used_mib_mean: float | None
    memory_used_mib_max: float | None
    power_w_mean: float | None
    power_w_max: float | None
    utilization_pct_mean: float | None
    utilization_pct_max: float | None


@dataclass(frozen=True)
class CommandResult:
    command: str
    returncode: int | None
    elapsed_s: float
    stdout_tail: str
    stderr_tail: str
    timed_out: bool


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "[not supported]", "not supported"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_nvidia_smi_csv(output: str, *, timestamp: float | None = None) -> list[GpuSample]:
    now = time.time() if timestamp is None else timestamp
    samples: list[GpuSample] = []
    for index, line in enumerate(output.splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        samples.append(
            GpuSample(
                timestamp=now,
                gpu_index=index,
                gpu_name=parts[0],
                memory_used_mib=parse_float(parts[1]),
                memory_total_mib=parse_float(parts[2]),
                power_w=parse_float(parts[3]),
                utilization_pct=parse_float(parts[4]),
            )
        )
    return samples


def query_gpu() -> list[GpuSample]:
    completed = subprocess.run(NVIDIA_QUERY, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
    return parse_nvidia_smi_csv(completed.stdout)


def sample_for_seconds(seconds: float, interval_s: float) -> list[GpuSample]:
    if seconds <= 0:
        return []
    samples: list[GpuSample] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        samples.extend(query_gpu())
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(interval_s, remaining))
    return samples


def run_command_with_sampling(command: str, *, interval_s: float, timeout_s: float | None) -> tuple[CommandResult, list[GpuSample]]:
    started = time.monotonic()
    process = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    samples: list[GpuSample] = []
    timed_out = False
    while process.poll() is None:
        if timeout_s is not None and time.monotonic() - started > timeout_s:
            timed_out = True
            process.kill()
            break
        try:
            samples.extend(query_gpu())
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"GPU sampling stopped: {exc}", file=sys.stderr)
            break
        time.sleep(interval_s)
    stdout, stderr = process.communicate()
    elapsed_s = time.monotonic() - started
    return (
        CommandResult(
            command=command,
            returncode=process.returncode,
            elapsed_s=elapsed_s,
            stdout_tail=tail(stdout),
            stderr_tail=tail(stderr),
            timed_out=timed_out,
        ),
        samples,
    )


def tail(text: str, *, max_lines: int = 40) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def latest_json(directory: Path, prefix: str = "") -> Path | None:
    """Newest JSON in a directory, optionally restricted by file-name prefix.

    The prefix matters because results/ holds every probe's output side by side;
    without it the newest network or noise file would be picked as a latency run.
    """
    paths = sorted(directory.glob(f"{prefix}*.json"), key=lambda path: path.stat().st_mtime)
    return paths[-1] if paths else None


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None and math.isfinite(value)]


# Samples from every visible card are pooled into one summary. Exact on a
# single-card host; on a multi-GPU host it mixes idle and busy devices and
# understates both, so check nvidia-smi before reading these figures.
def summarize_samples(samples: list[GpuSample]) -> SampleSummary:
    power = numeric([sample.power_w for sample in samples])
    memory = numeric([sample.memory_used_mib for sample in samples])
    utilization = numeric([sample.utilization_pct for sample in samples])
    totals = numeric([sample.memory_total_mib for sample in samples])
    names = [sample.gpu_name for sample in samples if sample.gpu_name]
    return SampleSummary(
        samples=len(samples),
        gpu_name=names[0] if names else None,
        memory_total_mib=max(totals) if totals else None,
        memory_used_mib_mean=mean(memory) if memory else None,
        memory_used_mib_max=max(memory) if memory else None,
        power_w_mean=mean(power) if power else None,
        power_w_max=max(power) if power else None,
        utilization_pct_mean=mean(utilization) if utilization else None,
        utilization_pct_max=max(utilization) if utilization else None,
    )


def summarize_latency(payload: dict[str, Any] | None, *, active_elapsed_s: float | None = None) -> dict[str, Any]:
    if not payload:
        return {"turn_count": 0}
    turns = payload.get("turns") or []
    metrics = [
        turn.get("final_metrics") or {}
        for turn in turns
        if isinstance(turn, dict) and isinstance(turn.get("final_metrics"), dict)
    ]
    e2e_values = numeric([metric.get("e2e_first_audio_ms") for metric in metrics])
    duration_values = numeric(
        [
            (metric.get("first_audio_chunk_sent") - metric.get("utterance_end"))
            if isinstance(metric.get("first_audio_chunk_sent"), int | float)
            and isinstance(metric.get("utterance_end"), int | float)
            else None
            for metric in metrics
        ]
    )
    turn_count = int(payload.get("turn_count") or len(turns))
    # Harness wall-clock per turn, which includes the trailing silence the probe
    # sends to close the utterance and the pause between turns. Energy per query
    # is mean power times this duration, so both are upper bounds on the
    # assistant's busy time rather than estimates of it.
    mean_turn_duration_s = None
    if active_elapsed_s is not None and turn_count:
        mean_turn_duration_s = active_elapsed_s / turn_count
    elif duration_values:
        mean_turn_duration_s = mean(duration_values)
    elif e2e_values:
        mean_turn_duration_s = mean(e2e_values) / 1000.0
    return {
        "turn_count": turn_count,
        "mean_e2e_first_audio_ms": mean(e2e_values) if e2e_values else None,
        "p50_e2e_first_audio_ms": median(e2e_values) if e2e_values else None,
        "mean_turn_duration_s": mean_turn_duration_s,
    }


def collect_asr_realtime_factors(payloads: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for payload in payloads:
        for key in ("results", "asr_results"):
            rows = payload.get(key) or []
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict):
                    value = row.get("realtime_factor")
                    if isinstance(value, int | float) and math.isfinite(value):
                        values.append(float(value))
    return values


def summarize_asr(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    values = collect_asr_realtime_factors(payloads)
    return {
        "sample_count": len(values),
        "mean_realtime_factor": mean(values) if values else None,
        "median_realtime_factor": median(values) if values else None,
    }


def parse_llm_metrics(text: str) -> float | None:
    candidates: list[float] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(None, 1)[0]
        if "throughput" not in name.lower() and "tokens_per_second" not in name.lower():
            continue
        if "generation" not in name.lower() and "token" not in name.lower():
            continue
        try:
            candidates.append(float(line.rsplit(None, 1)[-1]))
        except ValueError:
            continue
    return max(candidates) if candidates else None


def fetch_llm_tokens_s(url: str, *, timeout_s: float) -> float | None:
    try:
        response = httpx.get(url, timeout=timeout_s)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    return parse_llm_metrics(response.text)


def parse_component_vram(values: list[str]) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"--component-vram must use LABEL=MIB, got {raw!r}")
        label, value = raw.split("=", 1)
        amount = parse_float(value)
        if amount is None:
            raise SystemExit(f"Invalid component VRAM value: {raw!r}")
        components.append({"component": label.strip(), "memory_mib": amount, "source": "manual approximate"})
    return components


def component_vram_summary(
    manual_components: list[dict[str, Any]],
    idle: SampleSummary,
    active: SampleSummary,
) -> list[dict[str, Any]]:
    if manual_components:
        return manual_components
    if idle.memory_used_mib_mean is None or active.memory_used_mib_max is None:
        return [{"component": "full pipeline", "memory_mib": None, "source": "n/a"}]
    return [
        {
            "component": "full pipeline delta",
            "memory_mib": max(0.0, active.memory_used_mib_max - idle.memory_used_mib_mean),
            "source": "active peak minus idle mean, approximate",
        }
    ]


def fmt(value: float | None, suffix: str = "", digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def markdown_table(profile: dict[str, Any]) -> str:
    gpu = profile["gpu"]
    idle = profile["idle_summary"]
    active = profile["active_summary"]
    latency = profile["latency_summary"]
    asr = profile["asr_summary"]
    components = profile["component_vram"]
    lines = [
        "| Metric | Value | Notes |",
        "|---|---:|---|",
        f"| GPU | {gpu.get('name') or 'n/a'} | total VRAM {fmt(gpu.get('memory_total_mib'), ' MiB', 0)} |",
        f"| Idle power | {fmt(idle.get('power_w_mean'), ' W')} | {idle.get('samples', 0)} sample(s) |",
        f"| Active power | {fmt(active.get('power_w_mean'), ' W')} | peak {fmt(active.get('power_w_max'), ' W')} |",
        f"| Active GPU utilization | {fmt(active.get('utilization_pct_mean'), '%')} | peak {fmt(active.get('utilization_pct_max'), '%')} |",
        f"| Active VRAM used | {fmt(active.get('memory_used_mib_max'), ' MiB', 0)} | peak during benchmark window |",
    ]
    for component in components:
        amount = component.get("memory_mib")
        lines.append(
            f"| VRAM: {component.get('component')} | {fmt(amount, ' MiB', 0)} | {component.get('source', 'approximate')} |"
        )
    lines.extend(
        [
            f"| LLM throughput | {fmt(profile.get('llm_tokens_s'), ' tok/s')} | LLM server metrics endpoint |",
            f"| ASR speed factor | {fmt(asr.get('median_realtime_factor'), 'x')} | median audio duration / transcribe time |",
            f"| Mean turn duration | {fmt(latency.get('mean_turn_duration_s'), ' s')} | benchmark wall time per turn when available |",
            f"| Energy per query | {fmt(profile.get('energy_per_query_wh'), ' Wh')} | active mean power x mean turn duration |",
        ]
    )
    return "\n".join(lines)


def build_profile(args: argparse.Namespace) -> dict[str, Any]:
    idle_samples: list[GpuSample] = []
    active_samples: list[GpuSample] = []
    command_result = None
    try:
        idle_samples = sample_for_seconds(args.idle_seconds, args.sample_interval_s)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Idle GPU sampling unavailable: {exc}", file=sys.stderr)

    if args.benchmark_command:
        try:
            command_result, active_samples = run_command_with_sampling(
                args.benchmark_command,
                interval_s=args.sample_interval_s,
                timeout_s=args.benchmark_timeout_s,
            )
        except FileNotFoundError as exc:
            print(f"Benchmark command failed to start: {exc}", file=sys.stderr)

    latency_path = Path(args.latency_json) if args.latency_json else latest_json(DEFAULT_LATENCY_DIR, "latency-")
    latency_payload = load_json(latency_path)
    asr_paths = [Path(path) for path in args.asr_json]
    if not asr_paths:
        asr_paths = [path for directory in DEFAULT_ASR_DIRS if (path := latest_json(directory, "noise-")) is not None]
    asr_payloads = [payload for path in asr_paths if (payload := load_json(path)) is not None]

    idle_summary = summarize_samples(idle_samples)
    active_summary = summarize_samples(active_samples)
    latency_summary = summarize_latency(
        latency_payload,
        active_elapsed_s=command_result.elapsed_s if command_result and command_result.returncode == 0 else None,
    )
    asr_summary = summarize_asr(asr_payloads)
    active_power = active_summary.power_w_mean
    turn_duration = latency_summary.get("mean_turn_duration_s")
    energy_per_query_wh = (
        active_power * turn_duration / 3600.0
        if isinstance(active_power, int | float) and isinstance(turn_duration, int | float)
        else None
    )
    llm_tokens_s = None if args.skip_llm_metrics else fetch_llm_tokens_s(args.llm_metrics_url, timeout_s=1.5)

    gpu_name = active_summary.gpu_name or idle_summary.gpu_name
    gpu_memory = active_summary.memory_total_mib or idle_summary.memory_total_mib
    profile: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    
        "gpu": {"name": gpu_name, "memory_total_mib": gpu_memory},
        "idle_seconds": args.idle_seconds,
        "sample_interval_s": args.sample_interval_s,
        "benchmark_command": args.benchmark_command,
        "benchmark_result": asdict(command_result) if command_result else None,
        "latency_json": str(latency_path) if latency_path else None,
        "asr_json": [str(path) for path in asr_paths],
        "idle_summary": asdict(idle_summary),
        "active_summary": asdict(active_summary),
        "latency_summary": latency_summary,
        "asr_summary": asr_summary,
        "llm_tokens_s": llm_tokens_s,
        "energy_per_query_wh": energy_per_query_wh,
        "component_vram": component_vram_summary(
            parse_component_vram(args.component_vram),
            idle_summary,
            active_summary,
        ),
        "notes": [
            "Component VRAM is approximate unless --component-vram LABEL=MIB values were supplied.",
            "Cost figures are not derived here; the paper reports energy only.",
        ],
    }
    profile["markdown_table"] = markdown_table(profile)
    return profile


def write_results(args: argparse.Namespace, profile: dict[str, Any]) -> Path:
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = output_dir / f"{stamp}.json"
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--idle-seconds", type=float, default=60.0)
    parser.add_argument("--sample-interval-s", type=float, default=1.0)
    parser.add_argument("--benchmark-command", default="", help="command to run while active GPU samples are captured")
    parser.add_argument("--benchmark-timeout-s", type=float, default=None)
    parser.add_argument("--latency-json", default="", help="latency benchmark JSON; defaults to the newest results/latency-*.json")
    parser.add_argument(
        "--asr-json",
        action="append",
        default=[],
        help="recognizer JSON; defaults to the newest results/noise-*.json",
    )
    parser.add_argument("--component-vram", action="append", default=[], help="manual approximate component VRAM, LABEL=MIB")
    parser.add_argument("--llm-metrics-url", default=DEFAULT_LLM_METRICS_URL)
    parser.add_argument("--skip-llm-metrics", action="store_true")
    args = parser.parse_args()

    profile = build_profile(args)
    print(profile["markdown_table"])
    path = write_results(args, profile)
    print(f"\nRaw JSON: {path}")
    if profile["benchmark_result"] and profile["benchmark_result"]["returncode"] not in (0, None):
        print("Benchmark command exited non-zero; resource profile was still written.", file=sys.stderr)


if __name__ == "__main__":
    main()
