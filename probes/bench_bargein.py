#!/usr/bin/env python
"""Measure barge-in reaction time against a running daemon.

The harness starts a normal turn (push-to-talk framing), waits for
reply audio to flow, then injects a speech WAV mid-reply with the browser's
128 ms framing and measures how quickly the server cancels the response:

- acoustic onset -> ``interrupted`` event received (response cancelled)
- acoustic onset -> last ``audio_chunk`` received (audio actually stops)

Usage:
    python probes/bench_bargein.py --runs 30
    python probes/bench_bargein.py --url wss://127.0.0.1:8765/speech --runs 30
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_latency import (
    AudioCase,
    chunk_bytes,
    fmt_ms,
    load_audio_cases,
    percentile,
    websocket_connect_kwargs,
)

DEFAULT_AUDIO_DIR = Path(__file__).resolve().parent.parent / "data" / "queries" / "audio"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
# Injection offsets after audio_started, cycled per run. All exceed
# barge_in_grace_ms (250 ms) so every injection is eligible to interrupt.
DEFAULT_OFFSETS_MS = (500, 1000, 1500, 2000, 3000)
DEFAULT_FRAME_MS = 128
DEFAULT_TRIM_THRESHOLD_DBFS = -40.0

# Kept out of the per-run event log: high-rate streaming events, and events a
# particular deployment emits that have nothing to do with the turn cycle.
NOISY_EVENTS = {"audio_chunk", "audio_level"} | set(
    filter(None, os.environ.get("PROBE_QUIET_EVENTS", "").split(",")))


@dataclass
class BargeInResult:
    query_file: str
    inject_file: str
    run: int
    offset_ms: int
    trimmed_leading_ms: int
    outcome: str  # interrupted | reply_ended_first | timeout
    inject_to_interrupted_ms: float | None
    inject_to_last_chunk_ms: float | None
    event_log: list[str]


def trim_leading_silence(case: AudioCase, threshold_dbfs: float) -> tuple[AudioCase, int]:
    """Trim leading synthetic silence using 20 ms RMS windows."""
    samples = memoryview(case.pcm).cast("h")
    window_samples = 320
    threshold = 10 ** (threshold_dbfs / 20.0)
    onset_sample: int | None = None
    for start in range(0, len(samples), window_samples):
        window = samples[start : start + window_samples]
        if not window:
            continue
        rms = math.sqrt(sum((sample / 32768.0) ** 2 for sample in window) / len(window))
        if rms >= threshold:
            onset_sample = start
            break
    if onset_sample is None:
        raise ValueError(f"{case.path} contains no audio above {threshold_dbfs:.0f} dBFS")
    trimmed_ms = round(onset_sample * 1000 / 16_000)
    pcm = case.pcm[onset_sample * 2 :]
    duration_ms = round(len(pcm) / 2 * 1000 / 16_000)
    return AudioCase(path=case.path, duration_ms=duration_ms, pcm=pcm), trimmed_ms


async def stream_frames(
    websocket: Any,
    pcm: bytes,
    frame_ms: int,
    *,
    capture_before_send: bool = False,
) -> None:
    """Stream PCM in real-time frames.

    Browser audio frames arrive only after they have been captured. The
    injection path therefore waits for each frame before sending it; otherwise
    the server receives the first 128 ms of speech instantaneously and the
    measured reaction time is biased low by one frame.
    """
    bytes_per_ms = 16_000 * 2 / 1000
    frame_size = max(2, int(bytes_per_ms * frame_ms))
    frame_size += frame_size % 2
    for frame in chunk_bytes(pcm, frame_size):
        frame_duration_s = len(frame) / (16_000 * 2)
        if capture_before_send:
            await asyncio.sleep(frame_duration_s)
        await websocket.send(frame)
        if not capture_before_send:
            await asyncio.sleep(frame_duration_s)


def reply_is_terminal(payload: dict[str, Any]) -> bool:
    kind = payload.get("type")
    return kind in {"assistant_done", "error", "input_error"} or (
        kind == "turn_metrics" and payload.get("final") is True
    )


async def run_bargein_turn(
    websocket: Any,
    query: AudioCase,
    inject: AudioCase,
    *,
    run_index: int,
    offset_ms: int,
    trimmed_leading_ms: int,
    frame_ms: int,
    timeout_s: float,
) -> BargeInResult:
    audio_started_at: float | None = None
    last_chunk_at: float | None = None
    interrupted_at: float | None = None
    reply_done = asyncio.Event()
    audio_flowing = asyncio.Event()
    interrupted_seen = asyncio.Event()
    injecting = asyncio.Event()
    event_log: list[str] = []
    turn_started_at = time.monotonic()

    async def receiver() -> None:
        nonlocal audio_started_at, last_chunk_at, interrupted_at
        while True:
            message = await websocket.recv()
            if not isinstance(message, str):
                continue
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue
            kind = payload.get("type")
            now = time.monotonic()
            # High-rate and deployment-specific events are left out of the log.
            if kind not in NOISY_EVENTS:
                event_log.append(f"+{now - turn_started_at:.2f}s {kind}")
            if kind == "audio_started":
                audio_started_at = now
                audio_flowing.set()
            elif kind == "audio_chunk":
                last_chunk_at = now
            elif kind == "interrupted":
                # Only count interruptions caused by our injection; a reset
                # between turns can emit a stale `interrupted` event.
                if injecting.is_set():
                    interrupted_at = now
                    interrupted_seen.set()
            elif reply_is_terminal(payload):
                reply_done.set()

    receive_task = asyncio.create_task(receiver())
    inject_first_sent: float | None = None
    outcome = "timeout"
    try:
        await websocket.send(json.dumps({"type": "start_recording"}))
        await stream_frames(websocket, query.pcm, frame_ms)
        await websocket.send(json.dumps({"type": "stop_recording"}))

        await asyncio.wait_for(audio_flowing.wait(), timeout=timeout_s)
        await asyncio.sleep(offset_ms / 1000)
        if reply_done.is_set():
            outcome = "reply_ended_first"
        else:
            inject_first_sent = time.monotonic()
            injecting.set()
            inject_task = asyncio.create_task(
                stream_frames(
                    websocket,
                    inject.pcm,
                    frame_ms,
                    capture_before_send=True,
                )
            )
            interrupted_wait = asyncio.create_task(interrupted_seen.wait())
            reply_wait = asyncio.create_task(reply_done.wait())
            try:
                done, _pending = await asyncio.wait(
                    {interrupted_wait, reply_wait},
                    timeout=timeout_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if interrupted_wait in done and interrupted_seen.is_set():
                    outcome = "interrupted"
                    # Let any trailing audio_chunk events drain before measuring.
                    await asyncio.sleep(0.5)
                elif reply_wait in done and reply_done.is_set():
                    outcome = "reply_ended_first"
                else:
                    outcome = "timeout"
            finally:
                for waiter in (interrupted_wait, reply_wait):
                    waiter.cancel()
                for waiter in (interrupted_wait, reply_wait):
                    with contextlib.suppress(asyncio.CancelledError):
                        await waiter
                inject_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await inject_task
    except asyncio.TimeoutError:
        outcome = "timeout"
    finally:
        receive_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receive_task

    to_interrupted = None
    to_last_chunk = None
    if outcome == "interrupted" and inject_first_sent is not None:
        if interrupted_at is not None:
            to_interrupted = (interrupted_at - inject_first_sent) * 1000
        if last_chunk_at is not None:
            to_last_chunk = max(0.0, (last_chunk_at - inject_first_sent) * 1000)
    return BargeInResult(
        query_file=query.path.name,
        inject_file=inject.path.name,
        run=run_index,
        offset_ms=offset_ms,
        trimmed_leading_ms=trimmed_leading_ms,
        outcome=outcome,
        inject_to_interrupted_ms=to_interrupted,
        inject_to_last_chunk_ms=to_last_chunk,
        event_log=event_log,
    )


async def run_benchmark(args: argparse.Namespace) -> list[BargeInResult]:
    cases = load_audio_cases(Path(args.audio_dir), limit=None)
    trimmed_cases: list[AudioCase] = []
    trimmed_leading_ms: list[int] = []
    for case in cases:
        trimmed, leading_ms = trim_leading_silence(case, args.trim_threshold_dbfs)
        trimmed_cases.append(trimmed)
        trimmed_leading_ms.append(leading_ms)
    results: list[BargeInResult] = []
    connect_kwargs = websocket_connect_kwargs(args.url)
    async with websockets.connect(args.url, max_size=None, **connect_kwargs) as websocket:
        # Drain the ready/suggestions burst before the first turn.
        await asyncio.sleep(0.5)
        for run_index in range(1, args.runs + 1):
            query = cases[(run_index - 1) % len(cases)]
            inject_index = run_index % len(cases)
            inject = trimmed_cases[inject_index]
            offset_ms = DEFAULT_OFFSETS_MS[(run_index - 1) % len(DEFAULT_OFFSETS_MS)]
            result = await run_bargein_turn(
                websocket,
                query,
                inject,
                run_index=run_index,
                offset_ms=offset_ms,
                trimmed_leading_ms=trimmed_leading_ms[inject_index],
                frame_ms=args.frame_ms,
                timeout_s=args.timeout_s,
            )
            results.append(result)
            print(
                f"run {run_index}: {result.outcome} offset={offset_ms}ms "
                f"cancel={fmt_ms(result.inject_to_interrupted_ms)}ms "
                f"last_chunk={fmt_ms(result.inject_to_last_chunk_ms)}ms",
                file=sys.stderr,
            )
            await websocket.send(json.dumps({"type": "reset"}))
            await asyncio.sleep(args.reset_pause_s)
    return results


def markdown_table(results: list[BargeInResult]) -> str:
    interrupted = [r for r in results if r.outcome == "interrupted"]
    timed_out = [r for r in results if r.outcome == "timeout"]
    eligible = len(interrupted) + len(timed_out)
    cancel = [
        r.inject_to_interrupted_ms
        for r in interrupted
        if r.inject_to_interrupted_ms is not None
    ]
    last_chunk = [
        r.inject_to_last_chunk_ms
        for r in interrupted
        if r.inject_to_last_chunk_ms is not None
    ]
    lines = [
        "| Barge-in metric | p50 ms | p95 ms | max ms |",
        "|---|---:|---:|---:|",
        f"| Acoustic onset → response cancelled | {fmt_ms(percentile(cancel, 50))} | "
        f"{fmt_ms(percentile(cancel, 95))} | {fmt_ms(max(cancel) if cancel else None)} |",
        f"| Acoustic onset → last server audio chunk | {fmt_ms(percentile(last_chunk, 50))} | "
        f"{fmt_ms(percentile(last_chunk, 95))} | "
        f"{fmt_ms(max(last_chunk) if last_chunk else None)} |",
        "",
        f"Successful eligible interruptions: {len(interrupted)}/{eligible} "
        f"({len(results)} scheduled attempts; "
        f"reply_ended_first: {sum(1 for r in results if r.outcome == 'reply_ended_first')}, "
        f"timeout: {len(timed_out)})",
    ]
    return "\n".join(lines)


def write_results(args: argparse.Namespace, results: list[BargeInResult], table: str) -> Path:
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = output_dir / f"{stamp}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": args.url,
        "runs": args.runs,
        "frame_ms": args.frame_ms,
        "trim_threshold_dbfs": args.trim_threshold_dbfs,
        "offsets_ms": list(DEFAULT_OFFSETS_MS),
        "summary_table": table,
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="wss://127.0.0.1:8765/speech", help="speech WebSocket URL")
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR), help="directory of 16 kHz mono PCM16 WAVs")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="directory for raw JSON output")
    parser.add_argument("--runs", type=int, default=30, help="number of interruption attempts")
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=DEFAULT_FRAME_MS,
        help="streaming frame size and pacing (browser AudioWorklet default: 128 ms)",
    )
    parser.add_argument(
        "--trim-threshold-dbfs",
        type=float,
        default=DEFAULT_TRIM_THRESHOLD_DBFS,
        help="RMS threshold used to remove synthetic leading silence",
    )
    parser.add_argument("--timeout-s", type=float, default=60.0, help="per-phase timeout")
    parser.add_argument("--reset-pause-s", type=float, default=1.0, help="pause after reset between turns")
    args = parser.parse_args()

    results = asyncio.run(run_benchmark(args))
    table = markdown_table(results)
    print(table)
    path = write_results(args, results, table)
    print(f"\nRaw JSON: {path}")


if __name__ == "__main__":
    main()
