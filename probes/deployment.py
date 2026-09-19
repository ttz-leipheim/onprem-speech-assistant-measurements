"""The single bind point between these probes and a deployment's own code.

Two probes need more than the WebSocket. `eval_noise.py` and `eval_f2_gate.py`
measure a deployed recognizer, wake-word detector and voice-activity detector
*at the thresholds the running system uses*, so they load the deployment's own
components rather than constructing fresh ones. Building a second copy here
would measure a different system. Every other probe talks to the assistant over
its WebSocket and needs nothing from this module.

This file names five roles, not any particular implementation:

    Config          frozen settings object exposing `from_env()`, plus the
                    fields the probes read: `language_mode`, `vad_onset_ms`,
                    `min_speech_ms`, `asr_model`, `asr_compute_type`,
                    `asr_beam_size`
    Transcriber     constructed from a Config. Provides the deployed decode
                    path, which returns a result object carrying the final
                    text and, when a turn was rejected, the reason
    WakeWord        constructed from a Config; consumes float32 frames
    VoiceActivity   constructed from a Config; exposes the frame rate,
                    per-frame speech probability and the decision threshold
    codec           float32 <-> PCM16 byte conversion

Bind them with environment variables, each `module:attribute` relative to
ASSISTANT_PACKAGE:

    export ASSISTANT_PACKAGE=your_package
    export ASSISTANT_CONFIG=config:Config
    export ASSISTANT_TRANSCRIBER=components:Transcriber
    export ASSISTANT_WAKEWORD=components:WakeWord
    export ASSISTANT_VAD=components:VoiceActivity
    export ASSISTANT_CODEC=audio
    export ASSISTANT_MISSING_DEP=components:MissingDependencyError

If the package is not installed or the names do not resolve, importing either
offline probe raises `DeploymentNotBound` with the variable that is missing.
The published result files in `results/` carry every number the paper prints
from these two probes, so they can be checked without binding anything.
"""

from __future__ import annotations

import importlib
import os

PACKAGE_VAR = "ASSISTANT_PACKAGE"

# Role -> (environment variable, default "module:attribute" spelling).
ROLES = {
    "Config": ("ASSISTANT_CONFIG", "config:Config"),
    "Transcriber": ("ASSISTANT_TRANSCRIBER", "components:Transcriber"),
    "WakeWord": ("ASSISTANT_WAKEWORD", "components:WakeWord"),
    "VoiceActivity": ("ASSISTANT_VAD", "components:VoiceActivity"),
    "MissingSpeechDependencyError": ("ASSISTANT_MISSING_DEP",
                                     "components:MissingDependencyError"),
}
# Optional: the config field holding a decode-time prompt, if the deployment
# uses one. The gate-off decode must pass the same value as the deployed path,
# or the two decodes would not be comparable.
PROMPT_FIELD_VAR = "ASSISTANT_PROMPT_FIELD"
PROMPT_FIELD_DEFAULT = "initial_prompt"

CODEC_VAR = "ASSISTANT_CODEC"
CODEC_DEFAULT = "audio"
CODEC_FUNCTIONS = ("float32_to_pcm16_bytes", "pcm16_bytes_to_float32")


class DeploymentNotBound(ImportError):
    """Raised when the deployment's components cannot be located."""


def _package() -> str:
    package = os.environ.get(PACKAGE_VAR)
    if not package:
        raise DeploymentNotBound(
            f"{PACKAGE_VAR} is not set. These two probes measure a deployed "
            f"recognizer and detector at their configured thresholds, so they "
            f"run on the deployment. See the module docstring for the five "
            f"roles to bind, and results/ for the measurements they produced."
        )
    return package


def _resolve(spec: str, variable: str):
    """Import `module:attribute`, relative to the assistant package."""
    module_name, _, attribute = spec.partition(":")
    full = f"{_package()}.{module_name}"
    try:
        module = importlib.import_module(full)
    except ImportError as exc:
        raise DeploymentNotBound(f"{variable}: cannot import {full} ({exc})") from exc
    if not attribute:
        return module
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise DeploymentNotBound(f"{variable}: {full} has no {attribute}") from exc


def __getattr__(name: str):
    """Resolve a role on first use, so `--help` works without a deployment."""
    if name in ROLES:
        variable, default = ROLES[name]
        return _resolve(os.environ.get(variable, default), variable)
    if name in CODEC_FUNCTIONS:
        module = _resolve(os.environ.get(CODEC_VAR, CODEC_DEFAULT), CODEC_VAR)
        try:
            return getattr(module, name)
        except AttributeError as exc:
            raise DeploymentNotBound(f"{CODEC_VAR}: module has no {name}") from exc
    raise AttributeError(name)


def initial_prompt(config) -> str | None:
    """Decode-time prompt the deployment configures, or None if it has none."""
    field = os.environ.get(PROMPT_FIELD_VAR, PROMPT_FIELD_DEFAULT)
    return getattr(config, field, None) or None
