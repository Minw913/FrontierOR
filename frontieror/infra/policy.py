"""Fail-closed policy selection for FrontierOR execution profiles."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Sequence


class EvaluationProfile(StrEnum):
    RESEARCH = "research"
    AGENT = "agent"


DEFAULT_AGENT_DOCKER_IMAGE = "frontieror-candidate:1"
DEFAULT_CORAL_AGENT_IMAGE = "frontieror-coral-agent:0.1"


@dataclass(frozen=True)
class AgentModePolicy:
    """Non-overridable settings for evaluating an untrusted agent system."""

    framework: str = "coral"
    execution_backend: str = "docker"
    scorer: str = "staged_qte"
    agent_isolation: str = "docker"
    model_access: str = "proxy"
    final_feedback: str = "aggregate_after_agent_shutdown"

    def validate_split(
        self,
        *,
        stage1_instances: Iterable[str],
        dev_instances: Iterable[str],
        final_instances: Iterable[str],
    ) -> None:
        stage1 = list(stage1_instances)
        dev = list(dev_instances)
        final = list(final_instances)
        for label, values in (("stage1", stage1), ("dev", dev), ("final", final)):
            if len(values) != len(set(values)):
                raise ValueError(f"agent mode {label} instances must not contain duplicates")
        public = set(stage1) | set(dev)
        if not final:
            raise ValueError("agent mode requires a non-empty final instance set")
        overlap = sorted(public & set(final))
        if overlap:
            raise ValueError(
                "agent mode requires dev/final isolation with disjoint instances; overlap: "
                + ", ".join(overlap)
            )


def validate_anti_hack_runtime(
    *,
    enabled: bool,
    exec_mode: str,
    final_test_instances: Iterable[str] | None = None,
    scorer: str | None = None,
) -> None:
    """Compatibility validator used by the upstream runner adapters."""
    if not enabled:
        return
    if exec_mode != "docker":
        raise ValueError(
            "anti-hack mode requires --exec-mode docker; agent evaluation "
            "cannot use a host candidate process"
        )
    if final_test_instances is not None and not list(final_test_instances):
        raise ValueError("agent evaluation requires a non-empty final test set")
    if scorer == "aocc":
        raise ValueError(
            "agent evaluation does not accept candidate-written AOCC "
            "timestamps; use staged_qte"
        )


@functools.lru_cache(maxsize=16)
def _resolve_image(image_ref: str) -> str:
    from frontieror.infra.execution import resolve_docker_image

    return resolve_docker_image(image_ref)


# Name retained for callers from the pre-package implementation.
_resolve_docker_image_cached = _resolve_image


def with_anti_hack_exec_cfg(exec_cfg: dict | None, enabled: bool) -> dict:
    """Attach the immutable candidate-sandbox settings to an exec config."""
    out = dict(exec_cfg or {})
    if not enabled:
        return out
    image_ref = str(out.get("docker_image", DEFAULT_AGENT_DOCKER_IMAGE))
    out.update(
        {
            "anti_hack": True,
            "docker_image_ref": image_ref,
            "docker_image": _resolve_image(image_ref),
        }
    )
    return out


_FORCED_OPTIONS = {
    "--framework": "coral",
    "--exec-mode": "docker",
    "--stage2-scorer": "staged_qte",
    "--coral-agent-isolation": "docker",
    "--coral-model-access": "proxy",
    "--coral-agent-image": DEFAULT_CORAL_AGENT_IMAGE,
}


def _option_values(argv: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(argv):
        if token == option and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
    return values


def hardened_agent_argv(argv: Sequence[str]) -> list[str]:
    """Validate user arguments and append the non-overridable agent profile."""
    incoming = list(argv)
    for option, expected in _FORCED_OPTIONS.items():
        values = _option_values(incoming, option)
        if any(value != expected for value in values):
            raise ValueError(
                f"agent mode fixes {option}={expected}; received {values[-1]!r}"
            )

    modes = _option_values(incoming, "--modes")
    if modes and any(mode != "self_evolve" for mode in modes):
        raise ValueError("agent mode supports only --modes self_evolve")
    if "--coral-gateway" in incoming:
        raise ValueError("agent mode does not permit the legacy CORAL gateway")

    return [
        *incoming,
        "--modes",
        "self_evolve",
        "--framework",
        "coral",
        "--exec-mode",
        "docker",
        "--stage2-scorer",
        "staged_qte",
        "--coral-agent-isolation",
        "docker",
        "--coral-model-access",
        "proxy",
        "--coral-agent-image",
        DEFAULT_CORAL_AGENT_IMAGE,
        "--anti-hack",
    ]
