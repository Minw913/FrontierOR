"""Compatibility and security wrapper around CORAL's CLI."""

from __future__ import annotations

import os
from pathlib import Path


def _patch_codex_settings():
    from coral.agent import manager as manager_module
    from coral.workspace import worktree as worktree_module
    import coral.workspace as workspace_module

    def setup_codex_settings(
        worktree_path: Path,
        coral_dir: Path,
        *,
        research: bool = True,
        gateway_url: str | None = None,
        gateway_api_key: str | None = None,
    ) -> None:
        del coral_dir, research, gateway_api_key
        codex_dir = worktree_path / ".codex"
        codex_dir.mkdir(exist_ok=True)
        sandbox_mode = (
            "workspace-write"
            if os.environ.get("FRONTIER_OR_ANTI_HACK") == "1"
            else "danger-full-access"
        )
        lines = [
            'model = "gpt-5.4"',
            'approval_policy = "never"',
            f'sandbox_mode = "{sandbox_mode}"',
            'personality = "pragmatic"',
        ]
        if gateway_url:
            lines += [
                'model_provider = "litellm"',
                "",
                "[model_providers.litellm]",
                'name = "LiteLLM Proxy"',
                f'base_url = "{gateway_url}/v1"',
                'wire_api = "responses"',
                'env_key = "OPENAI_API_KEY"',
            ]
        (codex_dir / "config.toml").write_text("\n".join(lines) + "\n")

    worktree_module.setup_codex_settings = setup_codex_settings
    workspace_module.setup_codex_settings = setup_codex_settings
    manager_module.setup_codex_settings = setup_codex_settings


def _patch_anti_hack_runtime() -> None:
    if os.environ.get("FRONTIER_OR_ANTI_HACK") != "1":
        return
    from coral.agent import manager as manager_module
    from coral.agent import registry
    from frontieror.infra.agent.instructions import generate_secure_coral_md
    from frontieror.infra.agent.runtime import SecureCodexRuntime

    registry.register_runtime("codex", SecureCodexRuntime, default_model="gpt-5.4")
    # AgentManager imported this function by value, so patch its module binding.
    manager_module.generate_coral_md = generate_secure_coral_md


def main():
    _patch_codex_settings()
    _patch_anti_hack_runtime()
    from coral.cli import main as coral_main

    coral_main()


if __name__ == "__main__":
    main()
