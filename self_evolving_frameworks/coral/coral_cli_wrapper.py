"""Compatibility wrapper around CORAL's CLI.

Only patches the generated Codex ``config.toml`` (the pinned CORAL revision
writes ``[tools].web_search = "disabled"``, which the locally-installed Codex
CLI rejects). The instruction template (``CORAL.md`` / ``AGENTS.md``) is kept
upstream so single/multi-agent auto-switching works and the full collaborative
workflow is intact. Project-specific guardrails (e.g. forbid reading
``gurobi_solution/``) are injected via ``task.tips`` in ``task.yaml``, which
CORAL renders as a ``## Tips`` section appended to the original template.
"""

from __future__ import annotations

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
        lines = [
            'model = "gpt-5.4"',
            'approval_policy = "never"',
            'sandbox_mode = "danger-full-access"',
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


def main():
    _patch_codex_settings()
    from coral.cli import main as coral_main

    coral_main()


if __name__ == "__main__":
    main()
