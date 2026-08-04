#!/usr/bin/env python3
"""Emit a deterministic Codex JSONL stream for container lifecycle checks."""

import json


print(json.dumps({"type": "thread.started", "thread_id": "fixture-session"}))
for index in range(15):
    print(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": f"fixture-{index}",
                    "type": "command_execution",
                    "command": f"fixture-command-{index}",
                    "status": "completed",
                },
            }
        )
    )
print(json.dumps({"type": "turn.completed"}))
