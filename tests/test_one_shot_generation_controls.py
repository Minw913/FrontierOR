import one_shot_eval as ose


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_glm52_generation_options_are_explicit():
    assert ose._model_generation_options("z-ai/glm-5.2") == {
        "reasoning_effort": "high",
        "max_completion_tokens": 65536,
    }
    assert ose._model_generation_options("openai/gpt-5.6-sol") == {}


def test_call_openrouter_reports_truncation_and_sends_model_options(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(json)
        return _Response({
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "```python\nprint('partial')"},
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 65536},
        })

    monkeypatch.setattr(ose.requests, "post", fake_post)
    content, usage = ose.call_openrouter(
        [{"role": "user", "content": "test"}],
        {"OPENROUTER_API_KEY": "test-key"},
        "z-ai/glm-5.2",
    )

    assert content.startswith("```python")
    assert captured["reasoning_effort"] == "high"
    assert captured["max_completion_tokens"] == 65536
    assert usage["finish_reason"] == "length"
    assert usage["truncated"] is True


def test_initial_generation_repairs_a_truncated_response(monkeypatch, tmp_path):
    replies = iter([
        ("```python\nprint('partial')", {"truncated": True}),
        ("```python\nprint('complete')\n```", {"truncated": False}),
    ])

    def fake_call(*args, **kwargs):
        text, flags = next(replies)
        return text, {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cached_tokens": 0,
            "finish_reason": "length" if flags["truncated"] else "stop",
            **flags,
        }

    monkeypatch.setattr(ose, "call_openrouter", fake_call)
    code_path = tmp_path / "code.py"
    attempt_path = tmp_path / "code_attempt0.py"
    code, _ = ose._generate_initial_code(
        "prompt",
        {"OPENROUTER_API_KEY": "test-key"},
        "z-ai/glm-5.2",
        code_path,
        attempt_path,
        init_gen_max=1,
    )

    assert code == "print('complete')"
    assert code_path.read_text() == code
    assert attempt_path.read_text() == code


def test_token_usage_accumulates_call_and_truncation_metadata():
    total = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    ose._add_token_usage(total, {
        "prompt_tokens": 10, "completion_tokens": 20, "cached_tokens": 3,
        "api_calls": 1, "truncated_responses": 1, "finish_reason": "length",
    })
    ose._add_token_usage(total, {
        "prompt_tokens": 5, "completion_tokens": 8, "cached_tokens": 1,
        "api_calls": 1, "truncated_responses": 0, "finish_reason": "stop",
    })
    assert total == {
        "prompt_tokens": 15, "completion_tokens": 28, "cached_tokens": 4,
        "api_calls": 2, "truncated_responses": 1,
        "finish_reasons": ["length", "stop"],
    }
