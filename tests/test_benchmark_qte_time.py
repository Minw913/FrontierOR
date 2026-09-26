import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compute_benchmark_main_metrics.py"
SPEC = importlib.util.spec_from_file_location("benchmark_metrics", SCRIPT)
METRICS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(METRICS)


def test_timeout_overhead_is_capped_to_declared_budget():
    assert METRICS.qte_time_is_fast_enough(3604.2, 3601.7)


def test_small_clock_jitter_is_tolerated():
    assert METRICS.qte_time_is_fast_enough(100.08, 100.0)
    assert not METRICS.qte_time_is_fast_enough(102.0, 100.0)


def test_time_tolerance_is_configurable():
    assert not METRICS.qte_time_is_fast_enough(
        100.8, 100.0, tolerance_seconds=0.5, tolerance_fraction=0.0
    )
    assert METRICS.qte_time_is_fast_enough(
        102.0, 100.0, tolerance_seconds=2.0, tolerance_fraction=0.0
    )


def test_invalid_time_does_not_pass_qte_time_gate():
    assert not METRICS.qte_time_is_fast_enough(float("nan"), 100.0)
    assert not METRICS.qte_time_is_fast_enough(10.0, 0.0)


def test_negative_time_tolerance_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="non-negative"):
        METRICS.qte_time_is_fast_enough(
            10.0, 10.0, tolerance_seconds=-1.0
        )
