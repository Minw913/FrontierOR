import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "frontieror_task_builder/scripts/check_artifact_contract.py"
)


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_artifact_contract", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_start_time_assignment_must_be_executable(tmp_path):
    checker = _load_checker()
    script = tmp_path / "gurobi_code.py"
    script.write_text(
        'r"""\n_GUROBI_CODE_START_TIME = time.time()\n"""\n'
        "import time\n"
        "print(time.time() - _GUROBI_CODE_START_TIME)\n",
        encoding="utf-8",
    )

    assert not checker.script_defines_name_before_use(
        script, "_GUROBI_CODE_START_TIME"
    )


def test_start_time_real_assignment_passes(tmp_path):
    checker = _load_checker()
    script = tmp_path / "gurobi_code.py"
    script.write_text(
        "import time\n"
        "_GUROBI_CODE_START_TIME = time.time()\n"
        "print(time.time() - _GUROBI_CODE_START_TIME)\n",
        encoding="utf-8",
    )

    assert checker.script_defines_name_before_use(script, "_GUROBI_CODE_START_TIME")


def test_start_time_annotation_only_does_not_define_name(tmp_path):
    checker = _load_checker()
    script = tmp_path / "gurobi_code.py"
    script.write_text(
        "import time\n"
        "_GUROBI_CODE_START_TIME: float\n"
        "print(time.time() - _GUROBI_CODE_START_TIME)\n",
        encoding="utf-8",
    )

    assert not checker.script_defines_name_before_use(
        script, "_GUROBI_CODE_START_TIME"
    )
