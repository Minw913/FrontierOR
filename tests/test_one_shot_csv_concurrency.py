import csv
import threading
import time

import pytest

import one_shot_eval


def _row():
    row = {column: "" for column in one_shot_eval.RESULTS_CSV_COLUMNS}
    row.update(
        paper_id="paper1",
        model="model1",
        instance="tiny",
        status="pass",
        feasible="True",
        obj="1.0",
        time="2.0",
        gap="0.0",
        first_status="pass",
        first_feasible="True",
        first_obj="1.0",
        first_time="2.0",
    )
    return row


@pytest.mark.parametrize(
    ("reader", "expected"),
    [
        (lambda: one_shot_eval._get_csv_done_instances("paper1", "model1"), {"tiny"}),
        (lambda: one_shot_eval._read_prev_result_rows("paper1", "model1")["tiny"]["status"], "pass"),
        (lambda: one_shot_eval._read_prev_first_results("paper1", "model1")["tiny"]["status"], "pass"),
    ],
)
def test_resume_csv_readers_wait_for_in_progress_rewrite(
    tmp_path, monkeypatch, reader, expected
):
    csv_path = tmp_path / "results.csv"
    monkeypatch.setattr(one_shot_eval, "_RESULTS_CSV_OVERRIDE", str(csv_path))
    writer_started = threading.Event()
    allow_writer_to_finish = threading.Event()

    def rewrite():
        with one_shot_eval._csv_file_lock(str(csv_path)):
            with csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=one_shot_eval.RESULTS_CSV_COLUMNS
                )
                writer.writeheader()
                stream.flush()
                writer_started.set()
                assert allow_writer_to_finish.wait(timeout=2)
                writer.writerow(_row())

    writer_thread = threading.Thread(target=rewrite)
    writer_thread.start()
    assert writer_started.wait(timeout=2)

    result = []
    reader_thread = threading.Thread(target=lambda: result.append(reader()))
    reader_thread.start()
    time.sleep(0.05)
    assert reader_thread.is_alive(), "reader bypassed the in-progress writer lock"

    allow_writer_to_finish.set()
    writer_thread.join(timeout=2)
    reader_thread.join(timeout=2)
    assert not writer_thread.is_alive()
    assert not reader_thread.is_alive()
    assert result == [expected]
