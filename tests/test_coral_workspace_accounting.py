from test_time_self_evolution.coral import runner


def test_trusted_grader_inputs_do_not_consume_agent_workspace(tmp_path):
    private = tmp_path / '.coral/private/grader_checkouts'
    private.mkdir(parents=True)
    with (private / 'input.json').open('wb') as f:
        f.truncate(2_000_000_000)
    (tmp_path / 'code.py').write_bytes(b'x' * 100)
    assert runner._workspace_usage(str(tmp_path), 1000)[0] > 1000
    assert runner._workspace_usage(str(tmp_path), 1000, [private])[0] == 100


def test_agent_created_files_and_symlinks_do_not_escape_accounting(tmp_path):
    private = tmp_path / '.coral/private/grader_checkouts'
    private.mkdir(parents=True)
    agent = tmp_path / 'agents/agent-1/grader_checkouts'
    agent.mkdir(parents=True)
    (agent / 'large').write_bytes(b'x' * 2000)
    (tmp_path / 'private-alias').symlink_to(private, target_is_directory=True)
    assert runner._workspace_usage(str(tmp_path), 1000, [private])[0] > 1000


def test_repeated_sessions_without_submissions_are_bounded():
    guard = runner._SubmissionProgressGuard(max_restarts=3)
    rows = [{'commit_hash': 'a', 'status': 'baseline'}]
    assert not guard.exhausted(rows, 1)
    assert not guard.exhausted(rows, 3)
    assert guard.exhausted(rows, 4)


def test_pending_waits_and_new_submissions_reset_progress_guard():
    guard = runner._SubmissionProgressGuard(max_restarts=3)
    rows = [{'commit_hash': 'a', 'status': 'pending'}]
    assert not guard.exhausted(rows, 1)
    assert not guard.exhausted(rows, 1000)
    rows[0]['status'] = 'baseline'
    assert not guard.exhausted(rows, 1000)
    assert not guard.exhausted(rows, 1002)
    rows.append({'commit_hash': 'b', 'status': 'baseline'})
    assert not guard.exhausted(rows, 1003)
