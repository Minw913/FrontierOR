"""Persistent, atomic admission count: seed initialization is not evolution."""
import contextlib
import fcntl
import json
from pathlib import Path
import uuid


class EvaluationBudget:
    def __init__(self, path, limit, seed_hash, *, resume=False, context=None):
        self.path = Path(path)
        if limit < 1:
            raise ValueError('Candidate evaluation budget must be positive')
        self.limit, self.seed_hash, self.context = limit, seed_hash, context
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.locked():
            if self.path.exists():
                row = self.read()
                if not resume:
                    raise RuntimeError('Budget ledger already exists; use --resume or a new run ID')
                if (row['limit'], row['seed_hash'], row.get('context')) != (limit, seed_hash, context):
                    raise RuntimeError('Cannot resume with a different budget, seed or evaluation context')
            else:
                self.write(dict(version=1, limit=limit, seed_hash=seed_hash,
                                context=context, submissions=[]))

    @contextlib.contextmanager
    def locked(self):
        with self.path.with_suffix('.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def read(self):
        return json.loads(self.path.read_text())

    def write(self, row):
        temporary = self.path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(row, indent=2) + '\n')
        temporary.replace(self.path)

    @property
    def remaining(self):
        with self.locked():
            return self.limit - len(self.read()['submissions'])

    def admit(self, code_hash):
        if code_hash == self.seed_hash:
            return True
        with self.locked():
            row = self.read()
            if len(row['submissions']) >= self.limit:
                return False
            # Reserve before execution; failures/interrupted evaluations count.
            row['submissions'].append({'code_hash': code_hash})
            self.write(row)
            return True


def patch_batch_admission(interface_class):
    """Trim the final offspring batch before generation; preserve EoH operators."""
    original = interface_class.get_algorithm
    if getattr(original, '_budget_admission', False):
        return

    def get_algorithm(self, population, operator):
        budget = getattr(self.interface_eval, 'candidate_budget', None)
        if budget is None:
            return original(self, population, operator)
        remaining = budget.remaining
        if remaining <= 0 and operator != 'i1':
            return [], []
        previous = self.pop_size
        # Initialization contains the shared seed, which consumes no admission.
        self.pop_size = min(previous, remaining + (1 if operator == 'i1' else 0))
        try:
            return original(self, population, operator)
        finally:
            self.pop_size = previous
    get_algorithm._budget_admission = True
    interface_class.get_algorithm = get_algorithm
