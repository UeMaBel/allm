"""Cooperative run limits and atomic saves of the best evaluated model."""
import json
import os
from pathlib import Path
import tempfile
import time


def save_checkpoint(path, state):
    """Replace a checkpoint only after its complete JSON has reached disk."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(state, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RunControl:
    """Keep deadline and persistence out of the optimizer and worker protocols.

    Checks run between minibatches and evaluations. An evaluation already in
    progress finishes before stopping; pytest's own timeout still applies.
    """

    def __init__(self, config, checkpoint=None, *, clock=None):
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.deadline = (self.started + config.time_budget_seconds
                         if config.time_budget_seconds else None)
        self.interval = config.checkpoint_interval_seconds
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.last_save = self.started
        self.state = None

    @property
    def expired(self):
        return self.deadline is not None and self.clock() >= self.deadline

    def retain(self, state):
        first = self.state is None
        self.state = state
        self.save(force=first)

    def save(self, *, force=False):
        if self.checkpoint is not None and self.state is not None:
            if force or self.clock() - self.last_save >= self.interval:
                save_checkpoint(self.checkpoint, self.state)
                self.last_save = self.clock()
                print(f"  Checkpoint saved: {self.checkpoint}", flush=True)

    def pulse(self):
        self.save()
        return not self.expired

    def reason(self, converged):
        return "converged" if converged else "time_budget" if self.expired else "evaluation_budget"
