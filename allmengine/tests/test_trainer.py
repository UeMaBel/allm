import pytest
from allmengine import alu
from allmengine.trainer import Trainer


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    (tmp_path / "candidate_model.py").write_text('''
from allmengine.reference import ReferenceModel
class Model(ReferenceModel):
    def __init__(self):
        self.weight = 0
    def state_dict(self):
        return {"weight": self.weight}
    def load_state_dict(self, state):
        self.weight = state["weight"]
    def mutate(self, rng):
        self.weight += 1
    def forward(self, opcode, lhs, rhs=None):
        return super().forward(opcode, lhs, rhs) + self.weight - 2
''', encoding="utf-8")
    (tmp_path / "test_target.py").write_text('''
from allmengine import NeuralValue
def test_add():
    assert (NeuralValue(3) + 4).value == 7
''', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    previous = alu.model
    yield tmp_path
    alu.set_model(previous)


def test_mutations_reach_external_suite(experiment):
    result = Trainer("candidate_model:Model", experiment).train(4)
    assert result.converged
    assert result.evaluations == 3
    assert result.state == {"weight": 2}
    assert alu.model.state_dict() == result.state


def test_budget_exhaustion_is_not_success(experiment):
    result = Trainer("candidate_model:Model", experiment).train(1)
    assert not result.converged
    assert result.feedback.failed == 1


def test_collection_errors_are_not_training_feedback(experiment):
    (experiment / "test_target.py").write_text("import nonexistent_allm_module\n")
    with pytest.raises(RuntimeError, match="infrastructure failure"):
        Trainer("candidate_model:Model", experiment).train(2)


def test_all_skipped_is_not_convergence(experiment):
    (experiment / "test_target.py").write_text(
        "import pytest\ndef test_skip():\n    pytest.skip('no observation')\n")
    result = Trainer("candidate_model:Model", experiment).train(1)
    assert not result.converged
    assert result.feedback.skipped == 1


def test_training_reports_configuration_and_selected_test_count(experiment, capsys):
    (experiment / "test_other.py").write_text("def test_other(): pass\n")
    Trainer("candidate_model:Model", experiment,
            pytest_args=("-k", "test_add"), seed=42).train(1)
    output = capsys.readouterr().out
    assert "Starting training: candidate_model:Model" in output
    assert str(experiment) in output
    assert "seed: 42" in output
    assert "Loading pytest and collecting tests" in output
    assert "Test count: 1 selected tests" in output
    assert "Passed: 0; failed: 1; skipped: 0" in output
    assert "evaluation budget exhausted" in output


def test_worker_timeout_stops_evaluation(experiment):
    (experiment / "test_target.py").write_text(
        "import time\ndef test_slow():\n    time.sleep(5)\n")
    with pytest.raises(RuntimeError, match="pytest evaluation timed out"):
        Trainer("candidate_model:Model", experiment, timeout=0.3).train(1)


def test_gradient_updates_reach_pytest_worker(experiment):
    (experiment / "gradient_model.py").write_text("""
from dataclasses import replace
from allmengine import BinaryNetwork, NetworkConfig, TrainingConfig
class Model(BinaryNetwork):
    def __init__(self):
        super().__init__(replace(NetworkConfig.load(), hidden_bits=(), output_bits=4),
                         TrainingConfig(learning_rate=0.1, epochs=20))
        state = self.state_dict()
        for row in state['layers'][-1]['weights']:
            row[:] = [0] * len(row)
        state['layers'][-1]['biases'][:] = [1, 1, 1, 1]
        self.load_state_dict(state)
""")
    (experiment / "test_target.py").write_text(
        "from allmengine import NeuralValue\n"
        "def test_add():\n    assert (NeuralValue(0) + 0).value == 0\n")
    result = Trainer("gradient_model:Model", experiment, seed=42).train(3)
    assert result.converged
    assert result.evaluations == 2
    assert result.feedback.operation_accuracy == 1
    assert alu.model.forward('add', 0, 0) == 0


def test_new_branches_rescore_incumbent_on_same_inputs(experiment):
    (experiment / "branch_model.py").write_text("""
from allmengine.reference import ReferenceModel
class Model(ReferenceModel):
    def __init__(self): self.weight = 0
    def state_dict(self): return {'weight': self.weight}
    def load_state_dict(self, state): self.weight = state['weight']
    def mutate(self, rng): self.weight = 1
    def forward(self, opcode, lhs, rhs=None):
        expected = super().forward(opcode, lhs, rhs)
        if opcode.value == 'add': return expected if self.weight else expected - 1
        return 0 if self.weight else expected
""")
    (experiment / "test_target.py").write_text(
        "from allmengine import NeuralValue\n"
        "def test_branches():\n"
        "    assert (NeuralValue(3) + 4).value == 7\n"
        "    first = (NeuralValue(5) - 2).value\n"
        "    second = (NeuralValue(7) - 2).value\n"
        "    assert (first, second) == (3, 5)\n")
    result = Trainer("branch_model:Model", experiment).train(2)
    assert result.state == {'weight': 0}
    assert result.feedback.scored_operations == 3
    assert result.feedback.score == pytest.approx(2 / 3)


def test_training_initializes_from_saved_weights(experiment, capsys):
    import json
    checkpoint = experiment / "weights.json"
    checkpoint.write_text(json.dumps({"weight": 1}))
    result = Trainer("candidate_model:Model", experiment, weights=checkpoint).train(2)
    assert result.converged
    assert result.evaluations == 2
    assert result.state == {"weight": 2}
    assert json.loads(checkpoint.read_text()) == {"weight": 1}
    assert f"Initial weights: {checkpoint}" in capsys.readouterr().out


def test_missing_initial_weights_fail_without_fresh_training(experiment):
    with pytest.raises(FileNotFoundError):
        Trainer("candidate_model:Model", experiment,
                weights=experiment / "missing.json").train(1)
