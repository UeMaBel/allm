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
