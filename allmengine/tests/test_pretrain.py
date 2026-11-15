from dataclasses import replace

import pytest

from allmengine import BinaryNetwork, NetworkConfig
from allmengine.training.examples import generate_examples
from allmengine.training.pretrain import pretrain


def test_generated_examples_are_reproducible_and_validation_is_disjoint():
    config = NetworkConfig.load()
    training = generate_examples(config, 30, seed=7)
    assert training == generate_examples(config, 30, seed=7)
    validation = generate_examples(config, 20, seed=8, exclude=training)
    assert not set(training) & set(validation)
    for example in training:
        expected = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b,
                    "mul": lambda a, b: a * b}[example.opcode](example.lhs, example.rhs)
        assert example.expected == expected


def test_exhausted_domain_fails_instead_of_looping():
    with pytest.raises(ValueError, match="enough distinct"):
        generate_examples(NetworkConfig.load(), 2, ("add",), 0, 0)


def test_pretraining_retains_model_and_reports_validation(capsys):
    model = BinaryNetwork(replace(NetworkConfig.load(), hidden_bits=(2,)))
    result = pretrain(model, examples=4, validation_examples=2, max_evaluations=2)
    assert result.state == model.state_dict()
    assert result.evaluations <= 2
    assert "Validation fitness:" in capsys.readouterr().out
