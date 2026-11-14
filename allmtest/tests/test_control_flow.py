import pytest
from allmengine import NeuralValue
from target import add_if_positive, choose


def test_if_else():
    assert choose(NeuralValue(True)) == "if branch"
    assert choose(NeuralValue(False)) == "else branch"


@pytest.mark.parametrize("left,right,expected", [(3, 4, 7), (-3, 1, 0), (0, 0, 0)])
def test_add_if_positive(left, right, expected):
    result = add_if_positive(NeuralValue(left), NeuralValue(right))
    assert isinstance(result, NeuralValue)
    assert result.value == expected
