"""Application logic stays engine-independent; callers wrap its inputs."""


def choose(condition):
    if condition:
        return "if branch"
    return "else branch"


def add_if_positive(left, right):
    total = left + right
    if total > 0:
        return total
    return left - left
