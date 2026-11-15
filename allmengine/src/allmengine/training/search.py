"""Shared mutation selection on an expanding, fairly scored corpus."""
from copy import deepcopy
from .fitness import operation_key
from .result import TrainingResult


def search(model, evaluate, score, report, rng, max_evaluations):
    """Optimize custom black-box models using observed calls.

    evaluate supplies feedback and observed operations; score rescores that
    feedback on common inputs. Neither callback needs to manage mutations.
    """
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be at least one")
    # Evaluate the initial model before making any mutations.
    print(f"Evaluation 1/{max_evaluations} (initial model)", flush=True)
    best = evaluate(model)
    report(best)
    corpus = {operation_key(op): op for op in best.operations}
    best_state = deepcopy(model.state_dict())
    evaluations = 1
    while not best.success and evaluations < max_evaluations:
        # Always mutate the best retained state, not a rejected candidate.
        model.load_state_dict(deepcopy(best_state))
        model.mutate(rng)
        print(f"Evaluation {evaluations + 1}/{max_evaluations}", flush=True)
        candidate = evaluate(model)
        candidate_state = deepcopy(model.state_dict())
        # A mutation can reach new branches. Add their calls to a shared
        # corpus and rescore BOTH states on it before comparing fitness.
        previous_count = len(corpus)
        corpus.update((operation_key(op), op) for op in candidate.operations)
        if len(corpus) > previous_count:
            print(f"  Shared operation corpus expanded: {previous_count} -> {len(corpus)} calls; "
                  "rescoring both models", flush=True)
        candidate = score(model, candidate, corpus.values())
        model.load_state_dict(best_state)
        best = score(model, best, corpus.values())
        report(candidate)
        print(f"  Incumbent fitness on same inputs: {best.score:.4%}", flush=True)
        evaluations += 1
        # Ties are kept too, allowing exploration without losing score.
        if candidate.score >= best.score:
            best = candidate
            best_state = candidate_state
            print("  Candidate kept", flush=True)
        else:
            print("  Candidate rejected", flush=True)
    model.load_state_dict(best_state)
    print(f"Training finished: {'converged' if best.success else 'evaluation budget exhausted'}; "
          f"{evaluations} evaluations; retained operation fitness: {best.score:.4%}", flush=True)
    return TrainingResult(best.success, evaluations, best, best_state)
