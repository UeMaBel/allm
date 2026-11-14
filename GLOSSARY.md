# allm

allm is an experimental, educational project exploring whether trainable operations can reproduce the behavior required by ordinary Python tests. Its engine, allmengine, learns from an external test suite, allmtest, while target application logic stays independent of the engine.

The names below are canonical; they supersede the earlier names in [handoff.md](handoff.md). These definitions describe the intended concepts, not implementation status.

## Language

### Projects and boundaries

**allm**:
The overall project encompassing allmengine, allmtest, and experiments with learned Python behavior.
_Avoid_: neural_engine when referring to the whole project.

**allmengine**:
The reusable engine that replaces selected Python operations with trainable operations and learns their parameters from external tests. It is independent of any particular target application or test suite.
_Avoid_: neural_engine, Project 1.

**allmtest**:
The project's test suite that specifies the behavior allmengine is trained to satisfy. It provides examples of required behavior through ordinary Python tests.
_Avoid_: test_calculator as the name of the whole suite.

**Target application**:
The ordinary Python code whose operations are selected for replacement by allmengine. Its business logic contains no engine-specific or neural training code.
_Avoid_: target_example as a general name for every target application.
