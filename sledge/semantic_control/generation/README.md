# Scene generation controls

This package owns everything that turns semantic intent into executable scene
changes and evaluates those changes.

`legacy/` contains the currently working parser/template/editor/evaluator chain.
It is isolated so the EventFrame research pipeline can evolve independently.

Future modules should be added here in the following order:

```text
HazardSemanticSpec
→ base-scene matcher
→ constraint solver
→ SceneEditPlan compiler
→ primitive executor
→ semantic and physical validator
```

