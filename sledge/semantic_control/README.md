# Semantic control packages

The semantic-control code contains two pipelines with different maturity and
responsibilities.

## EventFrame language pipeline

Location: `language/`

```text
text
→ narrative decomposition
→ hazard focus selection
→ EventFrame
→ ordered event sequence
→ verification and repair
→ compositional hazard specification
→ missing-information completion
```

This pipeline is used by the language benchmark scripts. It does not yet edit a
SLEDGE scene directly.

## Legacy executable generation pipeline

Location: `generation/legacy/`

```text
text
→ keyword parser
→ PromptSpec
→ crossing / cut-in / hard-brake editor
→ prompt-alignment evaluator
```

This is the pipeline currently used by scenario-cache construction. The old
top-level modules remain compatibility imports so existing commands do not need
to change.

## Integration boundary

The planned integration point is an EventFrame-to-edit-plan compiler under
`generation/`. It should consume the compositional hazard specification and
produce executable edit primitives without importing parser internals.

