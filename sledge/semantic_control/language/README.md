# EventFrame language understanding

This package owns natural-language decomposition and semantic representation.
It must not import scene editors, diffusion code, or simulation builders.

Main modules:

- `narrative_semantics.py`: clauses, candidate events, and hazard focus.
- `event_frame.py`: serializable intermediate representation.
- `event_frame_parser.py`: LLM/fallback parsing into EventFrame.
- `event_sequence_builder.py`: temporal event reconstruction.
- `event_frame_verifier.py`: consistency checks and deterministic repair.
- `event_frame_mapper.py`: compositional hazard specification.
- `missing_info_filler.py`: provenance-aware default/distribution completion.
- `direct_template_baseline.py`: no-EventFrame comparison baseline.

