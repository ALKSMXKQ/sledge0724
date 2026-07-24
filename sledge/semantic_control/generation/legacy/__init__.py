"""Executable legacy natural-language-to-scene pipeline.

The old keyword parser, fixed template, three editors, and alignment evaluators
are grouped here while the EventFrame-to-edit-plan compiler is developed.
Imports are deliberately explicit at call sites to avoid pulling simulator
dependencies into lightweight language-only workflows.
"""

