"""Lightweight diffusion-mode constants shared by CLI and model runners."""

RAW_DIFFUSION_BASELINE = "raw_diffusion_baseline"
SEMANTIC_PROTECTED = "semantic_protected"
TOPOLOGY_ADAPTIVE = "topology_adaptive"

SUPPORTED_DIFFUSION_MODES = frozenset(
    {
        RAW_DIFFUSION_BASELINE,
        SEMANTIC_PROTECTED,
        TOPOLOGY_ADAPTIVE,
    }
)

__all__ = [
    "RAW_DIFFUSION_BASELINE",
    "SEMANTIC_PROTECTED",
    "TOPOLOGY_ADAPTIVE",
    "SUPPORTED_DIFFUSION_MODES",
]
