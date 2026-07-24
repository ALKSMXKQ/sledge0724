"""Compact BEV diagnostics for original and semantically edited scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
import numpy as np

from sledge.semantic_control.occluded_pedestrian_pipeline.generation.geometry_metrics import (
    oriented_box_corners_from_state,
)


def save_scene_comparison(
    original_scene: Any,
    edited_scene: Any,
    edit_result: Dict[str, Any],
    output_png: Path,
    *,
    prompt: str = "",
    xlim: Tuple[float, float] = (-8.0, 35.0),
    ylim: Tuple[float, float] = (-15.0, 15.0),
) -> Path:
    """Save an original/B1 side-by-side plot with actor, occluder, LOS and ROIs."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharex=True, sharey=True)
    _draw_scene(axes[0], original_scene, {}, title="B0 original")
    _draw_scene(axes[1], edited_scene, edit_result, title="B1 semantic edit")
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.6)
        ax.set_xlabel("longitudinal x / m")
    axes[0].set_ylabel("lateral y / m")
    if prompt:
        fig.suptitle(prompt, fontsize=10)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png


def save_three_stage_comparison(
    original_scene: Any,
    edited_scene: Any,
    generated_scene: Any,
    edit_result: Dict[str, Any],
    generated_result: Dict[str, Any],
    output_png: Path,
    *,
    prompt: str = "",
    xlim: Tuple[float, float] = (-8.0, 35.0),
    ylim: Tuple[float, float] = (-15.0, 15.0),
) -> Path:
    """Render B0, B1 and the actual diffusion-produced B2 vector together."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), sharex=True, sharey=True)
    _draw_scene(axes[0], original_scene, {}, title="B0 original")
    _draw_scene(axes[1], edited_scene, edit_result, title="B1 semantic template edit")
    _draw_scene(axes[2], generated_scene, generated_result, title="B2 diffusion + protected semantics")
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.6)
        ax.set_xlabel("longitudinal x / m")
    axes[0].set_ylabel("lateral y / m")
    if prompt:
        fig.suptitle(prompt, fontsize=10)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png


def _draw_scene(ax, scene: Any, edit_result: Dict[str, Any], title: str) -> None:
    _draw_lines(ax, scene.lines)
    ego = Rectangle((-2.4, -0.95), 4.8, 1.9, facecolor="#4c78a8", alpha=0.25, edgecolor="#1f4e79")
    ax.add_patch(ego)
    ax.text(0.0, 0.0, "ego", fontsize=7, ha="center", va="center")

    pedestrian_index = int(edit_result.get("pedestrian_index", -1))
    occluder_index = int(edit_result.get("occluder_index", -1))
    occluder_elem_name = str(edit_result.get("occluder_elem_name", "vehicles"))
    occluder_label = str(edit_result.get("occluder_source", "occluder")).upper()
    _draw_elements(ax, scene.vehicles, "vehicles", "#4c78a8", -1, occluder_index if occluder_elem_name == "vehicles" else -1, occluder_label)
    _draw_elements(ax, scene.pedestrians, "pedestrians", "#e45756", pedestrian_index, -1)
    _draw_elements(
        ax,
        scene.static_objects,
        "static_objects",
        "#79706e",
        -1,
        occluder_index if occluder_elem_name == "static_objects" else -1,
        occluder_label,
    )

    actor_xy = _state_xy(scene.pedestrians, pedestrian_index)
    if actor_xy is not None:
        ax.plot([0.0, actor_xy[0]], [0.0, actor_xy[1]], "--", color="#f58518", linewidth=1.4, label="ego-pedestrian LOS")
    conflict = edit_result.get("conflict_point_xy")
    if isinstance(conflict, Sequence) and len(conflict) >= 2:
        ax.plot(float(conflict[0]), float(conflict[1]), marker="x", color="black", markersize=7)
    for roi in edit_result.get("preserved_rois", []):
        try:
            rect = Rectangle(
                (float(roi["x_min"]), float(roi["y_min"])),
                float(roi["x_max"]) - float(roi["x_min"]),
                float(roi["y_max"]) - float(roi["y_min"]),
                fill=False,
                linestyle=":",
                linewidth=0.9,
                edgecolor="#54a24b",
            )
            ax.add_patch(rect)
        except (KeyError, TypeError, ValueError):
            continue
    ax.set_title(title)


def _draw_lines(ax, elem: Any) -> None:
    states = np.asarray(elem.states)
    masks = np.asarray(elem.mask)
    if states.ndim != 3:
        return
    for index, line in enumerate(states):
        line_mask = masks[index] if masks.ndim >= 2 else np.ones(len(line), dtype=bool)
        valid = np.asarray(line_mask) >= 0.3
        points = line[valid]
        if len(points) >= 2:
            ax.plot(points[:, 0], points[:, 1], color="#bab0ac", linewidth=0.65, alpha=0.8)


def _draw_elements(
    ax,
    elem: Any,
    name: str,
    color: str,
    actor_index: int,
    occluder_index: int,
    occluder_label: str = "occluder",
) -> None:
    states = np.asarray(elem.states)
    masks = np.asarray(elem.mask).reshape(-1)
    if states.ndim == 1:
        states = states[None, :]
    for index, state in enumerate(states):
        if index >= len(masks) or float(masks[index]) < 0.3 or len(state) < 5:
            continue
        is_actor = index == actor_index
        is_occluder = index == occluder_index
        linewidth = 2.8 if is_actor or is_occluder else 0.75
        face = color if is_actor or is_occluder else "none"
        alpha = 0.35 if is_actor or is_occluder else 1.0
        patch = MplPolygon(
            oriented_box_corners_from_state(np.asarray(state)),
            closed=True,
            facecolor=face,
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.add_patch(patch)
        if is_actor or is_occluder:
            role = "pedestrian" if is_actor else f"occluder\n{occluder_label}"
            ax.text(float(state[0]), float(state[1]), role, fontsize=7, ha="center", va="center")


def _state_xy(elem: Any, index: int) -> Optional[Tuple[float, float]]:
    states = np.asarray(elem.states)
    masks = np.asarray(elem.mask).reshape(-1)
    if index < 0 or index >= len(states) or index >= len(masks) or float(masks[index]) < 0.3:
        return None
    return float(states[index][0]), float(states[index][1])
