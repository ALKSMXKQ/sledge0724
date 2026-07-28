"""Export edited occluded-pedestrian scenes into diffusion cache format.

This utility converts accepted B1 edited raw scenes into the latent cache layout
consumed by SLEDGE latent diffusion training.
"""

from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path

import torch

from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.script.builders.model_builder import build_autoencoder_torch_module_wrapper
from sledge.semantic_control.io import load_raw_scene, load_gz_pickle, save_gz_pickle
from sledge.autoencoder.preprocessing.features.latent_feature import Latent


def encode_scene(scene_path: Path, model, cfg):
    raw, _ = load_raw_scene(scene_path)
    _, raster = sledge_raw_feature_processing(raw, cfg)
    tensor = raster.to_feature_tensor().data.unsqueeze(0)
    with torch.no_grad():
        latent = model.get_encoder()(tensor).mu
    return latent.squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--autoencoder-checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    cfg.autoencoder_checkpoint = args.autoencoder_checkpoint
    model = build_autoencoder_torch_module_wrapper(cfg)
    model.eval()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    for raw_path in sorted(input_root.glob("**/sledge_raw.gz")):
        rel = raw_path.parent.relative_to(input_root)
        out = output_root / rel
        out.mkdir(parents=True, exist_ok=True)

        latent = encode_scene(raw_path, model, cfg)
        save_gz_pickle(out / "rvae_latent", {"mu": latent, "log_var": latent * 0.0})

        label_path = raw_path.parent / "scenario_label.json"
        scenario_id = 4
        if label_path.exists():
            import json
            label = json.loads(label_path.read_text())
            scenario_id = int(label.get("scenario_id", label.get("map_id", 4)))
        save_gz_pickle(out / "scenario_type", {"id": scenario_id})


if __name__ == "__main__":
    main()
