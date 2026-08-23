"""Build the deterministic, ONNX-friendly RVAE inference graph."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Mapping, Tuple

import torch
from torch import Tensor, nn
import yaml

from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.modeling.models.rvae.rvae_decoder import patchify
from sledge.autoencoder.modeling.models.rvae.rvae_model import RVAEModel

from .contract import OUTPUT_NAMES


@contextmanager
def _disable_torchvision_weight_download(model_name: str) -> Iterator[None]:
    """Construct the backbone without downloading weights; the checkpoint replaces them."""
    import torchvision.models

    original = getattr(torchvision.models, model_name)

    def factory(*args, **kwargs):
        kwargs["weights"] = None
        return original(*args, **kwargs)

    setattr(torchvision.models, model_name, factory)
    try:
        yield
    finally:
        setattr(torchvision.models, model_name, original)


@contextmanager
def deployment_inference_mode() -> Iterator[None]:
    """Use the same portable attention path traced into the ONNX graph."""
    fastpath_was_enabled = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.inference_mode():
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath_was_enabled)


def load_config(path: Path) -> RVAEConfig:
    with path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    return RVAEConfig(**values)


def load_rvae(checkpoint: Path, config_path: Path, device: torch.device) -> RVAEModel:
    config = load_config(config_path)
    with _disable_torchvision_weight_download(config.model_name):
        model = RVAEModel(config)

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    raw_state = checkpoint_data.get("state_dict", checkpoint_data)
    state = {
        key.removeprefix("model."): value
        for key, value in raw_state.items()
        if key.startswith("model.")
    }
    if not state:
        state = dict(raw_state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval().to(device)
    return model


class DeterministicRVAE(nn.Module):
    """RVAE graph using latent mean and export-friendly functional box heads."""

    def __init__(self, model: RVAEModel):
        super().__init__()
        self.encoder = model.get_encoder()
        self.decoder = model.get_decoder()
        self.config = model._config
        # Input and batch are fixed for the vehicle package. Precomputing these
        # tensors removes ScatterND/CumSum from ONNX, which is safer for the
        # TensorRT 8.6.1 parser and avoids needless runtime shape work.
        config = self.config
        device = next(model.parameters()).device
        static_queries = sum(config.num_queries_list[:1])
        dynamic_queries = sum(config.num_queries_list[1:])
        static_patches = config.num_patches
        dynamic_patches = config.num_patches
        zero_static = torch.zeros(static_queries, static_patches, device=device)
        zero_dynamic = torch.zeros(dynamic_queries, dynamic_patches, device=device)
        blocked_static = torch.full((static_queries, dynamic_patches), float("-inf"), device=device)
        blocked_dynamic = torch.full((dynamic_queries, static_patches), float("-inf"), device=device)
        self.register_buffer(
            "fixed_decoder_mask",
            torch.cat(
                (
                    torch.cat((zero_static, blocked_static), dim=1),
                    torch.cat((blocked_dynamic, zero_dynamic), dim=1),
                ),
                dim=0,
            ),
            persistent=False,
        )
        dummy = torch.zeros(
            1,
            config.latent_channel // 2,
            config.latent_frame[0],
            config.latent_frame[1],
            device=device,
        )
        with torch.no_grad():
            position = self.decoder._position_encoding(dummy).flatten(-2).permute(2, 0, 1)
        self.register_buffer("fixed_position_encoding", position, persistent=False)

    @staticmethod
    def _line_head(head: nn.Module, queries: Tensor) -> Tuple[Tensor, Tensor]:
        batch, count = queries.shape[:2]
        states = head._ffn_states(queries).tanh()
        states = states.reshape(batch, count, head._num_line_poses, 2) * head._frame_transform
        logits = head._ffn_mask(queries).squeeze(-1)
        return states, logits

    @staticmethod
    def _agent_head(head: nn.Module, queries: Tensor, max_velocity: float) -> Tuple[Tensor, Tensor]:
        raw = head._ffn_states(queries)
        point = raw[..., 0:2].tanh() * head._frame_transform
        heading = raw[..., 2:3].tanh() * torch.pi
        dimensions = raw[..., 3:5]
        velocity = raw[..., 5:6].sigmoid() * max_velocity
        return torch.cat((point, heading, dimensions, velocity), dim=-1), head._ffn_mask(queries).squeeze(-1)

    @staticmethod
    def _static_head(head: nn.Module, queries: Tensor) -> Tuple[Tensor, Tensor]:
        raw = head._ffn_states(queries)
        point = raw[..., 0:2].tanh() * head._frame_transform
        heading = raw[..., 2:3].tanh() * torch.pi
        dimensions = raw[..., 3:5]
        return torch.cat((point, heading, dimensions), dim=-1), head._ffn_mask(queries).squeeze(-1)

    def forward(self, raster: Tensor) -> Tuple[Tensor, ...]:
        # Training samples z = mu + eps*std. Deployment deliberately uses z = mu
        # so repeated inference and cross-backend comparisons are deterministic.
        latent = self.encoder(raster).mu
        decoder = self.decoder
        config = self.config
        batch = latent.shape[0]

        static_latent, dynamic_latent = torch.chunk(latent, 2, dim=1)
        static_patches = patchify(static_latent, config.patch_size)
        dynamic_patches = patchify(dynamic_latent, config.patch_size)

        type_embed = decoder._type_encoding.weight[None, ...].repeat(batch, 1, 1)
        type_embed = type_embed.repeat_interleave(config.num_patches, 1).permute(1, 0, 2)
        pos_embed = self.fixed_position_encoding.repeat(2, batch, 1) + type_embed

        static_patches = static_patches.flatten(-2).permute(2, 0, 1)
        dynamic_patches = dynamic_patches.flatten(-2).permute(2, 0, 1)
        projected = decoder._patch_projection(torch.cat((static_patches, dynamic_patches), dim=0))
        queries = decoder._query_embedding.weight[:, None].repeat(1, batch, 1)
        hidden = decoder._transformer(
            src=projected,
            query_embed=queries,
            pos_embed=pos_embed,
            memory_mask=self.fixed_decoder_mask,
        )[0].permute(1, 0, 2)

        line, vehicle, pedestrian, static, green, red, ego = hidden.split(config.num_queries_list, dim=1)
        lines_states, lines_logits = self._line_head(decoder._line_head, line)
        vehicles_states, vehicles_logits = self._agent_head(
            decoder._vehicle_head, vehicle, config.vehicle_max_velocity
        )
        pedestrians_states, pedestrians_logits = self._agent_head(
            decoder._pedestrian_head, pedestrian, config.pedestrian_max_velocity
        )
        static_states, static_logits = self._static_head(decoder._static_object_head, static)
        green_states, green_logits = self._line_head(decoder._green_line_head, green)
        red_states, red_logits = self._line_head(decoder._red_line_head, red)
        ego_states = decoder._ego_head._ffn_states(ego).squeeze(-1)
        ego_mask = torch.ones_like(ego_states)
        return (
            lines_states,
            lines_logits,
            vehicles_states,
            vehicles_logits,
            pedestrians_states,
            pedestrians_logits,
            static_states,
            static_logits,
            green_states,
            green_logits,
            red_states,
            red_logits,
            ego_states,
            ego_mask,
        )


def as_numpy_dict(outputs: Tuple[Tensor, ...]) -> Dict[str, object]:
    return {
        name: value.detach().cpu().numpy()
        for name, value in zip(OUTPUT_NAMES, outputs)
    }
