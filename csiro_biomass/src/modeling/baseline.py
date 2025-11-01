"""Model factory for the CSIRO biomass competition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Sequence, List

import torch
import torch.nn as nn
from torchvision import models

try:  # Optional dependency for extended backbones
    import timm
except ImportError:  # pragma: no cover - handled at runtime
    timm = None

# This should be in a shared location, but defining here for simplicity
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

@dataclass
class ModelSpec:
    model_name: str = "resnet18"
    pretrained: bool = True
    dropout: float = 0.2
    num_outputs: int = 4
    metadata_dim: int = 0
    image_entropy: float = 7.1
    metadata_entropy: float = 2.3
    tabular_hidden_dims: Tuple[int, ...] = (128, 64)
    fusion_hidden_dims: Tuple[int, ...] = (512, 256)
    fusion_dropout: float = 0.25
    use_layernorm: bool = True
    fusion_type: str = "mlp"
    perceiver_latents: int = 32
    perceiver_layers: int = 3
    perceiver_heads: int = 4
    perceiver_dropout: float = 0.1


class TabularEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        *,
        use_layernorm: bool,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim
        if use_layernorm:
            layers.append(nn.LayerNorm(input_dim))
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = current_dim

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.network(metadata)


class FusionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        num_outputs: int,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_outputs * 2))
        self.network = nn.Sequential(*layers)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.network(fused)

class BiomassRegressor(nn.Module):
    """Final regressor model that predicts mean and log_var for uncertainty."""
    
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        metadata_dim: int,
        dropout: float,
        num_outputs: int,
        image_entropy: float,
        metadata_entropy: float,
        tabular_hidden_dims: Sequence[int],
        fusion_hidden_dims: Sequence[int],
        fusion_dropout: float,
        use_layernorm: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.metadata_dim = metadata_dim
        self.feature_dim = feature_dim
        self.tabular_output_dim = metadata_dim

        total_entropy = float(image_entropy + (metadata_entropy if metadata_dim > 0 else 0.0))
        if total_entropy <= 0:
            total_entropy = 1.0
        self.image_scale = float(image_entropy / total_entropy)
        self.metadata_scale = (
            float(metadata_entropy / total_entropy) if metadata_dim > 0 else 0.0
        )

        self.tabular_encoder = (
            TabularEncoder(metadata_dim, tabular_hidden_dims, dropout, use_layernorm=use_layernorm)
            if metadata_dim > 0
            else None
        )
        if self.tabular_encoder is not None:
            self.tabular_output_dim = self.tabular_encoder.output_dim
        head_input = feature_dim + (self.tabular_output_dim if metadata_dim > 0 else 0)
        self.fusion_gate = (
            nn.Sequential(
                nn.Linear(head_input, max(64, head_input // 4)),
                nn.GELU(),
                nn.Linear(max(64, head_input // 4), 1),
                nn.Sigmoid(),
            )
            if metadata_dim > 0
            else None
        )
        self.regressor = FusionHead(
            head_input,
            fusion_hidden_dims,
            fusion_dropout,
            num_outputs,
        )
        
    def forward(self, images: torch.Tensor, metadata: Optional[torch.Tensor] = None):
        features = self.backbone(images)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        features = features * self.image_scale

        if metadata is not None and self.metadata_dim > 0:
            if metadata.ndim == 1:
                metadata = metadata.unsqueeze(0)
            metadata = metadata * self.metadata_scale
            encoded = self.tabular_encoder(metadata)
            combined = torch.cat([features, encoded], dim=1)
            if self.fusion_gate is not None:
                gate = self.fusion_gate(combined)
                encoded = encoded * gate
                combined = torch.cat([features, encoded], dim=1)
            fused = combined
        else:
            fused = features

        outputs = self.regressor(fused)
        mean, log_var = torch.chunk(outputs, 2, dim=-1)
        return mean, log_var


class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_latents: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.input_norm = nn.LayerNorm(latent_dim)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_ff = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.self_blocks = nn.ModuleList([
            nn.ModuleDict(
                {
                    "ln1": nn.LayerNorm(latent_dim),
                    "attn": nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True),
                    "ln2": nn.LayerNorm(latent_dim),
                    "ff": nn.Sequential(
                        nn.Linear(latent_dim, latent_dim * 2),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(latent_dim * 2, latent_dim),
                    ),
                }
            )
            for _ in range(num_layers)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bsz = inputs.size(0)
        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)
        inputs = self.input_norm(inputs)
        cross, _ = self.cross_attn(latents, inputs, inputs)
        latents = latents + cross
        latents = latents + self.cross_ff(latents)
        for block in self.self_blocks:
            attn_out, _ = block["attn"](block["ln1"](latents), block["ln1"](latents), block["ln1"](latents))
            latents = latents + attn_out
            latents = latents + block["ff"](block["ln2"](latents))
        return latents.mean(dim=1)


class PerceiverBiomassRegressor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        metadata_dim: int,
        num_outputs: int,
        latent_dim: int,
        num_latents: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.metadata_dim = metadata_dim
        self.image_proj = nn.Linear(feature_dim, latent_dim)
        self.metadata_proj = (
            nn.Linear(metadata_dim, latent_dim)
            if metadata_dim > 0
            else None
        )
        self.encoder = PerceiverEncoder(
            latent_dim=latent_dim,
            num_latents=num_latents,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.head = nn.Linear(latent_dim, num_outputs * 2)

    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        img_token = self.image_proj(features).unsqueeze(1)
        tokens = [img_token]
        if self.metadata_proj is not None:
            if metadata.ndim == 1:
                metadata = metadata.unsqueeze(0)
            meta_token = self.metadata_proj(metadata).unsqueeze(1)
            tokens.append(meta_token)
        inputs = torch.cat(tokens, dim=1)
        latent = self.encoder(inputs)
        outputs = self.head(latent)
        mean, log_var = torch.chunk(outputs, 2, dim=-1)
        return mean, log_var

def _convert_backbone(backbone: nn.Module, name: str) -> Tuple[nn.Module, int]:
    if "resnet" in name or "convnext" in name:
        feature_dim = backbone.fc.in_features if "resnet" in name else backbone.classifier[-1].in_features
        setattr(backbone, "fc" if "resnet" in name else "classifier", nn.Identity())
        return backbone, feature_dim
    elif "efficientnet" in name:
        feature_dim = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Identity()
        return backbone, feature_dim
    raise ValueError(f"Unsupported backbone type for conversion: {name}")

def build_model(spec: ModelSpec, metadata_dim: int) -> nn.Module:
    name = spec.model_name.lower()
    norm_mean = IMAGENET_MEAN
    norm_std = IMAGENET_STD
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if spec.pretrained else None
        backbone, feat_dim = _convert_backbone(models.resnet18(weights=weights), name)
        if weights is not None:
            norm_mean = tuple(weights.meta.get("mean", IMAGENET_MEAN))
            norm_std = tuple(weights.meta.get("std", IMAGENET_STD))
    elif name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if spec.pretrained else None
        backbone, feat_dim = _convert_backbone(models.resnet34(weights=weights), name)
        if weights is not None:
            norm_mean = tuple(weights.meta.get("mean", IMAGENET_MEAN))
            norm_std = tuple(weights.meta.get("std", IMAGENET_STD))
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if spec.pretrained else None
        backbone, feat_dim = _convert_backbone(models.efficientnet_b0(weights=weights), name)
        if weights is not None:
            norm_mean = tuple(weights.meta.get("mean", IMAGENET_MEAN))
            norm_std = tuple(weights.meta.get("std", IMAGENET_STD))
    elif name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if spec.pretrained else None
        backbone, feat_dim = _convert_backbone(models.efficientnet_b3(weights=weights), name)
        if weights is not None:
            norm_mean = tuple(weights.meta.get("mean", IMAGENET_MEAN))
            norm_std = tuple(weights.meta.get("std", IMAGENET_STD))
    else:
        if timm is None:
            raise ValueError(f"Unsupported model: {spec.model_name}")
        backbone = timm.create_model(
            spec.model_name,
            pretrained=spec.pretrained,
            num_classes=0,
            global_pool="avg",
        )
        if hasattr(backbone, "num_features"):
            feat_dim = int(backbone.num_features)  # type: ignore[assignment]
        else:  # pragma: no cover - fallback for uncommon timm models
            with torch.no_grad():
                sample = torch.zeros(1, 3, 224, 224)
                feat_dim = int(backbone(sample).shape[-1])
        data_cfg = getattr(backbone, "pretrained_cfg", None)
        if data_cfg is not None:
            norm_mean = tuple(data_cfg.get("mean", IMAGENET_MEAN))
            norm_std = tuple(data_cfg.get("std", IMAGENET_STD))

    if spec.fusion_type.lower() == "perceiver":
        latent_dim = spec.fusion_hidden_dims[0] if spec.fusion_hidden_dims else 256
        model = PerceiverBiomassRegressor(
            backbone=backbone,
            feature_dim=feat_dim,
            metadata_dim=metadata_dim,
            num_outputs=spec.num_outputs,
            latent_dim=latent_dim,
            num_latents=spec.perceiver_latents,
            num_layers=spec.perceiver_layers,
            num_heads=spec.perceiver_heads,
            dropout=spec.perceiver_dropout,
        )
    else:
        model = BiomassRegressor(
            backbone,
            feat_dim,
            metadata_dim,
            spec.dropout,
            spec.num_outputs,
            spec.image_entropy,
            spec.metadata_entropy,
            spec.tabular_hidden_dims,
            spec.fusion_hidden_dims,
            spec.fusion_dropout,
            spec.use_layernorm,
        )
    model.normalization_mean = norm_mean  # type: ignore[attr-defined]
    model.normalization_std = norm_std  # type: ignore[attr-defined]
    return model


@dataclass
class AdvancedModelSpec:
    model_name: str = "resnet34"
    pretrained: bool = True
    dropout: float = 0.2
    num_outputs: int = 4
    metadata_dim: int = 0
    image_entropy: float = 7.1
    metadata_entropy: float = 2.3
    tabular_hidden_dims: Tuple[int, ...] = (256, 128)
    fusion_hidden_dims: Tuple[int, ...] = ()
    fusion_dropout: float = 0.3
    use_layernorm: bool = True
    fusion_hidden_dim: int = 256
    use_information_fusion: bool = True


def build_advanced_model(spec: AdvancedModelSpec) -> nn.Module:
    """Construct an extended biomass regressor while reusing the baseline core."""
    if spec.fusion_hidden_dims:
        fusion_hidden_dims = spec.fusion_hidden_dims
    elif spec.use_information_fusion:
        secondary = max(spec.fusion_hidden_dim // 2, 1)
        fusion_hidden_dims = (spec.fusion_hidden_dim, secondary)
    else:
        fusion_hidden_dims = (spec.fusion_hidden_dim,)

    base_spec = ModelSpec(
        model_name=spec.model_name,
        pretrained=spec.pretrained,
        dropout=spec.dropout,
        num_outputs=spec.num_outputs,
        metadata_dim=spec.metadata_dim,
        image_entropy=spec.image_entropy,
        metadata_entropy=spec.metadata_entropy,
        tabular_hidden_dims=spec.tabular_hidden_dims,
        fusion_hidden_dims=fusion_hidden_dims,
        fusion_dropout=spec.fusion_dropout,
        use_layernorm=spec.use_layernorm,
    )
    return build_model(base_spec, metadata_dim=spec.metadata_dim)

def get_normalization_stats() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    return IMAGENET_MEAN, IMAGENET_STD
