"""Experiment runner for the CSIRO Image2Biomass competition."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold, GroupKFold
from PIL import Image
import pywt

from .config.experiment import (
    AugmentationConfig,
    BackboneConfig,
    CurriculumConfig,
    ExperimentConfig,
    FusionConfig,
    OptimizerConfig,
    RegularizationConfig,
    SnapshotConfig,
)
from .training.lightning_module import BiomassLightningModule
from .utils.metrics import compute_metrics
from .data.dataset import BiomassDataset, create_inference_loader
from .data.sampler import FractalCurriculumSampler
from .modeling.baseline import (
    AdvancedModelSpec,
    ModelSpec,
    build_advanced_model,
    build_model,
    get_normalization_stats,
)
from .modeling.loss import GaussianNLLLoss
from .postprocess.constraints import BiomassConstraintProcessor, ConstraintConfig
from .modeling.postprocessing import AdvancedBiomassConstraintProcessor
from .features.fractal import compute_fractal_dimension
from .utils.leaderboard import compare_leaderboard, print_leaderboard_summary

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image_column(df: pd.DataFrame, override: Optional[str]) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"Image column '{override}' not in dataframe")
        return override

    candidates = [col for col in df.columns if "image" in col.lower()]
    if "image_path" in df.columns:
        return "image_path"
    if candidates:
        return candidates[0]
    raise ValueError("Unable to infer image column. Provide --image-column in config.")


def decode_category_series(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
    decoder = {float(value): key for key, value in mapping.items()}
    return series.map(
        lambda code: decoder.get(float(code), "unknown")
        if pd.notnull(code)
        else "unknown"
    )


def normalise_metadata_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    metadata_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    train_df = train_df.copy()
    val_df = val_df.copy()
    if metadata_columns:
        for column in metadata_columns:
            train_mean = train_df[column].mean()
            if not np.isfinite(train_mean):
                train_mean = 0.0
            train_df[column] = train_df[column].fillna(train_mean).astype(np.float32)
            val_df[column] = val_df[column].fillna(train_mean).astype(np.float32)

        metadata_array = train_df[metadata_columns].to_numpy(dtype=np.float32)
        metadata_mean = metadata_array.mean(axis=0)
        metadata_std = metadata_array.std(axis=0)
        metadata_std = np.where(metadata_std == 0, 1.0, metadata_std)
    else:
        metadata_mean = np.array([], dtype=np.float32)
        metadata_std = np.array([], dtype=np.float32)
    return train_df, val_df, metadata_mean, metadata_std


def generate_inference_metadata(
    df: pd.DataFrame,
    *,
    image_column: str,
    image_dir: Path,
    metadata_columns: Sequence[str],
    use_metadata: bool,
    fractal_csv: Optional[Path],
    category_maps: Dict[str, Dict[str, int]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if use_metadata:
        test_fractal_features = _load_precomputed_fractals(
            fractal_csv,
            image_column,
        )
        test_meta_context, _ = _extract_metadata_features(
            df,
            image_column=image_column,
            image_dir=image_dir,
            precomputed_fractals=test_fractal_features,
            category_maps=category_maps,
        )
    else:
        test_meta_context = (
            df[[image_column]].drop_duplicates().reset_index(drop=True)
        )

    for column in metadata_columns:
        if column not in test_meta_context.columns:
            test_meta_context[column] = 0.0

    if "species_code" in test_meta_context.columns and "species" not in test_meta_context.columns:
        species_mapping = category_maps.get("species", {})
        if species_mapping:
            test_meta_context["species"] = decode_category_series(
                test_meta_context["species_code"],
                species_mapping,
            )
    elif "Species" in df.columns and "species" not in test_meta_context.columns:
        species_context = (
            df[[image_column, "Species"]]
            .drop_duplicates()
            .rename(columns={"Species": "species"})
        )
        species_context["species"] = (
            species_context["species"]
            .fillna("unknown")
            .astype(str)
            .str.lower()
        )
        test_meta_context = test_meta_context.merge(
            species_context,
            on=image_column,
            how="left",
        )

    if "species" not in test_meta_context.columns:
        test_meta_context["species"] = "unknown"
    test_meta_context["species"] = (
        test_meta_context["species"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
    )

    if metadata_columns:
        test_meta_numeric = test_meta_context[[image_column, *metadata_columns]].copy()
    else:
        test_meta_numeric = test_meta_context[[image_column]].copy()
    test_meta_numeric = test_meta_numeric.fillna(0.0)
    return test_meta_context, test_meta_numeric


def _filter_missing_images(
    df: pd.DataFrame,
    *,
    image_column: str,
    image_dir: Path,
    context: str,
) -> pd.DataFrame:
    def resolve_path(rel: str) -> Path:
        rel_path = Path(rel)
        if rel_path.is_absolute():
            return rel_path
        if rel_path.parts and rel_path.parts[0] in {"train", "test"}:
            return image_dir / rel_path
        return image_dir / rel_path

    paths = df[image_column].astype(str).map(resolve_path)
    mask = paths.apply(lambda p: p.exists())
    missing = int((~mask).sum())
    if missing:
        LOGGER.warning("Dropping %d samples with missing images for %s", missing, context)
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metadata feature engineering
# ---------------------------------------------------------------------------

def _encode_categories(
    meta: pd.DataFrame,
    column: str,
    key: str,
    category_maps: Dict[str, Dict[str, int]],
) -> pd.Series:
    values = meta.get(column, pd.Series(dtype=object)).fillna("unknown").astype(str).str.lower()
    mapping = category_maps.get(key)
    if mapping is None:
        mapping = {name: idx for idx, name in enumerate(sorted(values.unique()))}
        category_maps[key] = mapping
    return values.map(lambda v: mapping.get(v, -1)).astype(float)


def _compute_complexity_features(
    meta: pd.DataFrame,
    image_dir: Path,
    image_column: str,
) -> pd.DataFrame:
    fractal_values: List[float] = []
    wavelet_ratios: List[float] = []

    for image_name in meta[image_column].astype(str):
        image_path = image_dir / image_name
        try:
            with Image.open(image_path) as img:
                gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        except FileNotFoundError:
            LOGGER.warning("Missing image while computing fractal features: %s", image_path)
            fractal_values.append(np.nan)
            wavelet_ratios.append(np.nan)
            continue
        except OSError:
            LOGGER.warning("Failed to read image for fractal features: %s", image_path)
            fractal_values.append(np.nan)
            wavelet_ratios.append(np.nan)
            continue

        fractal_dim = compute_fractal_dimension(gray)
        try:
            cA, (cH, cV, cD) = pywt.dwt2(gray, "haar")
            low_energy = float(np.sum(np.abs(cA)))
            high_energy = float(
                np.sum(np.abs(cH)) + np.sum(np.abs(cV)) + np.sum(np.abs(cD))
            )
            ratio = low_energy / (low_energy + high_energy + 1e-8)
        except ValueError:
            ratio = 0.5

        fractal_values.append(fractal_dim)
        wavelet_ratios.append(ratio)

    meta = meta.copy()
    meta["fractal_dimension"] = fractal_values
    meta["wavelet_ratio"] = wavelet_ratios
    return meta


def _load_precomputed_fractals(
    fractal_path: Optional[Path], image_column: str
) -> Optional[pd.DataFrame]:
    if fractal_path is None:
        return None
    path = Path(fractal_path)
    if not path.exists():
        LOGGER.warning("Precomputed fractal CSV not found at %s", path)
        return None
    try:
        features = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load precomputed fractal features: %s", exc)
        return None

    if image_column not in features.columns and "image_path" in features.columns:
        features = features.rename(columns={"image_path": image_column})
    features[image_column] = features[image_column].astype(str)

    if "wavelet_ratio" not in features.columns:
        if "wavelet_energy_ratio" in features.columns:
            features = features.rename(columns={"wavelet_energy_ratio": "wavelet_ratio"})
        else:
            features["wavelet_ratio"] = 0.5

    required = {image_column, "fractal_dimension", "wavelet_ratio"}
    if not required.issubset(features.columns):
        LOGGER.warning(
            "Precomputed fractal CSV missing columns %s; ignoring file.",
            sorted(required - set(features.columns)),
        )
        return None

    return features[list(required)]


def _extract_metadata_features(
    df: pd.DataFrame,
    image_column: str,
    image_dir: Optional[Path] = None,
    precomputed_fractals: Optional[pd.DataFrame] = None,
    category_maps: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    meta = df.drop_duplicates(subset=[image_column]).copy()
    meta[image_column] = meta[image_column].astype(str)
    category_maps = category_maps or {}

    if precomputed_fractals is not None:
        meta = meta.merge(precomputed_fractals, on=image_column, how="left")
    elif image_dir is not None:
        meta = _compute_complexity_features(meta, image_dir, image_column)
    else:
        meta["fractal_dimension"] = np.nan
        meta["wavelet_ratio"] = np.nan

    # Ordinal date
    sampling_series = meta.get("Sampling_Date")
    if sampling_series is None:
        sampling_series = pd.Series([pd.NaT] * len(meta))
    meta["Sampling_Date_ordinal"] = pd.to_datetime(
        sampling_series, errors="coerce"
    ).map(lambda x: x.toordinal() if pd.notnull(x) else 0)

    # Numerical features
    ndvi_series = meta.get("Pre_GSHH_NDVI")
    if ndvi_series is None:
        ndvi_series = pd.Series([0.0] * len(meta))
    meta["NDVI"] = ndvi_series.fillna(0.0)

    height_series = meta.get("Height_Ave_cm")
    if height_series is None:
        height_series = pd.Series([0.0] * len(meta))
    meta["Height_cm"] = height_series.fillna(0.0)

    # Categorical encodings
    meta["species_code"] = _encode_categories(meta, "Species", "species", category_maps)
    meta["state_code"] = _encode_categories(meta, "State", "state", category_maps)

    feature_columns = [
        "Sampling_Date_ordinal",
        "NDVI",
        "Height_cm",
        "species_code",
        "state_code",
        "fractal_dimension",
        "wavelet_ratio",
    ]
    meta_features = meta[[image_column, *feature_columns]].copy()
    return meta_features, category_maps


def estimate_huber_delta(values: np.ndarray, quantile: float = 0.9) -> float:
    if values.size == 0:
        return 1.0
    diffs = np.abs(values - np.median(values, axis=0, keepdims=True))
    flattened = diffs.reshape(-1)
    delta = float(np.quantile(flattened, quantile))
    if not np.isfinite(delta) or delta <= 0:
        return 1.0
    return delta


def estimate_entropy(values: np.ndarray, bins: int = 32) -> float:
    if values.size == 0:
        return 0.0
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0
    counts, _ = np.histogram(finite_values, bins=bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts.astype(np.float64) / total
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def prepare_training_dataframe(
    train_df: pd.DataFrame,
    image_column: str,
    target_name_column: str,
    target_value_column: str,
    *,
    image_dir: Optional[Path] = None,
    precomputed_fractals: Optional[Path] = None,
    include_metadata: bool = True,
    category_maps: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[
    pd.DataFrame,
    List[str],
    List[str],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Dict[str, Dict[str, int]],
    pd.DataFrame,
]:
    target_names = sorted(train_df[target_name_column].unique())
    pivot = (
        train_df.pivot_table(
            index=image_column,
            columns=target_name_column,
            values=target_value_column,
        )
        .reindex(columns=target_names)
        .reset_index()
    )
    pivot[image_column] = pivot[image_column].astype(str)
    if pivot[target_names].isna().any().any():
        LOGGER.warning("NaN targets detected after pivot – filling with zeros.")
        pivot[target_names] = pivot[target_names].fillna(0.0)

    metadata_mean: Optional[np.ndarray] = None
    metadata_std: Optional[np.ndarray] = None

    if include_metadata:
        fractal_features_df = _load_precomputed_fractals(
            precomputed_fractals, image_column
        )

        metadata_features, category_maps = _extract_metadata_features(
            train_df,
            image_column=image_column,
            image_dir=image_dir,
            precomputed_fractals=fractal_features_df,
            category_maps=category_maps,
        )
        feature_columns = [col for col in metadata_features.columns if col != image_column]
        numeric_cols = metadata_features.select_dtypes(include=[np.number]).columns
        metadata_features[numeric_cols] = metadata_features[numeric_cols].fillna(0.0)
        pivot = pivot.merge(metadata_features, on=image_column, how="left")

        if feature_columns:
            pivot[feature_columns] = pivot[feature_columns].fillna(
                pivot[feature_columns].mean()
            ).fillna(0.0)
            metadata_array = pivot[feature_columns].to_numpy(dtype=np.float32)
            metadata_mean = metadata_array.mean(axis=0)
            metadata_std = metadata_array.std(axis=0)
            metadata_std = np.where(metadata_std == 0, 1.0, metadata_std)
    else:
        feature_columns = []
        metadata_features = pivot[[image_column]].copy()
        category_maps = category_maps or {}

    return (
        pivot,
        target_names,
        feature_columns,
        metadata_mean,
        metadata_std,
        category_maps,
        metadata_features,
    )


def _parse_int_list(value: str, *, default: Optional[Sequence[int]] = None) -> List[int]:
    if not value:
        return list(default or [])
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(f"Invalid integer list: {value}") from exc


# ---------------------------------------------------------------------------
# Augmentation builders
# ---------------------------------------------------------------------------

def build_transforms(
    cfg: AugmentationConfig,
    *,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    is_tta: bool = False,
) -> Tuple[Callable, Callable]:
    if mean is None or std is None:
        mean_default, std_default = get_normalization_stats()
        mean = mean or mean_default
        std = std or std_default

    policy = (cfg.policy or "standard").lower()
    aug_list: List[A.BasicTransform] = []
    if policy == "randaugment" and hasattr(A, "RandAugment"):
        aug_list.append(
            A.RandAugment(
                num_ops=cfg.randaugment_num_ops,
                magnitude=cfg.randaugment_magnitude,
            )
        )
    elif policy == "trivialaugment" and hasattr(A, "TrivialAugmentWide"):
        aug_list.append(
            A.TrivialAugmentWide(
                num_magnitude_bins=cfg.trivial_magnitude_bins,
            )
        )

    aug_list.extend(
        [
            A.RandomResizedCrop(
                size=(cfg.image_size, cfg.image_size),
                scale=tuple(cfg.random_resized_crop["scale"]),
                ratio=tuple(cfg.random_resized_crop["ratio"]),
            ),
            A.HorizontalFlip(p=cfg.horizontal_flip),
            A.VerticalFlip(p=cfg.vertical_flip),
            A.Affine(
                scale=tuple(cfg.affine_scale),
                translate_percent=(0.0, cfg.affine_translate),
                rotate=(-cfg.affine_rotate, cfg.affine_rotate),
                shear=(-8, 8),
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_contrast,
                contrast_limit=cfg.brightness_contrast,
                p=0.8,
            ),
            A.HueSaturationValue(
                hue_shift_limit=cfg.hue_shift,
                sat_shift_limit=cfg.hue_shift * 1.25,
                val_shift_limit=cfg.hue_shift * 0.75,
                p=0.7,
            ),
            A.GaussianBlur(blur_limit=(3, 9), p=cfg.blur_probability),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    train_aug = A.Compose(aug_list)
    val_aug = A.Compose(
        [
            A.Resize(height=int(cfg.image_size * 1.05), width=int(cfg.image_size * 1.05)),
            A.CenterCrop(height=cfg.image_size, width=cfg.image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    if is_tta:
        tta_transforms = A.Compose([
            A.Resize(height=int(cfg.image_size * 1.05), width=int(cfg.image_size * 1.05)),
            A.CenterCrop(height=cfg.image_size, width=cfg.image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
        return AlbumentationsWrapper(train_aug), AlbumentationsWrapper(val_aug), AlbumentationsWrapper(tta_transforms)

    return AlbumentationsWrapper(train_aug), AlbumentationsWrapper(val_aug)


class AlbumentationsWrapper:
    def __init__(self, transform: A.Compose) -> None:
        self.transform = transform

    def __call__(self, image):
        image_np = np.array(image)
        return self.transform(image=image_np)["image"]


# ---------------------------------------------------------------------------
# Training / evaluation primitives
# ---------------------------------------------------------------------------


def run_prediction_loop(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    return_targets: bool = False,
    use_tta: bool = False,
) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor]]:
    model.eval()
    identifiers: List[str] = []
    predictions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    with torch.no_grad():
        for images, target_tensor, metadata_tensor, ids in loader:
            images = images.to(device, non_blocking=True)
            metadata_tensor = metadata_tensor.to(device, non_blocking=True)
            if use_tta:
                # images is a list of augmented tensors
                tta_preds = []
                for img_aug in images:
                    img_aug = img_aug.to(device, non_blocking=True)
                    outputs = model(img_aug, metadata_tensor)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    tta_preds.append(outputs.cpu())
                predictions.append(torch.stack(tta_preds).mean(dim=0))
            else:
                outputs = model(images, metadata_tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                predictions.append(outputs.cpu())

            if return_targets and target_tensor.numel():
                targets.append(target_tensor.cpu())

            identifiers.extend(list(ids))

    stacked_preds = torch.cat(predictions, dim=0) if predictions else torch.empty(0)
    stacked_targets = (
        torch.cat(targets, dim=0) if targets and return_targets else None
    )
    return identifiers, stacked_preds, stacked_targets


def save_submission(
    predictions: Optional[torch.Tensor],
    test_df: pd.DataFrame,
    target_names: Sequence[str],
    image_column: str,
    target_name_column: str,
    id_column: str,
    run_dir: Path,
    sample_submission: Optional[Union[pd.DataFrame, Path, str]] = None,
    *,
    target_value_column: str = "target",
    repaired_df: Optional[pd.DataFrame] = None,
) -> Path:
    """Persist a submission file that mirrors the competition schema."""
    if predictions is None and repaired_df is None:
        raise ValueError("Either predictions or repaired_df must be provided.")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if repaired_df is not None:
        predictions_df = repaired_df.copy()
    else:
        preds_np = predictions.detach().cpu().numpy()
        predictions_df = pd.DataFrame(preds_np, columns=list(target_names))
        unique_images = (
            test_df[[image_column]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        if len(unique_images) != len(predictions_df):
            LOGGER.warning(
                "Mismatch between predictions (%d) and unique images (%d); "
                "falling back to head alignment.",
                len(predictions_df),
                len(unique_images),
            )
        predictions_df[image_column] = unique_images[image_column].astype(str).values[: len(predictions_df)]

    if image_column not in predictions_df.columns:
        raise ValueError(f"{image_column} must be present in predictions for submission.")

    melted = predictions_df.melt(
        id_vars=[image_column],
        value_vars=list(target_names),
        var_name=target_name_column,
        value_name=target_value_column,
    )

    if target_name_column not in test_df.columns:
        LOGGER.warning(
            "Target name column '%s' missing from test dataframe; assuming wide submission.",
            target_name_column,
        )
        id_lookup = test_df[[id_column, image_column]].drop_duplicates()
        submission_df = id_lookup.merge(
            melted,
            on=image_column,
            how="left",
        )
    else:
        id_lookup = (
            test_df[[id_column, image_column, target_name_column]]
            .drop_duplicates()
        )
        submission_df = id_lookup.merge(
            melted,
            on=[image_column, target_name_column],
            how="left",
        )

    template_df: Optional[pd.DataFrame] = None
    if sample_submission is not None:
        if isinstance(sample_submission, (str, Path)):
            template_df = pd.read_csv(sample_submission)
        else:
            template_df = sample_submission.copy()

    if template_df is not None:
        expected_cols = list(template_df.columns)
        for column in expected_cols:
            if column not in submission_df.columns:
                submission_df[column] = template_df[column]
        submission_df = submission_df[expected_cols]
    else:
        base_cols = [id_column]
        if target_name_column in submission_df.columns:
            base_cols.append(target_name_column)
        base_cols.append(target_value_column)
        submission_df = submission_df[base_cols]

    submission_path = run_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    return submission_path


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> Dict[str, object]:
    set_seed(config.seed)
    pl.seed_everything(config.seed, workers=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = config.output_dir / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(config.train_csv)
    image_column = resolve_image_column(train_raw, config.image_column)

    (
        train_df,
        target_names,
        metadata_columns,
        metadata_mean,
        metadata_std,
        category_maps,
        metadata_features_train,
    ) = prepare_training_dataframe(
        train_raw,
        image_column=image_column,
        target_name_column=config.target_name_column,
        target_value_column=config.target_value_column,
        image_dir=config.image_dir,
        precomputed_fractals=config.fractal_csv,
        include_metadata=config.use_metadata,
    )

    if "species_code" in metadata_features_train.columns and "species" not in metadata_features_train.columns:
        species_mapping = category_maps.get("species", {})
        if species_mapping:
            metadata_features_train["species"] = (
                decode_category_series(metadata_features_train["species_code"], species_mapping)
            )
    if "state_code" in metadata_features_train.columns and "state" not in metadata_features_train.columns:
        state_mapping = category_maps.get("state", {})
        if state_mapping:
            metadata_features_train["state"] = (
                decode_category_series(metadata_features_train["state_code"], state_mapping)
            )

    if "species" in metadata_features_train.columns:
        metadata_features_train["species"] = (
            metadata_features_train["species"]
            .fillna("unknown")
            .astype(str)
            .str.lower()
        )

    metadata_features_train.to_csv(run_dir / "train_metadata.csv", index=False)
    LOGGER.info("Saved training metadata to %s", run_dir / "train_metadata.csv")

    # Train / validation split
    val_size = max(1, int(len(train_df) * config.val_fraction))
    val_df = train_df.sample(n=val_size, random_state=config.seed).reset_index(drop=True)
    train_df_split = train_df.drop(val_df.index).reset_index(drop=True)

    if config.max_train_samples is not None:
        train_df_split = train_df_split.sample(
            n=min(config.max_train_samples, len(train_df_split)),
            random_state=config.seed,
        ).reset_index(drop=True)
    if config.max_val_samples is not None:
        val_df = val_df.sample(
            n=min(config.max_val_samples, len(val_df)),
            random_state=config.seed,
        ).reset_index(drop=True)

    train_df_split = _filter_missing_images(
        train_df_split,
        image_column=image_column,
        image_dir=Path(config.image_dir),
        context="training",
    )
    val_df = _filter_missing_images(
        val_df,
        image_column=image_column,
        image_dir=Path(config.image_dir),
        context="validation",
    )

    train_df_split, val_df, metadata_mean, metadata_std = normalise_metadata_split(
        train_df_split,
        val_df,
        metadata_columns,
    )

    metadata_entropy = 0.0
    if metadata_columns:
        entropy_values = []
        for column in metadata_columns:
            column_entropy = estimate_entropy(
                train_df_split[column].to_numpy(dtype=np.float32)
            )
            if np.isfinite(column_entropy):
                entropy_values.append(column_entropy)
        if entropy_values:
            metadata_entropy = float(np.mean(entropy_values))
    LOGGER.info("Estimated metadata entropy: %.3f", metadata_entropy)

    model_spec = ModelSpec(
        model_name=config.backbone.name,
        pretrained=config.backbone.pretrained,
        dropout=config.backbone.dropout,
        num_outputs=len(target_names),
        metadata_dim=len(metadata_columns),
        metadata_entropy=max(metadata_entropy, 1e-3),
        tabular_hidden_dims=tuple(config.fusion.tabular_hidden_dims),
        fusion_hidden_dims=tuple(config.fusion.fusion_hidden_dims),
        fusion_dropout=config.fusion.fusion_dropout,
        use_layernorm=config.fusion.use_layernorm,
        fusion_type=config.fusion.fusion_type,
        perceiver_latents=config.fusion.perceiver_latents,
        perceiver_layers=config.fusion.perceiver_layers,
        perceiver_heads=config.fusion.perceiver_heads,
        perceiver_dropout=config.fusion.perceiver_dropout,
    )
    model = build_model(model_spec, metadata_dim=len(metadata_columns))
    requested_device = str(config.device).lower()
    accelerator = "cpu"
    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(config.device)
            accelerator = "gpu"
        else:
            LOGGER.warning("CUDA requested but not available; falling back to CPU.")
            device = torch.device("cpu")
    elif requested_device.startswith("mps"):
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
            accelerator = "mps"
        else:
            LOGGER.warning("MPS requested but not available; falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    model.to(device)

    norm_mean = getattr(model, "normalization_mean", None)
    norm_std = getattr(model, "normalization_std", None)
    train_transforms, val_transforms = build_transforms(
        config.augmentation,
        mean=norm_mean,
        std=norm_std,
    )

    train_dataset = BiomassDataset(
        train_df_split,
        image_dir=config.image_dir,
        image_column=image_column,
        transforms=train_transforms,
        target_columns=target_names,
        metadata_columns=metadata_columns,
        metadata_mean=metadata_mean,
        metadata_std=metadata_std,
    )
    val_dataset = BiomassDataset(
        val_df,
        image_dir=config.image_dir,
        image_column=image_column,
        transforms=val_transforms,
        target_columns=target_names,
        metadata_columns=metadata_columns,
        metadata_mean=metadata_mean,
        metadata_std=metadata_std,
    )

    curriculum_sampler = None
    if config.curriculum.enable:
        stage_dicts = [
            {
                "fractal_range": stage.fractal_range,
                "biomass_range": stage.biomass_range,
                "epochs": stage.epochs,
            }
            for stage in config.curriculum.stages
        ]
        biomass_column = config.curriculum.target_column
        if biomass_column and biomass_column not in train_df_split.columns:
            LOGGER.warning(
                "Curriculum target column '%s' missing; biomass filtering disabled.",
                biomass_column,
            )
            biomass_column = None
        curriculum_sampler = FractalCurriculumSampler(
            train_df_split,
            batch_size=config.batch_size,
            shuffle=True,
            stages=stage_dicts,
            total_epochs=config.epochs,
            biomass_column=biomass_column,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=curriculum_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    target_array = train_df_split[target_names].to_numpy(dtype=np.float32)
    huber_delta = estimate_huber_delta(target_array)
    LOGGER.info("Using SmoothL1Loss with beta=%.3f", huber_delta)

    lightning_module = BiomassLightningModule(
        model=model,
        optimizer_cfg=config.optimizer,
        target_names=target_names,
        huber_beta=huber_delta,
        train_sampler=curriculum_sampler,
        mixup_alpha=config.regularization.mixup_alpha,
        mixup_prob=config.regularization.mixup_prob,
        save_oof=config.save_oof,
    )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_count = max(int(config.snapshots.num_snapshots), 0)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch{epoch:02d}-rmse{val_rmse:.4f}",
        monitor="val_rmse",
        mode="min",
        save_top_k=max(snapshot_count, 1),
        save_last=True,
    )
    callbacks = [checkpoint_callback]

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        default_root_dir=str(run_dir),
        accelerator=accelerator,
        devices=1,
        logger=False,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_model_summary=False,
        log_every_n_steps=25,
        num_sanity_val_steps=0,
    )

    trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    snapshot_paths: List[str] = []
    if snapshot_count > 0 and checkpoint_callback.best_k_models:
        ordered = list(checkpoint_callback.best_k_models.keys())
        snapshot_paths = ordered[:snapshot_count]

    if snapshot_paths:
        LOGGER.info("Averaging %d checkpoints for snapshot ensemble", len(snapshot_paths))
        averaged_states = average_checkpoints(snapshot_paths)
        lightning_module.load_state_dict(averaged_states["state_dict"])
        if (
            "ema_state_dict" in averaged_states
            and lightning_module.use_ema
            and lightning_module.ema_model is not None
        ):
            lightning_module.ema_model.load_state_dict(averaged_states["ema_state_dict"])
            lightning_module.model.load_state_dict(lightning_module.ema_model.module.state_dict())
    else:
        best_ckpt_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
        if best_ckpt_path:
            state = torch.load(best_ckpt_path, map_location=device)
            lightning_module.load_state_dict(state["state_dict"])
            if (
                "ema_state_dict" in state
                and lightning_module.use_ema
                and lightning_module.ema_model is not None
            ):
                lightning_module.ema_model.load_state_dict(state["ema_state_dict"])
                lightning_module.model.load_state_dict(lightning_module.ema_model.module.state_dict())
        else:
            LOGGER.warning("No checkpoint available after training; using in-memory weights.")

    lightning_module.model.to(device)
    lightning_module.model.eval()

    torch.save(lightning_module.model.state_dict(), run_dir / "best_model.pt")

    history: List[Dict[str, float]] = lightning_module.history
    best_rmse = lightning_module.best_val_rmse
    guard_flags: Dict[str, object] = {}
    if not np.isfinite(best_rmse):
        guard_flags["training_unstable"] = True

    # Validation repair summary
    model = lightning_module.model
    val_ids, val_preds_tensor, val_targets_tensor = run_prediction_loop(
        model, val_loader, device, return_targets=True
    )

    constraint_processor = BiomassConstraintProcessor(
        ConstraintConfig(tolerance=config.constraint_tolerance)
    )

    validation_summary: Dict[str, object] = {}
    guard_flags: Dict[str, object] = {}
    if val_targets_tensor is not None and val_preds_tensor.numel():
        val_predictions_df = pd.DataFrame(val_preds_tensor.numpy(), columns=list(target_names))
        val_predictions_df[image_column] = val_ids

        metadata_lookup = metadata_features_train.set_index(image_column)
        meta_subset = (
            metadata_lookup.reindex([str(_id) for _id in val_ids])
            .reset_index()
        )
        if metadata_columns:
            for column in metadata_columns:
                if column in meta_subset.columns:
                    meta_subset[column] = meta_subset[column].fillna(0.0)
        if "species_code" in meta_subset.columns:
            meta_subset["species_code"] = meta_subset["species_code"].fillna(-1.0)
        if "state_code" in meta_subset.columns:
            meta_subset["state_code"] = meta_subset["state_code"].fillna(-1.0)
        if "species" in meta_subset.columns:
            meta_subset["species"] = (
                meta_subset["species"]
                .fillna("unknown")
                .astype(str)
                .str.lower()
            )

        repaired_val_df = constraint_processor.repair_frame(
            val_predictions_df[[image_column, *target_names]],
            meta_subset,
            id_column=image_column,
        )

        y_true = val_targets_tensor.numpy()
        raw_pred = val_predictions_df[target_names].to_numpy()
        repaired_pred = repaired_val_df[target_names].to_numpy()

        def _np_metrics(pred: np.ndarray) -> Dict[str, object]:
            metrics = {
                "rmse": float(np.sqrt(np.mean((pred - y_true) ** 2))),
                "mae": float(np.mean(np.abs(pred - y_true))),
                "per_target": {},
            }
            for idx, name in enumerate(target_names):
                metrics["per_target"][name] = {
                    "rmse": float(np.sqrt(np.mean((pred[:, idx] - y_true[:, idx]) ** 2))),
                    "mae": float(np.mean(np.abs(pred[:, idx] - y_true[:, idx]))),
                }
            return metrics

        delta_abs = np.abs(repaired_pred - raw_pred)
        validation_summary = {
            "raw": _np_metrics(raw_pred),
            "repaired": _np_metrics(repaired_pred),
            "mean_abs_adjustment": float(delta_abs.mean()),
            "max_abs_adjustment": float(delta_abs.max()),
            "samples": int(repaired_pred.shape[0]),
        }

        metrics_path = run_dir / "validation_constraint_metrics.json"
        with metrics_path.open("w") as f:
            json.dump(validation_summary, f, indent=2)
        LOGGER.info(
            "Validation constraint repair: RMSE %.4f → %.4f (Δ=%.4f)",
            validation_summary["raw"]["rmse"],
            validation_summary["repaired"]["rmse"],
            validation_summary["raw"]["rmse"] - validation_summary["repaired"]["rmse"],
        )
        if (
            validation_summary["repaired"]["rmse"]
            > validation_summary["raw"]["rmse"] + 0.5
        ):
            guard_flags["repair_degraded"] = True

    submission_path = None
    if config.test_csv:
        test_df = pd.read_csv(config.test_csv)
        sample_submission_df = (
            pd.read_csv(config.sample_submission)
            if config.sample_submission
            else None
        )
        test_meta_context, test_meta_numeric = generate_inference_metadata(
            test_df,
            image_column=image_column,
            image_dir=config.image_dir,
            metadata_columns=metadata_columns,
            use_metadata=config.use_metadata,
            fractal_csv=config.fractal_csv,
            category_maps=category_maps,
        )
        test_meta_numeric.to_csv(run_dir / "test_metadata.csv", index=False)

        inference_loader = create_inference_loader(
            test_meta_numeric,
            config.image_dir,
            image_column=image_column,
            transforms=val_transforms,
            metadata_columns=metadata_columns,
            metadata_mean=metadata_mean,
            metadata_std=metadata_std,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        _, test_predictions, _ = run_prediction_loop(
            model, inference_loader, device, return_targets=False
        )
        predictions_df = pd.DataFrame(test_predictions.numpy(), columns=list(target_names))
        predictions_df[image_column] = test_meta_numeric[image_column].values

        repaired_df = constraint_processor.repair_frame(
            predictions_df[[image_column, *target_names]],
            test_meta_context,
            id_column=image_column,
        )

        deltas = repaired_df[target_names].values - predictions_df[target_names].values
        LOGGER.info(
            "Constraint repair applied to test set: mean abs delta=%.4f (max=%.4f)",
            float(np.mean(np.abs(deltas))),
            float(np.max(np.abs(deltas))),
        )

        submission_path = save_submission(
            None,
            test_df,
            target_names,
            image_column,
            config.target_name_column,
            config.id_column,
            run_dir,
            sample_submission=sample_submission_df,
            target_value_column=config.target_value_column,
            repaired_df=repaired_df,
        )
        latest_submission = config.output_dir / "latest_submission.csv"
        shutil.copyfile(submission_path, latest_submission)

    # Persist config & history
    with (run_dir / "config.json").open("w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with (run_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    return {
        "run_dir": str(run_dir),
        "val_rmse": best_rmse,
        "validation_summary": validation_summary,
        "submission": str(submission_path) if submission_path else None,
        "guard_flags": guard_flags,
    }


class AdvancedTrainer:
    """Lightweight trainer for the advanced pipeline with uncertainty-aware loss."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        device: Union[str, torch.device],
        use_physics_loss: bool = False,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.model_spec: AdvancedModelSpec = config["model_spec"]
        self.model = build_advanced_model(self.model_spec)
        self.model.to(self.device)

        self.grad_clip = float(config.get("grad_clip", 1.0))
        if use_physics_loss:
            LOGGER.warning(
                "Physics-informed training is not fully supported yet; using GaussianNLLLoss."
            )
        self.criterion = GaussianNLLLoss()
        params = list(self.model.parameters())
        self.optimizer = optim.AdamW(
            params,
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0
        for images, targets, metadata, _ in dataloader:
            if targets.numel() == 0:
                continue
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            metadata = metadata.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            mean, log_var = self.model(images, metadata)
            loss = self.criterion(mean, log_var, targets)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
            batches += 1
        if batches == 0:
            return 0.0
        return total_loss / batches

    def validate(
        self,
        dataloader: DataLoader,
        mc_dropout_samples: int = 0,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        all_means: List[np.ndarray] = []
        all_log_vars: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        with torch.no_grad():
            for images, targets, metadata, _ in dataloader:
                images = images.to(self.device, non_blocking=True)
                metadata = metadata.to(self.device, non_blocking=True)
                mean, log_var = self.model(images, metadata)
                all_means.append(mean.cpu().numpy())
                all_log_vars.append(log_var.cpu().numpy())
                if targets.numel():
                    all_targets.append(targets.cpu().numpy())

        means_array = (
            np.vstack(all_means)
            if all_means
            else np.empty((0, self.model_spec.num_outputs), dtype=np.float32)
        )
        if all_log_vars:
            log_vars_array = np.vstack(all_log_vars)
            aleatoric_uncertainty = np.exp(log_vars_array).mean(axis=0)
        else:
            aleatoric_uncertainty = np.zeros(self.model_spec.num_outputs, dtype=np.float32)

        if all_targets:
            targets_array = np.vstack(all_targets)
            rmse = float(np.sqrt(np.mean((means_array - targets_array) ** 2)))
        else:
            rmse = float("nan")

        epistemic_uncertainty = np.zeros(self.model_spec.num_outputs, dtype=np.float32)
        if mc_dropout_samples > 0 and means_array.size:
            mc_predictions: List[np.ndarray] = []
            self.model.train()
            with torch.no_grad():
                for _ in range(mc_dropout_samples):
                    fold_preds: List[np.ndarray] = []
                    for images, _, metadata, _ in dataloader:
                        images = images.to(self.device, non_blocking=True)
                        metadata = metadata.to(self.device, non_blocking=True)
                        mean, _ = self.model(images, metadata)
                        fold_preds.append(mean.cpu().numpy())
                    if fold_preds:
                        mc_predictions.append(np.vstack(fold_preds))
            self.model.eval()
            if mc_predictions:
                mc_array = np.stack(mc_predictions, axis=0)
                epistemic_uncertainty = mc_array.var(axis=0).mean(axis=0)

        return rmse, aleatoric_uncertainty, epistemic_uncertainty, means_array


def build_advanced_config(
    args: argparse.Namespace,
    *,
    target_count: int,
    metadata_dim: int,
) -> Dict[str, Any]:
    tab_hidden_dims = tuple(
        _parse_int_list(args.tab_hidden_dims, default=[128, 64])
    )
    fusion_hidden_dims = tuple(
        _parse_int_list(args.fusion_hidden_dims, default=[512, 256])
    )
    fusion_hidden_dim = fusion_hidden_dims[0] if fusion_hidden_dims else 256
    model_spec = AdvancedModelSpec(
        model_name=args.model,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        num_outputs=target_count,
        metadata_dim=metadata_dim,
        tabular_hidden_dims=tab_hidden_dims,
        fusion_hidden_dims=fusion_hidden_dims,
        fusion_dropout=args.fusion_dropout,
        use_layernorm=not args.no_tab_layernorm,
        fusion_hidden_dim=fusion_hidden_dim,
    )
    return {
        "model_spec": model_spec,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.advanced_grad_clip,
        "constraint_config": {
            "tolerance": args.constraint_tolerance,
            "repair_strategy": "adaptive",
            "enable_species_constraints": True,
        },
    }


def run_advanced_experiment(args: argparse.Namespace) -> Dict[str, object]:
    n_folds = max(int(args.n_folds), 2)
    if n_folds != args.n_folds:
        LOGGER.warning("Advanced mode requires at least 2 folds; overriding n_folds to %d.", n_folds)
    LOGGER.info("Running advanced pipeline with %d folds.", n_folds)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    image_dir = Path(args.image_dir)
    train_raw = pd.read_csv(args.train_csv)
    image_column = resolve_image_column(train_raw, args.image_column)
    (
        train_df,
        target_names,
        metadata_columns,
        _,
        _,
        category_maps,
        metadata_features_train,
    ) = prepare_training_dataframe(
        train_raw,
        image_column=image_column,
        target_name_column=args.target_name_column,
        target_value_column=args.target_value_column,
        image_dir=image_dir,
        precomputed_fractals=args.fractal_csv,
        include_metadata=not args.no_metadata,
    )

    if "species_code" in metadata_features_train.columns and "species" not in metadata_features_train.columns:
        species_mapping = category_maps.get("species", {})
        if species_mapping:
            metadata_features_train["species"] = decode_category_series(
                metadata_features_train["species_code"],
                species_mapping,
            )
    if "species" in metadata_features_train.columns:
        metadata_features_train["species"] = (
            metadata_features_train["species"]
            .fillna("unknown")
            .astype(str)
            .str.lower()
        )

    metadata_features_train.to_csv(run_dir / "train_metadata.csv", index=False)

    trainer_config = build_advanced_config(
        args,
        target_count=len(target_names),
        metadata_dim=len(metadata_columns),
    )

    aug_cfg = AugmentationConfig(
        image_size=args.image_size,
        policy=args.aug_policy,
        randaugment_num_ops=args.randaugment_num_ops,
        randaugment_magnitude=args.randaugment_magnitude,
        trivial_magnitude_bins=args.trivial_magnitude_bins,
    )
    train_transforms, val_transforms = build_transforms(aug_cfg)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    fold_histories: List[List[Dict[str, float]]] = []
    fold_metrics: List[float] = []
    checkpoint_paths: List[Path] = []

    for fold_idx, (train_index, val_index) in enumerate(kf.split(train_df)):
        LOGGER.info("Fold %d/%d", fold_idx + 1, n_folds)
        fold_dir = run_dir / f"fold-{fold_idx+1:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_fold = train_df.iloc[train_index].reset_index(drop=True)
        val_fold = train_df.iloc[val_index].reset_index(drop=True)
        if args.max_train_samples is not None:
            train_fold = train_fold.sample(
                n=min(args.max_train_samples, len(train_fold)),
                random_state=args.seed,
            ).reset_index(drop=True)
        if args.max_val_samples is not None:
            val_fold = val_fold.sample(
                n=min(args.max_val_samples, len(val_fold)),
                random_state=args.seed,
            ).reset_index(drop=True)
        train_fold = _filter_missing_images(
            train_fold,
            image_column=image_column,
            image_dir=image_dir,
            context=f"advanced fold {fold_idx+1} train",
        )
        val_fold = _filter_missing_images(
            val_fold,
            image_column=image_column,
            image_dir=image_dir,
            context=f"advanced fold {fold_idx+1} val",
        )
        if config.max_train_samples is not None:
            train_fold = train_fold.sample(
                n=min(config.max_train_samples, len(train_fold)),
                random_state=config.seed,
            ).reset_index(drop=True)
        if config.max_val_samples is not None:
            val_fold = val_fold.sample(
                n=min(config.max_val_samples, len(val_fold)),
                random_state=config.seed,
            ).reset_index(drop=True)
        train_fold = _filter_missing_images(
            train_fold,
            image_column=image_column,
            image_dir=Path(config.image_dir),
            context=f"cv fold {fold_idx+1} train",
        )
        val_fold = _filter_missing_images(
            val_fold,
            image_column=image_column,
            image_dir=Path(config.image_dir),
            context=f"cv fold {fold_idx+1} val",
        )
        train_fold, val_fold, fold_mean, fold_std = normalise_metadata_split(
            train_fold,
            val_fold,
            metadata_columns,
        )

        train_dataset = BiomassDataset(
            train_fold,
            image_dir=image_dir,
            image_column=image_column,
            transforms=train_transforms,
            target_columns=target_names,
            metadata_columns=metadata_columns,
            metadata_mean=fold_mean if metadata_columns else None,
            metadata_std=fold_std if metadata_columns else None,
        )
        val_dataset = BiomassDataset(
            val_fold,
            image_dir=image_dir,
            image_column=image_column,
            transforms=val_transforms,
            target_columns=target_names,
            metadata_columns=metadata_columns,
            metadata_mean=fold_mean if metadata_columns else None,
            metadata_std=fold_std if metadata_columns else None,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        trainer = AdvancedTrainer(
            trainer_config,
            device=args.device,
            use_physics_loss=args.use_physics_loss,
        )

        best_rmse = float("inf")
        best_state: Optional[Dict[str, Any]] = None
        history: List[Dict[str, float]] = []

        for epoch in range(1, args.epochs + 1):
            train_loss = trainer.train_epoch(train_loader, epoch)
            val_rmse, _, _, _ = trainer.validate(val_loader, mc_dropout_samples=0)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_rmse": float(val_rmse) if np.isfinite(val_rmse) else float("nan"),
                }
            )
            LOGGER.info(
                "Fold %d Epoch %d/%d - train_loss=%.4f val_rmse=%.4f",
                fold_idx + 1,
                epoch,
                args.epochs,
                train_loss,
                val_rmse,
            )
            if np.isfinite(val_rmse) and val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = {
                    "model_state": trainer.model.state_dict(),
                    "metadata_mean": fold_mean.tolist() if metadata_columns else [],
                    "metadata_std": fold_std.tolist() if metadata_columns else [],
                }
        if best_state is None:
            best_state = {
                "model_state": trainer.model.state_dict(),
                "metadata_mean": fold_mean.tolist() if metadata_columns else [],
                "metadata_std": fold_std.tolist() if metadata_columns else [],
            }

        checkpoint_path = fold_dir / "best_model.pt"
        torch.save(best_state, checkpoint_path)
        checkpoint_paths.append(checkpoint_path)
        fold_histories.append(history)
        fold_metrics.append(best_rmse)

    submission_path: Optional[Path] = None
    if args.test_csv:
        test_df = pd.read_csv(args.test_csv)
        test_meta_context, test_meta_numeric = generate_inference_metadata(
            test_df,
            image_column=image_column,
            image_dir=args.image_dir,
            metadata_columns=metadata_columns,
            use_metadata=not args.no_metadata,
            fractal_csv=args.fractal_csv,
            category_maps=category_maps,
        )
        test_meta_numeric.to_csv(run_dir / "test_metadata.csv", index=False)

        fold_predictions: List[np.ndarray] = []
        fold_epistemic: List[np.ndarray] = []
        for checkpoint_path in checkpoint_paths:
            state = torch.load(checkpoint_path, map_location=args.device)
            trainer = AdvancedTrainer(
                trainer_config,
                device=args.device,
                use_physics_loss=False,
            )
            trainer.model.load_state_dict(state["model_state"])
            fold_mean = (
                np.array(state.get("metadata_mean", []), dtype=np.float32)
                if metadata_columns
                else None
            )
            fold_std = (
                np.array(state.get("metadata_std", []), dtype=np.float32)
                if metadata_columns
                else None
            )
            inference_dataset = BiomassDataset(
                test_meta_numeric,
                image_dir=image_dir,
                image_column=image_column,
                transforms=val_transforms,
                target_columns=None,
                metadata_columns=metadata_columns,
                metadata_mean=fold_mean,
                metadata_std=fold_std,
            )
            inference_loader = DataLoader(
                inference_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            _, _, epistemic_unc, means = trainer.validate(
                inference_loader,
                mc_dropout_samples=args.mc_dropout_samples,
            )
            fold_predictions.append(means)
            fold_epistemic.append(epistemic_unc)

        if fold_predictions:
            avg_preds = np.mean(fold_predictions, axis=0)
            if fold_epistemic and fold_epistemic[0].size:
                avg_epistemic = np.mean(fold_epistemic, axis=0)
            else:
                avg_epistemic = np.zeros_like(avg_preds)

            predictions_df = pd.DataFrame(avg_preds, columns=list(target_names))
            predictions_df[image_column] = test_meta_numeric[image_column].values

            processor = AdvancedBiomassConstraintProcessor(
                trainer_config.get("constraint_config", {})
            )
            epistemic_df = pd.DataFrame(avg_epistemic, columns=list(target_names))
            repaired_df = processor.apply(predictions_df, test_meta_context, epistemic_df)

            submission_path = save_submission(
                None,
                test_df,
                target_names,
                image_column,
                args.target_name_column,
                args.id_column,
                run_dir,
                sample_submission=args.sample_submission,
                target_value_column=args.target_value_column,
                repaired_df=repaired_df,
            )

    with (run_dir / "advanced_history.json").open("w") as f:
        json.dump(fold_histories, f, indent=2)

    payload = {
        "run_dir": str(run_dir),
        "fold_rmse": [float(rmse) for rmse in fold_metrics],
        "submission": str(submission_path) if submission_path else None,
    }
    if submission_path:
        LOGGER.info("Advanced submission saved to %s", submission_path)
    return payload


def average_checkpoints(paths: Sequence[Union[str, Path]]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Average state dictionaries from multiple Lightning checkpoints."""
    if not paths:
        raise ValueError("No checkpoint paths provided for averaging.")

    avg_state: Dict[str, torch.Tensor] = {}
    avg_ema_state: Optional[Dict[str, torch.Tensor]] = None
    count = 0
    has_ema = True

    for path in paths:
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("state_dict")
        if state_dict is None:
            raise KeyError(f"Checkpoint at {path} missing 'state_dict'.")
        if not avg_state:
            avg_state = {k: v.clone() for k, v in state_dict.items()}
        else:
            for key, value in state_dict.items():
                avg_state[key] += value

        ema_state_dict = ckpt.get("ema_state_dict")
        if ema_state_dict is None:
            has_ema = False
        else:
            if avg_ema_state is None:
                avg_ema_state = {k: v.clone() for k, v in ema_state_dict.items()}
            else:
                for key, value in ema_state_dict.items():
                    avg_ema_state[key] += value
        count += 1

    for key in avg_state:
        avg_state[key] /= count

    result = {"state_dict": avg_state}
    if has_ema and avg_ema_state is not None:
        for key in avg_ema_state:
            avg_ema_state[key] /= count
        result["ema_state_dict"] = avg_ema_state
    return result


def run_cross_validation(
    config: ExperimentConfig,
    *,
    n_folds: int,
    group_column: Optional[str] = None,
) -> Dict[str, object]:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2 for cross-validation")

    set_seed(config.seed)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = config.output_dir / f"cv-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(config.train_csv)
    image_column = resolve_image_column(train_raw, config.image_column)
    (
        train_df,
        target_names,
        metadata_columns,
        _,
        _,
        category_maps,
        _,
    ) = prepare_training_dataframe(
        train_raw,
        image_column=image_column,
        target_name_column=config.target_name_column,
        target_value_column=config.target_value_column,
        image_dir=config.image_dir,
        precomputed_fractals=config.fractal_csv,
        include_metadata=config.use_metadata,
    )

    groups = None
    if group_column:
        if group_column not in train_df.columns:
            LOGGER.warning(
                "Group column '%s' not found; reverting to standard KFold.",
                group_column,
            )
        else:
            groups = train_df[group_column].astype(str).values

    if groups is not None:
        splitter = GroupKFold(n_splits=n_folds)
        split_iter = splitter.split(train_df, groups=groups)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
        split_iter = splitter.split(train_df)

    aug_cfg = config.augmentation
    train_transforms, val_transforms = build_transforms(aug_cfg)

    fold_histories: List[List[Dict[str, float]]] = []
    fold_metrics: List[float] = []
    fold_dirs: List[Path] = []

    for fold_idx, (train_index, val_index) in enumerate(split_iter):
        LOGGER.info("Cross-validation fold %d/%d", fold_idx + 1, n_folds)
        fold_dir = run_dir / f"fold-{fold_idx + 1:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_fold = train_df.iloc[train_index].reset_index(drop=True)
        val_fold = train_df.iloc[val_index].reset_index(drop=True)
        train_fold, val_fold, metadata_mean, metadata_std = normalise_metadata_split(
            train_fold,
            val_fold,
            metadata_columns,
        )

        train_dataset = BiomassDataset(
            train_fold,
            image_dir=config.image_dir,
            image_column=image_column,
            transforms=train_transforms,
            target_columns=target_names,
            metadata_columns=metadata_columns,
            metadata_mean=metadata_mean if metadata_columns else None,
            metadata_std=metadata_std if metadata_columns else None,
        )
        val_dataset = BiomassDataset(
            val_fold,
            image_dir=config.image_dir,
            image_column=image_column,
            transforms=val_transforms,
            target_columns=target_names,
            metadata_columns=metadata_columns,
            metadata_mean=metadata_mean if metadata_columns else None,
            metadata_std=metadata_std if metadata_columns else None,
        )

        train_sampler = None
        if config.curriculum.enable:
            stage_dicts = [
                {
                    "fractal_range": stage.fractal_range,
                    "biomass_range": stage.biomass_range,
                    "epochs": stage.epochs,
                }
                for stage in config.curriculum.stages
            ]
            biomass_column = (
                config.curriculum.target_column
                if config.curriculum.target_column in train_fold.columns
                else None
            )
            if config.curriculum.target_column and biomass_column is None:
                LOGGER.warning(
                    "Curriculum target column '%s' missing in fold; skipping biomass filter.",
                    config.curriculum.target_column,
                )
            train_sampler = FractalCurriculumSampler(
                train_fold,
                batch_size=config.batch_size,
                shuffle=True,
                stages=stage_dicts,
                total_epochs=config.epochs,
                biomass_column=biomass_column,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=config.num_workers,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True,
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        metadata_entropy = 0.0
        if metadata_columns:
            entropies: List[float] = []
            for column in metadata_columns:
                column_entropy = estimate_entropy(train_fold[column].to_numpy(dtype=np.float32))
                if np.isfinite(column_entropy):
                    entropies.append(column_entropy)
            if entropies:
                metadata_entropy = float(np.mean(entropies))

        model_spec = ModelSpec(
            model_name=config.backbone.name,
            pretrained=config.backbone.pretrained,
            dropout=config.backbone.dropout,
            num_outputs=len(target_names),
            metadata_dim=len(metadata_columns),
            metadata_entropy=max(metadata_entropy, 1e-3),
            tabular_hidden_dims=tuple(config.fusion.tabular_hidden_dims),
            fusion_hidden_dims=tuple(config.fusion.fusion_hidden_dims),
            fusion_dropout=config.fusion.fusion_dropout,
            use_layernorm=config.fusion.use_layernorm,
            fusion_type=config.fusion.fusion_type,
            perceiver_latents=config.fusion.perceiver_latents,
            perceiver_layers=config.fusion.perceiver_layers,
            perceiver_heads=config.fusion.perceiver_heads,
            perceiver_dropout=config.fusion.perceiver_dropout,
        )
        model = build_model(model_spec, metadata_dim=len(metadata_columns))
        device = torch.device(config.device)
        model.to(device)

        target_array = train_fold[target_names].to_numpy(dtype=np.float32)
        huber_beta = estimate_huber_delta(target_array)

        lightning_module = BiomassLightningModule(
            model=model,
            optimizer_cfg=config.optimizer,
            target_names=target_names,
            huber_beta=huber_beta,
            train_sampler=train_sampler,
            mixup_alpha=config.regularization.mixup_alpha,
            mixup_prob=config.regularization.mixup_prob,
            save_oof=config.save_oof,
        )

        checkpoint_dir = fold_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        snapshot_count = max(int(config.snapshots.num_snapshots), 0)
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch{epoch:02d}-rmse{val_rmse:.4f}",
            monitor="val_rmse",
            mode="min",
            save_top_k=max(snapshot_count, 1),
            save_last=True,
        )
        callbacks = [checkpoint_callback]

        accelerator = "gpu" if config.device.startswith("cuda") and torch.cuda.is_available() else "cpu"

        trainer = pl.Trainer(
            max_epochs=config.epochs,
            default_root_dir=str(fold_dir),
            accelerator=accelerator,
            devices=1,
            logger=False,
            callbacks=callbacks,
            enable_checkpointing=True,
            enable_model_summary=False,
            log_every_n_steps=25,
            num_sanity_val_steps=0,
        )

        trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        snapshot_paths: List[str] = []
        if snapshot_count > 0 and checkpoint_callback.best_k_models:
            ordered_paths = list(checkpoint_callback.best_k_models.keys())
            snapshot_paths = ordered_paths[:snapshot_count]

        if snapshot_paths:
            averaged_states = average_checkpoints(snapshot_paths)
            lightning_module.load_state_dict(averaged_states["state_dict"])
            if (
                "ema_state_dict" in averaged_states
                and lightning_module.use_ema
                and lightning_module.ema_model is not None
            ):
                lightning_module.ema_model.load_state_dict(averaged_states["ema_state_dict"])
                lightning_module.model.load_state_dict(lightning_module.ema_model.module.state_dict())
        else:
            best_ckpt_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
            if best_ckpt_path:
                state = torch.load(best_ckpt_path, map_location=device)
                lightning_module.load_state_dict(state["state_dict"])
                if (
                    "ema_state_dict" in state
                    and lightning_module.use_ema
                    and lightning_module.ema_model is not None
                ):
                    lightning_module.ema_model.load_state_dict(state["ema_state_dict"])
                    lightning_module.model.load_state_dict(lightning_module.ema_model.module.state_dict())

        lightning_module.model.to(device)
        lightning_module.model.eval()

        torch.save(lightning_module.model.state_dict(), fold_dir / "best_model.pt")
        history = lightning_module.history
        fold_histories.append(history)
        best_rmse = lightning_module.best_val_rmse
        fold_metrics.append(float(best_rmse))
        fold_dirs.append(fold_dir)

    mean_rmse = float(np.mean(fold_metrics)) if fold_metrics else float("nan")
    std_rmse = float(np.std(fold_metrics)) if fold_metrics else float("nan")

    with (run_dir / "cv_history.json").open("w") as f:
        json.dump(fold_histories, f, indent=2)

    LOGGER.info(
        "Cross-validation completed: mean RMSE=%.4f (std=%.4f)",
        mean_rmse,
        std_rmse,
    )

    return {
        "run_dir": str(run_dir),
        "fold_rmse": [float(rmse) for rmse in fold_metrics],
        "rmse_mean": mean_rmse,
        "rmse_std": std_rmse,
        "fold_dirs": [str(path) for path in fold_dirs],
    }


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    backbone = BackboneConfig(
        name=args.model,
        pretrained=not args.no_pretrained,
        dropout=args.dropout if args.backbone_dropout is None else args.backbone_dropout,
    )
    augmentation = AugmentationConfig(
        image_size=args.image_size,
        policy=args.aug_policy,
        randaugment_num_ops=args.randaugment_num_ops,
        randaugment_magnitude=args.randaugment_magnitude,
        trivial_magnitude_bins=args.trivial_magnitude_bins,
    )
    fusion = FusionConfig(
        tabular_hidden_dims=_parse_int_list(args.tab_hidden_dims, default=[128, 64]),
        fusion_hidden_dims=_parse_int_list(args.fusion_hidden_dims, default=[512, 256]),
        fusion_dropout=args.fusion_dropout,
        use_layernorm=not args.no_tab_layernorm,
        fusion_type=args.fusion_type,
        perceiver_latents=args.perceiver_latents,
        perceiver_layers=args.perceiver_layers,
        perceiver_heads=args.perceiver_heads,
        perceiver_dropout=args.perceiver_dropout,
    )
    ema_decay = args.ema_decay if args.ema_decay and args.ema_decay > 0 else None
    optimizer = OptimizerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_scheduler=not args.no_scheduler,
        scheduler_t_max=args.scheduler_t_max,
        ema_decay=ema_decay,
        warmup_epochs=args.warmup_epochs,
    )
    curriculum = CurriculumConfig()
    if args.curriculum_target:
        curriculum.target_column = args.curriculum_target
    if args.no_curriculum:
        curriculum.enable = False
    snapshots = SnapshotConfig(num_snapshots=max(int(args.snapshot_count), 0))
    regularization = RegularizationConfig(
        mixup_alpha=max(float(args.mixup_alpha), 0.0),
        mixup_prob=float(min(max(args.mixup_prob, 0.0), 1.0)),
    )

    return ExperimentConfig(
        train_csv=Path(args.train_csv),
        image_dir=Path(args.image_dir),
        output_dir=Path(args.output_dir),
        test_csv=Path(args.test_csv) if args.test_csv else None,
        sample_submission=Path(args.sample_submission) if args.sample_submission else None,
        fractal_csv=Path(args.fractal_csv) if args.fractal_csv else None,
        use_metadata=not args.no_metadata,
        image_column=args.image_column,
        id_column=args.id_column,
        target_name_column=args.target_name_column,
        target_value_column=args.target_value_column,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        constraint_tolerance=args.constraint_tolerance,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        backbone=backbone,
        fusion=fusion,
        curriculum=curriculum,
        snapshots=snapshots,
        augmentation=augmentation,
        optimizer=optimizer,
        regularization=regularization,
        save_oof=args.save_oof,
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a biomass experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core execution mode
    parser.add_argument(
        "--mode",
        choices=["baseline", "advanced"],
        default="baseline",
        help="Select baseline Lightning training or the advanced custom loop.",
    )

    # Data locations
    parser.add_argument("--train-csv", required=True, help="Path to training metadata CSV.")
    parser.add_argument("--image-dir", required=True, help="Directory containing training images.")
    parser.add_argument("--test-csv", help="Optional path to test metadata CSV for inference.")
    parser.add_argument("--sample-submission", help="Optional path to sample submission template.")
    parser.add_argument("--fractal-csv", help="Optional path to precomputed fractal features CSV.")
    parser.add_argument("--image-column", help="Column name that references image files.")
    parser.add_argument("--target-name-column", default="target_name", help="Column containing target names.")
    parser.add_argument("--target-value-column", default="target", help="Column containing target values.")
    parser.add_argument("--id-column", default="sample_id", help="Identifier column for submissions.")

    # Training schedule
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of data reserved for validation.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--scheduler-t_max", type=int, default=15, help="T_max parameter for cosine scheduler.")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay factor (set <=0 to disable).")
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Number of warmup epochs before cosine decay.")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable learning rate scheduler.")
    parser.add_argument("--max-train-samples", type=int, help="Optional cap on number of training samples per epoch.")
    parser.add_argument("--max-val-samples", type=int, help="Optional cap on number of validation samples.")
    parser.add_argument("--n-folds", type=int, default=1, help="Number of folds for cross-validation or advanced training.")
    parser.add_argument("--cv-group-column", type=str, help="Optional column name for grouped cross-validation.")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation during inference.")
    parser.add_argument("--save-oof", action="store_true", help="Persist out-of-fold validation predictions.")

    # Model configuration
    parser.add_argument("--model", default="efficientnet_b3", help="Backbone / architecture identifier.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability for backbone / model heads.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights for the backbone.")
    parser.add_argument(
        "--backbone-dropout",
        type=float,
        help="Override backbone dropout (e.g. for Noisy Student training).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--constraint-tolerance", type=float, default=0.5, help="Tolerance for constraint repair heuristics.")
    parser.add_argument("--use-physics-loss", action="store_true", help="Enable experimental physics-informed loss (advanced mode).")
    parser.add_argument("--advanced-grad-clip", type=float, default=1.0, help="Gradient clipping value for the advanced trainer.")
    parser.add_argument("--mc-dropout-samples", type=int, default=0, help="MC dropout samples for advanced inference uncertainty.")

    # Fusion / metadata configuration
    parser.add_argument("--no-metadata", action="store_true", help="Disable metadata/tabular features for image-only baseline.")
    parser.add_argument("--curriculum-target", type=str, help="Column name for biomass-based curriculum stages.")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum sampler.")
    parser.add_argument("--snapshot-count", type=int, default=0, help="Number of best checkpoints to average for snapshot ensembling.")
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Mixup Beta distribution alpha (0 disables mixup).")
    parser.add_argument("--mixup-prob", type=float, default=0.0, help="Probability of applying mixup to a batch.")

    # Augmentation configuration
    parser.add_argument("--image-size", type=int, default=352, help="Input image size for transforms.")
    parser.add_argument(
        "--aug-policy",
        choices=["standard", "randaugment", "trivialaugment"],
        default="standard",
        help="Augmentation policy applied before the standard Albumentations stack.",
    )
    parser.add_argument("--randaugment-num-ops", type=int, default=2, help="Number of operations applied by RandAugment (when enabled).")
    parser.add_argument("--randaugment-magnitude", type=int, default=9, help="Magnitude of RandAugment operations (when enabled).")
    parser.add_argument("--trivial-magnitude-bins", type=int, default=31, help="Magnitude bins for TrivialAugmentWide (when enabled).")

    # Tabular fusion architecture
    parser.add_argument("--tab-hidden-dims", default="128,64", help="Comma-separated hidden sizes for the tabular encoder.")
    parser.add_argument("--fusion-hidden-dims", default="512,256", help="Comma-separated hidden sizes for the fusion head.")
    parser.add_argument("--fusion-dropout", type=float, default=0.25, help="Dropout applied in the fusion head.")
    parser.add_argument("--no-tab-layernorm", action="store_true", help="Disable layer normalization in the tabular encoder.")
    parser.add_argument("--fusion-type", choices=["mlp", "perceiver"], default="mlp", help="Fusion strategy for combining image and metadata features.")
    parser.add_argument("--perceiver-latents", type=int, default=32, help="Number of latent tokens for Perceiver fusion.")
    parser.add_argument("--perceiver-layers", type=int, default=3, help="Number of Perceiver self-attention blocks.")
    parser.add_argument("--perceiver-heads", type=int, default=4, help="Number of attention heads used in Perceiver fusion.")
    parser.add_argument("--perceiver-dropout", type=float, default=0.1, help="Dropout applied inside the Perceiver fusion blocks.")

    # Output / bookkeeping
    parser.add_argument(
        "--output-dir",
        default="competitions/csiro_biomass/outputs/baseline",
        help="Directory where experiment outputs will be stored.",
    )
    parser.add_argument("--leaderboard-competition", default="csiro-biomass", help="Competition slug for leaderboard comparison.")
    parser.add_argument("--leaderboard-team", help="Team name to highlight on the leaderboard.")
    parser.add_argument("--leaderboard-top-k", type=int, default=5, help="Number of top leaderboard entries to display.")
    parser.add_argument("--leaderboard-metrics", help="Optional metrics file to use for local RMSE (defaults to latest run summary).")
    parser.add_argument("--leaderboard-distribution", action="store_true", help="Print leaderboard distribution statistics.")
    parser.add_argument("--compare-leaderboard", action="store_true", help="Display latest leaderboard snapshot after training.")

    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)
    if args.mode == "advanced":
        result = run_advanced_experiment(args)
        metrics_candidate = Path(result["run_dir"]) / "advanced_history.json"
    else:
        config = build_config_from_args(args)
        if args.n_folds > 1:
            LOGGER.info("Starting %d-fold cross-validation", args.n_folds)
            result = run_cross_validation(
                config,
                n_folds=args.n_folds,
                group_column=args.cv_group_column,
            )
        else:
            result = run_experiment(config)
        metrics_candidate = Path(result["run_dir"]) / "history.json"
    LOGGER.info("Run complete: %s", json.dumps(result, indent=2))
    if args.compare_leaderboard:
        metrics_path = Path(args.leaderboard_metrics) if args.leaderboard_metrics else metrics_candidate
        summary = compare_leaderboard(
            competition=args.leaderboard_competition,
            top_k=args.leaderboard_top_k,
            team=args.leaderboard_team,
            metrics_file=metrics_path if metrics_path.exists() else None,
            local_rmse=result.get("val_rmse") if "val_rmse" in result else None,
            print_distribution=args.leaderboard_distribution,
        )
        print()
        print_leaderboard_summary(summary)
        if args.leaderboard_team and summary["team_entry"] is None:
            print(f"\nTeam '{args.leaderboard_team}' not found on the public leaderboard.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
