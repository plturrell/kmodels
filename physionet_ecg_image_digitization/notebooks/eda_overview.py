"""Quick exploratory overview for the PhysioNet ECG Image Digitization dataset.

Run with:

    python -m physionet_ecg_image_digitization.notebooks.eda_overview
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from pathlib import Path

from physionet_ecg_image_digitization.src.data import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_METADATA_CSV,
    DEFAULT_SIGNAL_DIR,
    ECGDigitizationDataset,
    discover_samples,
    load_samples_from_metadata,
)
from physionet_ecg_image_digitization.src.features.transforms import (
    build_eval_transform,
)


def _load_samples():
    train_csv = DEFAULT_METADATA_CSV
    if train_csv.exists():
        try:
            return load_samples_from_metadata(
                train_csv,
                image_root=DEFAULT_IMAGE_DIR,
                signal_root=DEFAULT_SIGNAL_DIR,
            )
        except FileNotFoundError as exc:
            print(f"Metadata found but assets missing: {exc}")
            return []
    try:
        return discover_samples(
            DEFAULT_IMAGE_DIR,
            signal_dir=DEFAULT_SIGNAL_DIR,
        )
    except FileNotFoundError as exc:
        print(f"Data not ready yet: {exc}")
        return []


def main() -> None:
    samples = _load_samples()
    print(f"Discovered {len(samples)} samples.")
    if not samples:
        print("Populate `data/physionet_ecg_image_digitization/raw/` and rerun.")
        return
    leads = Counter(sample.lead or "unknown" for sample in samples)
    print("Lead distribution (top 10):")
    for lead, count in leads.most_common(10):
        print(f"  {lead:<12} {count}")

    dataset = ECGDigitizationDataset(
        samples[: min(len(samples), 64)],
        transforms=build_eval_transform(image_size=512),
        preload_signals=True,
    )
    signal_lengths = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        signal = item.get("signal")
        if signal is not None:
            signal_lengths.append(signal.shape[-1] if signal.ndim == 2 else signal.numel())
    if signal_lengths:
        print(
            f"Signal length stats — min: {min(signal_lengths)}, "
            f"max: {max(signal_lengths)}, avg: {int(mean(signal_lengths))}"
        )
    else:
        print("No digitised waveforms located alongside the images.")


if __name__ == "__main__":
    main()
