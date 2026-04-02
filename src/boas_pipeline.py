from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from pyedflib import EdfReader


STAGE_NAMES = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}

BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("sigma", 12.0, 16.0),
    ("beta", 16.0, 30.0),
)


@dataclass
class RecordingSpec:
    subject: str
    headband_edf: Path
    headband_events: Path
    psg_events: Path


def _is_readable_headband_file(edf_path: Path) -> bool:
    reader = None
    try:
        reader = EdfReader(str(edf_path))
        labels = set(reader.getSignalLabels())
        return "HB_1" in labels and "HB_2" in labels
    except Exception:
        return False
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass


def list_complete_recordings(dataset_root: Path, validate_edf: bool = True) -> List[RecordingSpec]:
    records: List[RecordingSpec] = []
    for subject_dir in sorted(
        [path for path in dataset_root.glob("sub-*") if path.is_dir()],
        key=lambda path: int(path.name.split("-")[1]),
    ):
        eeg_dir = subject_dir / "eeg"
        headband = list(eeg_dir.glob("*acq-headband_eeg.edf"))
        headband_events = list(eeg_dir.glob("*acq-headband_events.tsv"))
        psg_events = list(eeg_dir.glob("*acq-psg_events.tsv"))
        if headband and headband_events and psg_events:
            if validate_edf and not _is_readable_headband_file(headband[0]):
                continue
            records.append(
                RecordingSpec(
                    subject=subject_dir.name,
                    headband_edf=headband[0],
                    headband_events=headband_events[0],
                    psg_events=psg_events[0],
                )
            )
    return records


def select_smoke_split(records: Sequence[RecordingSpec]) -> Tuple[List[RecordingSpec], List[RecordingSpec]]:
    if len(records) < 2:
        raise RuntimeError("At least two complete BOAS recordings are required for the smoke split.")
    return list(records[:-1]), [records[-1]]


def robust_normalize(signal: np.ndarray) -> np.ndarray:
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = np.std(signal)
    if scale < 1e-6:
        scale = 1.0
    return ((signal - median) / scale).astype(np.float32)


def _single_channel_features(signal: np.ndarray, sample_rate: float) -> np.ndarray:
    signal = signal.astype(np.float32, copy=False)
    variance = float(np.var(signal))
    rms = float(np.sqrt(np.mean(signal**2)))
    line_length = float(np.mean(np.abs(np.diff(signal))))
    zero_cross = float(np.mean((signal[:-1] * signal[1:]) < 0))

    diff_signal = np.diff(signal)
    diff_variance = float(np.var(diff_signal)) if diff_signal.size else 0.0
    hjorth_mobility = float(np.sqrt(diff_variance / variance)) if variance > 1e-8 else 0.0
    second_diff = np.diff(diff_signal)
    second_variance = float(np.var(second_diff)) if second_diff.size else 0.0
    if diff_variance > 1e-8 and hjorth_mobility > 1e-8:
        hjorth_complexity = float(np.sqrt(second_variance / diff_variance) / hjorth_mobility)
    else:
        hjorth_complexity = 0.0

    window = np.hanning(signal.shape[0]).astype(np.float32)
    spectrum = np.fft.rfft(signal * window)
    power = (np.abs(spectrum) ** 2).astype(np.float64)
    freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / sample_rate)
    valid = (freqs >= 0.5) & (freqs <= 30.0)
    total_power = float(power[valid].sum()) + 1e-8
    band_powers: List[float] = []
    for _, low, high in BANDS:
        band_mask = (freqs >= low) & (freqs < high)
        band_powers.append(float(power[band_mask].sum() / total_power))

    normalized_power = power[valid] / total_power
    spectral_entropy = float(
        -np.sum(normalized_power * np.log(normalized_power + 1e-8)) / np.log(normalized_power.size + 1e-8)
    )
    log_total_power = float(np.log(total_power))

    return np.asarray(
        [
            variance,
            rms,
            line_length,
            zero_cross,
            hjorth_mobility,
            hjorth_complexity,
            *band_powers,
            spectral_entropy,
            log_total_power,
        ],
        dtype=np.float32,
    )


def extract_epoch_features(epoch: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, int]:
    features = [_single_channel_features(epoch[idx], sample_rate) for idx in range(epoch.shape[0])]
    per_channel_dim = int(features[0].shape[0])
    return np.concatenate(features, axis=0).astype(np.float32), per_channel_dim


def _read_normalized_headband_channels(edf_path: Path) -> Tuple[Dict[str, np.ndarray], float]:
    reader = EdfReader(str(edf_path))
    try:
        labels = list(reader.getSignalLabels())
        sample_rate = float(reader.getSampleFrequency(0))
        hb_1 = reader.readSignal(labels.index("HB_1")).astype(np.float32)
        hb_2 = reader.readSignal(labels.index("HB_2")).astype(np.float32)
    finally:
        reader.close()

    hb_1_norm = robust_normalize(hb_1)
    hb_2_norm = robust_normalize(hb_2)
    hb_diff = robust_normalize(hb_1_norm - hb_2_norm)
    return {"HB_1": hb_1_norm, "HB_2": hb_2_norm, "HB_DIFF": hb_diff}, sample_rate


def load_subject_features(
    recording: RecordingSpec,
    label_column: str,
    valid_labels: Iterable[int],
    include_headband_stage_ai_feature: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    channels, sample_rate = _read_normalized_headband_channels(recording.headband_edf)
    allowed = set(valid_labels)
    feature_rows: List[np.ndarray] = []
    labels: List[int] = []
    per_channel_dim = None
    skipped = 0

    headband_stage_ai_by_onset: Dict[str, int] = {}
    if include_headband_stage_ai_feature:
        with recording.headband_events.open("r", encoding="utf-8", errors="ignore") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                stage_text = row.get("stage_ai")
                onset_text = row.get("onset")
                if stage_text is None or stage_text == "" or onset_text is None:
                    continue
                stage = int(stage_text)
                if stage in allowed:
                    headband_stage_ai_by_onset[onset_text] = stage

    with recording.psg_events.open("r", encoding="utf-8", errors="ignore") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        for row in rows:
            label_text = row.get(label_column)
            if label_text is None or label_text == "":
                skipped += 1
                continue
            label = int(label_text)
            if label not in allowed:
                skipped += 1
                continue

            start = int(row["begsample"]) - 1
            end = int(row["endsample"])
            if start < 0 or end <= start:
                skipped += 1
                continue

            epoch = np.stack(
                [channels["HB_1"][start:end], channels["HB_2"][start:end], channels["HB_DIFF"][start:end]],
                axis=0,
            )
            if epoch.shape[1] == 0 or end > channels["HB_1"].shape[0]:
                skipped += 1
                continue

            features, per_channel_dim = extract_epoch_features(epoch, sample_rate)
            if include_headband_stage_ai_feature:
                onset_text = row.get("onset")
                stage = headband_stage_ai_by_onset.get(onset_text, -1)
                one_hot = np.zeros(len(STAGE_NAMES), dtype=np.float32)
                if stage in allowed:
                    one_hot[stage] = 1.0
                features = np.concatenate([features, one_hot], axis=0)
            feature_rows.append(features)
            labels.append(label)

    if not feature_rows or per_channel_dim is None:
        raise RuntimeError(f"No valid labeled epochs found for {recording.subject}.")

    counts = {stage: int(np.sum(np.asarray(labels) == stage)) for stage in sorted(allowed)}
    metadata = {
        "subject": recording.subject,
        "epochs": len(labels),
        "stage_counts": counts,
        "skipped_rows": skipped,
        "sample_rate": sample_rate,
        "per_channel_feature_dim": per_channel_dim,
    }
    return np.stack(feature_rows).astype(np.float32), np.asarray(labels, dtype=np.int64), metadata


def standardize_features(
    train_features: np.ndarray,
    eval_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    train_standard = ((train_features - mean) / std).astype(np.float32)
    eval_standard = ((eval_features - mean) / std).astype(np.float32)
    return train_standard, eval_standard, {"mean": mean.squeeze(0), "std": std.squeeze(0)}


def build_context_windows(features: np.ndarray, context_radius: int) -> np.ndarray:
    if context_radius <= 0:
        return features[:, None, :]
    padded = np.pad(features, ((context_radius, context_radius), (0, 0)), mode="edge")
    windows = []
    width = 2 * context_radius + 1
    for index in range(features.shape[0]):
        windows.append(padded[index : index + width])
    return np.stack(windows).astype(np.float32)
