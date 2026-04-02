#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.boas_pipeline import (  # noqa: E402
    STAGE_NAMES,
    build_context_windows,
    list_complete_recordings,
    load_subject_features,
    select_smoke_split,
    standardize_features,
)
from src.frontal_dual_view_net import FrontalDualViewNet  # noqa: E402
from src.metrics import classification_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BOAS frontal smoke experiment.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--output-json", help="Optional metrics output path.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def compute_class_weights(labels: np.ndarray, num_classes: int, power: float) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float32)
    counts[counts < 1.0] = 1.0
    weights = (counts.sum() / counts) ** power
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch_inputs, batch_targets in loader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_inputs)
        loss = loss_fn(logits, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch_targets.shape[0]
        total_items += int(batch_targets.shape[0])
    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, object], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    preds = []
    truths = []
    for batch_inputs, batch_targets in loader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        logits = model(batch_inputs)
        loss = loss_fn(logits, batch_targets)
        total_loss += float(loss.item()) * batch_targets.shape[0]
        total_items += int(batch_targets.shape[0])
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        truths.extend(batch_targets.cpu().tolist())
    y_pred = np.asarray(preds, dtype=np.int64)
    y_true = np.asarray(truths, dtype=np.int64)
    metrics = classification_metrics(y_true, y_pred, num_classes=len(STAGE_NAMES))
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics["loss"], metrics, y_true, y_pred


def majority_baseline(train_labels: np.ndarray, val_labels: np.ndarray) -> Dict[str, object]:
    majority_label = int(np.bincount(train_labels, minlength=len(STAGE_NAMES)).argmax())
    preds = np.full_like(val_labels, fill_value=majority_label)
    metrics = classification_metrics(val_labels, preds, num_classes=len(STAGE_NAMES))
    metrics["majority_label"] = majority_label
    return metrics


def merge_metadata(items: List[Dict[str, object]]) -> Dict[str, object]:
    stage_counts = {stage: 0 for stage in range(len(STAGE_NAMES))}
    total_epochs = 0
    skipped_rows = 0
    per_channel_feature_dim = 0
    sample_rate = 0.0
    for item in items:
        total_epochs += int(item["epochs"])
        skipped_rows += int(item["skipped_rows"])
        per_channel_feature_dim = int(item["per_channel_feature_dim"])
        sample_rate = float(item["sample_rate"])
        for stage, count in item["stage_counts"].items():
            stage_counts[int(stage)] += int(count)
    return {
        "subjects": [item["subject"] for item in items],
        "epochs": total_epochs,
        "stage_counts": stage_counts,
        "skipped_rows": skipped_rows,
        "sample_rate": sample_rate,
        "per_channel_feature_dim": per_channel_feature_dim,
    }


def main() -> int:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["training"]["seed"])
    set_seed(seed)
    device = choose_device(config["training"].get("device", "auto"))

    dataset_root = PROJECT_ROOT / config["dataset"]["root"]
    all_records = list_complete_recordings(dataset_root, validate_edf=False)
    records = list_complete_recordings(dataset_root, validate_edf=True)
    all_subjects = {record.subject for record in all_records}
    readable_subjects = {record.subject for record in records}
    skipped_unreadable_subjects = sorted(all_subjects - readable_subjects, key=lambda name: int(name.split("-")[1]))
    train_records, val_records = select_smoke_split(records)

    train_feature_blocks = []
    train_label_blocks = []
    train_meta_blocks = []
    use_ai_feature = bool(config["dataset"].get("include_headband_stage_ai_feature", False))
    for train_record in train_records:
        features, labels, metadata = load_subject_features(
            train_record,
            label_column=config["dataset"]["label_column"],
            valid_labels=config["dataset"]["valid_labels"],
            include_headband_stage_ai_feature=use_ai_feature,
        )
        train_feature_blocks.append(features)
        train_label_blocks.append(labels)
        train_meta_blocks.append(metadata)

    train_features = np.concatenate(train_feature_blocks, axis=0)
    train_labels = np.concatenate(train_label_blocks, axis=0)
    train_meta = merge_metadata(train_meta_blocks)
    val_features, val_labels, val_meta = load_subject_features(
        val_records[0],
        label_column=config["dataset"]["label_column"],
        valid_labels=config["dataset"]["valid_labels"],
        include_headband_stage_ai_feature=use_ai_feature,
    )

    train_features, val_features, scaler = standardize_features(train_features, val_features)
    context_radius = int(config["model"]["context_radius"])
    train_windows = build_context_windows(train_features, context_radius)
    val_windows = build_context_windows(val_features, context_radius)

    train_tensor = torch.tensor(train_windows, dtype=torch.float32)
    train_targets = torch.tensor(train_labels, dtype=torch.long)
    val_tensor = torch.tensor(val_windows, dtype=torch.float32)
    val_targets = torch.tensor(val_labels, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(train_tensor, train_targets),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor, val_targets),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    num_classes = len(STAGE_NAMES)
    weights = compute_class_weights(
        train_labels,
        num_classes,
        power=float(config["training"].get("class_weight_power", 1.0)),
    ).to(device)
    model = FrontalDualViewNet(
        channel_feature_dim=int(train_meta["per_channel_feature_dim"]),
        total_feature_dim=int(train_features.shape[1]),
        context_size=train_windows.shape[1],
        num_classes=num_classes,
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    history = []
    best_state = None
    best_metrics = None
    best_epoch = -1

    for epoch in range(int(config["training"]["epochs"])):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device)
        _, val_metrics, _, _ = evaluate(model, val_loader, loss_fn, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_kappa": val_metrics["kappa"],
            }
        )
        if best_metrics is None or val_metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = val_metrics
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    _, final_metrics, final_truths, final_preds = evaluate(model, val_loader, loss_fn, device)
    majority_metrics = majority_baseline(train_labels, val_labels)

    output = {
        "config_path": str(Path(args.config).resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "complete_subjects_seen": [record.subject for record in records],
        "all_complete_subjects_seen": [record.subject for record in all_records],
        "skipped_unreadable_subjects": skipped_unreadable_subjects,
        "include_headband_stage_ai_feature": use_ai_feature,
        "train_subjects": [record.subject for record in train_records],
        "val_subjects": [record.subject for record in val_records],
        "train_metadata": train_meta,
        "val_metadata": val_meta,
        "context_size": int(train_windows.shape[1]),
        "feature_dim": int(train_features.shape[1]),
        "device": str(device),
        "best_epoch": best_epoch,
        "history": history,
        "majority_baseline": majority_metrics,
        "best_val_metrics": best_metrics,
        "final_val_metrics": final_metrics,
        "final_val_truths": final_truths.tolist(),
        "final_val_predictions": final_preds.tolist(),
        "scaler_feature_dim": int(scaler["mean"].shape[0]),
        "stage_names": STAGE_NAMES,
    }

    output_path = Path(args.output_json) if args.output_json else Path(config["output"]["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    summary = {
        "train_subjects": output["train_subjects"],
        "val_subjects": output["val_subjects"],
        "majority_macro_f1": majority_metrics["macro_f1"],
        "macro_f1": best_metrics["macro_f1"],
        "accuracy": best_metrics["accuracy"],
        "kappa": best_metrics["kappa"],
        "best_epoch": best_epoch,
        "output_json": str(output_path.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
