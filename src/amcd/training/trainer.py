"""train stage: fit CNN on energy tensors, select checkpoint on valid loss."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import EnergyDataset
from ..models.cnn import build_model  # noqa: F401 — import also triggers registration
from ..provenance import select_device
from ..runtime import Verbosity, emit
from .loss import build_criterion


def run_train(config: Config, run_dir: Path, verbosity: Verbosity) -> None:
    preprocessed_dir = run_dir / "preprocessed"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load split metadata
    meta_path = preprocessed_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    n_channels = meta["n_channels"]
    split_counts = meta["split_counts"]

    # Split names are role-derived from config (never hardcoded). Cardinality is
    # guaranteed at config load (REQUIRED_ROLE_COUNTS), so this cannot silently take
    # the first of two `valid` splits or raise a bare StopIteration for zero (F-44).
    train_split = config.the_split_with_role("train")
    valid_split = config.the_split_with_role("valid")

    if split_counts.get(train_split, 0) == 0:
        raise RuntimeError(f"Training split {train_split!r} is empty — cannot train.")
    if split_counts.get(valid_split, 0) == 0:
        raise RuntimeError(
            f"Validation split {valid_split!r} is empty — cannot select checkpoint. "
            "Increase scenes.n_id or adjust split fracs."
        )

    # Seed weight init and DataLoader shuffle from independent named seeds (inv #5)
    torch.manual_seed(config.seed("weight_init"))

    # One selector, shared with `infer` and with the `versions.json` stamp, so the
    # device a checkpoint was trained on cannot differ from the device recorded
    # for the run (F-74).
    device = select_device()
    emit(verbosity, "metrics", f"  Device: {device}")

    # Datasets + loaders (num_workers=0 for MPS compatibility)
    train_ds = EnergyDataset(preprocessed_dir, train_split)
    valid_ds = EnergyDataset(preprocessed_dir, valid_split)
    _shuffle_gen = torch.Generator()
    _shuffle_gen.manual_seed(config.seed("data_shuffle"))
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0,
        generator=_shuffle_gen,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    # Model — params validated against the model's own schema (config.model.params)
    model = build_model(config.model.name, n_channels, config.model.params).to(device)

    # δ is a config value in dB; build_criterion scales it into the z-scored loss
    # domain by 1/high_std so the Huber knee stays O(1)-meaningful in dB (inv #7).
    criterion = build_criterion(config, meta["norm_stats"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_valid_loss = float("inf")
    patience_counter = 0
    log_rows: list[dict] = []

    for epoch in range(config.n_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for low, high in train_loader:
            low, high = low.to(device), high.to(device)
            optimizer.zero_grad()
            pred = low + model(low)  # residual framing
            loss = criterion(pred, high)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- Validate (checkpoint selection only — invariant #2) ---
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for low, high in valid_loader:
                low, high = low.to(device), high.to(device)
                pred = low + model(low)
                valid_loss += criterion(pred, high).item()
        valid_loss /= len(valid_loader)

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                emit(verbosity, "progress", f"  Early stopping at epoch {epoch}")
                break

        if (epoch + 1) % max(1, config.n_epochs // 10) == 0:
            emit(
                verbosity, "progress",
                f"  Epoch {epoch+1}/{config.n_epochs} | "
                f"train={train_loss:.4f}  valid={valid_loss:.4f}",
            )

    # Per-epoch loss curve: observability only (nothing downstream reads it,
    # F-23) — checkpoint selection above already consumed the losses live.
    if verbosity.saves("metrics"):
        log_path = checkpoint_dir / "train_log.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "valid_loss"])
            writer.writeheader()
            writer.writerows(log_rows)

    emit(verbosity, "metrics", f"  Best valid loss: {best_valid_loss:.4f} → {checkpoint_dir / 'best.pt'}")
