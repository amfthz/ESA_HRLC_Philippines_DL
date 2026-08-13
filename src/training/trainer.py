# src/training/trainer.py

import torch
from tqdm import tqdm
from typing import Dict, Optional
import os
import math
import time

from src.training.metrics import (
    accuracy_fn,
    f1_macro_fn,
    miou_fn,
)

def train_one_epoch(
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    ignore_class_ids: list,
) -> Dict[str, float]:
    """
    Train model for one epoch.
    """
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    total_miou = 0.0
    n_used = 0
    ignore_ids_t = torch.tensor(ignore_class_ids, device=device)

    for X, y in trainloader:
        X = X.to(device)
        y = y.to(device)

        targets = y.squeeze(1).long()
        valid_mask = ~torch.isin(targets, ignore_ids_t)

        if valid_mask.sum() == 0:
            continue

        logits = model(X)
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
        targets_flat = targets.view(-1)

        logits_masked = logits_flat[valid_mask.view(-1)]
        targets_masked = targets_flat[valid_mask.view(-1)]

        loss = loss_fn(logits_masked, targets_masked)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_fn(logits, y.squeeze(1), ignore_class_ids)
        total_f1 += f1_macro_fn(logits, y.squeeze(1), ignore_class_ids)
        total_miou += miou_fn(logits, y.squeeze(1), ignore_class_ids)
        n_used += 1

    if n_used == 0:
        return {"loss": 0.0, "accuracy": 0.0, "f1": 0.0, "miou": 0.0}

    return {
        "loss": total_loss / n_used,
        "accuracy": total_acc / n_used,
        "f1": total_f1 / n_used,
        "miou": total_miou / n_used,
    }

@torch.inference_mode()
def validate_one_epoch(
    model: torch.nn.Module,
    valloader: torch.utils.data.DataLoader,
    loss_fn,
    device: torch.device,
    ignore_class_ids: list,
) -> Dict[str, float]:
    """
    Validate model for one epoch.
    """
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    total_miou = 0.0
    n_used = 0
    ignore_ids_t = torch.tensor(ignore_class_ids, device=device)

    for X, y in valloader:
        X = X.to(device)
        y = y.to(device)

        targets = y.squeeze(1).long()
        valid_mask = ~torch.isin(targets, ignore_ids_t)

        if valid_mask.sum() == 0:
            continue

        logits = model(X)
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
        targets_flat = targets.view(-1)

        logits_masked = logits_flat[valid_mask.view(-1)]
        targets_masked = targets_flat[valid_mask.view(-1)]

        loss = loss_fn(logits_masked, targets_masked)

        total_loss += loss.item()
        total_acc += accuracy_fn(logits, y.squeeze(1), ignore_class_ids)
        total_f1 += f1_macro_fn(logits, y.squeeze(1), ignore_class_ids)
        total_miou += miou_fn(logits, y.squeeze(1), ignore_class_ids)
        n_used += 1

    if n_used == 0:
        return {"loss": 0.0, "accuracy": 0.0, "f1": 0.0, "miou": 0.0}

    return {
        "loss": total_loss / n_used,
        "accuracy": total_acc / n_used,
        "f1": total_f1 / n_used,
        "miou": total_miou / n_used,
    }


def train(
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    valloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    scheduler,
    device: torch.device,
    cfg_train: dict,
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, list]:
    """
    Full training loop driven by training.yaml.
    """
    epochs = cfg_train["training"]["epochs"]
    ignore_class_ids = cfg_train["loss"]["ignore_class_ids"]
    use_scheduler = cfg_train["scheduler"]["enabled"]

    # ---- TRAINING CONTROL CONFIG ----
    monitor_name = cfg_train["training"].get("monitor", "f1")
    if monitor_name.startswith("val_"):
        monitor_name = monitor_name.replace("val_", "")
    monitor_tag = f"val_{monitor_name}"

    mode = cfg_train["training"].get("mode", "max")

    es_cfg = cfg_train.get("early_stopping", {})
    es_enabled = es_cfg.get("enabled", False)
    es_patience = es_cfg.get("patience", 0)
    es_min_delta = es_cfg.get("min_delta", 0.0)
    es_warmup = es_cfg.get("warmup_epochs", 0)

    patience_counter = 0

    ckpt_cfg = cfg_train.get("checkpoint", {})
    save_ckpt = ckpt_cfg.get("enabled", False)

    if mode == "max":
        best_metric = -math.inf
        is_improvement = lambda cur, best: cur > best + es_min_delta
    else:
        best_metric = math.inf
        is_improvement = lambda cur, best: cur < best - es_min_delta

    results = {
        "train_loss": [],
        "train_accuracy": [],
        "train_f1": [],
        "train_miou": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "val_miou": [],
    }

    for epoch in range(epochs):

        epoch_start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_metrics = train_one_epoch(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            ignore_class_ids=ignore_class_ids,
        )

        val_metrics = validate_one_epoch(
            model=model,
            valloader=valloader,
            loss_fn=loss_fn,
            device=device,
            ignore_class_ids=ignore_class_ids,
        )

        if monitor_name not in val_metrics:
            raise KeyError(
                f"Monitored metric '{monitor_name}' not found in val_metrics. "
                f"Available: {list(val_metrics.keys())}"
            )

        current_metric = val_metrics[monitor_name]

        # ---- IMPROVEMENT CHECK (ONCE PER EPOCH) ----
        improved = is_improvement(current_metric, best_metric)

        if improved:
            best_metric = current_metric
            patience_counter = 0

            # ---- CHECKPOINTING (BEST MODEL) ----
            if save_ckpt and checkpoint_dir is not None:
                ckpt_filename = f"best_model.pt"
                ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric": best_metric,
                        "monitor": monitor_name,
                        "cfg_train": cfg_train,
                    },
                    ckpt_path,
                )
                print(
                    f"Checkpoint saved ({ckpt_filename}) "
                    f"(best {monitor_tag} = {best_metric:.4f})"
                )
        else:
            patience_counter += 1

        # ---- EARLY STOPPING LOGIC ----
        if es_enabled and (epoch + 1) > es_warmup:
            if patience_counter >= es_patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1} "
                    f"(best {monitor_name} = {best_metric:.4f})"
                )
                break

        if use_scheduler and scheduler is not None:
            scheduler.step()
            print(f"LR updated to {scheduler.get_last_lr()}")

        epoch_elapsed = time.time() - epoch_start_time
        epoch_minutes = int(epoch_elapsed // 60)
        epoch_seconds = int(epoch_elapsed % 60)
        print(f"Epoch time: {epoch_minutes} min {epoch_seconds} sec")

        # ---- LOGGING ----
        print(
            f"Train | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Acc: {train_metrics['accuracy']:.2f}% | "
            f"F1: {train_metrics['f1']:.3f}"
        )

        print(
            f"Val   | "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.2f}% | "
            f"F1: {val_metrics['f1']:.3f}"
        )

        # ---- STORE ----
        for k, v in train_metrics.items():
            results[f"train_{k}"].append(v)

        for k, v in val_metrics.items():
            results[f"val_{k}"].append(v)

    return results