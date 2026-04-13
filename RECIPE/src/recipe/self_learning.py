from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from .utils import safe_r2


@dataclass
class EarlyStopping:
    patience: int = 50
    counter: int = 0
    best_loss: float = float("inf")
    early_stop: bool = False

    def step(self, loss: float) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def fit_on_indices(
    model,
    data,
    target,
    train_idx: torch.Tensor,
    lr: float = 7e-2,
    patience: int = 50,
    max_epochs: int = 10000,
) -> dict[str, float]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience)
    last_loss = float("nan")
    stop_epoch = max_epochs
    target_flat = target.view(-1)

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out, _ = model(data)
        loss = criterion(out.view(-1)[train_idx], target_flat[train_idx])
        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())
        if early_stopping.step(last_loss):
            stop_epoch = epoch
            break

    print(f"Self-learning fit finished at epoch {stop_epoch} with loss {last_loss:.4f}")
    return {"loss": last_loss, "stop_epoch": float(stop_epoch)}


def evaluate_on_indices(model, data, target, eval_idx: torch.Tensor) -> dict[str, float]:
    criterion = nn.MSELoss()
    target_flat = target.view(-1)
    model.eval()
    with torch.no_grad():
        out, _ = model(data)
        pred = out.view(-1)[eval_idx]
        gold = target_flat[eval_idx]
        loss = float(criterion(pred, gold).item())
        r2 = safe_r2(
            gold.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    return {"loss": loss, "r2": r2}


def select_pseudo_label_indices(
    outputs: torch.Tensor,
    pool_idx: torch.Tensor,
    batch_size: int,
    strategy: str = "sequential",
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(pool_idx) == 0:
        return pool_idx, pool_idx

    select_size = min(batch_size, len(pool_idx))
    if strategy == "confidence":
        confidence = outputs[pool_idx].detach().abs().view(-1)
        topk = torch.topk(confidence, k=select_size).indices
        selected_idx = pool_idx[topk]
        keep_mask = torch.ones(len(pool_idx), dtype=torch.bool, device=pool_idx.device)
        keep_mask[topk] = False
        remaining_idx = pool_idx[keep_mask]
        return selected_idx, remaining_idx

    selected_idx = pool_idx[:select_size]
    remaining_idx = pool_idx[select_size:]
    return selected_idx, remaining_idx


def run_self_learning(
    model,
    data,
    target,
    initial_labeled_idx: torch.Tensor,
    val_idx: torch.Tensor,
    pool_idx: torch.Tensor,
    test_idx: torch.Tensor | None = None,
    batch_size: int = 300,
    max_rounds: int = 100,
    lr: float = 7e-2,
    patience: int = 50,
    max_epochs: int = 10000,
    selection_strategy: str = "sequential",
):
    print("Starting self-learning")
    working_target = target.view(-1).clone()
    train_idx = initial_labeled_idx.clone()
    current_pool = pool_idx.clone()

    training_runs: list[dict[str, float]] = []
    initial_fit = fit_on_indices(
        model,
        data,
        working_target,
        train_idx,
        lr=lr,
        patience=patience,
        max_epochs=max_epochs,
    )
    training_runs.append({"round": 0, "train_size": float(len(train_idx)), **initial_fit})
    print(f"Round 0 complete: train_size={len(train_idx)}, loss={initial_fit['loss']:.4f}")

    for round_num in range(1, max_rounds + 1):
        if len(current_pool) == 0:
            print("No unlabeled pool remaining; stopping self-learning")
            break

        model.eval()
        with torch.no_grad():
            outputs, _ = model(data)

        selected_idx, current_pool = select_pseudo_label_indices(
            outputs,
            current_pool,
            batch_size=batch_size,
            strategy=selection_strategy,
        )
        if len(selected_idx) == 0:
            print(f"Round {round_num}: no pseudo labels selected; stopping")
            break

        working_target[selected_idx] = outputs.view(-1)[selected_idx].detach()
        train_idx = torch.cat([train_idx, selected_idx], dim=0)
        print(
            f"Round {round_num}: selected {len(selected_idx)} pseudo labels, "
            f"train_size={len(train_idx)}, remaining_pool={len(current_pool)}"
        )

        fit_result = fit_on_indices(
            model,
            data,
            working_target,
            train_idx,
            lr=lr,
            patience=patience,
            max_epochs=max_epochs,
        )
        training_runs.append(
            {
                "round": float(round_num),
                "new_pseudo_labels": float(len(selected_idx)),
                "train_size": float(len(train_idx)),
                **fit_result,
            }
        )
        print(
            f"Round {round_num} complete: stop_epoch={fit_result['stop_epoch']:.0f}, "
            f"loss={fit_result['loss']:.4f}"
        )

    summary = {
        "train_size": int(len(train_idx)),
        "remaining_pool_size": int(len(current_pool)),
        "val_metrics": evaluate_on_indices(model, data, working_target, val_idx),
        "rounds": training_runs,
    }
    if test_idx is not None:
        summary["test_metrics"] = evaluate_on_indices(model, data, working_target, test_idx)
    print(
        f"Self-learning finished: train_size={summary['train_size']}, "
        f"remaining_pool={summary['remaining_pool_size']}"
    )
    return model, working_target, summary
