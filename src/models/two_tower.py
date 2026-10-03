from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.eval.tracking import log_model_run


class InteractionDataset(Dataset):
    def __init__(self, user_idx: np.ndarray, item_idx: np.ndarray):
        self.user_idx = torch.as_tensor(user_idx, dtype=torch.long)
        self.item_idx = torch.as_tensor(item_idx, dtype=torch.long)

    def __len__(self):
        return len(self.user_idx)

    def __getitem__(self, i):
        return self.user_idx[i], self.item_idx[i]


class Tower(nn.Module):
    def __init__(self, num_ids: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_ids, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids)
        x = x + self.mlp(x)
        return nn.functional.normalize(x, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_tower = Tower(num_users, embedding_dim)
        self.item_tower = Tower(num_items, embedding_dim)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor):
        return self.user_tower(user_ids), self.item_tower(item_ids)


def in_batch_negative_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    user_ids: torch.Tensor | None = None,
    item_ids: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    logits = user_emb @ item_emb.T / temperature
    if user_ids is not None and item_ids is not None:
        same_user = user_ids.unsqueeze(1) == user_ids.unsqueeze(0)
        same_item = item_ids.unsqueeze(1) == item_ids.unsqueeze(0)
        diagonal = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill((same_user | same_item) & ~diagonal, float("-inf"))
    labels = torch.arange(logits.shape[0], device=logits.device)
    return nn.functional.cross_entropy(logits, labels)


@dataclass
class IdMaps:
    user_id_to_idx: dict
    item_id_to_idx: dict
    idx_to_user_id: dict
    idx_to_item_id: dict


def build_id_maps(train: pl.DataFrame) -> IdMaps:
    user_ids = train["userId"].unique().sort().to_list()
    item_ids = train["movieId"].unique().sort().to_list()
    user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_id_to_idx = {m: i for i, m in enumerate(item_ids)}
    return IdMaps(
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
        idx_to_user_id={i: u for u, i in user_id_to_idx.items()},
        idx_to_item_id={i: m for m, i in item_id_to_idx.items()},
    )


def save_checkpoint(path: Path, model: TwoTowerModel, optimizer, epoch: int, id_maps: IdMaps) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    embedding_dim = model.user_tower.embedding.embedding_dim
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "num_users": len(id_maps.user_id_to_idx),
            "num_items": len(id_maps.item_id_to_idx),
            "embedding_dim": embedding_dim,
            "user_id_to_idx": id_maps.user_id_to_idx,
            "item_id_to_idx": id_maps.item_id_to_idx,
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu"):
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    id_maps = IdMaps(
        user_id_to_idx=ckpt["user_id_to_idx"],
        item_id_to_idx=ckpt["item_id_to_idx"],
        idx_to_user_id={i: u for u, i in ckpt["user_id_to_idx"].items()},
        idx_to_item_id={i: m for m, i in ckpt["item_id_to_idx"].items()},
    )
    model = TwoTowerModel(ckpt["num_users"], ckpt["num_items"], embedding_dim=ckpt["embedding_dim"])
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["optimizer_state"], ckpt["epoch"], id_maps


def train(
    train_df: pl.DataFrame,
    checkpoint_path: Path,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    embedding_dim: int = 64,
    device: str | None = None,
    run_name: str = "two_tower",
) -> tuple[TwoTowerModel, IdMaps]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    resumed = load_checkpoint(checkpoint_path, device=device)
    if resumed is not None:
        model, optimizer_state, start_epoch, id_maps = resumed
        print(f"resuming from checkpoint at epoch {start_epoch}")
        start_epoch += 1
        missing_users = set(train_df["userId"].unique().to_list()) - set(id_maps.user_id_to_idx)
        missing_items = set(train_df["movieId"].unique().to_list()) - set(id_maps.item_id_to_idx)
        if missing_users or missing_items:
            raise ValueError(
                f"checkpoint at {checkpoint_path} was trained on different data: "
                f"{len(missing_users)} users and {len(missing_items)} items in the training frame are unknown to it"
            )
    else:
        id_maps = build_id_maps(train_df)
        model = TwoTowerModel(len(id_maps.user_id_to_idx), len(id_maps.item_id_to_idx), embedding_dim)
        optimizer_state = None
        start_epoch = 0

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        for group in optimizer.param_groups:
            group["lr"] = lr

    user_idx = train_df["userId"].replace_strict(id_maps.user_id_to_idx).to_numpy()
    item_idx = train_df["movieId"].replace_strict(id_maps.item_id_to_idx).to_numpy()
    loader = DataLoader(InteractionDataset(user_idx, item_idx), batch_size=batch_size, shuffle=True, drop_last=True)

    final_avg_loss = None
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for u, i in loader:
            u, i = u.to(device), i.to(device)
            user_emb, item_emb = model(u, i)
            loss = in_batch_negative_loss(user_emb, item_emb, u, i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        final_avg_loss = avg_loss
        print(f"epoch {epoch}: avg_loss={avg_loss:.4f}")
        save_checkpoint(checkpoint_path, model, optimizer, epoch, id_maps)

    if final_avg_loss is not None:
        with log_model_run(
            run_name,
            params={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "embedding_dim": model.user_tower.embedding.embedding_dim,
                "device": device,
            },
            metrics={"final_train_loss": final_avg_loss},
        ):
            pass

    return model, id_maps


def export_embeddings(
    model: TwoTowerModel, id_maps: IdMaps, output_dir: Path, device: str = "cpu", prefix: str = "two_tower"
) -> None:
    model = model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        user_ids_sorted = sorted(id_maps.user_id_to_idx, key=lambda u: id_maps.user_id_to_idx[u])
        user_idx_tensor = torch.arange(len(user_ids_sorted), device=device)
        user_emb = model.user_tower(user_idx_tensor).cpu().numpy()

        item_ids_sorted = sorted(id_maps.item_id_to_idx, key=lambda m: id_maps.item_id_to_idx[m])
        item_idx_tensor = torch.arange(len(item_ids_sorted), device=device)
        item_emb = model.item_tower(item_idx_tensor).cpu().numpy()

    user_df = pl.DataFrame({"userId": user_ids_sorted, "embedding": user_emb.tolist()})
    item_df = pl.DataFrame({"movieId": item_ids_sorted, "embedding": item_emb.tolist()})

    user_df.write_parquet(output_dir / f"{prefix}_user_embeddings.parquet")
    item_df.write_parquet(output_dir / f"{prefix}_item_embeddings.parquet")
    print(f"exported {len(user_ids_sorted)} user and {len(item_ids_sorted)} item embeddings to {output_dir}")

    n_nan_users = int((~np.isfinite(user_emb)).any(axis=1).sum())
    if n_nan_users > 0:
        print(f"WARNING: {n_nan_users} of {len(user_ids_sorted)} exported user embeddings contain NaN or infinity. "
              f"Run scripts/check_embeddings_for_nan.py on the output to identify affected users.")

    n_nan_items = int((~np.isfinite(item_emb)).any(axis=1).sum())
    if n_nan_items > 0:
        print(f"WARNING: {n_nan_items} of {len(item_ids_sorted)} exported item embeddings contain NaN or infinity. "
              f"Run scripts/check_embeddings_for_nan.py on the output to identify affected items.")
