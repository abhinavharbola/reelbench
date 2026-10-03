from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.eval.tracking import log_model_run

PAD_TOKEN = 0


def build_attention_mask(input_seq: torch.Tensor, num_heads: int) -> torch.Tensor:
    B, L = input_seq.shape
    device = input_seq.device
    causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
    pad_key = input_seq == PAD_TOKEN
    eye = torch.eye(L, dtype=torch.bool, device=device)
    block = causal.unsqueeze(0) | (pad_key.unsqueeze(1) & ~eye.unsqueeze(0))
    additive = torch.zeros(B, L, L, device=device, dtype=torch.float32)
    additive.masked_fill_(block, float("-inf"))
    return additive.unsqueeze(1).expand(B, num_heads, L, L).reshape(B * num_heads, L, L)


@dataclass
class IdMaps:
    item_id_to_idx: dict
    idx_to_item_id: dict


def build_id_maps(train: pl.DataFrame) -> IdMaps:
    item_ids = train["movieId"].unique().sort().to_list()
    item_id_to_idx = {m: i + 1 for i, m in enumerate(item_ids)}
    return IdMaps(item_id_to_idx=item_id_to_idx, idx_to_item_id={i: m for m, i in item_id_to_idx.items()})


def build_user_sequences(train: pl.DataFrame, id_maps: IdMaps) -> dict[int, list[int]]:
    by_user = (
        train.sort(["userId", "timestamp"])
        .group_by("userId", maintain_order=True)
        .agg(pl.col("movieId"))
        .to_dict(as_series=False)
    )
    sequences = {}
    for uid, items in zip(by_user["userId"], by_user["movieId"]):
        sequences[uid] = [id_maps.item_id_to_idx[m] for m in items]
    return sequences


def build_windows(sequence_length: int, max_seq_len: int) -> list[tuple[int, int]]:
    window = max_seq_len + 1
    windows = []
    end = sequence_length
    while end >= 2:
        start = max(0, end - window)
        windows.append((start, end))
        if start == 0:
            break
        end -= max_seq_len
    return windows


class SequenceDataset(Dataset):
    def __init__(self, sequences: dict[int, list[int]], max_seq_len: int = 50):
        self.max_seq_len = max_seq_len
        self.sequences = [seq for seq in sequences.values() if len(seq) >= 2]
        self.windows = [
            (seq_idx, start, end)
            for seq_idx, seq in enumerate(self.sequences)
            for start, end in build_windows(len(seq), max_seq_len)
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        seq_idx, start, end = self.windows[i]
        seq = self.sequences[seq_idx][start:end]
        input_seq = seq[:-1]
        target_seq = seq[1:]

        pad_len = self.max_seq_len - len(input_seq)
        input_padded = [PAD_TOKEN] * pad_len + input_seq
        target_padded = [PAD_TOKEN] * pad_len + target_seq

        return (
            torch.tensor(input_padded, dtype=torch.long),
            torch.tensor(target_padded, dtype=torch.long),
        )


class SASRec(nn.Module):
    def __init__(self, num_items: int, max_seq_len: int = 50, embedding_dim: int = 64, num_heads: int = 2, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=PAD_TOKEN)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=embedding_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        B, L = input_seq.shape
        positions = torch.arange(L, device=input_seq.device).unsqueeze(0).expand(B, L)

        x = self.item_embedding(input_seq) + self.position_embedding(positions)
        x = self.dropout(x)

        mask = build_attention_mask(input_seq, self.num_heads)
        hidden = self.encoder(x, mask=mask)
        return self.layer_norm(hidden)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.item_embedding.weight.T


def save_checkpoint(path: Path, model: SASRec, optimizer, epoch: int, id_maps: IdMaps, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "item_id_to_idx": id_maps.item_id_to_idx,
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu"):
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    id_maps = IdMaps(
        item_id_to_idx=ckpt["item_id_to_idx"],
        idx_to_item_id={i: m for m, i in ckpt["item_id_to_idx"].items()},
    )
    model = SASRec(num_items=len(id_maps.item_id_to_idx), **ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["optimizer_state"], ckpt["epoch"], id_maps, ckpt["config"]


def train(
    train_df: pl.DataFrame,
    checkpoint_path: Path,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_seq_len: int = 50,
    embedding_dim: int = 64,
    device: str | None = None,
    run_name: str = "sasrec",
) -> tuple[SASRec, IdMaps, dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    resumed = load_checkpoint(checkpoint_path, device=device)
    if resumed is not None:
        model, optimizer_state, start_epoch, id_maps, config = resumed
        print(f"resuming from checkpoint at epoch {start_epoch}")
        start_epoch += 1
        max_seq_len = config["max_seq_len"]
        unknown_items = set(train_df["movieId"].unique().to_list()) - set(id_maps.item_id_to_idx)
        if unknown_items:
            raise ValueError(
                f"checkpoint at {checkpoint_path} was trained on different data: "
                f"{len(unknown_items)} items in the training frame are unknown to it"
            )
    else:
        id_maps = build_id_maps(train_df)
        config = {"max_seq_len": max_seq_len, "embedding_dim": embedding_dim}
        model = SASRec(num_items=len(id_maps.item_id_to_idx), **config)
        optimizer_state = None
        start_epoch = 0

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        for group in optimizer.param_groups:
            group["lr"] = lr

    sequences = build_user_sequences(train_df, id_maps)
    dataset = SequenceDataset(sequences, max_seq_len=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    final_avg_loss = None
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for input_seq, target_seq in loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            hidden = model(input_seq)
            logits = model.logits(hidden)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target_seq.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        final_avg_loss = avg_loss
        print(f"epoch {epoch}: avg_loss={avg_loss:.4f}")
        save_checkpoint(checkpoint_path, model, optimizer, epoch, id_maps, config)

    if final_avg_loss is not None:
        with log_model_run(
            run_name,
            params={"epochs": epochs, "batch_size": batch_size, "lr": lr, "max_seq_len": max_seq_len,
                    "embedding_dim": config["embedding_dim"], "device": device},
            metrics={"final_train_loss": final_avg_loss},
        ):
            pass

    return model, id_maps, config


def export_embeddings(
    model: SASRec, id_maps: IdMaps, sequences: dict[int, list[int]], output_dir: Path,
    max_seq_len: int = 50, device: str = "cpu", prefix: str = "sasrec", batch_size: int = 512,
) -> None:
    model = model.to(device)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    user_ids = list(sequences)
    user_chunks = []
    with torch.no_grad():
        for start in range(0, len(user_ids), batch_size):
            chunk = user_ids[start:start + batch_size]
            padded = []
            for uid in chunk:
                seq = sequences[uid][-max_seq_len:]
                padded.append([PAD_TOKEN] * (max_seq_len - len(seq)) + seq)
            input_tensor = torch.tensor(padded, dtype=torch.long, device=device)
            hidden = model(input_tensor)
            user_chunks.append(hidden[:, -1, :].cpu().numpy())

        item_ids_sorted = sorted(id_maps.item_id_to_idx, key=lambda m: id_maps.item_id_to_idx[m])
        item_idx_tensor = torch.tensor([id_maps.item_id_to_idx[m] for m in item_ids_sorted], device=device)
        item_emb = model.item_embedding(item_idx_tensor).cpu().numpy()

    embedding_dim = item_emb.shape[1]
    user_emb = np.concatenate(user_chunks) if user_chunks else np.zeros((0, embedding_dim), dtype=np.float32)

    user_df = pl.DataFrame({"userId": user_ids, "embedding": user_emb.tolist()})
    item_df = pl.DataFrame({"movieId": item_ids_sorted, "embedding": item_emb.tolist()})

    user_df.write_parquet(output_dir / f"{prefix}_user_embeddings.parquet")
    item_df.write_parquet(output_dir / f"{prefix}_item_embeddings.parquet")
    print(f"exported {len(user_ids)} user and {len(item_ids_sorted)} item embeddings to {output_dir}")

    n_nan_users = int((~np.isfinite(user_emb)).any(axis=1).sum())
    if n_nan_users > 0:
        print(f"WARNING: {n_nan_users} of {len(user_ids)} exported user embeddings contain NaN or infinity. "
              f"Run scripts/check_embeddings_for_nan.py on the output to identify affected users.")

    n_nan_items = int((~np.isfinite(item_emb)).any(axis=1).sum())
    if n_nan_items > 0:
        print(f"WARNING: {n_nan_items} of {len(item_ids_sorted)} exported item embeddings contain NaN or infinity. "
              f"Run scripts/check_embeddings_for_nan.py on the output to identify affected items.")
