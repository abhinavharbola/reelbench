"""
SASRec: self-attentive sequential recommendation. Next-item prediction over
each user's chronological interaction sequence, causal (position i can only
attend to positions <= i).

Same checkpointing rationale as two_tower.py: trains on Colab/Kaggle GPU,
free-tier sessions can be cut off mid-run, so save/resume every epoch
rather than assuming one uninterrupted session.

Index convention: item ids are shifted by +1 so that 0 is reserved as the
padding token (0 never occurs as a real item id after this shift).
"""

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
    """Combined causal + padding mask, as one explicit additive float
    tensor of shape (B*num_heads, L, L) -- built so every row, including
    padding query positions, always keeps its own diagonal entry unmasked.

    This fixes a real NaN bug (confirmed empirically, not theoretical):
    the previous version passed `mask=causal_bool` and
    `src_key_padding_mask=padding_bool` separately and let PyTorch combine
    them. For a padding query position, that combination excludes every
    key including the position's own diagonal entry, so its attention row
    is entirely -inf and softmax produces NaN there -- expected and
    harmless on its own, since nothing reads a padding position's output.
    But with 2+ encoder layers (this model uses 2), that NaN becomes a
    *key's value* for real, non-padded query positions in the next layer.
    Even though the attention *weight* assigned to that key is correctly
    ~0 (properly masked), `0 * NaN = NaN` in IEEE floating point, so the
    weighted sum -- and with it every real position's hidden state --
    still comes out NaN. This reproduces for literally any sequence
    shorter than max_seq_len (i.e. nearly every user), confirmed by
    feeding sequences of every length 1..50 through the model and checking
    the last (real) position's hidden state.

    The fix: always allow a query position to attend to itself, regardless
    of padding. This is a no-op for real positions (self-attendance was
    never excluded for them in the first place -- confirmed bit-identical
    output vs. the old masking approach whenever no NaN was present to
    compare against), and it prevents a padding position's row from ever
    being fully masked, so it never produces NaN, so there is nothing left
    for later layers to pick up.
    """
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
    item_id_to_idx: dict  # real movieId -> shifted idx (1..num_items), 0 reserved for padding
    idx_to_item_id: dict


def build_id_maps(train: pl.DataFrame) -> IdMaps:
    item_ids = train["movieId"].unique().sort().to_list()
    item_id_to_idx = {m: i + 1 for i, m in enumerate(item_ids)}  # +1: reserve 0 for padding
    return IdMaps(item_id_to_idx=item_id_to_idx, idx_to_item_id={i: m for m, i in item_id_to_idx.items()})


def build_user_sequences(train: pl.DataFrame, id_maps: IdMaps) -> dict[int, list[int]]:
    """userId -> chronological list of shifted item indices."""
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


class SequenceDataset(Dataset):
    """Each sample: one user's sequence, left-padded/truncated to max_seq_len.
    Input is sequence[:-1], target is sequence[1:] (next-item prediction at
    every position), padding positions excluded from loss via ignore_index."""

    def __init__(self, sequences: dict[int, list[int]], max_seq_len: int = 50):
        self.max_seq_len = max_seq_len
        self.samples = [seq for seq in sequences.values() if len(seq) >= 2]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        seq = self.samples[i][-(self.max_seq_len + 1):]  # keep most recent activity
        input_seq = seq[:-1]
        target_seq = seq[1:]

        pad_len = self.max_seq_len - len(input_seq)
        input_padded = [PAD_TOKEN] * pad_len + input_seq
        target_padded = [PAD_TOKEN] * pad_len + target_seq  # PAD_TOKEN doubles as ignore_index

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
        """input_seq: (B, L) shifted item indices, 0 = padding.
        Returns hidden states (B, L, D); logits = hidden @ item_embedding.weight.T"""
        B, L = input_seq.shape
        positions = torch.arange(L, device=input_seq.device).unsqueeze(0).expand(B, L)

        x = self.item_embedding(input_seq) + self.position_embedding(positions)
        x = self.dropout(x)

        # See build_attention_mask's docstring: this must be one explicit
        # additive mask, not a separate causal `mask` + `src_key_padding_mask`
        # pair -- that combination produces NaN at every non-fully-padded
        # sequence length once you have 2+ encoder layers (confirmed
        # empirically, this isn't a hypothetical edge case).
        mask = build_attention_mask(input_seq, self.num_heads)
        hidden = self.encoder(x, mask=mask)
        return self.layer_norm(hidden)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Tied-weight projection back to vocab: (B, L, num_items+1)."""
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
    # explicit weights_only=False: checkpoints store id_maps/config dicts
    # alongside tensors, not just tensors, so the safer weights_only=True
    # default (which PyTorch is moving toward) can't load this file --
    # being explicit here avoids a silent break on a future torch upgrade
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
) -> tuple[SASRec, IdMaps, dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    resumed = load_checkpoint(checkpoint_path, device=device)
    if resumed is not None:
        model, optimizer_state, start_epoch, id_maps, config = resumed
        print(f"resuming from checkpoint at epoch {start_epoch}")
        start_epoch += 1
        max_seq_len = config["max_seq_len"]  # must match the resumed model's shapes
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
            logits = model.logits(hidden)  # (B, L, num_items+1)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target_seq.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        final_avg_loss = avg_loss
        print(f"epoch {epoch}: avg_loss={avg_loss:.4f}")
        save_checkpoint(checkpoint_path, model, optimizer, epoch, id_maps, config)

    # log once, after training completes -- per spec, only the final
    # checkpoint per model is logged, never intermediate epochs
    if final_avg_loss is not None:
        with log_model_run(
            "sasrec",
            params={"epochs": epochs, "batch_size": batch_size, "lr": lr, "max_seq_len": max_seq_len,
                    "embedding_dim": embedding_dim, "device": device},
            metrics={"final_train_loss": final_avg_loss},
        ):
            pass

    # return config alongside the model: if this call resumed from a
    # checkpoint, max_seq_len above was overridden from the checkpoint's
    # saved config and may not match the max_seq_len argument the caller
    # passed in. Callers must use this returned config (not their own CLI
    # arg) for export_embeddings(), or a resume with a mismatched
    # --max-seq-len crashes export with "IndexError: index out of range
    # in self" from position_embedding -- confirmed by reproducing a
    # train(max_seq_len=8) checkpoint, then resuming with
    # train(max_seq_len=50) and exporting with max_seq_len=50.
    return model, id_maps, config


def export_embeddings(
    model: SASRec, id_maps: IdMaps, sequences: dict[int, list[int]], output_dir: Path,
    max_seq_len: int = 50, device: str = "cpu",
) -> None:
    """User embedding = hidden state at the last real (non-padded) position
    of their sequence -- this is SASRec's standard "next item" representation.
    Item embeddings come straight from the tied embedding table."""
    model = model.to(device)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    user_ids, user_embs = [], []
    with torch.no_grad():
        for uid, seq in sequences.items():
            seq = seq[-max_seq_len:]
            pad_len = max_seq_len - len(seq)
            padded = [PAD_TOKEN] * pad_len + seq
            input_tensor = torch.tensor([padded], dtype=torch.long, device=device)
            hidden = model(input_tensor)  # (1, L, D)
            last_hidden = hidden[0, -1, :].cpu().numpy()  # last position = next-item representation
            user_ids.append(uid)
            user_embs.append(last_hidden)

        item_ids_sorted = sorted(id_maps.item_id_to_idx, key=lambda m: id_maps.item_id_to_idx[m])
        item_idx_tensor = torch.tensor([id_maps.item_id_to_idx[m] for m in item_ids_sorted], device=device)
        item_emb = model.item_embedding(item_idx_tensor).cpu().numpy()

    user_df = pl.DataFrame({"userId": user_ids, "embedding": [e.tolist() for e in user_embs]})
    item_df = pl.DataFrame({"movieId": item_ids_sorted, "embedding": item_emb.tolist()})

    user_df.write_parquet(output_dir / "sasrec_user_embeddings.parquet")
    item_df.write_parquet(output_dir / "sasrec_item_embeddings.parquet")
    print(f"exported {len(user_ids)} user and {len(item_ids_sorted)} item embeddings to {output_dir}")

    # Safety net: catches a NaN regression immediately at export time
    # instead of it surfacing three steps downstream as an opaque FAISS/
    # ranker crash (a NaN embedding makes FAISS return zero search results,
    # which build_training_table now skips silently -- fine for one bad
    # user, but worth a loud warning if it happens at all).
    n_nan_users = int(np.isnan(np.array(user_embs)).any(axis=1).sum())
    if n_nan_users > 0:
        print(f"WARNING: {n_nan_users} of {len(user_ids)} exported user embeddings contain NaN. "
              f"Run scripts/check_embeddings_for_nan.py on the output to identify affected users.")

    # item embeddings all go into the shared FAISS index every user's
    # query searches against, not just one user's query -- worth checking
    # separately from the user-embedding check above.
    n_nan_items = int(np.isnan(item_emb).any(axis=1).sum())
    if n_nan_items > 0:
        print(f"WARNING: {n_nan_items} of {len(item_ids_sorted)} exported item embeddings contain NaN. "
              f"Run scripts/check_embeddings_for_nan.py on the output to identify affected items.")
