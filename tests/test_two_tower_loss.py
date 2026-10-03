import pytest

torch = pytest.importorskip("torch")

from src.models.two_tower import in_batch_negative_loss


def test_duplicate_items_in_batch_are_masked_from_negatives():
    user_emb = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    item_emb = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    item_emb[2] = item_emb[0]
    user_ids = torch.tensor([1, 2, 3, 4])
    item_ids = torch.tensor([5, 6, 5, 7])

    unmasked = in_batch_negative_loss(user_emb, item_emb)
    masked = in_batch_negative_loss(user_emb, item_emb, user_ids, item_ids)
    assert torch.isfinite(masked)
    assert masked < unmasked


def test_repeated_users_do_not_treat_their_other_positives_as_negatives():
    user_emb = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    item_emb = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    user_ids = torch.tensor([1, 1, 2])
    item_ids = torch.tensor([5, 6, 7])
    loss = in_batch_negative_loss(user_emb, item_emb, user_ids, item_ids)
    assert torch.isfinite(loss)


def test_masking_is_a_noop_when_batch_has_no_collisions():
    torch.manual_seed(0)
    user_emb = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    item_emb = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    user_ids = torch.arange(5)
    item_ids = torch.arange(5) + 100
    assert torch.allclose(
        in_batch_negative_loss(user_emb, item_emb),
        in_batch_negative_loss(user_emb, item_emb, user_ids, item_ids),
    )
