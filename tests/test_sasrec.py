import pytest

torch = pytest.importorskip("torch")

from src.models.sasrec import PAD_TOKEN, SequenceDataset, build_attention_mask, build_windows


def test_attention_mask_is_causal_and_finite_for_padded_queries():
    input_seq = torch.tensor([[PAD_TOKEN, PAD_TOKEN, 5, 6]])
    mask = build_attention_mask(input_seq, num_heads=2)
    assert mask.shape == (2, 4, 4)
    head = mask[0]
    assert torch.isinf(head[0, 1]) and head[0, 1] < 0
    assert head[2, 3] < 0
    assert head[3, 2] == 0 and head[3, 3] == 0
    assert head[3, 0] < 0 and head[3, 1] < 0
    for query in range(4):
        assert torch.isfinite(head[query]).any()
    assert head[0, 0] == 0 and head[1, 1] == 0


def test_attention_mask_has_one_slice_per_head_and_batch_row():
    input_seq = torch.tensor([[PAD_TOKEN, 1, 2], [3, 4, 5]])
    mask = build_attention_mask(input_seq, num_heads=4)
    assert mask.shape == (8, 3, 3)
    assert torch.equal(mask[0], mask[3])
    assert not torch.equal(mask[0], mask[4])


@pytest.mark.parametrize("length", [2, 3, 10, 51, 52, 53, 101, 102, 103, 200])
def test_windows_cover_every_transition_exactly_once(length):
    max_seq_len = 50
    covered = []
    for start, end in build_windows(length, max_seq_len):
        assert 2 <= end - start <= max_seq_len + 1
        covered.extend(range(start, end - 1))
    assert sorted(covered) == list(range(length - 1))


def test_dataset_uses_full_history_and_pads_left():
    sequences = {1: list(range(1, 121)), 2: [7]}
    dataset = SequenceDataset(sequences, max_seq_len=50)
    assert len(dataset) == len(build_windows(120, 50))
    seen_targets = set()
    for i in range(len(dataset)):
        inputs, targets = dataset[i]
        assert inputs.shape == (50,) and targets.shape == (50,)
        seen_targets.update(t for t in targets.tolist() if t != PAD_TOKEN)
    assert seen_targets == set(range(2, 121))
