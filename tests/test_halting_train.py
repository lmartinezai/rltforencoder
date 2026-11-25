"""halting_train: head fitting on synthetic shards with a known answer, folded raw-state weights, exact scoring, baking into a checkpoint. CPU, tiny."""
import pytest
import torch

import halting
import halting_train as ht
from arch import BortConfig, BortForMaskedLM

D, H = 8, 16
TINY = dict(hidden_size=H, intermediate_size=32, num_attention_heads=2, vocab_size=120, pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1,
            sep_token_id=2, local_attention=8, global_rope_theta=1000.0, local_rope_theta=1000.0)


def synthetic(n=600, k=4, seed=0):
    """Sequences whose state feature 0 says how many iterations are still worth running: loss(d) = base + 0.1 * max(0, need - d)."""
    g = torch.Generator().manual_seed(seed)
    need = torch.randint(1, D + 1, (n,), generator=g)
    depth = torch.arange(1, D + 1)
    loss = 2.0 + 0.1 * (need[:, None] - depth[None, :]).clamp_min(0).float()  # best from depth `need` on
    state = torch.randn(n, D, H, generator=g) * 0.05
    state[:, :, 0] = (need[:, None] - depth[None, :]).float()  # the remaining need, visible in the state
    tok_loss = loss[:, :, None].repeat(1, 1, k)
    tok_state = state[:, :, None, :].repeat(1, 1, k, 1)
    return {"seq_state": state.half(), "seq_loss": loss, "n_voters": torch.full((n,), 100), "tok_state": tok_state.half(), "tok_loss": tok_loss}


def test_the_sequence_head_learns_a_visible_signal_and_its_folded_weights_apply_to_raw_states():
    data = synthetic()
    saved, head, mean, std = ht.fit_sequence_head(data, epochs=20, lr=0.05)
    assert saved["weight"].shape == (1, H) and saved["bias"].shape == (1,) and saved["level"] == "sequence"
    x = data["seq_state"].float().reshape(-1, H)
    with torch.no_grad():
        standardised = head((x - mean) / std)["logits"]
    raw = x @ saved["weight"].squeeze(0) + saved["bias"]
    assert torch.allclose(standardised, raw, atol=1e-3)  # the folded head on raw states == the trained head on standardised states
    y = ht.targets(data["seq_loss"], "v1", ht.SEQ_TOLERANCE).reshape(-1)
    assert ((raw > 0) == (y > 0.5)).float().mean() > 0.9  # feature 0 (the remaining need) separates the labels


def test_scoring_is_exact_and_a_perfect_signal_reaches_the_ceiling_that_no_head_can_pass():
    data = synthetic()
    saved, *_ = ht.fit_sequence_head(data, epochs=20, lr=0.05)
    s = ht.score_sequence_head(saved, data)
    assert len(s["fixed"]) == D and s["fixed"][-1] == pytest.approx(2.0)
    assert s["head_best_gain"] <= s["ceiling_best"] + 1e-6 and s["head_best_gain"] > 0.02  # a real gain, bounded by the hindsight ceiling
    for point in s["operating_points"]:
        assert 1 <= point["loops"] <= D and point["gain"] == pytest.approx(point["fixed_loss"] - point["loss"])


def test_token_rows_order_and_labels_and_the_token_head_is_scored_on_held_out_rows():
    data = synthetic(n=50, k=3)
    x, y = ht.token_rows(data)
    assert x.shape == (50 * D * 3, H) and y.shape == (50 * D * 3,)
    assert torch.equal(x.reshape(50, D, 3, H)[7, 2, 1], data["tok_state"][7, 2, 1].float())  # rows are ordered (sequence, depth, token)
    assert y.reshape(50, D, 3)[:, -1].min() == 1.0  # the deepest depth is within tolerance of the best for these losses
    head, accuracy, rate = ht.fit_token_head(synthetic(n=400, k=3), synthetic(n=100, k=3, seed=1), epochs=3, lr=0.05)
    assert head["level"] == "token" and 0.0 <= rate <= 1.0 and accuracy > 0.8


def test_baking_puts_the_trained_head_into_a_copy_and_leaves_the_forward_pass_alone(tmp_path):
    torch.manual_seed(0)
    model = BortForMaskedLM(BortConfig(**TINY, prelude_layers=1, looped_layers=2, coda_layers=1)).eval()
    model.save_pretrained(tmp_path / "ckpt")
    head = {"weight": torch.randn(1, H), "bias": torch.tensor([0.5]), "level": "sequence", "tolerance": 0.02}
    torch.save(head, tmp_path / "head.pt")
    ht.bake(tmp_path / "ckpt", tmp_path / "head.pt", tmp_path / "ckpt-halting")  # raises if the forward pass or the head changed
    baked = BortForMaskedLM.from_pretrained(tmp_path / "ckpt-halting")
    assert torch.equal(baked.model.halting_head.weight.detach(), head["weight"]) and baked.model.halting_head.bias.item() == pytest.approx(0.5)
    original = BortForMaskedLM.from_pretrained(tmp_path / "ckpt")
    assert not torch.equal(original.model.halting_head.weight.detach(), head["weight"])  # the original checkpoint is untouched
    assert halting.loop_bounds(baked) == (1, 2)
