import pytest
import torch
from torch.nn import functional as F

from targets import decode, full_kl, make_targets, target_kl

V, H, N = 300, 32, 40


@pytest.fixture()
def setup():
    torch.manual_seed(0)
    W, b = torch.randn(V, H) * 0.5, torch.randn(V) * 0.1
    z = (torch.randn(N, H) * 2).bfloat16().float()  # bf16-valued, like the teacher's real hidden state
    s = decode(z, W, b) + torch.randn(N, V) * 1.5   # a student that is wrong by a fair amount
    return z, W, b, s


def per_position_full_kl(s, z, W, b, T):
    pt = F.log_softmax(decode(z, W, b) / T, -1)
    return (pt.exp() * (pt - F.log_softmax(s / T, -1))).sum(-1)


@pytest.mark.parametrize("T", [1.0, 2.0, 4.0])
def test_k_equals_vocab_recovers_full_kl(setup, T):
    z, W, b, s = setup
    tg = make_targets(z, W, b, T, k=V, p=0.0)  # p=0 -> nothing is dense
    assert not tg.dense.any() and tg.tail.max() < 1e-5
    assert torch.allclose(target_kl(s, tg, W, b, T), per_position_full_kl(s, z, W, b, T), atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize("k", [4, 16, 64])
def test_bucketed_kl_never_exceeds_full_kl(setup, k):
    """Lumping outcomes can only lose information (log-sum inequality), so the truncated KL is a lower bound."""
    z, W, b, s = setup
    tg = make_targets(z, W, b, 2.0, k=k, p=0.0)
    bucket, full = target_kl(s, tg, W, b, 2.0), per_position_full_kl(s, z, W, b, 2.0)
    assert (bucket <= full + 1e-5).all() and (bucket >= 0).all()


def test_identical_student_gives_zero_kl(setup):
    z, W, b, _ = setup
    tg = make_targets(z, W, b, 2.0, k=16, p=0.0)
    assert target_kl(decode(z, W, b), tg, W, b, 2.0).abs().max() < 1e-5


def test_dense_rule_and_fp16_hidden_is_exact_for_bf16_values(setup):
    z, W, b, s = setup
    k, p, T = 8, 0.9, 2.0
    lp = F.log_softmax(decode(z, W, b) / T, -1)
    covered = lp.topk(k, -1).values.exp().sum(-1)
    tg = make_targets(z, W, b, T, k, p)
    assert torch.equal(tg.dense, covered < p) and tg.dense.any() and (~tg.dense).any()  # both paths exercised
    assert tg.ids.shape == (int((~tg.dense).sum()), k) and tg.hidden.shape == (int(tg.dense.sum()), H)
    assert torch.equal(tg.hidden.float(), z[tg.dense])  # fp16 holds bf16 values exactly -> lossless
    # hybrid targets: dense rows are exact full KL, sparse rows are the bucketed lower bound
    kl, full = target_kl(s, tg, W, b, T), per_position_full_kl(s, z, W, b, T)
    assert torch.allclose(kl[tg.dense], full[tg.dense], atol=1e-5)
    assert (kl[~tg.dense] <= full[~tg.dense] + 1e-5).all()


def test_stored_log_probs_rounded_to_fp16_barely_change_kl(setup):
    z, W, b, s = setup
    tg = make_targets(z, W, b, 2.0, k=32, p=0.0)
    ref = target_kl(s, tg, W, b, 2.0)
    rounded = tg._replace(logp=tg.logp.half().float())
    assert torch.allclose(target_kl(s, rounded, W, b, 2.0), ref, rtol=5e-3, atol=5e-4)


def test_gradients_flow_to_student_only_and_are_finite(setup):
    z, W, b, s = setup
    s = s.clone().requires_grad_(True)
    tg = make_targets(z, W, b, 2.0, k=8, p=0.9)
    target_kl(s, tg, W, b, 2.0).mean().backward()
    assert torch.isfinite(s.grad).all() and s.grad.abs().sum() > 0


def test_gradient_direction_matches_full_kl(setup):
    z, W, b, s = setup
    g = []
    for kl in (lambda s: target_kl(s, make_targets(z, W, b, 2.0, 64, 0.0), W, b, 2.0).mean(),
               lambda s: full_kl(s, z, W, b, 2.0)):
        s_ = s.clone().requires_grad_(True)
        kl(s_).backward()
        g.append(s_.grad.flatten())
    assert F.cosine_similarity(g[0], g[1], dim=0) > 0.97


def test_full_kl_matches_reference_formula(setup):
    z, W, b, s = setup
    assert torch.allclose(full_kl(s, z, W, b, 2.0), per_position_full_kl(s, z, W, b, 2.0).mean(), atol=1e-5)


def test_chunking_does_not_change_targets(setup):
    z, W, b, _ = setup
    a, c = make_targets(z, W, b, 2.0, 16, 0.9, chunk=7), make_targets(z, W, b, 2.0, 16, 0.9, chunk=10_000)
    assert torch.equal(a.ids, c.ids) and torch.equal(a.dense, c.dense) and torch.equal(a.hidden, c.hidden)
    assert torch.allclose(a.logp, c.logp, atol=1e-5) and torch.allclose(a.tail, c.tail, atol=1e-5)


def test_k_zero_is_dense_only_and_equals_full_kl(setup):
    z, W, b, s = setup
    tg = make_targets(z, W, b, 2.0, k=0, p=0.99)
    assert tg.dense.all() and tg.ids.numel() == 0 and tg.hidden.shape == (N, H)
    assert torch.allclose(target_kl(s, tg, W, b, 2.0), per_position_full_kl(s, z, W, b, 2.0), atol=1e-5)
    assert torch.allclose(target_kl(s, tg, W, b, 4.0).mean(), full_kl(s, z, W, b, 4.0), atol=1e-5)  # T chosen later


def test_decode_is_fp32_even_under_autocast(setup):
    z, W, b, _ = setup
    ref = decode(z, W, b)
    with torch.autocast("cpu", torch.bfloat16):
        got = decode(z, W, b)
    assert got.dtype == torch.float32 and torch.equal(got, ref)


def test_confident_student_tail_term_is_exact(setup):
    """A student with <1e-6 mass outside the stored top-K: the bucketed KL must match a float64 reference (the old
    1 - sum(q) tail floored at 1e-6 and lost both value and gradient here)."""
    z, W, b, _ = setup
    k = 4
    tg = make_targets(z, W, b, 1.0, k=k, p=0.0)
    s = torch.full((N, V), -5.0).scatter_(1, tg.ids[:, :1], 25.0)  # tail mass ~1e-9
    s.requires_grad_(True)
    kl = target_kl(s, tg, W, b, 1.0)
    lq = F.log_softmax(s.double(), -1)
    q_top, q_tail = lq.gather(1, tg.ids), (1 - lq.gather(1, tg.ids).exp().sum(-1)).clamp_min(1e-300)
    p, pt = tg.logp.double().exp(), tg.tail.double()
    ref = (p * (tg.logp.double() - q_top)).sum(-1) + pt * (pt.clamp_min(1e-30).log() - q_tail.log())
    assert torch.allclose(kl.double(), ref, rtol=2e-2, atol=1e-3)
    kl.sum().backward()
    assert torch.isfinite(s.grad).all()


def test_mixed_batch_matches_row_by_row(setup):
    z, W, b, s = setup
    tg = make_targets(z, W, b, 2.0, k=8, p=0.9)
    assert tg.dense.any() and (~tg.dense).any()
    whole = target_kl(s, tg, W, b, 2.0)
    for i in range(N):
        one = make_targets(z[i:i + 1], W, b, 2.0, k=8, p=0.9)
        assert torch.allclose(target_kl(s[i:i + 1], one, W, b, 2.0), whole[i:i + 1], atol=1e-5)


def test_extremely_confident_student_keeps_tail_loss_and_gradient():
    W, b = torch.eye(4), torch.zeros(4)
    z = torch.tensor([[1., 2., 3., 4.]])
    tg = make_targets(z, W, b, 1.0, k=1, p=0.0)
    s = torch.tensor([[0., 0., 0., 100.]], requires_grad=True)
    actual = target_kl(s, tg, W, b, 1.0).sum()
    logq = F.log_softmax(s.double(), -1)
    pt = tg.tail.double()
    ref = (tg.logp.double().exp() * (tg.logp.double() - logq[:, 3:])).sum() + \
        (pt * (pt.log() - logq[:, :3].logsumexp(-1))).sum()
    g_actual = torch.autograd.grad(actual, s, retain_graph=True)[0]
    g_ref = torch.autograd.grad(ref, s)[0]
    assert torch.allclose(actual.double(), ref, atol=1e-5)
    assert torch.allclose(g_actual, g_ref, atol=1e-6)


def test_full_vocab_zero_probability_after_fp16_storage_has_finite_kl():
    W, b = torch.eye(4), torch.zeros(4)
    z = torch.tensor([[0., -70000., -80000., -90000.]])
    tg = make_targets(z, W, b, 1.0, k=4, p=0.0)
    tg = tg._replace(logp=tg.logp.half().float())
    s = torch.zeros(1, 4, requires_grad=True)
    loss = target_kl(s, tg, W, b, 1.0).sum()
    loss.backward()
    assert torch.allclose(loss, torch.tensor(4.).log())
    assert torch.isfinite(s.grad).all()


def test_no_masked_positions_fail_clearly():
    W, b, z = torch.eye(4), torch.zeros(4), torch.empty(0, 4)
    tg = make_targets(z, W, b, 1.0, k=0, p=0.99)
    with pytest.raises(ValueError, match="no masked positions"):
        target_kl(z, tg, W, b, 1.0)
    with pytest.raises(ValueError, match="no masked positions"):
        full_kl(z, z, W, b, 1.0)


@pytest.mark.parametrize("value", [100000., 1e-9, 1e-6])
def test_dense_bf16_values_preserve_their_range(value):
    W, b = torch.eye(4), torch.zeros(4)
    z = torch.tensor([[value, 0., 0., 0.]], dtype=torch.bfloat16)
    tg = make_targets(z, W, b, 1.0, k=0, p=0.99)
    assert tg.hidden.dtype == torch.bfloat16 and torch.equal(tg.hidden, z)
    assert torch.isfinite(target_kl(torch.zeros(1, 4), tg, W, b, 1.0)).all()


def test_hidden_states_that_cannot_be_stored_are_rejected(setup):
    _, W, b, _ = setup
    bad_bf16 = torch.full((2, H), float("nan")).bfloat16()
    with pytest.raises(ValueError, match="non-finite"):
        make_targets(bad_bf16, W, b, 2.0, k=0, p=1.0)
    with pytest.raises(ValueError, match="overflow or underflow"):
        make_targets(torch.full((2, H), 1e6), W, b, 2.0, k=0, p=1.0)  # fp32 beyond fp16 range
    with pytest.raises(ValueError, match="overflow or underflow"):
        make_targets(torch.full((2, H), 1e-9), W, b, 2.0, k=0, p=1.0)  # fp32 that rounds to zero in fp16
    ok = make_targets(torch.randn(2, H).bfloat16(), W, b, 2.0, k=0, p=1.0)
    assert ok.hidden.dtype == torch.bfloat16  # BF16 is preserved exactly, not squeezed through fp16
