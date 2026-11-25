"""Compact teacher targets for masked-LM distillation.

At every supervised position the teacher distribution (softmax of logits / T) is kept as either
  sparse: its top-K token ids, their log-probs, and the tail mass 1 - sum(p_topk), or
  dense : the post-head hidden state z in bf16 or fp16 (logits = z @ W.T + b), used when top-K covers < P of the mass
          (K=0: always dense). BF16 states retain their full range and precision on disk and on the wire.
The student's KL is then taken over K+1 outcomes: the K stored tokens plus one lumped "everything else" bucket,
so only the teacher side is truncated; the student's full-vocab softmax is already computed.
"""
from typing import NamedTuple

import torch
from torch.nn import functional as F


class Targets(NamedTuple):
    ids: torch.Tensor     # [Ns, K] int64    top-K token ids        (sparse positions only)
    logp: torch.Tensor    # [Ns, K] float    log p_T(token)         (sparse positions only)
    tail: torch.Tensor    # [Ns]    float    teacher mass outside the top-K
    dense: torch.Tensor   # [N]     bool     position stores a hidden state instead
    hidden: torch.Tensor  # [Nd, H] bf16/fp16 post-head hidden state (dense positions only)


def decode(z, W, b):
    """Canonical teacher logits: fp32 decoder applied to the post-head hidden state (bf16- or fp16-valued).
    Autocast is disabled so extraction, stored and live teachers all produce identical numerics."""
    with torch.autocast(device_type=z.device.type, enabled=False):
        return F.linear(z.float(), W.float(), b.float())


def _dense_hidden(z):
    """Preserve BF16 exactly; reject FP32 values that overflow or underflow FP16."""
    if z.dtype == torch.bfloat16:
        if not torch.isfinite(z).all():
            raise ValueError("teacher hidden states contain non-finite values")
        return z
    h = z.half()
    if not torch.isfinite(h).all() or ((z != 0) & (h == 0)).any():
        raise ValueError("teacher hidden states overflow or underflow fp16")
    return h


@torch.no_grad()
def make_targets(z, W, b, T, k, p, chunk=2048):
    """Teacher side. z: [N, H] post-head hidden. Chunked so memory is independent of N."""
    if k == 0:  # dense only: the hidden state is the whole target (valid for any T, K, P later)
        e = z.new_empty(0)
        return Targets(e.long().reshape(0, 0), e.float().reshape(0, 0), e.float(), z.new_ones(len(z), dtype=torch.bool),
                       _dense_hidden(z))
    ids, logp, tail = [], [], []
    for zc in z.split(chunk):
        top, idx = F.log_softmax(decode(zc, W, b) / T, -1).topk(k, -1)
        ids.append(idx), logp.append(top), tail.append((1 - top.exp().sum(-1)).clamp_min(0))
    ids, logp, tail = torch.cat(ids), torch.cat(logp), torch.cat(tail)
    dense = tail > 1 - p  # top-K covers < P of the mass  <=>  the top-P set has more than K tokens
    sparse = ~dense
    return Targets(ids[sparse], logp[sparse], tail[sparse], dense, _dense_hidden(z[dense]))


def _kl_sparse(lsm, ids, logp, tail, rows):
    """KL(teacher || student) over K+1 outcomes for the sparse rows of the student's log-softmax `lsm`."""
    p = logp.exp()
    # With the full vocabulary there is no tail, even if fp32 roundoff left a small positive residual.
    full_vocab = ids.shape[1] == lsm.shape[1]
    if full_vocab:
        tail = torch.zeros_like(tail)
    norm = p.sum(-1) + tail  # ~1; absorbs fp16 rounding of stored log-probs
    lq = lsm[rows[:, None], ids]
    # A serialized -inf represents zero teacher probability: 0 * log(0/q) is zero.
    safe_logp = torch.where(torch.isneginf(logp), torch.zeros_like(logp), logp)
    top_kl = ((p / norm[:, None]) * (safe_logp - norm.log()[:, None] - lq)).sum(-1)
    if full_vocab:
        return top_kl
    log_q_tail = lsm[rows].scatter(1, ids, -torch.inf).logsumexp(-1)
    p_tail = tail / norm
    return top_kl + p_tail * (p_tail.clamp_min(1e-30).log() - log_q_tail)


def target_kl(student_logits, tg, W, b, T):
    """Per-position KL(teacher || student) at temperature T from stored targets (no T^2 factor). -> [N]."""
    if student_logits.shape[0] == 0:
        raise ValueError("distillation batch has no masked positions")
    lsm = F.log_softmax(student_logits.float() / T, -1)
    n_sparse, n_dense = tg.ids.shape[0], tg.hidden.shape[0]  # shapes, not masks: no device sync
    if n_sparse == 0:  # all dense (every streamed batch): no [N, V] gather copy of the student's log-softmax
        pt = F.log_softmax(decode(tg.hidden, W, b) / T, -1)
        return (pt.exp() * (pt - lsm)).sum(-1)
    kl = lsm.new_zeros(lsm.shape[0])
    sparse_rows, dense_rows = (~tg.dense).nonzero().squeeze(-1), tg.dense.nonzero().squeeze(-1)
    kl = kl.index_put((sparse_rows,), _kl_sparse(lsm, tg.ids, tg.logp, tg.tail, sparse_rows))
    if n_dense:
        pt = F.log_softmax(decode(tg.hidden, W, b) / T, -1)
        kl = kl.index_put((dense_rows,), (pt.exp() * (pt - lsm[dense_rows])).sum(-1))
    return kl


def full_kl(student_logits, z, W, b, T):
    """Reference: full-vocabulary forward KL(teacher || student) at temperature T. -> scalar (batchmean)."""
    if student_logits.shape[0] == 0:
        raise ValueError("distillation batch has no masked positions")
    t_lsm = F.log_softmax(decode(z, W, b) / T, -1)
    return F.kl_div(F.log_softmax(student_logits.float() / T, -1), t_lsm.detach(), log_target=True,
                    reduction="batchmean")


@torch.no_grad()
def teacher_hidden(teacher, batch):
    """Teacher's post-head hidden state z at the labelled positions: [N, H] (its own dtype, e.g. bf16)."""
    h = teacher.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    return teacher.head(h[batch["labels"] != -100])
