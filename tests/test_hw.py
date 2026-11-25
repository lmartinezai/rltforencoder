"""Hardware study helpers: quantisation, the masked-logits wrapper, depth fits. CPU, tiny."""

import pytest
import torch

import hw_report as hr
import ov_export
import quant_eval as qe
from arch import BortConfig, BortForMaskedLM

H = 16
TINY = dict(hidden_size=H, intermediate_size=32, num_attention_heads=2, vocab_size=120, pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1,
            sep_token_id=2, local_attention=8, global_rope_theta=1000.0, local_rope_theta=1000.0)


def test_quantisation_error_is_bounded_by_half_a_step_and_levels_are_few():
    w = torch.randn(8, 128)
    q8, q4 = qe.fake_quantise(w, "int8"), qe.fake_quantise(w, "int4")
    assert ((q8 - w).abs().amax(1) <= w.abs().amax(1) / 127 / 2 + 1e-6).all()
    groups = q4.reshape(8, 2, 64)
    assert all(g.unique().numel() <= 16 for row in groups for g in row)
    assert ((q4 - w).abs().reshape(8, 2, 64).amax(2) <= w.abs().reshape(8, 2, 64).amax(2) / 7 / 2 + 1e-6).all()


def test_only_block_linears_are_quantised_and_the_shared_block_once():
    torch.manual_seed(0)
    model = BortForMaskedLM(BortConfig(**TINY, prelude_layers=1, looped_layers=2, coda_layers=1)).eval()
    decoder, embedding = model.decoder.weight.clone(), model.model.embeddings.tok_embeddings.weight.clone()
    block = model.model.layers[1].mlp.Wi.weight.clone()
    n = qe.quantise_encoder(model, "int4")
    assert n == 4 * 4 and torch.equal(model.decoder.weight, decoder) and torch.equal(model.model.embeddings.tok_embeddings.weight, embedding)
    assert not torch.equal(model.model.layers[1].mlp.Wi.weight, block)


def test_masked_logits_wrapper_matches_the_model_at_the_masked_positions():
    torch.manual_seed(0)
    model = BortForMaskedLM(BortConfig(**TINY, prelude_layers=1, looped_layers=2, coda_layers=1)).eval()
    ids = torch.randint(3, 120, (2, 20))
    wrapper = ov_export.MaskedLogits(model, 20, depth=3)
    with torch.no_grad():
        out = wrapper(ids)
        ref = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), recurrence_depth=3).last_hidden_state[:, :6]
        ref = model.decoder(model.head(ref))
    assert out.shape == (2, 6, 120) and torch.allclose(out, ref, atol=1e-5)


def test_depth_fit_recovers_a_line_and_fills_untimed_depths():
    icpt, slope, residual = hr.fit_depth([(1, 12.0), (4, 30.0), (8, 54.0)])
    assert slope == pytest.approx(6.0) and icpt == pytest.approx(6.0) and residual < 1e-9
    rows = [{"backend": "openvino", "device": "NPU", "model": "m", "seq": 512, "batch": 1, "depth": d, "p50_ms": 6 + 6 * d} for d in (1, 4, 8)]
    table = hr.latency_table(rows)[("openvino", "NPU", "m", 512, 1)]
    assert table[4] == (30, True) and table[5][1] is False and table[5][0] == pytest.approx(36)
