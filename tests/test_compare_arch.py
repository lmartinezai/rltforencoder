"""The non-recurrent baseline: same blocks as Bort, run once, trained through distill.py unchanged."""
import pytest
import torch

import compare_arch
import distill
from arch import BortConfig, BortForMaskedLM, StealEmbeddingsFromModernBert

TINY = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_attention_heads=4, max_position_embeddings=64,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2, global_attn_every_n_layers=3)


def tiny_pair():
    bort_config = BortConfig(**TINY)
    bort_config.sparse_prediction = True
    return bort_config, compare_arch.flat_config(config=bort_config)


def test_flat_config_keeps_bort_layout_and_drops_the_recurrence_settings():
    bort_config, flat = tiny_pair()
    assert flat.num_hidden_layers == 8 and flat.layer_types == bort_config.layer_types
    assert [i for i, t in enumerate(flat.layer_types) if t == "full_attention"] == [0, 2, 5, 7]
    assert flat.model_type == "modernbert" and not hasattr(flat, "recurrence_depth") and flat.sparse_prediction
    assert (flat.hidden_size, flat.intermediate_size, flat.num_attention_heads) == (32, 64, 4)


def test_flat_student_is_bort_without_the_merge_and_runs_as_distill_step_calls_it():
    bort_config, flat_config = tiny_pair()
    torch.manual_seed(0)
    weight = torch.randn(64, 32)
    flat = compare_arch.build_flat_student(None, weight, flat_config)
    bort = BortForMaskedLM(bort_config)
    trainable = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    StealEmbeddingsFromModernBert(bort, distill._EmbeddingSource(weight), freeze=True)
    merge = sum(p.numel() for p in bort.model.merge.parameters())  # the halting head is frozen, so it is not trained either way
    assert trainable(bort) - merge == trainable(flat)
    # embeddings: copied, frozen and tied to the decoder
    assert torch.equal(flat.model.embeddings.tok_embeddings.weight, weight) and not flat.model.embeddings.tok_embeddings.weight.requires_grad
    assert flat.decoder.weight is flat.model.embeddings.tok_embeddings.weight
    ids = torch.randint(5, 64, (2, 16))
    labels = torch.full_like(ids, -100)
    labels[:, 3] = ids[:, 3]
    out = flat(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, recurrence_depth=5)  # the depth is ignored
    assert out.logits.shape == (2, 64) and torch.isfinite(out.loss)
    for depth in (1, 8):
        again = flat(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, recurrence_depth=depth)
        torch.testing.assert_close(again.logits, out.logits)
    out.loss.backward()
    assert all(p.grad is not None for n, p in flat.named_parameters() if p.requires_grad and "tok_embeddings" not in n)


def test_main_trains_through_distill_with_one_fixed_depth(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(distill, "train", lambda args, build: seen.update(args=args, build=build))
    compare_arch.main(["--output_dir", str(tmp_path), "--max_depth", "8", "--eval_depths", "1", "4", "8", "--max_steps", "3"])
    args = seen["args"]
    assert args.max_depth == 1 and args.eval_depths == [1] and args.max_steps == 3
    assert seen["build"] is compare_arch.build_flat_student  # passed to distill.train, which is left as it was
    monkeypatch.undo()  # the real distill.train hands the builder it was given to _train, and Bort's when it was given none
    monkeypatch.setattr(distill, "_train", lambda args, callbacks, trainer_class, build, stack: build)
    plain = distill.DistillArguments(output_dir=str(tmp_path), report_to="none")
    assert distill.train(plain, build=compare_arch.build_flat_student) is compare_arch.build_flat_student
    assert distill.train(plain) is distill.build_student
    with pytest.raises(SystemExit, match="only trains"):
        compare_arch.main(["--output_dir", str(tmp_path), "--stage", "extract", "--targets_dir", "x"])
