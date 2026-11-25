"""The three arms of the 200M-token comparison: Bort with the gated merge, Bort without it, and the flat baseline, each optionally started
from the nearest compatible ModernBERT-large layers."""
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_model
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
from transformers.models.modernbert.modeling_modernbert import ModernBertForMaskedLM

import compare_arch
import distill
from arch import BortConfig, BortForMaskedLM, MakeBortConfig
from distill import DistillArguments, init_from_teacher, teacher_layer_map, validate_args

TINY = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_attention_heads=4, max_position_embeddings=64,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2, global_attn_every_n_layers=3)
STUDENT_TYPES = ["full_attention" if i in (0, 2, 5, 7) else "sliding_attention" for i in range(8)]


def types(n):
    return ["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(n)]


# ---------------------------------------------------------------- which teacher layer each student layer starts from

def test_the_real_mapping_matches_attention_types_and_is_as_even_as_the_types_allow():
    mapping = teacher_layer_map(STUDENT_TYPES, types(28))
    assert mapping == [0, 4, 9, 11, 16, 18, 23, 27]
    assert all(STUDENT_TYPES[j] == types(28)[t] for j, t in enumerate(mapping))  # full<-full, sliding<-sliding
    assert mapping == sorted(set(mapping)) and mapping.count(0) == 1  # distinct, increasing, only layer 0 takes teacher layer 0
    assert all(abs(t - j * 27 / 7) <= 2 for j, t in enumerate(mapping))  # near the evenly spaced positions


def test_the_mapping_refuses_what_cannot_be_matched():
    with pytest.raises(ValueError, match="first layer"):
        teacher_layer_map(["sliding_attention", "full_attention"], types(4))
    with pytest.raises(ValueError, match="no teacher layer"):
        teacher_layer_map(STUDENT_TYPES, types(9))  # 9 teacher layers cannot host four full-attention layers after layer 0


# ---------------------------------------------------------------- the gated merge switch

def bort(use_merge, seed=0):
    torch.manual_seed(seed)
    config = BortConfig(**TINY)
    config.use_merge = use_merge
    config.sparse_prediction = True
    return BortForMaskedLM(config)


def batch():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(5, 64, (2, 12), generator=g)
    labels = torch.full_like(ids, -100)
    labels[:, 3] = ids[:, 3]
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)


def test_the_gated_merge_can_be_turned_off_and_the_loop_still_depends_on_depth():
    on, off = bort(True), bort(False)
    assert on.model.merge is not None and off.model.merge is None
    assert any(k.startswith("model.merge.") for k in on.state_dict()) and not any(k.startswith("model.merge.") for k in off.state_dict())
    merge = sum(p.numel() for p in on.model.merge.parameters())
    assert sum(p.numel() for p in on.parameters()) - merge == sum(p.numel() for p in off.parameters())
    off.eval()
    with torch.no_grad():
        shallow, deep = (off(**batch(), recurrence_depth=d).logits for d in (1, 4))
    assert torch.isfinite(shallow).all() and not torch.allclose(shallow, deep)  # still a loop: more iterations change the answer


def test_a_zero_initialised_merge_is_exactly_no_merge():
    """So a looped student with the merge starts as the same function as one without it, whatever else it starts from."""
    on, off = bort(True), bort(False)
    off.load_state_dict({k: v for k, v in on.state_dict().items() if not k.startswith("model.merge.")}, strict=True)
    torch.nn.init.zeros_(on.model.merge.proj.weight)
    on.eval(), off.eval()
    with torch.no_grad():
        for depth in (1, 3):
            torch.testing.assert_close(on(**batch(), recurrence_depth=depth).logits, off(**batch(), recurrence_depth=depth).logits)


def test_the_rlt_form_of_the_merge_reinjects_the_prelude_output_every_iteration(monkeypatch):
    """merge_base="anchor" is RLT's u = e + alpha g W RMSNorm(s): the base of every iteration is the prelude output, the state only arrives through the
    gated feedback. With a zero feedback projection every iteration therefore starts from the prelude output alone, and the default ("state") is the
    original Bort, unchanged."""
    from arch import GatedMerge

    merge = GatedMerge(32, 0.1, "anchor")
    anchor, state = torch.randn(2, 5, 32), torch.randn(2, 5, 32)
    torch.nn.init.zeros_(merge.proj.weight)
    torch.testing.assert_close(merge(anchor, state), anchor)
    torch.nn.init.normal_(merge.proj.weight)
    gate = torch.sigmoid(merge.gate(torch.cat((anchor, merge.norm(state)), -1)))
    torch.testing.assert_close(merge(anchor, state), anchor + 0.1 * gate * merge.proj(merge.norm(state)))
    assert GatedMerge(32, 0.1).base == "state" and BortConfig(**TINY).merge_base == "state"
    with pytest.raises(ValueError, match="merge_base"):
        GatedMerge(32, 0.1, "input")
    monkeypatch.setattr(distill, "MakeBortConfig", lambda _, p, l, c: BortConfig(**TINY, prelude_layers=p, looped_layers=l, coda_layers=c))
    student = distill.build_student(SimpleNamespace(gated_merge=True, merge_base="anchor", init="random", teacher="t", prelude_layers=1,
                                                    looped_layers=2, coda_layers=1), torch.randn(64, 32))
    assert student.model.merge.base == "anchor" and student.config.merge_base == "anchor"
    student.eval()
    with torch.no_grad():
        shallow, deep = (student(**batch(), recurrence_depth=d).logits for d in (1, 4))
    assert torch.isfinite(deep).all() and not torch.allclose(shallow, deep)
    with pytest.raises(ValueError, match="merge_base anchor needs the gated merge"):
        validate_args(DistillArguments(output_dir="x", merge_base="anchor", gated_merge=False))


def test_the_merge_flag_reaches_the_student_through_the_arguments(monkeypatch):
    monkeypatch.setattr(distill, "MakeBortConfig", lambda _, p, l, c: BortConfig(**TINY, prelude_layers=p, looped_layers=l, coda_layers=c))
    weight = torch.randn(64, 32)
    for flag in (True, False):
        student = distill.build_student(SimpleNamespace(gated_merge=flag, init="random", teacher="t", prelude_layers=2, looped_layers=4,
                                                        coda_layers=2), weight)
        assert (student.model.merge is not None) == flag
        assert not student.model.embeddings.tok_embeddings.weight.requires_grad
        assert len(student.model.layers) == 8
    small = distill.build_student(SimpleNamespace(gated_merge=True, init="random", teacher="t", prelude_layers=1, looped_layers=2,
                                                  coda_layers=1), weight)  # the split in the arguments reaches the model
    assert len(small.model.layers) == 4 and small.config.looped_layers == 2


# ---------------------------------------------------------------- starting from the teacher's layers

@pytest.fixture()
def teacher_checkpoint(tmp_path):
    torch.manual_seed(7)
    teacher = ModernBertForMaskedLM(ModernBertConfig(**TINY, num_hidden_layers=12))
    path = tmp_path / "model.safetensors"
    save_model(teacher, str(path))
    return teacher, path


def check_copied(student, teacher, mapping):
    t, s = teacher.state_dict(), student.state_dict()
    for j, tj in enumerate(mapping):
        for key in [k for k in t if k.startswith(f"model.layers.{tj}.")]:
            torch.testing.assert_close(s[key.replace(f"model.layers.{tj}.", f"model.layers.{j}.")], t[key])
    for key in ("model.embeddings.norm.weight", "model.final_norm.weight", "head.dense.weight", "head.norm.weight", "decoder.bias"):
        torch.testing.assert_close(s[key], t[key])


@pytest.mark.parametrize("use_merge", [True, False])
def test_bort_starts_from_the_mapped_teacher_layers(teacher_checkpoint, use_merge):
    teacher, path = teacher_checkpoint
    student = bort(use_merge)
    own_embedding = student.model.embeddings.tok_embeddings.weight.detach().clone()
    mapping = init_from_teacher(student, SimpleNamespace(teacher="t"), path, teacher.config.layer_types)
    assert mapping == teacher_layer_map(STUDENT_TYPES, teacher.config.layer_types) and len(set(mapping)) == 8
    check_copied(student, teacher, mapping)
    torch.testing.assert_close(student.model.embeddings.tok_embeddings.weight, own_embedding)  # the embeddings are stolen separately
    if use_merge:
        assert student.model.merge.proj.weight.abs().sum() == 0  # the merge starts as the identity


def test_the_flat_baseline_starts_from_the_same_teacher_layers_as_bort(teacher_checkpoint):
    teacher, path = teacher_checkpoint
    flat = compare_arch.build_flat_student(None, torch.randn(64, 32), compare_arch.flat_config(config=BortConfig(**TINY)))
    mapping = init_from_teacher(flat, SimpleNamespace(teacher="t"), path, teacher.config.layer_types)
    assert mapping == teacher_layer_map(STUDENT_TYPES, teacher.config.layer_types)
    check_copied(flat, teacher, mapping)


def test_an_incompatible_teacher_is_refused_not_half_loaded(tmp_path):
    wider = ModernBertForMaskedLM(ModernBertConfig(**(TINY | {"hidden_size": 64, "intermediate_size": 96}), num_hidden_layers=12))
    path = tmp_path / "wide.safetensors"
    save_model(wider, str(path))
    with pytest.raises(RuntimeError, match="size mismatch"):
        init_from_teacher(bort(True), SimpleNamespace(teacher="t"), path, wider.config.layer_types)


def test_init_option_is_validated(tmp_path):
    validate_args(DistillArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none", init="teacher_layers"))
    with pytest.raises(ValueError, match="--init"):
        validate_args(DistillArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none", init="pretrained"))


# ---------------------------------------------------------------- against the real checkpoint

@pytest.mark.parametrize("gated_merge", [True, False])
def test_the_real_student_starts_from_the_real_teacher_layers(tmp_path, gated_merge):
    args = DistillArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none", init="teacher_layers", gated_merge=gated_merge)
    student = distill.build_student(args)
    path = distill.hf_hub_download(args.teacher, "model.safetensors")
    mapping = [0, 4, 9, 11, 16, 18, 23, 27]
    with safe_open(path, "pt") as f:
        for j, tj in enumerate(mapping):
            for name in ("attn.Wqkv.weight", "attn.Wo.weight", "mlp.Wi.weight", "mlp.Wo.weight", "mlp_norm.weight"):
                torch.testing.assert_close(student.model.layers[j].get_parameter(name), f.get_tensor(f"model.layers.{tj}.{name}").float())
        torch.testing.assert_close(student.model.final_norm.weight, f.get_tensor("model.final_norm.weight").float())
        torch.testing.assert_close(student.model.embeddings.norm.weight, f.get_tensor("model.embeddings.norm.weight").float())
        torch.testing.assert_close(student.head.norm.weight, f.get_tensor("head.norm.weight").float())
        torch.testing.assert_close(student.head.dense.weight, f.get_tensor("head.dense.weight").float())
        torch.testing.assert_close(student.decoder.bias, f.get_tensor("decoder.bias").float())
    assert (student.model.merge is not None) == gated_merge
    assert not student.model.embeddings.tok_embeddings.weight.requires_grad and student.decoder.weight is student.model.embeddings.tok_embeddings.weight


def test_reloaded_bort_matches_original_at_every_depth(tmp_path):
    """StepEncoding used to keep its frequencies in a non-persistent buffer, which from_pretrained left uninitialised:
    a saved Bort reloaded with garbage at every recurrence step after the first."""
    config = MakeBortConfig("answerdotai/ModernBERT-large")
    config.num_hidden_layers = 8
    model = BortForMaskedLM(config).eval()
    model.save_pretrained(tmp_path)
    loaded = BortForMaskedLM.from_pretrained(tmp_path).eval()
    ids = torch.randint(0, 1000, (1, 16))
    with torch.no_grad():
        for depth in (1, 2, 5, 8):
            want = model.model(input_ids=ids, recurrence_depth=depth).last_hidden_state
            got = loaded.model(input_ids=ids, recurrence_depth=depth).last_hidden_state
            assert torch.allclose(got, want, atol=1e-5), depth
