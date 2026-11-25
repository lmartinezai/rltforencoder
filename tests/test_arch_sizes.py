"""Bort with a configurable prelude / looped core / coda split (the default 2 + 4 + 2 = 8 blocks is unchanged). Tiny random models, CPU."""
import pytest
import torch
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
from transformers.models.modernbert.modeling_modernbert import ModernBertForMaskedLM

import compare_arch
import distill
import halting
from arch import BortConfig, BortForMaskedLM
from targets import make_targets, teacher_hidden

TINY = dict(hidden_size=32, intermediate_size=64, num_attention_heads=2, vocab_size=120, pad_token_id=0, bos_token_id=1, eos_token_id=2,
            cls_token_id=1, sep_token_id=2, local_attention=8, global_rope_theta=1000.0, local_rope_theta=1000.0)
FULL, SLIDING = "full_attention", "sliding_attention"


def bort(prelude, looped, coda, **extra):
    config = BortConfig(**TINY, prelude_layers=prelude, looped_layers=looped, coda_layers=coda, **extra)
    config.sparse_prediction = True
    return BortForMaskedLM(config).eval()


def test_default_split_is_the_8_block_bort_with_full_attention_at_0_2_5_7():
    config = BortConfig(**TINY)
    assert (config.prelude_layers, config.looped_layers, config.coda_layers, config.num_hidden_layers) == (2, 4, 2, 8)
    assert [i for i, t in enumerate(config.layer_types) if t == FULL] == [0, 2, 5, 7]


@pytest.mark.parametrize(("split", "blocks", "full"), [((1, 2, 1), 4, [0, 1, 2, 3]), ((1, 3, 1), 5, [0, 1, 3, 4]), ((2, 3, 1), 6, [0, 2, 4, 5]),
                                                       ((3, 4, 3), 10, [0, 3, 6, 9])])
def test_other_splits_name_their_blocks_and_put_full_attention_at_the_ends_of_the_core(split, blocks, full):
    config = BortConfig(**TINY, prelude_layers=split[0], looped_layers=split[1], coda_layers=split[2])
    assert config.num_hidden_layers == blocks and len(config.layer_types) == blocks
    assert [i for i, t in enumerate(config.layer_types) if t == FULL] == full


def test_a_split_needs_at_least_one_block_in_each_part_and_layer_types_must_fit():
    with pytest.raises(ValueError, match="at least one"):
        BortConfig(**TINY, prelude_layers=0)
    with pytest.raises(ValueError, match="every block"):
        BortConfig(**TINY, prelude_layers=1, looped_layers=2, coda_layers=1, layer_types=[FULL] * 5)
    custom = BortConfig(**TINY, prelude_layers=1, looped_layers=2, coda_layers=1, layer_types=[FULL, SLIDING, SLIDING, FULL])
    assert custom.layer_types == [FULL, SLIDING, SLIDING, FULL]


def test_a_config_written_by_the_old_code_loads_as_the_default_split():
    old = BortConfig(**TINY).to_dict()
    for key in ("prelude_layers", "looped_layers", "coda_layers"):
        old.pop(key)
    config = BortConfig(**old)
    assert (config.prelude_layers, config.looped_layers, config.coda_layers) == (2, 4, 2)


@pytest.mark.parametrize("split", [(1, 2, 1), (2, 3, 1), (1, 1, 2)])
def test_forward_runs_the_right_blocks_and_the_loop_state_does_not_depend_on_the_total_depth(split):
    model = bort(*split)
    assert len(model.model.layers) == sum(split)
    ids = torch.randint(3, 120, (2, 24))
    first, last = halting.loop_bounds(model)
    assert (first, last) == (split[0], split[0] + split[1] - 1)
    with torch.no_grad():
        with halting.loop_states(model) as deep:
            model.model(input_ids=ids, recurrence_depth=5)
        assert len(deep) == 5
        for d in (1, 3):
            with halting.loop_states(model) as shallow:
                model.model(input_ids=ids, recurrence_depth=d)
            assert len(shallow) == d and torch.equal(shallow[-1], deep[d - 1])


@pytest.mark.parametrize("split", [(1, 2, 1), (2, 3, 1)])
def test_one_pass_depth_table_equals_per_depth_runs_for_other_splits(split):
    """The halting machinery (hooks on the loop's last block, captured masks, coda decode) follows the split."""
    torch.manual_seed(0)
    student = bort(*split)
    teacher = ModernBertForMaskedLM(ModernBertConfig(hidden_size=32, intermediate_size=64, num_attention_heads=2, num_hidden_layers=2,
                                                     vocab_size=120, pad_token_id=0, local_attention=8)).eval()
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, 120, (3, 24), generator=g)
    labels = torch.full_like(ids, -100)
    masked = torch.rand(3, 24, generator=g) < 0.3
    masked[:, 0] = True
    labels[masked] = ids[masked]
    batch = {"input_ids": ids.masked_fill(masked, 4), "attention_mask": torch.ones_like(ids), "labels": labels}
    stored = make_targets(teacher_hidden(teacher, batch), student.decoder.weight, teacher.decoder.bias, 2.0, 0, 1.0)
    args = distill.DistillArguments(output_dir="runs/test")
    slow = halting.depth_table(student, batch, stored, teacher.decoder.bias, args, range(1, 6), signals=True)
    fast = halting.depth_table_fast(student, batch, stored, teacher.decoder.bias, args, range(1, 6), signals=True)
    for a, b in zip(slow, fast, strict=True):
        finite = torch.isfinite(a)
        assert torch.equal(finite, torch.isfinite(b)) and torch.allclose(a[finite], b[finite], atol=1e-4, rtol=1e-4)


def test_flat_baseline_has_as_many_blocks_as_the_bort_it_is_compared_with():
    config = BortConfig(**TINY, prelude_layers=1, looped_layers=2, coda_layers=1)
    flat = compare_arch.flat_config(config=config)
    assert flat.num_hidden_layers == 4 and flat.layer_types == config.layer_types
    assert not hasattr(flat, "prelude_layers") and not hasattr(flat, "use_merge")
    default = compare_arch.flat_config(layers=(2, 4, 2))
    assert default.num_hidden_layers == 8
