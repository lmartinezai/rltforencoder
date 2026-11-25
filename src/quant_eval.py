"""Does quantising the weights hurt a looped model more than a flat one? A looped model quantises its shared blocks once and applies them up to 8 times, so
rounding error can compound across iterations; a flat model's blocks are each applied once.

    uv run python src/quant_eval.py --models a_s1 c_s1 ... --out results/hardware

Round-to-nearest, symmetric, weight-only quantisation of every Linear layer of the encoder blocks (embeddings, the prediction head, the decoder and the
looped model's gated merge, whose two Linears are applied once per iteration, stay in bf16: the merge is NOT quantised), simulated by quantise-then-dequantise: `int8` per output channel, `int4` in groups of 64 input features. No calibration data and no training: nothing
is fitted. Scored with the cross-tokenizer masked-LM bits per byte of src/crossmodel_eval.py on its cached held-out texts (all 4 sources, every loop depth).
Reported: the mean BPB and its change against the same model unquantised."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

import distill
from crossmodel_eval import DATASETS, load_model, load_texts, score_texts, uses_bf16
from hw_models import available

SCHEMES = ("fp", "int8", "int4")
GROUP = 64


def fake_quantise(weight, scheme):
    """Round-to-nearest symmetric quantise-dequantise of a [out, in] weight."""
    w = weight.float()
    if scheme == "int8":
        scale = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
        return ((w / scale).round().clamp(-127, 127) * scale).to(weight.dtype)
    if scheme == "int4":
        out, inp = w.shape
        size = min(GROUP, inp)  # (a layer narrower than one group is one group)
        if inp % size:
            raise ValueError(f"{inp} input features do not split into groups of {size}")
        g = w.reshape(out, inp // size, size)
        scale = g.abs().amax(2, keepdim=True).clamp_min(1e-8) / 7
        return ((g / scale).round().clamp(-8, 7) * scale).reshape(out, inp).to(weight.dtype)
    raise ValueError(scheme)


def quantise_encoder(model, scheme):
    """In place; -> number of Linear layers quantised."""
    if scheme == "fp":
        return 0
    n = 0
    for name, module in model.base_model.named_modules():
        if isinstance(module, nn.Linear) and {"layer", "layers"} & set(name.split(".")):  # the blocks' Linears, whatever the family calls them
            with torch.no_grad():
                module.weight.copy_(fake_quantise(module.weight, scheme))
            n += 1
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--schemes", nargs="+", default=list(SCHEMES))
    p.add_argument("--out", default="results/hardware")
    p.add_argument("--texts", type=int, default=128)
    p.add_argument("--chars", type=int, default=1000)
    p.add_argument("--texts_file", default="results/mlm_crossmodel/texts.json", help="the held-out texts cached by crossmodel_eval.py")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    distill.configure_cuda()
    out = Path(a.out)
    texts = load_texts(Path(a.texts_file), a.texts, a.chars)
    path = out / "quant.json"
    raw = json.loads(path.read_text()) if path.exists() else {}
    registry = available()
    for name in a.models:
        kind, checkpoint, _, _ = registry[name]
        for scheme in a.schemes:
            if f"{name}/{scheme}" in raw:
                continue
            model, tok = load_model(kind, checkpoint, "cuda")
            layers = quantise_encoder(model, scheme)
            per_depth = {}
            for dataset in DATASETS:
                totals = score_texts(kind, model, tok, texts[dataset], 8, 512, "cuda", bf16=uses_bf16(model))
                for depth, t in totals.items():
                    per_depth.setdefault(str(depth), {})[dataset] = t["nats"] / math.log(2) / t["bytes"]
            raw[f"{name}/{scheme}"] = {"quantised_linears": layers, "bpb": per_depth}
            path.write_text(json.dumps(raw, indent=1))
            means = {d: sum(v.values()) / len(v) for d, v in per_depth.items()}
            print(f"[{name}/{scheme}] {layers} linears quantised; mean BPB " + " ".join(f"d{d}:{m:.4f}" for d, m in means.items()), flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
