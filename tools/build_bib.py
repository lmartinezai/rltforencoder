"""Build references.bib from the arXiv API and check every ID against the title it is supposed to have (a wrong ID fails loudly).

    uv run python tools/build_bib.py          # writes references.bib at the repository root

Titles, authors and years come from arXiv itself, so the BibTeX cannot contain a mistyped author or title. `venue` is the published version
(prefer citing it over the arXiv record); the few works that are not on arXiv are written out by hand at the end."""
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# key: (arXiv id, a phrase the arXiv title must contain, published venue or "")
ARXIV = {
    # the architecture and its lineage
    "zhang2026rlt": ("2610.07591", "Recurrent Looped Transformer", ""),
    "geiping2025recurrent": ("2502.05171", "Recurrent Depth Approach", "NeurIPS 2025"),
    "saunshi2025latent": ("2502.17416", "Power of Looped Transformers", "ICLR 2025"),
    "dehghani2019universal": ("1807.03819", "Universal Transformers", "ICLR 2019"),
    "graves2016act": ("1603.08983", "Adaptive Computation Time", ""),
    "lan2020albert": ("1909.11942", "ALBERT", "ICLR 2020"),
    "bae2025relaxed": ("2410.20672", "Relaxed Recursive Transformers", "ICLR 2025"),
    "bae2025mor": ("2507.10524", "Mixture-of-Recursions", "NeurIPS 2025"),
    "csordas2024moeut": ("2405.16039", "MoEUT", "NeurIPS 2024"),
    "giannou2023looped": ("2301.13196", "Programmable Computers", "ICML 2023"),
    "yang2024looped": ("2311.12424", "Looped Transformers are Better", "ICLR 2024"),
    "takase2021lessons": ("2104.06022", "Parameter Sharing across Layers", ""),
    "dabre2019recurrent": ("1807.05353", "Recurrent Stacking", "AAAI 2019"),
    "fan2020feedback": ("2002.09402", "Feedback Memory", ""),
    "bai2019deq": ("1909.01377", "Deep Equilibrium Models", "NeurIPS 2019"),
    "srivastava2015highway": ("1505.00387", "Highway Networks", ""),
    "parisotto2020gtrxl": ("1910.06764", "Stabilizing Transformers for Reinforcement Learning", "ICML 2020"),
    "bachlechner2021rezero": ("2003.04887", "ReZero", "UAI 2021"),
    # encoders, teacher, references
    "vaswani2017attention": ("1706.03762", "Attention Is All You Need", "NeurIPS 2017"),
    "devlin2019bert": ("1810.04805", "BERT", "NAACL 2019"),
    "liu2019roberta": ("1907.11692", "RoBERTa", ""),
    "warner2024modernbert": ("2412.13663", "Smarter, Better, Faster, Longer", "ACL 2025"),
    "he2021deberta": ("2006.03654", "DeBERTa", "ICLR 2021"),
    "su2021roformer": ("2104.09864", "RoFormer", "Neurocomputing 2024"),
    "shazeer2020glu": ("2002.05202", "GLU Variants", ""),
    "zhang2019rmsnorm": ("1910.07467", "Root Mean Square Layer Normalization", "NeurIPS 2019"),
    "beltagy2020longformer": ("2004.05150", "Longformer", ""),
    "xiong2020prelayernorm": ("2002.04745", "Layer Normalization in the Transformer", "ICML 2020"),
    # distillation and masked-LM training
    "hinton2015distilling": ("1503.02531", "Distilling the Knowledge", "NIPS 2014 Deep Learning Workshop"),
    "sanh2019distilbert": ("1910.01108", "DistilBERT", "EMC2 Workshop at NeurIPS 2019"),
    "jiao2020tinybert": ("1909.10351", "TinyBERT", "Findings of EMNLP 2020"),
    "sun2019pkd": ("1908.09355", "Patient Knowledge Distillation", "EMNLP 2019"),
    "wang2020minilm": ("2002.10957", "MiniLM", "NeurIPS 2020"),
    "anshumann2025sparse": ("2503.16870", "Sparse Logit Sampling", "ACL 2025"),
    "sajjad2023dropping": ("2004.03844", "Dropping Layers", "Computer Speech & Language"),
    "wettig2023mask": ("2202.08005", "Should You Mask 15", "EACL 2023"),
    "krell2021packing": ("2107.02027", "Sequence Packing", ""),
    "loshchilov2019adamw": ("1711.05101", "Decoupled Weight Decay", "ICLR 2019"),
    "loshchilov2017sgdr": ("1608.03983", "SGDR", "ICLR 2017"),
    "micikevicius2018mixed": ("1710.03740", "Mixed Precision Training", "ICLR 2018"),
    "kalamkar2019bf16": ("1905.12322", "BFLOAT16", ""),
    "chen2016checkpointing": ("1604.06174", "Sublinear Memory Cost", ""),
    "huang2016stochastic": ("1603.09382", "Stochastic Depth", "ECCV 2016"),
    # adaptive computation and early exit
    "schuster2022calm": ("2207.07061", "Confident Adaptive Language Modeling", "NeurIPS 2022"),
    "banino2021pondernet": ("2107.05407", "PonderNet", ""),
    "xin2020deebert": ("2004.12993", "DeeBERT", "ACL 2020"),
    "zhou2020pabee": ("2006.04152", "BERT Loses Patience", "NeurIPS 2020"),
    "elbayad2020depthadaptive": ("1910.10073", "Depth-Adaptive Transformer", "ICLR 2020"),
    # compute, efficiency and reporting
    "kaplan2020scaling": ("2001.08361", "Scaling Laws for Neural Language Models", ""),
    "hoffmann2022chinchilla": ("2203.15556", "Compute-Optimal Large Language Models", "NeurIPS 2022"),
    "dehghani2022misnomer": ("2110.12894", "Efficiency Misnomer", "ICLR 2022"),
    "dettmers2022int8": ("2208.07339", "LLM.int8()", "NeurIPS 2022"),
    "frantar2023gptq": ("2210.17323", "GPTQ", "ICLR 2023"),
    "henderson2020energy": ("2002.05651", "Systematic Reporting of the Energy", "JMLR 2020"),
    "strubell2019energy": ("1906.02243", "Energy and Policy Considerations", "ACL 2019"),
    "reimers2017seeds": ("1707.09861", "Reporting Score Distributions", "EMNLP 2017"),
    "dodge2020finetuning": ("2002.06305", "Fine-Tuning Pretrained Language Models", ""),
    # data and evaluation
    "penedo2024fineweb": ("2406.17557", "FineWeb", "NeurIPS 2024 Datasets and Benchmarks"),
    "cohan2018arxiv": ("1804.05685", "Discourse-Aware Attention Model", "NAACL 2018"),
    "gao2020pile": ("2101.00027", "The Pile", ""),
    "salazar2020mlmscoring": ("1910.14659", "Masked Language Model Scoring", "ACL 2020"),
    "wang2019glue": ("1804.07461", "GLUE", "ICLR 2019"),
    "williams2018mnli": ("1704.05426", "Broad-Coverage Challenge Corpus", "NAACL 2018"),
    "rajpurkar2016squad": ("1606.05250", "SQuAD", "EMNLP 2016"),
    "schick2021pet": ("2001.07676", "Cloze Questions", "EACL 2021"),
    "gao2021lmbff": ("2012.15723", "Better Few-shot Learners", "ACL 2021"),
    "zhao2021calibrate": ("2102.09690", "Calibrate Before Use", "ICML 2021"),
    # software
    "paszke2019pytorch": ("1912.01703", "PyTorch", "NeurIPS 2019"),
    "wolf2020transformers": ("1910.03771", "Transformers", "EMNLP 2020 System Demonstrations"),
    "lhoest2021datasets": ("2109.02846", "Datasets", "EMNLP 2021 System Demonstrations"),
}

MANUAL = r"""
@inproceedings{ansel2024pytorch2,
  title = {{PyTorch} 2: Faster Machine Learning Through Dynamic {Python} Bytecode Transformation and Graph Compilation},
  author = {Ansel, Jason and others},
  booktitle = {Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '24)},
  pages = {929--947},
  year = {2024},
  doi = {10.1145/3620665.3640366}
}

@inproceedings{dagan2006rte,
  title = {The {PASCAL} Recognising Textual Entailment Challenge},
  author = {Dagan, Ido and Glickman, Oren and Magnini, Bernardo},
  booktitle = {Machine Learning Challenges: Evaluating Predictive Uncertainty, Visual Object Classification, and Recognising Textual Entailment},
  series = {Lecture Notes in Computer Science},
  volume = {3944},
  pages = {177--190},
  publisher = {Springer},
  year = {2006}
}

@misc{lehmann2026recurrence,
  title = {Does Recurrence Pay? A Controlled Evaluation of the Recurrent Looped Transformer},
  author = {Lehmann, Leon and Nakamura, Casie},
  year = {2026},
  howpublished = {alphaXiv preprint (not on arXiv as of 2026-11-16)},
  url = {https://www.alphaxiv.org/abs/2609.recurrence-looped-transformer-evaluation},
  note = {Page dated 15 Sep 2026, before the RLT arXiv v1 (6 Oct 2026); confirm the date and a stable identifier before citing}
}

@misc{wikimedia2023wikipedia,
  title = {Wikipedia dump 20231101.en},
  author = {{Wikimedia Foundation}},
  year = {2023},
  howpublished = {Hugging Face dataset wikimedia/wikipedia, revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa},
  url = {https://huggingface.co/datasets/wikimedia/wikipedia}
}

@misc{codeparrot2022clean,
  title = {codeparrot-clean-valid},
  author = {{CodeParrot}},
  year = {2022},
  howpublished = {Hugging Face dataset codeparrot/codeparrot-clean-valid, revision 4db92d2ec0c1b4c41eeb439cfae16854511d9dcd},
  url = {https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid}
}

@misc{openvino,
  title = {{OpenVINO} Toolkit},
  author = {{Intel Corporation}},
  year = {2026},
  howpublished = {Software, version 2026.4},
  url = {https://github.com/openvinotoolkit/openvino}
}
"""

NS = {"a": "http://www.w3.org/2005/Atom"}


def fetch(ids):
    url = "http://export.arxiv.org/api/query?max_results=200&id_list=" + ",".join(ids)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "citation-check"}), timeout=60) as r:
        return ET.fromstring(r.read())


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    ids = [v[0] for v in ARXIV.values()]
    entries = {}
    for i in range(0, len(ids), 40):  # the API answers long id lists in chunks
        for e in fetch(ids[i:i + 40]).findall("a:entry", NS):
            arxiv_id = e.find("a:id", NS).text.rsplit("/", 1)[-1].split("v")[0]
            entries[arxiv_id] = e
        time.sleep(3)  # arXiv asks for 3 s between requests
    bib, bad = [], []
    tex = lambda s: re.sub(r"(?<!\\)([&%#_])", r"\\\1", s)  # LaTeX specials in arXiv metadata ("15%", "Speech & Language") break the bibliography
    for key, (arxiv_id, phrase, venue) in ARXIV.items():
        e = entries.get(arxiv_id)
        if e is None:
            bad.append(f"{key}: {arxiv_id} not returned")
            continue
        title = " ".join(e.find("a:title", NS).text.split())
        if norm(phrase) not in norm(title):
            bad.append(f"{key}: {arxiv_id} is '{title}', expected '{phrase}'")
            continue
        authors = " and ".join(a.find("a:name", NS).text for a in e.findall("a:author", NS))
        year = e.find("a:published", NS).text[:4]
        fields = [f"  title = {{{{{tex(title)}}}}}", f"  author = {{{tex(authors)}}}", f"  year = {{{year}}}", f"  eprint = {{{arxiv_id}}}",
                  "  archivePrefix = {arXiv}"]
        if venue:
            fields.append(f"  note = {{Published at {tex(venue)}}}")
        bib.append(f"@misc{{{key},\n" + ",\n".join(fields) + "\n}\n")
    if bad:
        raise SystemExit("ID check failed:\n  " + "\n  ".join(bad))
    out = Path(__file__).resolve().parents[1] / "references.bib"
    out.write_text("% Generated by tools/build_bib.py from the arXiv API (titles, authors, years) + hand-written entries for works\n"
                   "% not on arXiv. Every arXiv ID was checked against its expected title.\n\n" + "\n".join(bib) + MANUAL, encoding="utf-8")
    print(f"{len(bib)} arXiv entries verified, {MANUAL.count('@')} manual entries; wrote {out}")


if __name__ == "__main__":
    main()
