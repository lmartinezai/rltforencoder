"""The Hugging Face revisions every reported result used, and the switch that makes a reproduction use exactly them.

With REPRO_PINNED=1 in the environment (reproduce/reproduce.py sets it for every step), every load from the Hub in src/ asks for the commit
below, so a repository changed after the runs cannot change a reproduction; a repository missing here is an error, not a silent fallback.
Without it, loads ask for the newest commit, as the reported runs did: each repository's newest commit predates every run (checked
2026-11-18), so the two agree. src/data.py takes the FineWeb revision through --dataset_revision instead, because the teacher server's stream
fingerprint covers that module's source. src/arch.py (the author's file) reads ModernBERT-large's configuration unpinned; reproduce/audit.py
compares the configuration it builds with the one each trained model stored.
"""
import os

REVISIONS = {
    "answerdotai/ModernBERT-large": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",  # teacher, tokenizer, embeddings, layer initialization
    "FacebookAI/roberta-base": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
    "microsoft/deberta-base": "0d1b43ccf21b5acd9f4e5f7b077fa698f05cf195",
    "distilbert/distilbert-base-cased": "6ea81172465e8b0ad3fddeed32b986cdcdcffcf0",
    "HuggingFaceFW/fineweb": "9bb295ddab0e05d785b879661af7260fed5140fc",  # sample-10BT: training data and the FineWeb hold-out
    "ccdv/arxiv-summarization": "240aaf1a969b3f8cd0ade6986bfad0cd730ee288",
    "wikimedia/wikipedia": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",  # 20231101.en
    "codeparrot/codeparrot-clean-valid": "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd",
    "nyu-mll/glue": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
}


def pinned():
    return os.environ.get("REPRO_PINNED") == "1"


def revision(repo):
    """The commit to load `repo` at: None (the newest) unless pinned; a local path is never pinned."""
    if not pinned() or os.path.exists(str(repo)):
        return None
    if repo not in REVISIONS:
        raise KeyError(f"{repo} has no pinned revision in src/pins.py; a reproduction loads only pinned repositories")
    return REVISIONS[repo]


def kw(repo):
    """Keyword arguments for a Hub load of `repo`: {"revision": commit} when pinned, else none (the call is then exactly as before)."""
    rev = revision(repo)
    return {} if rev is None else {"revision": rev}
