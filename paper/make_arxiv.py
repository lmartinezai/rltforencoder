"""Build the arXiv submission and compile it the way arXiv does.

    uv run python paper/make_arxiv.py          # needs Docker; first run downloads the TeX Live 2025 image (several GB)

arXiv compiles with pdflatex from TeX Live 2025 (its default) and does not run BibTeX, so the upload must carry the bibliography as main.bbl.
This writes paper/arxiv/ (main.tex with the bibliography path made local, the figures, and main.bbl produced by BibTeX here), compiles it with
pdflatex and BibTeX in the official texlive/texlive:TL2025-historic image (pdflatex, bibtex, pdflatex, pdflatex), reports undefined references
or citations and overfull boxes from the log, and packs the files to upload into paper/arxiv.tar.gz. paper/arxiv/main.pdf is what arXiv will
produce.
"""
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
OUT = PAPER / "arxiv"
IMAGE = "texlive/texlive:TL2025-historic"
UPLOAD = ("main.tex", "main.bbl", "figures")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    if tex.count("\\bibliography{../references}") != 1:
        sys.exit("main.tex: expected \\bibliography{../references}")
    (OUT / "main.tex").write_text(tex.replace("\\bibliography{../references}", "\\bibliography{references}"), encoding="utf-8")
    shutil.copy(PAPER.parent / "references.bib", OUT / "references.bib")
    for fig in (PAPER / "figures").glob("*.pdf"):
        shutil.copy(fig, OUT / "figures" / fig.name)
    run = "pdflatex -interaction=nonstopmode main && bibtex main && pdflatex -interaction=nonstopmode main && pdflatex -interaction=nonstopmode main"
    p = subprocess.run(["docker", "run", "--rm", "-v", f"{OUT}:/w", "-w", "/w", IMAGE, "sh", "-c", run], check=False, capture_output=True, text=True)
    log = (OUT / "main.log").read_text(encoding="latin-1") if (OUT / "main.log").exists() else ""
    problems = re.findall(r"(?:LaTeX|Package natbib) Warning: (?:Citation|Reference).*undefined.*", log) + re.findall(r"Overfull \\hbox.*", log)
    blg = (OUT / "main.blg").read_text(encoding="latin-1") if (OUT / "main.blg").exists() else ""
    problems += [line for line in blg.splitlines() if line.startswith(("Warning--", "I couldn't", "Repeated entry"))]
    if p.returncode or not (OUT / "main.pdf").exists():
        print(p.stdout[-3000:], p.stderr[-2000:])
        sys.exit(f"compilation failed (exit {p.returncode})")
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    print(f"compiled with {IMAGE}: {pages.group(1) if pages else '?'} pages -> {OUT / 'main.pdf'}")
    print("\n".join(problems) if problems else "no undefined references or citations, no overfull boxes, no BibTeX warnings")
    with tarfile.open(PAPER / "arxiv.tar.gz", "w:gz") as tar:
        for name in UPLOAD:
            tar.add(OUT / name, arcname=name)
    print(f"upload: {PAPER / 'arxiv.tar.gz'} ({', '.join(UPLOAD)})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
