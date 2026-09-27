"""Build the final submission zip in the layout the challenge README requires.

    python -m src.evaluation.package --team <team_name>

<team_name>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, models/, tests/, reports/, README.md, requirements.txt}
└── Documentation_template.md
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from ..preprocessing.io import ROOT

CODE = "code/business_entity_resolution"
MODEL_FILES = ("lgbm_matcher.txt", "matcher_config.json", "feature_importance.csv")   # final model only


def _add_tree(z: zipfile.ZipFile, src: Path, dest: str) -> None:
    for p in sorted(src.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != ".DS_Store":
            z.write(p, f"{dest}/{p.relative_to(src)}")


def build(team: str) -> Path:
    out = ROOT / f"{team}_submission.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(ROOT / "output" / name, f"output/{name}")
        for d in ("src", "tests", "reports"):
            _add_tree(z, ROOT / d, f"{CODE}/{d}")
        for name in MODEL_FILES:
            z.write(ROOT / "models" / name, f"{CODE}/models/{name}")
        z.write(ROOT / "README.md", f"{CODE}/README.md")
        z.write(ROOT / "requirements.txt", f"{CODE}/requirements.txt")
        z.write(ROOT / "Documentation_template.md", "Documentation_template.md")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="team")
    path = build(ap.parse_args().team)
    print(f"wrote {path} ({path.stat().st_size / 1e6:,.0f} MB)")


if __name__ == "__main__":
    main()
