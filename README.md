# More Than a Hack 2027 — Phase 1: ball possession

Rule-based possession labelling for FC Barcelona vs Athletic Club (tactical camera).
For every video frame it writes `H`, `A` or `D`. No training, no hand labels, numpy only.

## Install

    python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

Tested with Python 3.13 and numpy 2.5.3.

## Data

The match data is not in this repository (it is not redistributable). Put the three tactical
files from the organisers' bundle in `data/tactical/`:

    data/tactical/BAR-ATH_tactical_atd.txt
    data/tactical/BAR-ATH_tactical_atd.xml
    data/tactical/BAR-ATH_tactical_smart_tagging.xml

## Regenerate the submission

    python src/possession.py --data data/tactical --out out/submission.csv

Runs in about 10 seconds and is deterministic. Validate and score with the organisers' tools:

    python evaluation/validate_submission.py out/submission.csv
    python evaluation/run_evaluation.py --results out

## Method

See [TECHNICAL.md](TECHNICAL.md).
