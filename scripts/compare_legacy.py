"""
Field-by-field comparison: adapter output (from a saved v1.2 extraction) vs the other app's own
extraction of the SAME RFP. Free: no API call.

    uv run python -m scripts.compare_legacy data/extractions/a9378dcd-8470-4843-a963-de558c32f7bc__pdf.json eval/legacy/bhopal.json
    uv run python -m scripts.compare_legacy <v12.json> <legacy.json> --all     # also show equal fields
    uv run python -m scripts.compare_legacy <v12.json> <legacy.json> --skip scoring_rules

`legacy.json` = the `extraction` object returned by the other app's /rfp-extraction/preview (or the
plain dict from extract_rfp). Both sides are flattened to `path -> value`; personnel rows are keyed
by role name so a different order does not matter.

Status per field:
  same        equal after normalising case / spaces / number type
  DIFF        both have a value and they differ        <- read these first
  only_legacy the other app has a value, the adapter has none   <- possible loss
  only_new    the adapter has a value, the other app has none    <- possible gain
A DIFF is not always an error: the new value may be the correct one. The report only tells you where to look.
"""

import argparse
import json
import math
import re
from pathlib import Path

from app.services.legacy_adapter import to_legacy

DEFAULT_SKIP = ("ExtractionSource", "ExtractionReviewFlags", "scoring_rules", "PersonnelSummary", "missing_fields", "extraction_confidence")
EMPTY = (None, "", [], {})


def flatten(node, prefix: str = "") -> dict:
    """dict/list -> {path: leaf}. A list of dicts that have a 'role' is keyed by role name."""
    out: dict = {}

    if isinstance(node, dict):
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else key))
    elif isinstance(node, list) and node and all(isinstance(i, dict) and i.get("role") for i in node):
        for item in node:
            out.update(flatten(item, f"{prefix}[{item['role']}]"))
    else:
        out[prefix] = node

    return out


def norm(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = " ".join(value.split()).lower()
        if re.fullmatch(r"-?[\d,]*\.?\d+", text):  # "3500000" and 3500000 are the same figure
            return float(text.replace(",", ""))
        return text
    return json.dumps(value, sort_keys=True, ensure_ascii=False).lower()


def same(a, b) -> bool:
    a, b = norm(a), norm(b)
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
    return a == b


def short(value, limit: int = 90) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("v12", type=Path, help="saved v1.2 extraction (data/extractions/*.json)")
    parser.add_argument("legacy", type=Path, help="the other app's extraction of the same RFP")
    parser.add_argument("--all", action="store_true", help="also list fields that are equal")
    parser.add_argument("--skip", nargs="*", default=list(DEFAULT_SKIP), help="path parts to ignore")
    args = parser.parse_args()

    new = flatten(to_legacy(json.loads(args.v12.read_text(encoding="utf-8"))))
    old = flatten(json.loads(args.legacy.read_text(encoding="utf-8")))

    rows = {"DIFF": [], "only_legacy": [], "only_new": [], "same": []}

    for path in sorted(set(new) | set(old)):
        if any(part in path for part in args.skip):
            continue
        a, b = new.get(path), old.get(path)
        if a in EMPTY and b in EMPTY:
            continue
        if b in EMPTY:
            rows["only_new"].append((path, a, b))
        elif a in EMPTY:
            rows["only_legacy"].append((path, a, b))
        elif same(a, b):
            rows["same"].append((path, a, b))
        else:
            rows["DIFF"].append((path, a, b))

    print(f"{args.v12.name}  vs  {args.legacy.name}")
    print("  " + " | ".join(f"{name}: {len(items)}" for name, items in rows.items()))

    for name in ("DIFF", "only_legacy", "only_new") + (("same",) if args.all else ()):
        if not rows[name]:
            continue
        print(f"\n== {name} ({len(rows[name])})")
        for path, a, b in rows[name]:
            if name == "same":
                print(f"  {path} = {short(a)}")
            elif name == "only_legacy":
                print(f"  {path}\n      legacy : {short(b)}")
            elif name == "only_new":
                print(f"  {path}\n      adapter: {short(a)}")
            else:
                print(f"  {path}\n      adapter: {short(a)}\n      legacy : {short(b)}")


if __name__ == "__main__":
    main()