"""
Field-level extraction eval against hand-verified reference checks.

    uv run python -m scripts.eval_extraction --check-refs      # free: validates reference paths
    uv run python -m scripts.eval_extraction --saved           # free: score saved outputs in data/extractions
    uv run python -m scripts.eval_extraction                   # runs Gemini on every reference PDF (~$0.07-0.10 each)
    uv run python -m scripts.eval_extraction --mode text --only lone_pine

Reference files live in eval/references/*.json:

    {"name": "...", "pdf": "uploads/....pdf", "checks": [
        {"path": "dates.submission_end", "expect": "2020-01-28", "page": 3},
        {"path": "dates.pre_bid.date", "expect": null, "trap": true, "page": 4}
    ]}

Check kinds (one per check):
    "expect": value        exact match after normalising (case/space for text, tolerance for numbers);
                           null means "must be empty" (null / [] / "")
    "expect_any": [..]     any of these values
    "contains": "text"     text contains it (for lists: any item contains it)
    "contains_any": [..]   text contains any of these
    "min": number          the value is a number >= this (use with path#len for "at least N items")
"trap": true marks a hallucination test: the document does NOT state this value (blank,
placeholder, sample page ...). A failing trap means the model invented something.

Path syntax:  a.b.c   |  list[0]  |  list[name~text] (contains)  |  list[kind=turnover] (equals)
              |  path#len (length of a list)
"""

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args, get_origin

from pydantic import BaseModel

REF_DIR = Path("eval/references")
OUT_DIR = Path("data/extractions")
RESULT_DIR = Path("data/eval")

# List prices per 1M tokens (verify on Google's pricing page; same numbers as extract_one).
PRICE_IN_PER_M = 0.30
PRICE_OUT_PER_M = 2.50

_TOKEN = re.compile(r"\[[^\]]*\]|[^.\[\]]+")
_FILTER = re.compile(r"(\w+)([~=])(.*)", re.DOTALL)


# ---------------------------------------------------------------- paths


def tokenize(path: str) -> tuple[list[str], bool]:
    length = path.endswith("#len")
    if length:
        path = path[: -len("#len")]
    return _TOKEN.findall(path), length


def _text(value) -> str:
    return " ".join(str(value).split()).lower()


def _matches(value, op: str, wanted: str) -> bool:
    if value is None:
        return False
    return (
        (wanted.lower() in _text(value))
        if op == "~"
        else (_text(value) == _text(wanted))
    )


def resolve(data, path: str):
    tokens, length = tokenize(path)
    node = data

    for token in tokens:
        if node is None:
            break

        if token.startswith("["):
            body = token[1:-1]
            if not isinstance(node, list):
                node = None
            elif body.isdigit():
                node = node[int(body)] if int(body) < len(node) else None
            else:
                key, op, wanted = _FILTER.fullmatch(body).groups()
                node = next(
                    (
                        item
                        for item in node
                        if isinstance(item, dict)
                        and _matches(item.get(key), op, wanted)
                    ),
                    None,
                )
        else:
            node = node.get(token) if isinstance(node, dict) else None

    if length:
        return len(node) if isinstance(node, list) else 0

    return node


def _unwrap(annotation):
    """Annotation -> (BaseModel class or None, is_list)."""
    origin = get_origin(annotation)

    if origin is list:
        model, _ = _unwrap(get_args(annotation)[0])
        return model, True

    if origin is not None:  # X | None, Optional[X]
        for arg in get_args(annotation):
            if arg is not type(None):
                return _unwrap(arg)

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation, False

    return None, False


def path_problem(model: type[BaseModel], path: str) -> str | None:
    """Static check: does `path` exist in the schema? Returns a message or None."""
    tokens, _ = tokenize(path)
    current, is_list = model, False

    for token in tokens:
        if token.startswith("["):
            if not is_list:
                return f"'{token}' used on something that is not a list"
            body = token[1:-1]
            if current is not None and not body.isdigit():
                match = _FILTER.fullmatch(body)
                if not match or match.group(1) not in current.model_fields:
                    return f"unknown filter field in '{token}'"
            is_list = False
            continue

        if current is None:
            return f"cannot go below a scalar at '{token}'"
        if token not in current.model_fields:
            return f"no field '{token}' in {current.__name__}"
        current, is_list = _unwrap(current.model_fields[token].annotation)

    return None


# ---------------------------------------------------------------- compare


def _equal(got, expected) -> bool:
    if expected is None:
        return got in (None, "", [], {})
    if isinstance(expected, bool):
        return got is expected
    if isinstance(expected, (int, float)):
        return (
            isinstance(got, (int, float))
            and not isinstance(got, bool)
            and math.isclose(float(got), float(expected), rel_tol=1e-6, abs_tol=1e-6)
        )
    if isinstance(expected, str):
        return isinstance(got, str) and _text(got) == _text(expected)
    return got == expected


def _contains(got, needle: str) -> bool:
    if isinstance(got, list):
        return any(_contains(item, needle) for item in got)
    return got is not None and _text(needle) in _text(got)


def run_check(data: dict, check: dict) -> tuple[bool, object]:
    got = resolve(data, check["path"])

    if "expect" in check:
        return _equal(got, check["expect"]), got
    if "expect_any" in check:
        return any(_equal(got, option) for option in check["expect_any"]), got
    if "contains" in check:
        return _contains(got, check["contains"]), got
    if "contains_any" in check:
        return any(_contains(got, option) for option in check["contains_any"]), got

    if "min" in check:
        return (
            isinstance(got, (int, float))
            and not isinstance(got, bool)
            and got >= check["min"],
            got,
        )

    raise ValueError(f"check has no expectation: {check}")


def expected_text(check: dict) -> str:
    for key in ("expect", "expect_any", "contains", "contains_any", "min"):
        if key in check:
            return f"{key}={json.dumps(check[key], ensure_ascii=False)}"
    return "?"


def evaluate(data: dict, checks: list[dict]) -> list[dict]:
    rows = []
    for check in checks:
        ok, got = run_check(data, check)
        rows.append(
            {
                "path": check["path"],
                "ok": ok,
                "trap": bool(check.get("trap")),
                "page": check.get("page"),
                "expected": expected_text(check),
                "got": got,
            }
        )
    return rows


# ---------------------------------------------------------------- runner


def load_references(only: str | None) -> list[dict]:
    refs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(REF_DIR.glob("*.json"))
    ]
    if only:
        refs = [r for r in refs if only.lower() in r["name"].lower()]
    return refs


def check_refs(refs: list[dict]) -> int:
    from app.schemas.extraction import TenderExtraction

    problems = 0
    for ref in refs:
        for check in ref["checks"]:
            message = path_problem(TenderExtraction, check["path"])
            if message:
                problems += 1
                print(f"[{ref['name']}] {check['path']}: {message}")

    total = sum(len(r["checks"]) for r in refs)
    print(f"{len(refs)} reference files, {total} checks, {problems} bad path(s)")
    return problems


def short(value, limit: int = 70) -> str:
    text = (
        json.dumps(value, ensure_ascii=False)
        if not isinstance(value, str)
        else repr(value)
    )
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pdf", "text"], default="pdf")
    parser.add_argument(
        "--saved", action="store_true", help="score saved outputs, no API calls"
    )
    parser.add_argument(
        "--check-refs", action="store_true", help="only validate reference paths"
    )
    parser.add_argument("--only", help="run only references whose name contains this")
    args = parser.parse_args()

    refs = load_references(args.only)
    if not refs:
        raise SystemExit(f"No reference files found in {REF_DIR}/")

    if args.check_refs:
        raise SystemExit(1 if check_refs(refs) else 0)

    service = None
    summary = []
    failed: list[str] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    for ref in refs:
        pdf = Path(ref["pdf"])
        saved = OUT_DIR / f"{pdf.stem}__{args.mode}.json"
        usage_text = "saved output"

        if args.saved:
            if not saved.exists():
                print(f"\n[{ref['name']}] skipped: {saved} not found")
                continue
            data = json.loads(saved.read_text(encoding="utf-8"))
        else:
            if not pdf.exists():
                print(f"\n[{ref['name']}] skipped: {pdf} not found")
                continue
            if service is None:
                from app.services.extraction_service import ExtractionService

                service = ExtractionService()
            try:
                result = service.extract(pdf, mode=args.mode)
            except Exception as exc:  # keep going: the other PDFs already cost money
                failed.append(ref["name"])
                print(f"\n[{ref['name']}] extraction FAILED: {exc}")
                continue
            data = result.extraction.model_dump(mode="json")
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            saved.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            u = result.usage
            cost = (
                u.prompt_tokens * PRICE_IN_PER_M
                + (u.output_tokens + u.thinking_tokens) * PRICE_OUT_PER_M
            ) / 1_000_000
            usage_text = (
                f"{result.seconds:.0f}s, {u.prompt_tokens:,} in / {u.output_tokens:,} out tokens, "
                f"~${cost:.3f}, repairs={result.repairs}"
            )

        rows = evaluate(data, ref["checks"])
        passed = sum(r["ok"] for r in rows)
        trap_rows = [r for r in rows if r["trap"]]
        trap_fail = sum(not r["ok"] for r in trap_rows)
        meta = data.get("meta", {})

        print(
            f"\n=== {ref['name']}: {passed}/{len(rows)} passed | traps failed {trap_fail}/{len(trap_rows)}"
        )
        print(
            f"    {usage_text} | prompt_hash={meta.get('prompt_hash')} | flags={len(meta.get('review_flags', []))}"
        )
        for row in rows:
            if not row["ok"]:
                tag = "TRAP " if row["trap"] else ""
                page = f" (p.{row['page']})" if row["page"] else ""
                print(f"  x {tag}{row['path']}{page}")
                print(f"      expected {row['expected']}")
                print(f"      got      {short(row['got'])}")

        summary.append(
            {
                "name": ref["name"],
                "passed": passed,
                "total": len(rows),
                "trap_failed": trap_fail,
                "traps": len(trap_rows),
                "usage": usage_text,
                "review_flags": meta.get("review_flags", []),
                "rows": rows,
            }
        )

    if failed:
        print(f"\nFAILED extractions (not scored): {', '.join(failed)}")

    if not summary:
        return

    print("\n" + "=" * 64)
    print(f"{'reference':<22}{'passed':>12}{'accuracy':>10}{'traps failed':>16}")
    for item in summary:
        print(
            f"{item['name']:<22}{item['passed']:>6}/{item['total']:<5}"
            f"{item['passed'] / item['total']:>10.0%}{item['trap_failed']:>9}/{item['traps']:<6}"
        )
    total_pass = sum(i["passed"] for i in summary)
    total_all = sum(i["total"] for i in summary)
    total_trap = sum(i["trap_failed"] for i in summary)
    total_traps = sum(i["traps"] for i in summary)
    print(
        f"{'ALL':<22}{total_pass:>6}/{total_all:<5}{total_pass / total_all:>10.0%}{total_trap:>9}/{total_traps:<6}"
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    target = RESULT_DIR / f"{stamp}__{args.mode}{'__saved' if args.saved else ''}.json"
    target.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nsaved: {target}")


if __name__ == "__main__":
    main()
