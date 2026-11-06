"""
Pick the evidence for ONE extraction call from a big RFP (flow doc, stages 5 and 6).

No user question: the schema decides what to look for. Eight query GROUPS (one per field family)
each score every chunk, then ONE shared character budget is split between them:

  score  = BM25 of the group's queries (normalised per query)  +  a bonus per keyword/heading pattern
  pick   = per group: the first chunks of the document if asked (cover / data sheet), then the best
           chunks up to 60% of the group's share, then their neighbours (a table continues in the next
           chunk, maybe on the next page)
  fill   = budget left over goes to the best chunks of any group

This first version is lexical on purpose (no embeddings, no Qdrant): it can be measured for free with
scripts/rag_recall.py. Dense scores can be added next to the BM25 score in _group_scores().
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.services.rag_chunker import EvidenceChunk

DEFAULT_BUDGET_CHARS = 160_000
SELECTOR_VERSION = (
    "1"  # bump when queries / scoring change: it is part of the extraction cache key
)
SEED_SHARE = 0.6  # part of a group's quota spent on its best chunks; the rest on their neighbours
KEYWORD_BONUS = 0.3  # per matching pattern, at most 3 patterns count
MIN_SCORE = 0.5  # a chunk must score this much to use its group's protected share


@dataclass(frozen=True)
class QueryGroup:
    name: str
    queries: tuple[str, ...]
    keywords: tuple[str, ...]  # regex, case-insensitive
    share: float  # protected share of the budget
    front_chunks: int = 0  # always take the first N chunks of the document


GROUPS: tuple[QueryGroup, ...] = (
    QueryGroup(
        "identity_dates",
        (
            "RFP reference number tender title name of authority client",
            "key dates submission deadline proposal due date technical proposal opening pre-proposal meeting",
            "cost of tender document fee",
            "proposal validity period days",
            "e-procurement portal registration",
        ),
        (
            r"\b(?:rfp|tender|nit)\b.{0,40}\b(?:no|number|ref)\b",
            r"pre-?bid|pre-?proposal",
            r"last date|due date|bid opening|proposal opening",
            r"validity",
            r"data sheet|tender information summary|notice inviting|summary",
        ),
        0.12,
        front_chunks=6,
    ),
    QueryGroup(
        "securities",
        (
            "bid security earnest money deposit amount",
            "performance security performance bank guarantee percentage of contract price",
            "additional performance security abnormally low bid",
        ),
        (
            r"earnest money|bid security|\bemd\b",
            r"performance (?:bank )?guarantee|performance security|\bpbg\b",
            r"bank guarantee",
        ),
        0.06,
    ),
    QueryGroup(
        "eligibility",
        (
            "eligibility requirements consultant entity registration blacklisted conflict of interest",
            "joint venture consortium members lead member share",
            "sub-contracting limit percent prior approval",
            "only one proposal per consultant participation",
        ),
        (
            r"eligib|qualification criteria|blacklist|debarred|insolven",
            r"joint venture|consorti|\bjv\b|lead (?:member|partner)",
            r"sub-?contract",
        ),
        0.12,
    ),
    QueryGroup(
        "qualification",
        (
            "average annual turnover last years financial capacity net worth",
            "similar projects experience aggregate length lookback years",
            "residual bid capacity formula",
            "multiplication factor two-lane experience",
        ),
        (
            r"turnover|net worth|crore|lakh",
            r"aggregate length|similar (?:project|assignment)|experience of",
            r"bid capacity",
        ),
        0.10,
    ),
    QueryGroup(
        "evaluation",
        (
            "evaluation of technical proposal marking scheme maximum marks criteria",
            "detailed marking and evaluation scheme further break-up sub-points",
            "weightage technical score financial score combined score formula",
            "minimum technical score qualifying marks",
            "selection method QCBS QBS quality and cost based",
        ),
        (
            r"marking scheme|evaluation (?:criteria|scheme)|scoring criteria",
            r"sub-?\s*points?",
            r"max(?:imum)?\s+(?:points|marks)",
            r"weightage|combined score|\bqcbs\b|\bqbs\b|\bfbs\b|\blcs\b",
            r"technical score|minimum (?:technical|qualifying)",
        ),
        0.26,
    ),
    QueryGroup(
        "personnel",
        (
            "key personnel key experts man-months staff months deployment site office",
            "team leader qualification minimum experience years",
            "CV evaluation marking key professionals points",
            "conditional experts additional personnel in case tunnels hilly terrain",
            "remuneration rate per man month staff months amount",
        ),
        (
            r"man.?months?|staff.?months?|person.?months?",
            r"key (?:personnel|experts?|professionals?)|team leader",
            r"remuneration|rate per",
            r"in case of|hilly|mountainous|tunnel",
        ),
        0.16,
    ),
    QueryGroup(
        "contract",
        (
            "payment schedule stage-wise percentage of contract price milestones",
            "liquidated damages penalty for delay per day maximum percent",
            "penalty for delay in completion of services per day of contract price",
            "contract duration period of services months commencement",
            "price escalation",
        ),
        (
            r"liquidated damages|penalty",
            r"payment (?:schedule|stage|terms)|milestone",
            r"duration|period of (?:the )?(?:contract|services)",
        ),
        0.12,
    ),
    QueryGroup(
        "value",
        (
            "estimated cost of consultancy assignment estimated project cost",
            "financial proposal summary of costs total remuneration",
            "format of financial proposal cost estimate",
        ),
        (
            r"estimated (?:project |contract |assignment )?(?:cost|value)|cost estimate",
            r"summary of costs?|financial proposal",
        ),
        0.06,
    ),
)


# ---------------------------------------------------------------- BM25

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from in is it of on or that the this to with shall will".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class _Bm25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(doc) for doc in docs]
        self.length = [len(doc) for doc in docs]
        self.avg = (sum(self.length) / len(docs)) if docs else 1.0
        self.avg = self.avg or 1.0

        df: Counter = Counter()
        for tf in self.tf:
            df.update(tf.keys())

        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens: list[str]) -> list[float]:
        terms = [t for t in set(query_tokens) if t in self.idf]
        out = [0.0] * len(self.tf)

        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.length[i] / self.avg)
            score = 0.0

            for term in terms:
                f = tf.get(term)
                if f:
                    score += self.idf[term] * f * (self.k1 + 1) / (f + norm)

            out[i] = score

        return out


# ---------------------------------------------------------------- selection


@dataclass
class EvidenceSelection:
    chunks: list[EvidenceChunk]  # the chosen chunks, in document order
    groups: dict[str, list[int]]  # group name -> chunk indexes it asked for
    budget_chars: int
    total_chars: int  # size of the whole document

    @property
    def chars(self) -> int:
        return sum(len(c.text) for c in self.chunks)

    @property
    def pages(self) -> list[int]:
        return sorted({c.page_number for c in self.chunks})


def _group_scores(
    index: _Bm25, chunks: list[EvidenceChunk], group: QueryGroup
) -> list[float]:
    best = [0.0] * len(chunks)

    for query in group.queries:
        raw = index.scores(tokens(query))
        top = max(raw, default=0.0)

        if top <= 0:
            continue

        for i, value in enumerate(raw):
            normal = value / top
            if normal > best[i]:
                best[i] = normal

    patterns = [re.compile(p, re.IGNORECASE) for p in group.keywords]

    for i, chunk in enumerate(chunks):
        hits = sum(1 for pattern in patterns if pattern.search(chunk.text))
        if hits:
            best[i] += KEYWORD_BONUS * min(hits, 3)

    return best


def _pick(
    chunks: list[EvidenceChunk],
    scores: list[float],
    quota: int,
    front: int,
    min_score: float,
) -> list[int]:
    picked: list[int] = []
    taken: set[int] = set()
    used = 0

    def add(i: int) -> bool:
        nonlocal used
        cost = len(chunks[i].text)

        if i in taken or used + cost > quota:
            return False

        taken.add(i)
        picked.append(i)
        used += cost
        return True

    for i in range(min(front, len(chunks))):
        add(i)

    ranked = [
        i
        for i in sorted(range(len(chunks)), key=lambda i: -scores[i])
        if scores[i] > 0 and scores[i] >= min_score
    ]
    seeds: list[int] = []

    for i in ranked:
        if used >= quota * SEED_SHARE:
            break
        if add(i):
            seeds.append(i)

    for i in seeds:  # best seeds first
        for j in (i - 1, i + 1):
            if 0 <= j < len(chunks):
                add(j)

    return picked


def select_evidence(
    chunks: list[EvidenceChunk],
    budget_chars: int = DEFAULT_BUDGET_CHARS,
    groups: tuple[QueryGroup, ...] = GROUPS,
    min_score: float = MIN_SCORE,
) -> EvidenceSelection:
    total = sum(len(c.text) for c in chunks)

    if total <= budget_chars:  # small document: everything fits
        return EvidenceSelection(list(chunks), {}, budget_chars, total)

    index = _Bm25([tokens(c.text) for c in chunks])
    chosen: set[int] = set()
    per_group: dict[str, list[int]] = {}
    all_scores: list[list[float]] = []

    for group in groups:
        scores = _group_scores(index, chunks, group)
        all_scores.append(scores)
        picked = _pick(
            chunks,
            scores,
            int(group.share * budget_chars),
            group.front_chunks,
            min_score,
        )
        per_group[group.name] = picked
        chosen.update(picked)

    used = sum(len(chunks[i].text) for i in chosen)

    if used < budget_chars:  # leftover budget: best remaining chunk of any group
        best = [max(s[i] for s in all_scores) for i in range(len(chunks))]

        for i in sorted(range(len(chunks)), key=lambda i: -best[i]):
            if best[i] <= 0:
                break
            if i in chosen:
                continue

            cost = len(chunks[i].text)
            if used + cost <= budget_chars:
                chosen.add(i)
                used += cost

    return EvidenceSelection(
        [chunks[i] for i in sorted(chosen)], per_group, budget_chars, total
    )


def score_groups(
    chunks: list[EvidenceChunk], groups: tuple[QueryGroup, ...] = GROUPS
) -> dict[str, list[float]]:
    """Score of every chunk for every group (for diagnostics: scripts/rag_recall.py --explain)."""
    index = _Bm25([tokens(c.text) for c in chunks])
    return {group.name: _group_scores(index, chunks, group) for group in groups}


# ---------------------------------------------------------------- the text the model gets

NOTE = (
    "SELECTED EXCERPTS. The text below is NOT the whole document: it is a set of excerpts chosen as "
    'the most relevant, in document order, with page markers. "[... omitted ...]" marks skipped text. '
    "A section that is not shown may still exist in the document. If the excerpts do not state a "
    "value, use null; never guess from what is missing."
)


def render_evidence(selection: EvidenceSelection) -> str:
    lines = [NOTE]
    previous: EvidenceChunk | None = None

    for chunk in selection.chunks:
        contiguous = previous is not None and chunk.index == previous.index + 1

        if previous is not None and not contiguous:
            lines.append("[... omitted ...]")

        if not contiguous or chunk.page_number != previous.page_number:
            lines.append(f"=== PAGE {chunk.page_number} ===")

        lines.append(chunk.text)
        previous = chunk

    return "\n\n".join(lines)
