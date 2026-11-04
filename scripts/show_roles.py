import json
import sys
from pathlib import Path

path = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "data/extractions/c41b1645-96b3-46ae-b0d2-4830f272d1fe__pdf.json"
)
d = json.loads(path.read_text(encoding="utf-8"))
kp = d["key_personnel"]

print("key_person_months_total:", kp["key_person_months_total"])
for r in kp["roles"]:
    print(
        f"  {r['role'][:45]:<45} pm={r['person_months']} "
        f"cond={r['is_conditional']} rate={r['rate_per_month']}"
    )

print("delay_damages:", d["contract"]["delay_damages"])
print(
    "criteria (name, subs, bands):",
    [
        (
            c["name"][:25],
            len(c["sub_criteria"]),
            sum(len(s["bands"]) for s in c["sub_criteria"]),
        )
        for c in d["evaluation"]["criteria"]
    ],
)
print("flags:")
for flag in d["meta"]["review_flags"]:
    print("  -", flag)
