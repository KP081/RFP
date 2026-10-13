import argparse
import json
import os
import subprocess
import sys

BGE_PREFIX = "Represent this sentence for searching relevant passages: "

# baseline = tumhari current .env (1000/150, MiniLM, rfp_documents collection)
EXPERIMENTS = {
    "baseline": {},
    "minilm_600": {
        "CHUNK_SIZE": "600",
        "CHUNK_OVERLAP": "90",
        "QDRANT_COLLECTION": "exp_minilm_600",
    },
    "bge_1000": {
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "EMBEDDING_QUERY_PREFIX": BGE_PREFIX,
        "QDRANT_COLLECTION": "exp_bge_1000",
    },
    "bge_600": {
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "EMBEDDING_QUERY_PREFIX": BGE_PREFIX,
        "CHUNK_SIZE": "600",
        "CHUNK_OVERLAP": "90",
        "QDRANT_COLLECTION": "exp_bge_600",
    },
}

# Inme se kuch badle to re-ingest chahiye (aur naya collection bhi).
INGEST_KEYS = {
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "EMBEDDING_MODEL",
    "EMBEDDING_MAX_SEQ_LENGTH",
    "EMBEDDING_DOCUMENT_PREFIX",
}


def run(module: str, overrides: dict[str, str]) -> subprocess.CompletedProcess:
    # Har experiment alag process me chalta hai, to settings/model cache mix nahi hote.
    return subprocess.run(
        [sys.executable, "-m", module],
        env={**os.environ, **overrides},
        capture_output=True,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="experiment names (default: all)")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    rows = []

    for name in args.names or list(EXPERIMENTS):
        overrides = EXPERIMENTS[name]
        changes_ingest = bool(INGEST_KEYS & overrides.keys())

        if changes_ingest and "QDRANT_COLLECTION" not in overrides:
            raise SystemExit(f"{name}: set QDRANT_COLLECTION for ingest changes.")

        print(f"\n=== {name}: {overrides}")

        if changes_ingest and not args.skip_ingest:
            print("re-ingesting (kuch minute lag sakte hain)...")
            proc = run("scripts.reingest_all", overrides)

            if proc.returncode != 0:
                print(proc.stderr[-2000:])
                continue

        proc = run("scripts.run_eval", overrides)
        lines = proc.stdout.splitlines()
        result = next((ln for ln in lines if ln.startswith("RESULT ")), None)

        if proc.returncode != 0 or result is None:
            print(proc.stderr[-2000:] or proc.stdout[-2000:])
            continue

        print("\n".join(ln for ln in lines if not ln.startswith("RESULT ")))
        rows.append((name, json.loads(result[len("RESULT ") :])))

    print("\n" + "=" * 55)
    print(f"{'experiment':<14}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'mrr':>8}")
    for name, m in rows:
        print(
            f"{name:<14}{m['hit@1']:>8.1%}{m['hit@3']:>8.1%}"
            f"{m['hit@5']:>8.1%}{m['mrr']:>8.3f}"
        )


if __name__ == "__main__":
    main()
