"""
Probe: will Gemini accept the FULL TenderExtraction schema in ONE call?

Why: Google documents that complex response schemas (many optional properties,
many enums, nested arrays, min/max constraints) can be rejected with HTTP 400.
Our schema has all of those, and one-call extraction sends all of it at once.

Run from the project root (needs GEMINI_API_KEY in .env):

    uv run python -m scripts.probe_gemini_schema
    uv run python -m scripts.probe_gemini_schema --pdf uploads/<file>.pdf

Three levels are tried, from strictest to loosest:
  1. full   : response_schema=TenderExtraction (SDK converts the Pydantic model)
  2. lean   : same schema with constraints stripped (min/max/maxItems/...), $refs inlined
  3. prompt : no response_schema, JSON mime type only, schema text in the prompt

Levels 2-3 are safe only because Pydantic re-validates everything afterwards.
With --pdf it also uploads the PDF and prints the REAL token count (free call),
so we stop guessing "258 tokens per page".
"""

import argparse
import copy
import json
import socket
import sys
import time

SAMPLE = """
NOTICE INVITING TENDER. NHIDCL invites proposals for Consultancy Services for
preparation of Detailed Project Report for construction of 4-Lane Access Controlled
High Speed Corridor from Siliguri to Guwahati in the States of West Bengal and Assam.
Cost of the document: non-refundable fee of Rs.5,000 (Rupees Five Thousand only) + 18% GST
(Rs. 5,900/-). Duration of the Project: 12 months. Pre-proposal conference:
Date: 19.06.2025, Time: 03:00 PM, Venue: World Trade Centre, Nauroji Nagar, New Delhi.
Consultants must register on the INFRACON portal (firm and all key personnel).
Sub-contracting is not allowed beyond the limit given in the Data Sheet.
"""

# keys that make Gemini schemas "complex" (or are not supported anyway)
DROP_KEYS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
    "default",
    "title",
    "additionalProperties",
}


def lean_schema(model) -> dict:
    """Pydantic model -> plain JSON schema: $refs inlined, constraints removed."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            out = {}
            for key, value in node.items():
                if key == "properties":
                    # property NAMES (e.g. a field called "title") must survive
                    out[key] = {name: resolve(sub) for name, sub in value.items()}
                elif key in DROP_KEYS:
                    continue
                else:
                    out[key] = resolve(value)
            return out
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


def llm_model(TenderExtraction):
    """LLM-facing root model = everything except `meta`.

    `meta` is filled by the backend and contains `pages_used: dict[str, list[int]]`,
    an open dict -> JSON schema `additionalProperties`, which the Gemini Developer
    API rejects. Uses TenderExtractionLLM if extraction.py already defines it.
    """
    from pydantic import create_model

    module = sys.modules[TenderExtraction.__module__]
    existing = getattr(module, "TenderExtractionLLM", None)
    if existing is not None:
        return existing

    fields = {
        name: (field.annotation, field)
        for name, field in TenderExtraction.model_fields.items()
        if name != "meta"
    }
    return create_model("TenderExtractionLLM", **fields)


def check_network(host: str = "generativelanguage.googleapis.com") -> None:
    try:
        socket.gethostbyname(host)
    except OSError as exc:
        raise SystemExit(
            f"DNS lookup for {host} failed ({exc}). Fix internet/VPN/DNS first, "
            "otherwise every level below fails with 'getaddrinfo failed'."
        )


def attempt(client, types, model_name, label, prompt, config, TenderExtraction):
    started = time.time()
    try:
        response = client.models.generate_content(
            model=model_name, contents=prompt, config=config
        )
    except Exception as exc:  # the interesting case: HTTP 400 "too many states"
        print(f"[{label}] REJECTED by server: {str(exc)[:400]}\n")
        return False

    usage = response.usage_metadata
    print(
        f"[{label}] accepted in {time.time() - started:.1f}s | "
        f"prompt_tokens={getattr(usage, 'prompt_token_count', None)} "
        f"output_tokens={getattr(usage, 'candidates_token_count', None)} "
        f"thinking_tokens={getattr(usage, 'thoughts_token_count', None)}"
    )

    try:
        doc = TenderExtraction.model_validate_json(response.text)
    except Exception as exc:
        print(f"[{label}] but Pydantic validation FAILED: {str(exc)[:400]}\n")
        return False

    print(f"   dates.pre_bid        = {doc.dates.pre_bid}")
    print(f"   securities.document_fee = {doc.securities.document_fee}")
    print(f"   contract.duration_months = {doc.contract.duration_months}")
    print(f"   identity.title       = {doc.identity.title}\n")
    return True


def main() -> None:
    from google import genai
    from google.genai import types

    from app.core.config import get_settings
    from app.schemas.extraction import TenderExtraction

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="optional: PDF to upload and token-count")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    check_network()

    settings = get_settings()
    model_name = args.model or settings.gemini_model
    LLMModel = llm_model(TenderExtraction)
    client = genai.Client(api_key=settings.gemini_api_key)
    print(f"model = {model_name}\n")

    base_prompt = (
        "Extract tender information from the text into the JSON schema. "
        "Use null or an empty list when a value is not stated. Never guess.\n\n"
        f"TEXT:\n{SAMPLE}"
    )

    lean = lean_schema(LLMModel)
    assert "additionalProperties" not in json.dumps(lean)
    print(f"lean schema size: {len(json.dumps(lean)):,} chars\n")

    attempt(
        client,
        types,
        model_name,
        "1 full",
        base_prompt,
        types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=LLMModel,
        ),
        LLMModel,
    )

    attempt(
        client,
        types,
        model_name,
        "2 lean",
        base_prompt,
        types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_json_schema=lean,
        ),
        LLMModel,
    )

    attempt(
        client,
        types,
        model_name,
        "3 prompt-only",
        base_prompt + "\n\nJSON SCHEMA:\n" + json.dumps(lean),
        types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json"
        ),
        LLMModel,
    )

    if args.pdf:
        uploaded = client.files.upload(file=args.pdf)
        while getattr(getattr(uploaded, "state", None), "name", "") == "PROCESSING":
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        counted = client.models.count_tokens(model=model_name, contents=[uploaded])
        from pypdf import PdfReader

        pages = len(PdfReader(args.pdf).pages)
        print(
            f"PDF {args.pdf}: pages = {pages}, total_tokens = {counted.total_tokens:,}, "
            f"tokens/page = {counted.total_tokens / pages:,.0f}"
        )


if __name__ == "__main__":
    main()
