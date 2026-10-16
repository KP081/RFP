import copy
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.core.exceptions import AppError, LLMError

logger = logging.getLogger(__name__)

# The Gemini Developer API rejected our full schema (HTTP 400), so the schema goes
# into the PROMPT instead of response_schema. Constraints are dropped from that text
# (Pydantic enforces them after the call); descriptions are kept: they carry rules.
_DROP_KEYS = {
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

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_RETRY_IN = re.compile(
    r"retry in ([\d.]+)\s*s|retryDelay['\"]?\s*:\s*['\"]?([\d.]+)s", re.IGNORECASE
)


def _rate_limit_wait(message: str) -> float | None:
    """Seconds to wait if this error is a per-minute rate limit (429), else None.

    Daily quota errors are not retried: waiting a minute would not help.
    """
    if "429" not in message and "RESOURCE_EXHAUSTED" not in message:
        return None

    if "PerDay" in message or "per day" in message.lower():
        return None

    match = _RETRY_IN.search(message)
    seconds = float(next(group for group in match.groups() if group)) if match else 60.0

    return min(seconds + 3, 120.0)


def lean_json_schema(model: type[BaseModel]) -> dict:
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
                elif key in _DROP_KEYS:
                    continue
                else:
                    out[key] = resolve(value)
            return out

        if isinstance(node, list):
            return [resolve(item) for item in node]

        return node

    return resolve(schema)


@dataclass
class Usage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    calls: int = 0

    def add(self, usage_metadata) -> None:
        self.calls += 1
        self.prompt_tokens += getattr(usage_metadata, "prompt_token_count", 0) or 0
        self.output_tokens += getattr(usage_metadata, "candidates_token_count", 0) or 0
        self.thinking_tokens += getattr(usage_metadata, "thoughts_token_count", 0) or 0


@dataclass
class StructuredResult[T: BaseModel]:
    data: T
    usage: Usage
    repairs: int


class LLMService:
    def __init__(self):
        settings = get_settings()

        if not settings.gemini_api_key:
            raise AppError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=settings.gemini_timeout_s * 1000),
        )
        self.model = settings.gemini_model
        self.temperature = settings.llm_temperature

    # ---------- plain text (RAG answers) ----------

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=self.temperature),
            )
        except Exception as e:
            raise LLMError(f"LLM request failed: {e}") from e

        if not response.text:
            raise LLMError("LLM returned an empty response.")

        return response.text

    # ---------- files (Gemini Files API, stored 48h, free) ----------

    def upload_file(self, path: Path):
        try:
            uploaded = self.client.files.upload(file=str(path))

            while _state(uploaded) == "PROCESSING":
                time.sleep(2)
                uploaded = self.client.files.get(name=uploaded.name)
        except Exception as e:
            raise LLMError(f"File upload failed: {e}") from e

        if _state(uploaded) == "FAILED":
            raise LLMError("Gemini could not process the uploaded file.")

        return uploaded

    def delete_file(self, uploaded) -> None:
        try:
            self.client.files.delete(name=uploaded.name)
        except Exception:
            logger.warning("Could not delete uploaded file %s", uploaded.name)

    # ---------- structured output ----------

    def generate_structured[T: BaseModel](
        self,
        instructions: str,
        schema: type[T],
        *,
        parts: list | None = None,
        max_output_tokens: int | None = None,
        max_repairs: int | None = None,
    ) -> StructuredResult[T]:
        """ONE call: document parts first, then instructions + schema text.

        Output is validated with Pydantic. On failure we make a cheap repair call
        that sends only the bad JSON and the errors, NOT the document again.
        """
        settings = get_settings()
        token_limit = max_output_tokens or settings.extract_max_output_tokens
        repairs_allowed = (
            settings.extract_repair_attempts if max_repairs is None else max_repairs
        )

        schema_text = json.dumps(lean_json_schema(schema), separators=(",", ":"))
        prompt = (
            f"{instructions}\n\n"
            "Return ONLY one JSON object that follows this JSON schema. "
            "No markdown, no commentary.\n\n"
            f"JSON SCHEMA:\n{schema_text}"
        )

        usage = Usage()
        text = self._call([*(parts or []), prompt], token_limit, usage)
        data, error = self._parse(text, schema)
        repairs = 0

        while data is None and repairs < repairs_allowed:
            repairs += 1
            logger.warning("Structured output invalid, repair %d: %s", repairs, error)

            repair_prompt = (
                "Your previous JSON failed validation. You can no longer see the "
                "document. Fix ONLY the listed problems using your previous JSON. "
                "Use null only for a plain value that cannot be fixed; an object must "
                "stay an object (with all its fields null) and a list must stay a list "
                "([]). Return the complete corrected JSON object.\n\n"
                f"ERRORS:\n{error}\n\n"
                f"PREVIOUS JSON:\n{text}\n\n"
                f"JSON SCHEMA:\n{schema_text}"
            )
            text = self._call([repair_prompt], token_limit, usage)
            data, error = self._parse(text, schema)

        if data is None:
            dumped = _dump_invalid(text)
            raise LLMError(
                f"Model output failed validation after {repairs} repair attempt(s): "
                f"{error} (raw output saved: {dumped})"
            )

        logger.info(
            "Structured call done: calls=%d prompt_tokens=%d output_tokens=%d repairs=%d",
            usage.calls,
            usage.prompt_tokens,
            usage.output_tokens,
            repairs,
        )

        return StructuredResult(data=data, usage=usage, repairs=repairs)

    def _call(self, contents: list, max_output_tokens: int, usage: Usage) -> str:
        retries = getattr(get_settings(), "gemini_rate_limit_retries", 3)
        attempt = 0

        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        response_mime_type="application/json",
                        max_output_tokens=max_output_tokens,
                    ),
                )
                break
            except Exception as e:
                wait = _rate_limit_wait(str(e))

                if wait is None or attempt >= retries:
                    raise LLMError(f"LLM request failed: {e}") from e

                attempt += 1
                logger.warning(
                    "Rate limited (429). Waiting %.0fs, retry %d/%d",
                    wait,
                    attempt,
                    retries,
                )
                time.sleep(wait)

        usage.add(response.usage_metadata)

        if _finish_reason(response) == "MAX_TOKENS":
            raise LLMError(
                "Model output was cut off (MAX_TOKENS); the JSON is incomplete."
            )

        if not response.text:
            raise LLMError("LLM returned an empty response.")

        return response.text

    @staticmethod
    def _parse(text: str, schema: type[BaseModel]):
        cleaned = _FENCE.sub("", text.strip())

        try:
            return schema.model_validate_json(cleaned), None
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}"
                for err in e.errors()[:20]
            )
            return None, problems


def _state(uploaded) -> str:
    return getattr(getattr(uploaded, "state", None), "name", "") or ""


def _finish_reason(response) -> str:
    candidates = getattr(response, "candidates", None) or []

    if not candidates:
        return ""

    reason = getattr(candidates[0], "finish_reason", None)
    return getattr(reason, "name", str(reason or ""))


def _dump_invalid(text: str) -> str:
    """Keep the last invalid model output, so we can see what the model really wrote."""
    try:
        folder = Path("data/debug")
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        target = folder / f"invalid_{stamp}.json"
        target.write_text(text, encoding="utf-8")
        return str(target)
    except OSError:
        return "could not write"
