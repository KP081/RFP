from google import genai
from google.genai import types

from app.core.config import get_settings
from app.core.exceptions import AppError, LLMError


class LLMService:
    def __init__(self):
        settings = get_settings()

        if not settings.gemini_api_key:
            raise AppError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model
        self.temperature = settings.llm_temperature

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
