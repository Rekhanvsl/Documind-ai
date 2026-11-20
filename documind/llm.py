"""Talks to the AI provider (Gemini by default, Claude optional)."""
import time

from . import config


class LLMError(Exception):
    """Friendly error shown to the user."""


class LLMClient:
    def __init__(self):
        self._client = None

    # ------------------------------------------------------------ public
    def complete(self, system: str, user: str, max_tokens: int = 1200) -> str:
        if config.PROVIDER == "claude":
            return self._claude(system, user, max_tokens)
        if config.PROVIDER == "gemini":
            return self._gemini(system, user, max_tokens)
        raise LLMError(f"Unknown DOCUMIND_PROVIDER '{config.PROVIDER}'. Use 'gemini' or 'claude'.")

    # ------------------------------------------------------------ Gemini
    def _gemini(self, system, user, max_tokens):
        if not config.GEMINI_API_KEY:
            raise LLMError(
                "No Gemini API key found.\n1. Get a free key at https://aistudio.google.com/apikey\n"
                "2. Put it in your .env file as GEMINI_API_KEY=...\n3. Restart the app."
            )
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as e:
            raise LLMError("Missing package. Run: pip install -r requirements.txt") from e

        if self._client is None:
            self._client = genai.Client(api_key=config.GEMINI_API_KEY)

        # Gemini counts its internal "thinking" tokens inside the output limit,
        # so we leave extra room to avoid empty answers.
        cfg = types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max_tokens * 4
        )
        delays = [5, 15, 30]  # wait and retry when the free tier is busy
        for attempt in range(len(delays) + 1):
            try:
                resp = self._client.models.generate_content(
                    model=config.GEMINI_MODEL, contents=user, config=cfg
                )
                text = (resp.text or "").strip()
                if not text:
                    raise LLMError("The AI returned an empty answer. Please try again.")
                return text
            except errors.APIError as e:
                code = getattr(e, "code", None)
                if code in (429, 500, 503) and attempt < len(delays):
                    time.sleep(delays[attempt])
                    continue
                if code == 429:
                    raise LLMError(
                        "Free-tier limit reached. Wait a minute and try again, or set "
                        "GEMINI_MODEL=gemini-3.5-flash-lite in your .env file."
                    ) from e
                if code in (400, 401, 403):
                    raise LLMError(
                        "Gemini rejected the request. Check that your GEMINI_API_KEY is "
                        f"correct and the model name is valid.\n\nDetails: {e}"
                    ) from e
                raise LLMError(f"Gemini error: {e}") from e
            except LLMError:
                raise
            except Exception as e:
                raise LLMError(f"Could not reach the AI service. Check your internet.\n\n{e}") from e

    # ------------------------------------------------------------ Claude
    def _claude(self, system, user, max_tokens):
        if not config.ANTHROPIC_API_KEY:
            raise LLMError("No ANTHROPIC_API_KEY found in your .env file.")
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("Run: pip install anthropic") from e
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        try:
            resp = self._client.messages.create(
                model=config.CLAUDE_MODEL, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.APIStatusError as e:
            raise LLMError(f"The AI service returned an error: {e}") from e
        except Exception as e:
            raise LLMError(f"Could not reach the AI service: {e}") from e
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
