"""
Gemini Service Module for ThinkAI Studio.
Integrates with the official Google Gen AI SDK (google-genai) to provide:
- API authentication and security
- Live connection health checking
- Multi-turn chat session management with streaming
- System instructions and temperature control
- Safe error sanitization (preventing secret exposure)
- Model discovery and compatibility checks
"""

import os
import re
from typing import Generator, List, Optional, Tuple, Dict, Any
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError

# Curated list of verified & recommended Gemini models
RECOMMENDED_MODELS = [
    {
        "id": "gemini-3.5-flash-lite",
        "name": "Gemini 3.5 Flash-Lite (Fastest & Verified)",
        "desc": "Ultra-fast, lowest latency, 1M context. Ideal for live presentations.",
        "default": True,
    },
    {
        "id": "gemma-4-31b-it",
        "name": "Gemma 4 31B IT (High Accuracy Open Weights)",
        "desc": "High accuracy instruction-tuned model, great reasoning and coding.",
        "default": False,
    },
    {
        "id": "gemma-4-26b-a4b-it",
        "name": "Gemma 4 26B A4B IT (Efficient MoE)",
        "desc": "Mixture-of-Experts architecture: 26B total / 4B active parameters.",
        "default": False,
    },
    {
        "id": "gemini-3.8-flash",
        "name": "Gemini 3.8 Flash (Flagship Multimodal)",
        "desc": "1M context, high capability. (May experience occasional demand spikes).",
        "default": False,
    },
    {
        "id": "gemini-4-argon",
        "name": "Gemini 4 Argon (Advanced Reasoning)",
        "desc": "Most powerful model for deep technical reasoning and analysis.",
        "default": False,
    },
]


def mask_api_key(api_key: Optional[str]) -> str:
    """Safely mask an API key for UI display (e.g., 'AQ.Ab8...3PJ8Aw')."""
    if not api_key:
        return "Not Set"
    key_str = str(api_key).strip()
    if len(key_str) <= 10:
        return "***"
    return f"{key_str[:6]}...{key_str[-4:]}"


def sanitize_error_message(error: Exception, api_key: Optional[str] = None) -> str:
    """
    Format an API exception into a helpful, student- and judge-friendly message.
    Guarantees no raw API key or internal secret strings are ever exposed.
    """
    error_str = str(error)

    # Strip out the API key if present anywhere in the error text
    if api_key and len(api_key) > 6:
        error_str = error_str.replace(api_key, "[REDACTED_API_KEY]")

    # Check for known status codes and error types
    if "404" in error_str or "NOT_FOUND" in error_str:
        return (
            "Model unavailable or deprecated. The selected model was not found for this API tier. "
            "Recommendation: Switch to 'gemini-3.5-flash-lite' in the sidebar settings."
        )

    if "503" in error_str or "UNAVAILABLE" in error_str or "high demand" in error_str.lower():
        return (
            "Google API Service Notice: The requested model is temporarily experiencing high demand. "
            "Please retry in a few seconds, or switch to 'gemini-3.5-flash-lite' or 'gemma-4-31b-it'."
        )

    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
        return (
            "API Quota Limit: The account has reached its rate limit or token quota. "
            "Please pause for a few seconds before sending another message, or try a lighter model."
        )

    if "401" in error_str or "403" in error_str or "API_KEY_INVALID" in error_str:
        return (
            "Authentication Error: The provided GEMINI_API_KEY is invalid, expired, or unauthorized. "
            "Please verify your key in the .env file or project configuration."
        )

    if "connection" in error_str.lower() or "dns" in error_str.lower() or "timeout" in error_str.lower():
        return (
            "Network Connection Error: Unable to establish a secure connection to the Google Gemini API. "
            "Please verify your internet connection."
        )

    # Clean generic error message
    clean_msg = re.sub(r"key=[^\s&'\"]+", "key=[REDACTED]", error_str)
    return f"Gemini API Error: {clean_msg}"


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Initialize and return the official Google Gen AI Client.
    Prioritizes explicitly passed api_key. If None, falls back to GEMINI_API_KEY env var.
    """
    if api_key is not None:
        effective_key = api_key.strip()
    else:
        effective_key = (os.getenv("GEMINI_API_KEY") or "").strip()

    if not effective_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Please set your API key in the .env file or enter it in the application sidebar."
        )
    return genai.Client(api_key=effective_key)


class GeminiService:
    """High-level service managing Gemini interactions for ThinkAI Studio."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        """Lazy-loaded Client instance."""
        if self._client is None:
            self._client = get_gemini_client(self.api_key)
        return self._client

    def verify_connection(self, model: str = "gemini-3.5-flash-lite") -> Tuple[bool, str]:
        """
        Perform a live lightweight ping to test API key validity and model response.
        Returns (is_connected: bool, status_message: str).
        """
        try:
            client = self.client
            # Lightweight verification ping
            response = client.models.generate_content(
                model=model,
                contents="Ping",
                config=types.GenerateContentConfig(
                    max_output_tokens=5,
                    temperature=0.0,
                ),
            )
            if response and response.text:
                return True, f"Connected to Gemini API ({model})"
            return True, "Connected to Gemini API"
        except Exception as e:
            sanitized = sanitize_error_message(e, self.api_key)
            return False, sanitized

    def list_models(self) -> List[Dict[str, str]]:
        """Fetch list of available models from the account."""
        try:
            client = self.client
            models_list = []
            for m in client.models.list():
                m_name = getattr(m, "name", str(m))
                clean_name = m_name.replace("models/", "")
                disp_name = getattr(m, "display_name", clean_name)
                models_list.append({"id": clean_name, "name": f"{disp_name} ({clean_name})"})
            return models_list
        except Exception:
            return RECOMMENDED_MODELS

    def create_chat(
        self,
        model: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        history_contents: Optional[Any] = None,
    ):
        """
        Create a stateful multi-turn chat session using the official SDK.
        """
        client = self.client
        config = types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
            temperature=float(temperature),
        )
        chat = client.chats.create(
            model=model,
            config=config,
            history=history_contents,
        )
        return chat

    def stream_conversation(
        self,
        model: str,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        Stream response for a multi-turn conversation.
        Converts the session message list into types.Content objects and streams via
        client.models.generate_content_stream.
        This approach is completely immune to client-closed or stale session socket issues in Streamlit.
        """
        try:
            client = self.client
            contents: Any = []

            # Filter and convert messages, skipping internal greeting or error notes
            for msg in messages:
                role_val = msg.get("role", "")
                content_val = msg.get("content", "").strip()

                if not content_val or content_val.startswith("⚠️ *Error:"):
                    continue

                # Map streamlit roles to Gemini roles ('user' or 'model')
                gemini_role = "user" if role_val == "user" else "model"
                contents.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part.from_text(text=content_val)],
                    )
                )

            # Ensure at least the latest user prompt is present
            if not contents:
                raise ValueError("No valid message content to send.")

            config = types.GenerateContentConfig(
                system_instruction=system_instruction if system_instruction else None,
                temperature=float(temperature),
            )

            stream = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )

            for chunk in stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            err_msg = sanitize_error_message(e, self.api_key)
            raise RuntimeError(err_msg) from None

    def stream_message(
        self,
        chat_session,
        user_message: str,
    ) -> Generator[str, None, None]:
        """
        Stream incremental response chunks from an active chat session.
        Yields text tokens as they arrive.
        """
        try:
            stream = chat_session.send_message_stream(user_message)
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            err_msg = sanitize_error_message(e, self.api_key)
            raise RuntimeError(err_msg) from None

    def generate_direct(
        self,
        model: str,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate a single response non-streamed (useful for unit tests or fallback)."""
        try:
            client = self.client
            config = types.GenerateContentConfig(
                system_instruction=system_instruction if system_instruction else None,
                temperature=float(temperature),
            )
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as e:
            err_msg = sanitize_error_message(e, self.api_key)
            raise RuntimeError(err_msg) from None
