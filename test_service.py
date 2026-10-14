"""
Unit and integration tests for ThinkAI Studio services.
"""

import os
import unittest
from dotenv import load_dotenv

# Load local environment
load_dotenv()

from services.gemini_service import (
    GeminiService,
    get_gemini_client,
    mask_api_key,
    sanitize_error_message,
    RECOMMENDED_MODELS,
)


class TestGeminiService(unittest.TestCase):
    """Test suite for Gemini service functions."""

    def test_mask_api_key(self):
        """Verify API key masking never leaks full keys."""
        self.assertEqual(mask_api_key(None), "Not Set")
        self.assertEqual(mask_api_key(""), "Not Set")
        self.assertEqual(mask_api_key("short"), "***")
        masked = mask_api_key("AQ.Ab8RN6I0ho4OCyMbO0MLQ5RBnrRXgwcAbwvXa3wkW8863PJ8Aw")
        self.assertTrue(masked.startswith("AQ.Ab8"))
        self.assertTrue(masked.endswith("J8Aw"))
        self.assertIn("...", masked)
        self.assertNotIn("RBnrRXgwc", masked)

    def test_sanitize_error_message_redacts_key(self):
        """Verify sensitive strings are scrubbed from errors."""
        secret = "AQ.SECRET_KEY_12345"
        raw_error = Exception(f"Failed to connect using key={secret} due to internal error.")
        sanitized = sanitize_error_message(raw_error, api_key=secret)
        self.assertNotIn(secret, sanitized)
        self.assertIn("[REDACTED", sanitized)

    def test_sanitize_error_message_known_codes(self):
        """Verify standard HTTP/API errors return user-friendly advice."""
        err_404 = Exception("404 NOT_FOUND. Model not found.")
        self.assertIn("Model unavailable or deprecated", sanitize_error_message(err_404))

        err_503 = Exception("503 UNAVAILABLE. High demand spikes.")
        self.assertIn("high demand", sanitize_error_message(err_503).lower())

        err_429 = Exception("429 RESOURCE_EXHAUSTED. Quota exceeded.")
        self.assertIn("quota", sanitize_error_message(err_429).lower())

        err_auth = Exception("403 API_KEY_INVALID. Forbidden.")
        self.assertIn("Authentication Error", sanitize_error_message(err_auth))

    def test_client_init_missing_key(self):
        """Verify missing key raises explicit ValueError."""
        with self.assertRaises(ValueError):
            get_gemini_client(api_key="")

    def test_recommended_models_structure(self):
        """Verify recommended models list is well-formed."""
        self.assertGreater(len(RECOMMENDED_MODELS), 0)
        has_default = any(m.get("default") for m in RECOMMENDED_MODELS)
        self.assertTrue(has_default)
        for m in RECOMMENDED_MODELS:
            self.assertIn("id", m)
            self.assertIn("name", m)


class TestLiveGeminiInference(unittest.TestCase):
    """Live inference integration test if GEMINI_API_KEY is configured."""

    def setUp(self):
        self.api_key = os.getenv("GEMINI_API_KEY")

    def test_live_ping_and_generation(self):
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY not configured. Skipping live inference test.")

        service = GeminiService(api_key=self.api_key)
        
        # Test connection verification
        is_connected, msg = service.verify_connection(model="gemini-3.5-flash-lite")
        self.assertTrue(is_connected, f"Connection verification failed: {msg}")

        # Test single-turn prompt
        answer = service.generate_direct(
            model="gemini-3.5-flash-lite",
            prompt="Respond with the single word 'VERIFIED'.",
            temperature=0.0,
        )
        self.assertIn("VERIFIED", answer.strip().upper())

        # Test multi-turn stateless conversation streaming
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Say the word 'READY'."},
        ]
        conv_chunks = list(service.stream_conversation(
            model="gemini-3.5-flash-lite",
            messages=messages,
            temperature=0.0,
        ))
        conv_reply = "".join(conv_chunks)
        self.assertIn("READY", conv_reply.upper())


if __name__ == "__main__":
    unittest.main()
