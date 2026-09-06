"""
Tests for the credential guard / offline allowlist mechanism.

Verifies that:
1. Allowlisted providers pass validation
2. Non-allowlisted providers are rejected
3. Environment scrubbing removes blocked vars
4. Safety checks detect dangerous configurations
"""

import os
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.credential_guard import (
    validate_provider,
    scrub_env,
    check_env_safety,
    CredentialGuardError,
    OFFLINE_ALLOWLIST,
    BLOCKED_ENV_VARS,
)


class TestValidateProvider:
    """Test provider validation against the allowlist."""

    def test_localhost_allowed(self):
        """Local endpoints should always be allowed."""
        assert validate_provider("http://localhost:8000", "llama-3.3-70b")
        assert validate_provider("http://127.0.0.1:8080", "any-model")
        assert validate_provider("http://localhost:11434", "ollama-model")

    def test_together_blocked(self):
        """Together AI should be BLOCKED until explicit approval."""
        with pytest.raises(CredentialGuardError, match="not allowlisted"):
            validate_provider(
                "https://api.together.xyz/v1",
                "meta-llama/Llama-3.3-70B-Instruct"
            )

    def test_fireworks_blocked(self):
        """Fireworks AI should be BLOCKED until explicit approval."""
        with pytest.raises(CredentialGuardError, match="not allowlisted"):
            validate_provider(
                "https://api.fireworks.ai/inference/v1",
                "accounts/fireworks/models/llama-v3p3-70b-instruct"
            )

    def test_mock_allowed(self):
        """Mock adapter should always be allowed."""
        assert validate_provider("mock://test", "any-model")

    def test_anthropic_blocked(self):
        """Anthropic API should NOT be allowlisted."""
        with pytest.raises(CredentialGuardError, match="not allowlisted"):
            validate_provider("https://api.anthropic.com/v1", "claude-3-opus")

    def test_openai_blocked(self):
        """OpenAI direct API should NOT be allowlisted (proprietary)."""
        with pytest.raises(CredentialGuardError, match="not allowlisted"):
            validate_provider("https://api.openai.com/v1", "gpt-4")

    def test_unknown_blocked(self):
        """Unknown endpoints should be blocked."""
        with pytest.raises(CredentialGuardError, match="not allowlisted"):
            validate_provider("https://evil.example.com/v1", "model")

    def test_custom_allowlist(self):
        """Custom allowlist should override default."""
        custom = {("https://custom.example.com", "*")}
        assert validate_provider(
            "https://custom.example.com/v1", "model", allowlist=custom
        )
        with pytest.raises(CredentialGuardError):
            validate_provider(
                "http://localhost:8000", "model", allowlist=custom
            )

    def test_case_insensitive_url(self):
        """URL matching should be case-insensitive."""
        assert validate_provider("HTTP://LOCALHOST:8000", "model")
        assert validate_provider("HTTP://127.0.0.1:8080", "model")

    def test_trailing_slash_handling(self):
        """Trailing slashes should not affect matching."""
        assert validate_provider("http://localhost:8000/", "model")
        assert validate_provider("http://127.0.0.1:9090/", "model")


class TestScrubEnv:
    """Test environment variable scrubbing."""

    def test_removes_blocked_vars(self):
        """Blocked vars should be removed from environment."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key-123"
        os.environ["ANTHROPIC_BASE_URL"] = "https://api.anthropic.com"
        os.environ["TOGETHER_API_KEY"] = "test-together-key"
        os.environ["FIREWORKS_API_KEY"] = "test-fireworks-key"

        results = scrub_env()

        assert results["ANTHROPIC_API_KEY"] is True  # was present
        assert results["ANTHROPIC_BASE_URL"] is True
        assert results["TOGETHER_API_KEY"] is True
        assert results["FIREWORKS_API_KEY"] is True
        assert "ANTHROPIC_API_KEY" not in os.environ
        assert "ANTHROPIC_BASE_URL" not in os.environ
        assert "TOGETHER_API_KEY" not in os.environ
        assert "FIREWORKS_API_KEY" not in os.environ

    def test_reports_absent_vars(self):
        """Should report vars that were not present."""
        # Ensure they're not set
        for var in BLOCKED_ENV_VARS:
            os.environ.pop(var, None)

        results = scrub_env()

        assert results["ANTHROPIC_API_KEY"] is False
        assert results["ANTHROPIC_BASE_URL"] is False
        assert results["TOGETHER_API_KEY"] is False
        assert results["FIREWORKS_API_KEY"] is False

    def test_dry_run(self):
        """Dry run should report but not remove."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key-456"

        results = scrub_env(dry_run=True)

        assert results["ANTHROPIC_API_KEY"] is True
        assert "ANTHROPIC_API_KEY" in os.environ  # still there

        # Clean up
        del os.environ["ANTHROPIC_API_KEY"]

    def test_custom_blocked_vars(self):
        """Custom blocked vars should work."""
        os.environ["CUSTOM_SECRET"] = "secret"

        results = scrub_env(blocked_vars={"CUSTOM_SECRET"})

        assert results["CUSTOM_SECRET"] is True
        assert "CUSTOM_SECRET" not in os.environ

    def test_preserves_allowed_vars(self):
        """Scrubbing should not touch allowed vars."""
        os.environ["OPENAI_API_KEY"] = "sk-test-allowed"

        scrub_env()

        assert os.environ.get("OPENAI_API_KEY") == "sk-test-allowed"

        # Clean up
        del os.environ["OPENAI_API_KEY"]


class TestCheckEnvSafety:
    """Test environment safety checks."""

    def test_clean_environment(self):
        """Clean environment should be safe."""
        for var in BLOCKED_ENV_VARS:
            os.environ.pop(var, None)

        result = check_env_safety()

        assert result["safe"] is True
        assert result["blocked_present"] == []
        assert result["warnings"] == []

    def test_blocked_vars_detected(self):
        """Should detect blocked vars."""
        os.environ["ANTHROPIC_API_KEY"] = "test"

        result = check_env_safety()

        assert result["safe"] is False
        assert "ANTHROPIC_API_KEY" in result["blocked_present"]
        assert len(result["warnings"]) > 0

        # Clean up
        del os.environ["ANTHROPIC_API_KEY"]

    def test_anthropic_proxy_in_openai_url(self):
        """Should warn if OPENAI_BASE_URL points to Anthropic."""
        os.environ["OPENAI_BASE_URL"] = "https://proxy.anthropic.internal/v1"

        result = check_env_safety()

        assert any("anthropic" in w.lower() for w in result["warnings"])

        # Clean up
        del os.environ["OPENAI_BASE_URL"]

    def test_allowed_vars_reported(self):
        """Should report which allowed vars are set."""
        os.environ["OPENAI_API_KEY"] = "sk-test-allowed"

        result = check_env_safety()

        assert "OPENAI_API_KEY" in result["allowed_present"]

        # Clean up
        del os.environ["OPENAI_API_KEY"]

    def test_together_fireworks_now_blocked(self):
        """Together and Fireworks keys should be detected as blocked."""
        os.environ["TOGETHER_API_KEY"] = "test-together"
        os.environ["FIREWORKS_API_KEY"] = "test-fireworks"

        result = check_env_safety()

        assert result["safe"] is False
        assert "TOGETHER_API_KEY" in result["blocked_present"]
        assert "FIREWORKS_API_KEY" in result["blocked_present"]

        # Clean up
        del os.environ["TOGETHER_API_KEY"]
        del os.environ["FIREWORKS_API_KEY"]


class TestIntegration:
    """Integration tests combining guard + scrub."""

    def test_scrub_then_validate(self):
        """Full workflow: scrub env, then validate provider."""
        os.environ["ANTHROPIC_API_KEY"] = "dangerous-key"
        os.environ["ANTHROPIC_BASE_URL"] = "https://api.anthropic.com"

        # Step 1: Scrub
        scrub_env()

        # Step 2: Validate approved provider
        assert validate_provider("http://localhost:8000", "llama-3.3-70b")

        # Step 3: Block non-approved provider
        with pytest.raises(CredentialGuardError):
            validate_provider("https://api.anthropic.com/v1", "claude-3")

    def test_blocked_env_vars_match_known_risks(self):
        """Verify the blocked vars list covers known risks."""
        assert "ANTHROPIC_API_KEY" in BLOCKED_ENV_VARS
        assert "ANTHROPIC_BASE_URL" in BLOCKED_ENV_VARS
        assert "TOGETHER_API_KEY" in BLOCKED_ENV_VARS
        assert "FIREWORKS_API_KEY" in BLOCKED_ENV_VARS

    def test_together_fireworks_endpoints_blocked(self):
        """Together and Fireworks endpoints should be blocked."""
        with pytest.raises(CredentialGuardError):
            validate_provider("https://api.together.xyz/v1", "any-model")
        with pytest.raises(CredentialGuardError):
            validate_provider("https://api.fireworks.ai/inference/v1", "any-model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
