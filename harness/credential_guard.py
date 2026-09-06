"""
Credential guard / offline allowlist mechanism for the inference harness.

PURPOSE: Prevent the eval harness from accidentally using platform proxy
credentials (e.g., ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL) for model
inference calls. Only explicitly allowlisted endpoints should be used.

MECHANISM:
  1. An ALLOWLIST of approved (base_url, model_id) pairs.
  2. A guard function that checks whether a provider's credentials match
     an allowlisted entry before allowing any API call.
  3. An environment scrubber that removes dangerous env vars before
     provider initialization.

USAGE:
  from harness.credential_guard import validate_provider, scrub_env

  # Before creating a provider:
  scrub_env()  # removes ANTHROPIC_* from os.environ

  # Before making any API call:
  validate_provider(base_url, model_id)  # raises if not allowlisted
"""

import os
from typing import Optional, Set, Tuple


# ================================================================
# ALLOWLIST
# ================================================================

# Approved (base_url_prefix, model_id_pattern) pairs.
# Only providers matching an entry here are allowed to make API calls.
# base_url_prefix is matched as a prefix (case-insensitive).
# model_id_pattern is matched as an exact string or "*" for any model.
#
# This list should be updated when new inference endpoints are approved.
OFFLINE_ALLOWLIST: Set[Tuple[str, str]] = {
    # Local/self-hosted endpoints
    ("http://localhost", "*"),
    ("http://127.0.0.1", "*"),
    ("http://0.0.0.0", "*"),

    # vLLM / TGI / Ollama typical ports
    ("http://localhost:8000", "*"),
    ("http://localhost:11434", "*"),

    # Together AI — DISABLED until explicit model-access approval
    # ("https://api.together.xyz", "*"),

    # Fireworks AI — DISABLED until explicit model-access approval
    # ("https://api.fireworks.ai", "*"),

    # Mock adapter (always allowed)
    ("mock://", "*"),
}

# Environment variables that must NEVER be used for inference.
# Anthropic vars are platform proxy credentials.
# All cloud API vars blocked until explicit model-access approval.
BLOCKED_ENV_VARS = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "HF_API_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
}

# Known real-provider URL prefixes that must be blocked.
# Used for deterministic testing that all real endpoints are rejected.
KNOWN_BLOCKED_ENDPOINTS = [
    "https://api.anthropic.com",
    "https://api.openai.com",
    "https://api.together.xyz",
    "https://api.fireworks.ai",
    "https://api-inference.huggingface.co",
    "https://generativelanguage.googleapis.com",
    "https://api.mistral.ai",
    "https://models.inference.ai.azure.com",
]

# Environment variables that are safe to use for inference credentials.
# Note: OPENAI_* vars are allowed because they are used by local/mock
# adapters that speak the OpenAI-compatible API protocol.
# Together/Fireworks vars have been moved to BLOCKED_ENV_VARS until
# explicit model-access approval is granted.
ALLOWED_CREDENTIAL_VARS = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
}


# ================================================================
# GUARD FUNCTIONS
# ================================================================

class CredentialGuardError(Exception):
    """Raised when a credential guard check fails."""
    pass


def validate_provider(
    base_url: str,
    model_id: str = "*",
    allowlist: Optional[Set[Tuple[str, str]]] = None,
) -> bool:
    """Validate that a provider's base_url and model_id are allowlisted.

    Args:
        base_url: The base URL of the API endpoint.
        model_id: The model identifier (default "*" matches any).
        allowlist: Optional custom allowlist. Defaults to OFFLINE_ALLOWLIST.

    Returns:
        True if the provider is allowlisted.

    Raises:
        CredentialGuardError: If the provider is not allowlisted.
    """
    if allowlist is None:
        allowlist = OFFLINE_ALLOWLIST

    base_url_lower = base_url.lower().rstrip("/")

    for allowed_url, allowed_model in allowlist:
        allowed_url_lower = allowed_url.lower().rstrip("/")

        # Check URL prefix match
        if base_url_lower.startswith(allowed_url_lower):
            # Check model match
            if allowed_model == "*" or model_id == allowed_model:
                return True

    raise CredentialGuardError(
        f"Provider not allowlisted: base_url={base_url!r}, model={model_id!r}. "
        f"Only the following endpoints are approved for inference: "
        f"{sorted(u for u, _ in allowlist)}. "
        f"If this endpoint should be allowed, add it to the OFFLINE_ALLOWLIST "
        f"in harness/credential_guard.py."
    )


def scrub_env(
    blocked_vars: Optional[Set[str]] = None,
    dry_run: bool = False,
) -> dict:
    """Remove blocked environment variables to prevent accidental use.

    This should be called BEFORE initializing any model provider to ensure
    platform proxy credentials are not accidentally picked up.

    Args:
        blocked_vars: Set of env var names to remove. Defaults to BLOCKED_ENV_VARS.
        dry_run: If True, report what would be removed without removing.

    Returns:
        Dict of {var_name: was_present} for each blocked var.
    """
    if blocked_vars is None:
        blocked_vars = BLOCKED_ENV_VARS

    results = {}
    for var in blocked_vars:
        was_present = var in os.environ
        results[var] = was_present
        if was_present and not dry_run:
            del os.environ[var]

    return results


def check_env_safety() -> dict:
    """Check environment for credential safety issues.

    Returns a dict with:
      - blocked_present: list of blocked vars that are set
      - allowed_present: list of allowed credential vars that are set
      - safe: True if no blocked vars are present
      - warnings: list of warning messages
    """
    blocked_present = [v for v in BLOCKED_ENV_VARS if v in os.environ]
    allowed_present = [v for v in ALLOWED_CREDENTIAL_VARS if v in os.environ]

    warnings_list = []
    if blocked_present:
        warnings_list.append(
            f"BLOCKED credential vars present in environment: {blocked_present}. "
            f"Call scrub_env() before provider initialization."
        )

    # Check if OPENAI vars might be pointing to Anthropic proxy
    openai_base = os.environ.get("OPENAI_BASE_URL", "")
    if "anthropic" in openai_base.lower():
        warnings_list.append(
            f"OPENAI_BASE_URL appears to point to Anthropic proxy: {openai_base!r}. "
            f"This may be a misconfiguration."
        )

    return {
        "blocked_present": blocked_present,
        "allowed_present": allowed_present,
        "safe": len(blocked_present) == 0,
        "warnings": warnings_list,
    }
