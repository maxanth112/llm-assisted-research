"""Robust JSON parsing with multiple repair strategies."""

import json
import re
from typing import Any, Optional, Type
from pydantic import BaseModel, ValidationError

from harness.schemas import (
    DirectAnswer,
    EnumerateOutput,
    TablePlaceboOutput,
    ProseDisconfirmOutput,
    FullACHOutput,
    FilterOutput,
    PRISMVerdictOutput,
    ParseResult,
)


# Schema mapping by condition ID
SCHEMA_MAP = {
    "000": DirectAnswer,
    "100": EnumerateOutput,
    "110": TablePlaceboOutput,
    "101": ProseDisconfirmOutput,
    "111": FullACHOutput,
    "filter_only": FilterOutput,
    "prism_full": PRISMVerdictOutput,
    "free_cot": DirectAnswer,  # Free CoT uses same schema as baseline
}


def extract_json_block(text: str) -> Optional[str]:
    """
    Extract JSON from markdown code fences or bare text.

    Tries multiple strategies:
    1. JSON in ```json ... ``` fence
    2. JSON in ``` ... ``` fence
    3. Bare JSON (find outermost braces/brackets)
    """
    # Strategy 1: JSON code fence
    json_fence_pattern = r'```json\s*\n(.*?)\n```'
    match = re.search(json_fence_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 2: Generic code fence
    code_fence_pattern = r'```\s*\n(.*?)\n```'
    match = re.search(code_fence_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 3: Bare JSON - find outermost { } or [ ]
    # Look for opening brace/bracket
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue

        # Find matching closing brace/bracket
        depth = 0
        for i in range(start_idx, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx:i+1]

    return None


def repair_json(text: str) -> str:
    """
    Attempt to repair common JSON formatting issues.

    Fixes:
    - Trailing commas before } or ]
    - Single quotes to double quotes (carefully)
    - Unquoted keys (simple cases)
    """
    # Remove trailing commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    # Replace single quotes with double quotes (simple heuristic)
    # This is imperfect but handles many cases
    text = text.replace("'", '"')

    # Try to fix unquoted keys (simple pattern: word followed by colon)
    # Only fix if not already quoted
    text = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)

    return text


def parse_condition_output(
    raw_text: str,
    condition_id: str,
    schema_class: Optional[Type[BaseModel]] = None
) -> ParseResult:
    """
    Parse raw model output into structured format for a given condition.

    NEVER silently drops failures - all attempts are logged.

    Args:
        raw_text: Raw model output text
        condition_id: Experimental condition ID
        schema_class: Pydantic schema to validate against (optional, will use condition mapping)

    Returns:
        ParseResult with success status, parsed data, and detailed attempt log
    """
    parse_log = []

    # Determine schema class
    if schema_class is None:
        schema_class = SCHEMA_MAP.get(condition_id)
        if schema_class is None:
            error_msg = f"No schema mapping found for condition '{condition_id}'"
            parse_log.append(error_msg)
            return ParseResult(
                success=False,
                data=None,
                raw_text=raw_text,
                error=error_msg,
                condition_id=condition_id,
                parse_attempt_log=parse_log
            )

    parse_log.append(f"Using schema: {schema_class.__name__}")

    # Strategy 1: Extract JSON block
    parse_log.append("Attempt 1: Extracting JSON block from text")
    json_text = extract_json_block(raw_text)

    if json_text is None:
        error_msg = "Failed to extract JSON block from text"
        parse_log.append(f"  FAILED: {error_msg}")
        return ParseResult(
            success=False,
            data=None,
            raw_text=raw_text,
            error=error_msg,
            condition_id=condition_id,
            parse_attempt_log=parse_log
        )

    parse_log.append(f"  Extracted {len(json_text)} characters")

    # Strategy 2: Try parsing extracted JSON directly
    parse_log.append("Attempt 2: Parsing extracted JSON directly")
    try:
        data_dict = json.loads(json_text)
        parse_log.append("  JSON parsed successfully")

        # Validate with Pydantic
        parse_log.append("Attempt 3: Validating with Pydantic schema")
        validated_data = schema_class(**data_dict)
        parse_log.append("  Validation successful")

        return ParseResult(
            success=True,
            data=validated_data.model_dump(),
            raw_text=raw_text,
            error=None,
            condition_id=condition_id,
            parse_attempt_log=parse_log
        )
    except json.JSONDecodeError as e:
        parse_log.append(f"  JSON parsing failed: {str(e)}")
    except ValidationError as e:
        parse_log.append(f"  Pydantic validation failed: {str(e)}")
        # If validation failed, the JSON was valid but schema didn't match
        # This is still a failure - return it
        error_msg = f"Schema validation failed: {str(e)}"
        return ParseResult(
            success=False,
            data=None,
            raw_text=raw_text,
            error=error_msg,
            condition_id=condition_id,
            parse_attempt_log=parse_log
        )
    except Exception as e:
        parse_log.append(f"  Unexpected error: {str(e)}")

    # Strategy 3: Try repairing JSON
    parse_log.append("Attempt 4: Repairing JSON and retrying")
    try:
        repaired_json = repair_json(json_text)
        parse_log.append("  JSON repair attempted")

        data_dict = json.loads(repaired_json)
        parse_log.append("  Repaired JSON parsed successfully")

        # Validate with Pydantic
        parse_log.append("Attempt 5: Validating repaired JSON with Pydantic schema")
        validated_data = schema_class(**data_dict)
        parse_log.append("  Validation successful")

        return ParseResult(
            success=True,
            data=validated_data.model_dump(),
            raw_text=raw_text,
            error=None,
            condition_id=condition_id,
            parse_attempt_log=parse_log
        )
    except json.JSONDecodeError as e:
        parse_log.append(f"  Repaired JSON parsing failed: {str(e)}")
    except ValidationError as e:
        parse_log.append(f"  Repaired JSON validation failed: {str(e)}")
        error_msg = f"Schema validation failed after repair: {str(e)}"
        return ParseResult(
            success=False,
            data=None,
            raw_text=raw_text,
            error=error_msg,
            condition_id=condition_id,
            parse_attempt_log=parse_log
        )
    except Exception as e:
        parse_log.append(f"  Unexpected error during repair: {str(e)}")

    # All strategies failed
    error_msg = "All parsing strategies exhausted - could not parse output"
    parse_log.append(f"FINAL: {error_msg}")

    return ParseResult(
        success=False,
        data=None,
        raw_text=raw_text,
        error=error_msg,
        condition_id=condition_id,
        parse_attempt_log=parse_log
    )
