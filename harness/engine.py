"""Experiment execution engine with caching, resume support, and multi-agent pipelines."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from harness.conditions import get_condition
from harness.parsers import parse_condition_output
from harness.providers.base import ModelProvider
from harness.randomization import ExperimentRandomizer
from harness.schemas import TrialRecord


class ResponseCache:
    """
    File-backed response cache keyed by prompt hash.

    Prevents redundant API calls for identical prompts.
    """

    def __init__(self, cache_dir: str):
        """
        Initialize response cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        params: dict[str, Any]
    ) -> str:
        """Generate cache key from prompt, model, and params."""
        # Create deterministic string from messages, model, and params
        cache_components = {
            "messages": messages,
            "model": model_id,
            "params": {k: v for k, v in sorted(params.items())}
        }

        cache_string = json.dumps(cache_components, sort_keys=True)
        cache_hash = hashlib.sha256(cache_string.encode("utf-8")).hexdigest()

        return cache_hash

    def get(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        params: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """
        Retrieve cached response if available.

        Args:
            messages: Prompt messages
            model_id: Model identifier
            params: Generation parameters

        Returns:
            Cached response dict or None if not cached
        """
        cache_key = self._get_cache_key(messages, model_id, params)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)

        return None

    def put(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        params: dict[str, Any],
        response: dict[str, Any]
    ) -> None:
        """
        Store response in cache.

        Args:
            messages: Prompt messages
            model_id: Model identifier
            params: Generation parameters
            response: Response to cache
        """
        cache_key = self._get_cache_key(messages, model_id, params)
        cache_file = self.cache_dir / f"{cache_key}.json"

        with open(cache_file, "w") as f:
            json.dump(response, f, indent=2)


class ExperimentLedger:
    """
    Append-only JSONL ledger for experiment trials.

    Supports resume by tracking completed trials.
    """

    def __init__(self, ledger_path: str):
        """
        Initialize experiment ledger.

        Args:
            ledger_path: Path to JSONL ledger file
        """
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file if it doesn't exist
        if not self.ledger_path.exists():
            self.ledger_path.touch()

    def load_completed(self) -> set[tuple[str, str, int]]:
        """
        Load set of completed trials from ledger.

        Returns:
            Set of (item_id, condition_id, run_index) tuples
        """
        completed = set()

        if not self.ledger_path.exists():
            return completed

        with open(self.ledger_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    trial_key = (
                        record["item_id"],
                        record["condition_id"],
                        record["run_index"]
                    )
                    completed.add(trial_key)
                except (json.JSONDecodeError, KeyError):
                    # Skip malformed lines
                    continue

        return completed

    def is_completed(
        self,
        item_id: str,
        condition_id: str,
        run_index: int
    ) -> bool:
        """
        Check if a trial is already completed.

        Args:
            item_id: Item identifier
            condition_id: Condition identifier
            run_index: Run index

        Returns:
            True if trial is completed
        """
        completed = self.load_completed()
        return (item_id, condition_id, run_index) in completed

    def record(self, trial_record: TrialRecord) -> None:
        """
        Append trial record to ledger.

        Args:
            trial_record: Trial record to append
        """
        with open(self.ledger_path, "a") as f:
            record_dict = trial_record.model_dump()
            f.write(json.dumps(record_dict) + "\n")


class ExperimentEngine:
    """
    Main experiment execution engine.

    Handles:
    - Prompt rendering and choice shuffling
    - Model API calls with caching
    - Response parsing
    - Multi-agent pipelines (filter_only, prism_full)
    - Resume support via ledger
    """

    def __init__(
        self,
        provider: ModelProvider,
        output_dir: str,
        master_seed: int = 42,
        model_params: Optional[dict[str, Any]] = None
    ):
        """
        Initialize experiment engine.

        Args:
            provider: Model provider instance
            output_dir: Output directory for cache and ledger
            master_seed: Master random seed
            model_params: Default model parameters (temperature, max_tokens, etc.)
        """
        self.provider = provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.randomizer = ExperimentRandomizer(master_seed)
        self.cache = ResponseCache(self.output_dir / "cache")
        self.ledger = ExperimentLedger(self.output_dir / "ledger.jsonl")

        self.model_params = model_params or {
            "temperature": 0.7,
            "max_tokens": 2048
        }

    def _compute_prompt_hash(self, messages: list[dict[str, str]]) -> str:
        """Compute SHA-256 hash of prompt messages."""
        prompt_string = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(prompt_string.encode("utf-8")).hexdigest()

    def _call_model(
        self,
        messages: list[dict[str, str]],
        params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Call model with caching.

        Args:
            messages: Prompt messages
            params: Generation parameters

        Returns:
            Dict with completion data
        """
        # Check cache
        cached = self.cache.get(messages, self.provider.provider_name(), params)
        if cached is not None:
            cached["cached"] = True
            return cached

        # Call provider
        completion = self.provider.complete(messages, params)

        # Prepare response dict
        response = {
            "text": completion.text,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "model": completion.model,
            "model_version": completion.model_version,
            "latency_ms": completion.latency_ms,
            "cached": False,
            "metadata": completion.metadata
        }

        # Cache response
        self.cache.put(messages, self.provider.provider_name(), params, response)

        return response

    def _render_prompt(
        self,
        template_name: str,
        item: dict[str, Any],
        shuffled_choices: Optional[list[str]] = None,
        extra_context: Optional[dict[str, Any]] = None
    ) -> list[dict[str, str]]:
        """
        Render prompt template with item data.

        Args:
            template_name: Name of prompt template
            item: Item data
            shuffled_choices: Shuffled choices (if applicable)
            extra_context: Additional context for multi-agent pipelines

        Returns:
            List of message dicts
        """
        # Import prompt templates
        from harness.prompts.templates import get_prompt_template

        template = get_prompt_template(template_name)

        # Prepare template context
        context = {
            "item": item,
            "choices": shuffled_choices or item.get("choices", []),
            **(extra_context or {})
        }

        # Render template (assumes template is a function that takes context)
        messages = template(context)

        return messages

    def run_single_trial(
        self,
        item: dict[str, Any],
        condition_id: str,
        run_index: int
    ) -> TrialRecord:
        """
        Run a single experimental trial.

        Args:
            item: Test item data
            condition_id: Experimental condition
            run_index: Run number

        Returns:
            TrialRecord with results
        """
        item_id = item["id"]

        # Get condition
        condition = get_condition(condition_id)
        if condition is None:
            raise ValueError(f"Unknown condition: {condition_id}")

        # Derive seed for this trial
        seed = self.randomizer.derive_seed(item_id, condition_id, run_index)

        # Shuffle choices
        original_choices = item.get("choices", [])
        shuffled_choices, permutation = self.randomizer.shuffle_choices(
            original_choices, item_id, condition_id
        )

        # Prepare params
        params = {
            **self.model_params,
            "item_id": item_id,
            "condition_id": condition_id,
            "seed": seed
        }

        errors = []

        # Execute based on condition type
        if condition.num_calls == 1:
            # Single-call condition
            raw_output, token_counts, latency_ms = self._execute_single_call(
                condition.prompt_template_name,
                item,
                shuffled_choices,
                params
            )
        elif condition_id == "filter_only":
            # Two-call pipeline: filter then answer
            raw_output, token_counts, latency_ms = self._execute_filter_pipeline(
                item,
                shuffled_choices,
                params
            )
        elif condition_id == "prism_full":
            # Four-call PRISM pipeline
            raw_output, token_counts, latency_ms = self._execute_prism_pipeline(
                item,
                shuffled_choices,
                params
            )
        else:
            raise ValueError(f"Unknown pipeline for condition: {condition_id}")

        # Compute prompt hash (for single call, use first prompt)
        messages = self._render_prompt(
            condition.prompt_template_name,
            item,
            shuffled_choices
        )
        prompt_hash = self._compute_prompt_hash(messages)

        # Parse output
        parse_result = parse_condition_output(raw_output, condition_id)

        if not parse_result.success:
            errors.append(f"Parse failed: {parse_result.error}")

        # Unshuffle answer if parse succeeded
        parsed_data = None
        if parse_result.success and parse_result.data:
            parsed_data = parse_result.data.copy()

            # Unshuffle answer
            answer = parsed_data.get("answer") or parsed_data.get("verdict")
            if answer:
                unshuffled_answer = self.randomizer.unshuffle_answer(
                    answer,
                    original_choices,
                    item_id,
                    condition_id
                )
                if "answer" in parsed_data:
                    parsed_data["answer"] = unshuffled_answer
                elif "verdict" in parsed_data:
                    parsed_data["verdict"] = unshuffled_answer

        # Estimate cost
        cost = self.provider.estimate_cost(
            token_counts["prompt_tokens"],
            token_counts["completion_tokens"]
        )

        # Create trial record
        trial_record = TrialRecord(
            item_id=item_id,
            condition_id=condition_id,
            run_index=run_index,
            seed=seed,
            model_id=self.provider.provider_name(),
            model_version="unknown",  # Will be filled by provider
            params=params,
            prompt_hash=prompt_hash,
            raw_output=raw_output,
            parsed_result={"success": parse_result.success, "data": parsed_data} if parsed_data else None,
            token_counts=token_counts,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            errors=errors,
            timestamp=time.time()
        )

        # Record in ledger
        self.ledger.record(trial_record)

        return trial_record

    def _execute_single_call(
        self,
        template_name: str,
        item: dict[str, Any],
        shuffled_choices: list[str],
        params: dict[str, Any]
    ) -> tuple[str, dict[str, int], float]:
        """Execute single-call condition."""
        messages = self._render_prompt(template_name, item, shuffled_choices)

        response = self._call_model(messages, params)

        token_counts = {
            "prompt_tokens": response["prompt_tokens"],
            "completion_tokens": response["completion_tokens"]
        }

        return response["text"], token_counts, response["latency_ms"]

    def _execute_filter_pipeline(
        self,
        item: dict[str, Any],
        shuffled_choices: list[str],
        params: dict[str, Any]
    ) -> tuple[str, dict[str, int], float]:
        """Execute two-call filter pipeline (A1 -> answer)."""
        total_latency = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Call 1: Filter evidence
        messages_filter = self._render_prompt("filter", item, shuffled_choices)
        response_filter = self._call_model(messages_filter, params)

        total_latency += response_filter["latency_ms"]
        total_prompt_tokens += response_filter["prompt_tokens"]
        total_completion_tokens += response_filter["completion_tokens"]

        # Parse filter output
        filter_parse = parse_condition_output(response_filter["text"], "filter_only")

        # Call 2: Answer with filtered evidence
        filtered_context = {}
        if filter_parse.success and filter_parse.data:
            filtered_context = {
                "filtered_evidence": filter_parse.data
            }

        messages_answer = self._render_prompt(
            "answer_filtered",
            item,
            shuffled_choices,
            extra_context=filtered_context
        )
        response_answer = self._call_model(messages_answer, params)

        total_latency += response_answer["latency_ms"]
        total_prompt_tokens += response_answer["prompt_tokens"]
        total_completion_tokens += response_answer["completion_tokens"]

        token_counts = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens
        }

        # Return final answer as raw output
        return response_answer["text"], token_counts, total_latency

    def _execute_prism_pipeline(
        self,
        item: dict[str, Any],
        shuffled_choices: list[str],
        params: dict[str, Any]
    ) -> tuple[str, dict[str, int], float]:
        """Execute four-call PRISM pipeline (A1 -> A2 -> A3 -> A4)."""
        total_latency = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Agent 1: Filter evidence
        messages_a1 = self._render_prompt("prism_a1_filter", item, shuffled_choices)
        response_a1 = self._call_model(messages_a1, params)

        total_latency += response_a1["latency_ms"]
        total_prompt_tokens += response_a1["prompt_tokens"]
        total_completion_tokens += response_a1["completion_tokens"]

        # Agent 2: Generate hypotheses
        messages_a2 = self._render_prompt(
            "prism_a2_hypothesize",
            item,
            shuffled_choices,
            extra_context={"a1_output": response_a1["text"]}
        )
        response_a2 = self._call_model(messages_a2, params)

        total_latency += response_a2["latency_ms"]
        total_prompt_tokens += response_a2["prompt_tokens"]
        total_completion_tokens += response_a2["completion_tokens"]

        # Agent 3: Build ACH matrix
        messages_a3 = self._render_prompt(
            "prism_a3_matrix",
            item,
            shuffled_choices,
            extra_context={
                "a1_output": response_a1["text"],
                "a2_output": response_a2["text"]
            }
        )
        response_a3 = self._call_model(messages_a3, params)

        total_latency += response_a3["latency_ms"]
        total_prompt_tokens += response_a3["prompt_tokens"]
        total_completion_tokens += response_a3["completion_tokens"]

        # Agent 4: Final verdict
        messages_a4 = self._render_prompt(
            "prism_a4_verdict",
            item,
            shuffled_choices,
            extra_context={
                "a1_output": response_a1["text"],
                "a2_output": response_a2["text"],
                "a3_output": response_a3["text"]
            }
        )
        response_a4 = self._call_model(messages_a4, params)

        total_latency += response_a4["latency_ms"]
        total_prompt_tokens += response_a4["prompt_tokens"]
        total_completion_tokens += response_a4["completion_tokens"]

        token_counts = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens
        }

        # Return final verdict as raw output
        return response_a4["text"], token_counts, total_latency

    def run_experiment(
        self,
        items: list[dict[str, Any]],
        condition_ids: list[str],
        k_runs: int
    ) -> list[TrialRecord]:
        """
        Run full experiment with resume support.

        Args:
            items: List of test items
            condition_ids: List of condition IDs to run
            k_runs: Number of runs per item-condition pair

        Returns:
            List of all trial records
        """
        # Generate trial schedule
        item_ids = [item["id"] for item in items]
        schedule = self.randomizer.generate_trial_schedule(item_ids, condition_ids, k_runs)

        # Load completed trials
        completed = self.ledger.load_completed()

        # Filter schedule to only pending trials
        pending = [
            trial for trial in schedule
            if (trial["item_id"], trial["condition_id"], trial["run_index"]) not in completed
        ]

        print(f"Total trials: {len(schedule)}")
        print(f"Completed: {len(completed)}")
        print(f"Pending: {len(pending)}")

        # Run pending trials
        results = []
        for i, trial_spec in enumerate(pending):
            print(f"Running trial {i+1}/{len(pending)}: {trial_spec['item_id']} / {trial_spec['condition_id']} / run {trial_spec['run_index']}")

            # Find item
            item = next((it for it in items if it["id"] == trial_spec["item_id"]), None)
            if item is None:
                print(f"  WARNING: Item {trial_spec['item_id']} not found, skipping")
                continue

            # Run trial
            try:
                trial_record = self.run_single_trial(
                    item,
                    trial_spec["condition_id"],
                    trial_spec["run_index"]
                )
                results.append(trial_record)

                if trial_record.errors:
                    print(f"  Completed with errors: {trial_record.errors}")
                else:
                    print(f"  Completed successfully")

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                # Continue with next trial

        return results

    def estimate_cost(
        self,
        items: list[dict[str, Any]],
        condition_ids: list[str],
        k_runs: int,
        pricing: dict[str, float]
    ) -> dict[str, Any]:
        """
        Estimate total cost for experiment.

        Args:
            items: List of test items
            condition_ids: List of condition IDs
            k_runs: Number of runs per item-condition pair
            pricing: Pricing dict with prompt_per_1k_tokens and completion_per_1k_tokens

        Returns:
            Dict with cost estimates
        """
        n_items = len(items)
        n_conditions = len(condition_ids)
        n_trials = n_items * n_conditions * k_runs

        # Estimate tokens per trial (rough heuristic)
        avg_prompt_tokens = 1500  # Typical for our prompts
        avg_completion_tokens = 500  # Typical structured output

        # Adjust for multi-call conditions
        total_calls = 0
        for cond_id in condition_ids:
            condition = get_condition(cond_id)
            if condition:
                total_calls += condition.num_calls * n_items * k_runs

        total_prompt_tokens = avg_prompt_tokens * total_calls
        total_completion_tokens = avg_completion_tokens * total_calls

        # Calculate cost
        prompt_cost = (total_prompt_tokens / 1000) * pricing["prompt_per_1k_tokens"]
        completion_cost = (total_completion_tokens / 1000) * pricing["completion_per_1k_tokens"]
        total_cost = prompt_cost + completion_cost

        return {
            "n_trials": n_trials,
            "n_api_calls": total_calls,
            "estimated_prompt_tokens": total_prompt_tokens,
            "estimated_completion_tokens": total_completion_tokens,
            "estimated_cost_usd": total_cost,
            "breakdown": {
                "prompt_cost_usd": prompt_cost,
                "completion_cost_usd": completion_cost
            }
        }
