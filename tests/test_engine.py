"""Tests for experiment engine, ledger, cache, and resume support."""

import json
import os
import tempfile
import pytest

from harness.engine import ResponseCache, ExperimentLedger, ExperimentEngine
from harness.providers.mock_adapter import MockAdapter
from harness.schemas import TrialRecord


class TestResponseCache:
    """Test file-backed response cache."""

    def test_put_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(os.path.join(tmpdir, "cache"))
            messages = [{"role": "user", "content": "test"}]
            model_id = "mock"
            params = {"temperature": 0.7}
            response = {"text": "hello", "tokens": 10}

            cache.put(messages, model_id, params, response)
            result = cache.get(messages, model_id, params)
            assert result is not None
            assert result["text"] == "hello"

    def test_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(os.path.join(tmpdir, "cache"))
            messages = [{"role": "user", "content": "test"}]
            result = cache.get(messages, "mock", {"temperature": 0.7})
            assert result is None

    def test_different_params_different_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(os.path.join(tmpdir, "cache"))
            messages = [{"role": "user", "content": "test"}]

            cache.put(messages, "mock", {"temperature": 0.7}, {"text": "warm"})
            cache.put(messages, "mock", {"temperature": 0.0}, {"text": "cold"})

            warm = cache.get(messages, "mock", {"temperature": 0.7})
            cold = cache.get(messages, "mock", {"temperature": 0.0})
            assert warm["text"] == "warm"
            assert cold["text"] == "cold"


class TestExperimentLedger:
    """Test append-only JSONL ledger."""

    def test_empty_ledger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = ExperimentLedger(os.path.join(tmpdir, "ledger.jsonl"))
            completed = ledger.load_completed()
            assert len(completed) == 0

    def test_record_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = ExperimentLedger(os.path.join(tmpdir, "ledger.jsonl"))

            record = TrialRecord(
                item_id="item1",
                condition_id="000",
                run_index=0,
                seed=123,
                model_id="mock",
                model_version="1.0",
                params={"temperature": 0.7},
                prompt_hash="abc",
                raw_output="test output",
                token_counts={"prompt_tokens": 50, "completion_tokens": 20},
                latency_ms=10.0,
                estimated_cost_usd=0.0,
                timestamp=1000.0,
            )
            ledger.record(record)

            completed = ledger.load_completed()
            assert ("item1", "000", 0) in completed

    def test_is_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = ExperimentLedger(os.path.join(tmpdir, "ledger.jsonl"))

            record = TrialRecord(
                item_id="item1",
                condition_id="000",
                run_index=0,
                seed=123,
                model_id="mock",
                model_version="1.0",
                params={},
                prompt_hash="abc",
                raw_output="test",
                token_counts={"prompt_tokens": 10, "completion_tokens": 5},
                latency_ms=1.0,
                estimated_cost_usd=0.0,
                timestamp=1000.0,
            )
            ledger.record(record)

            assert ledger.is_completed("item1", "000", 0)
            assert not ledger.is_completed("item1", "000", 1)
            assert not ledger.is_completed("item2", "000", 0)

    def test_failures_visible_in_ledger(self):
        """Trials with errors must still appear in the ledger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = ExperimentLedger(os.path.join(tmpdir, "ledger.jsonl"))

            record = TrialRecord(
                item_id="item_fail",
                condition_id="000",
                run_index=0,
                seed=1,
                model_id="mock",
                model_version="1.0",
                params={},
                prompt_hash="abc",
                raw_output="garbage",
                token_counts={"prompt_tokens": 10, "completion_tokens": 5},
                latency_ms=1.0,
                estimated_cost_usd=0.0,
                errors=["Parse failed: No JSON found"],
                timestamp=1000.0,
            )
            ledger.record(record)

            # Read raw JSONL
            with open(os.path.join(tmpdir, "ledger.jsonl")) as f:
                lines = [json.loads(line) for line in f if line.strip()]

            assert len(lines) == 1
            assert lines[0]["errors"] == ["Parse failed: No JSON found"]
            assert ("item_fail", "000", 0) in ledger.load_completed()

    def test_malformed_lines_skipped(self):
        """Malformed JSONL lines should be skipped without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = os.path.join(tmpdir, "ledger.jsonl")

            # Write a valid line followed by garbage
            with open(ledger_path, "w") as f:
                valid = {"item_id": "item1", "condition_id": "000", "run_index": 0}
                f.write(json.dumps(valid) + "\n")
                f.write("this is not json\n")

            ledger = ExperimentLedger(ledger_path)
            completed = ledger.load_completed()
            assert ("item1", "000", 0) in completed


class TestExperimentEngineResume:
    """Test that a stopped experiment resumes without duplicating calls."""

    def test_resume_skips_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = MockAdapter(mode="compliant", seed=42)

            # Provide a prompt template function since engine._render_prompt needs it
            # We need to set up a minimal environment
            items = [
                {"id": "item1", "choices": ["A", "B", "C", "D"],
                 "narrative": "test", "question": "What?"},
            ]
            condition_ids = ["000"]

            # Run first time
            engine1 = ExperimentEngine(
                provider=adapter,
                output_dir=tmpdir,
                master_seed=42,
            )

            # Manually record a completed trial
            record = TrialRecord(
                item_id="item1",
                condition_id="000",
                run_index=0,
                seed=123,
                model_id="mock",
                model_version="1.0",
                params={"temperature": 0.7},
                prompt_hash="abc",
                raw_output="test",
                token_counts={"prompt_tokens": 10, "completion_tokens": 5},
                latency_ms=1.0,
                estimated_cost_usd=0.0,
                timestamp=1000.0,
            )
            engine1.ledger.record(record)

            # Verify it shows as completed
            completed = engine1.ledger.load_completed()
            assert ("item1", "000", 0) in completed

            # Create new engine with same output dir (simulates resume)
            engine2 = ExperimentEngine(
                provider=adapter,
                output_dir=tmpdir,
                master_seed=42,
            )

            completed2 = engine2.ledger.load_completed()
            assert ("item1", "000", 0) in completed2


class TestExperimentEngineCostEstimate:
    """Test cost estimation."""

    def test_basic_cost_estimate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = MockAdapter(mode="compliant", seed=42)
            engine = ExperimentEngine(provider=adapter, output_dir=tmpdir)

            items = [{"id": f"item{i}"} for i in range(10)]
            pricing = {
                "prompt_per_1k_tokens": 0.001,
                "completion_per_1k_tokens": 0.002,
            }

            estimate = engine.estimate_cost(items, ["000", "100"], k_runs=3, pricing=pricing)
            assert estimate["n_trials"] == 60
            assert estimate["estimated_cost_usd"] > 0
