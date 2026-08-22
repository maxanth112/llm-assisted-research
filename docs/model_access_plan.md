# Model Access Plan: ETD-ACH Factorial Experiment

## Overview

This document outlines the model access strategy for the ETD-ACH factorial experiment investigating the effects of explicit hypothesis enumeration (E), evidence tabulation (T), and disconfirmation scoring (D) on LLM reasoning performance.

### Key Requirements
- **Model diversity**: Require ≥2 genuinely different model families to validate generalizability
- **Reproducibility priority**: Favor reproducible, stable models over bleeding-edge or prestige models
- **API compatibility**: All provider adapters are OpenAI-compatible (configured via base_url + model + API key from environment variables)
- **Determinism**: All models must support deterministic execution (temperature=0 or equivalent)
- **Data privacy**: Providers must not train on customer API data

---

## Proposed Models

### 1. Llama-3.3-70B-Instruct (Open-weight, via Together AI)

**Model Identifier**: `meta-llama/Llama-3.3-70B-Instruct-Turbo`

**Provider**: Together AI
**Provider URL**: https://www.together.ai/pricing

**Access Method**: OpenAI-compatible API

**Open/Proprietary**: Open-weight (Meta Llama 3.3 Community License)

**Availability**: Stable; widely hosted across multiple inference providers

**Pricing** (retrieved: August 2026 — **NOTE: MUST verify at spend time**):
- Input tokens: ~$0.88 per million tokens (Together Turbo tier)
- Output tokens: ~$0.88 per million tokens
- Source: https://www.together.ai/pricing

**Deterministic Execution**: `temperature=0` supported

**Data Retention Policy**: Together AI does not train on customer data per their Terms of Service (verify current policy at time of use)

**Rationale**: Llama 3.3 70B represents a strong open-weight baseline with widespread availability, good instruction-following, and competitive performance on reasoning tasks. The 70B parameter count provides sufficient capacity while remaining cost-effective.

---

### 2. Qwen2.5-72B-Instruct (Open-weight, via Together AI or Fireworks)

**Model Identifier**: `Qwen/Qwen2.5-72B-Instruct-Turbo`

**Provider**: Together AI (primary) or Fireworks AI (alternative)
**Provider URL**: https://www.together.ai/pricing

**Access Method**: OpenAI-compatible API

**Open/Proprietary**: Open-weight (Apache 2.0 License)

**Availability**: Stable; hosted by multiple providers

**Pricing** (retrieved: August 2026 — **NOTE: MUST verify at spend time**):
- Input tokens: ~$0.88 per million tokens (Together Turbo tier)
- Output tokens: ~$0.88 per million tokens
- Source: https://www.together.ai/pricing

**Deterministic Execution**: `temperature=0` supported

**Data Retention Policy**: Together AI does not train on customer data per their Terms of Service (verify current policy at time of use)

**Rationale**: Qwen 2.5 72B provides a genuinely different model family from Llama (different architecture, training data, and organizational origin). This cross-family comparison is essential for validating that observed effects are not model-specific artifacts.

---

### 3. GPT-4o-mini (Frontier Cross-Family Check)

**Model Identifier**: `gpt-4o-mini`

**Provider**: OpenAI
**Provider URL**: https://openai.com/api/pricing/

**Access Method**: OpenAI API

**Open/Proprietary**: Proprietary (closed-weight)

**Availability**: Production-stable OpenAI service

**Pricing** (retrieved: August 2026 — **NOTE: MUST verify at spend time**):
- Input tokens: ~$0.15 per million tokens
- Output tokens: ~$0.60 per million tokens
- Source: https://openai.com/api/pricing/

**Deterministic Execution**: `seed` parameter supported (note: OpenAI does not guarantee bit-identical outputs across model versions, but provides best-effort reproducibility)

**Data Retention Policy**: API data not used for training by default (opt-out is automatic for API usage as of 2024; verify current policy)

**Rationale**: GPT-4o-mini provides a low-cost proprietary baseline from a different model family (OpenAI GPT-4o architecture). While not the primary experimental models, it serves as a cross-family validation check. Its significantly lower cost makes it practical for subset validation.

---

## Token Budget Estimation

### Per-Item Token Estimates

**Single-call conditions** (baseline, E-only, T-only, etc.):
- Average prompt: ~1,500 tokens
- Average completion: ~500 tokens
- **Total per item**: ~2,000 tokens

**PRISM 4-call condition**:
- Total across 4 sequential calls: ~6,000 prompt tokens + ~2,000 completion tokens
- **Total per item**: ~8,000 tokens

**Free CoT condition**:
- Average prompt: ~1,500 tokens
- Average completion (longer due to unrestricted reasoning): ~800 tokens
- **Total per item**: ~2,300 tokens

### Dataset Sizes

**T2 dataset**:
- Pilot: 24 items (6 per regime: CLEAN, DECOY, CONFLICT, INSUFFICIENT)
- Full: 48-100 items (targeting 100 for adequate power)

**MuSR dataset**:
- murder_mystery subset: ~250 items (after exclusions)

**Total items** (confirmatory study): 350 items (100 T2 + 250 MuSR)

### Experimental Conditions

**Factorial conditions**: 5
- 000 (baseline)
- 100 (E only)
- 110 (E+T)
- 101 (E+D)
- 111 (E+T+D, full ACH)

**Reference conditions**: 3
- filter_only (minimal scaffolding)
- prism_full (4-call decomposed reasoning)
- free_cot (unrestricted chain-of-thought)

**Total conditions**: 8

**Repeat runs** (k_runs):
- Pilot: 1-3
- Confirmatory: 3-5

---

## Cost Estimates

### PILOT (LOW-COST) Option

**Purpose**: Verify harness works end-to-end, check parse rates, debug infrastructure

**Configuration**:
- Items: 24 (T2 items only, 6 per regime)
- Conditions: 5 (factorial only, skip reference conditions)
- k_runs: 1
- Models: 1 (Llama-3.3-70B-Instruct only)

**Call count calculation**:
- Most conditions: 24 items × 5 conditions × 1 run = 120 calls
- All are single-call conditions in pilot
- **Total calls**: 120

**Token calculation**:
- 120 calls × 2,000 tokens per call = 240,000 tokens

**Cost estimate** (Together AI at $0.88/M tokens):
- 240K tokens × ($0.88 / 1M) = **$0.21**

**Adjusted estimate** (including output tokens weighted higher if provider charges differently):
- With 1,500 input + 500 output per call
- 120 × 1,500 = 180K input tokens
- 120 × 500 = 60K output tokens
- At $0.88/M for both: (180K + 60K) / 1M × $0.88 = **$0.21**

**PILOT TOTAL**: ~**$0.42** (2× safety margin for prompt variations)

---

### CONFIRMATORY (FULL) Option

**Purpose**: Full factorial experiment with cross-model validation

**Configuration**:
- Items: 350 (100 T2 + 250 MuSR)
- Conditions: 8 (5 factorial + 3 reference)
- k_runs: 3
- Models: 2 (Llama-3.3-70B + Qwen2.5-72B)

**Call count calculation**:
- Single-call conditions: 7 (all except PRISM)
  - 350 items × 7 conditions × 3 runs × 2 models = 14,700 calls
- Multi-call condition (PRISM): 1 condition
  - 350 items × 1 condition × 4 calls × 3 runs × 2 models = 8,400 calls
- **Total calls**: 23,100 calls

**Token calculation**:
- Single-call: 14,700 calls × 2,000 tokens = 29.4M tokens
- PRISM: 8,400 calls × 2,000 tokens = 16.8M tokens (averaged per call)
- **Total tokens**: ~46.2M tokens

**Cost estimate** (Together AI at $0.88/M tokens for both models):
- 46.2M tokens × ($0.88 / 1M) = **$40.66**

**Adjusted estimate** (with safety margin for longer completions, retries):
- 2× safety margin: **$81.32**

**CONFIRMATORY SUBTOTAL** (Llama + Qwen): ~**$81**

---

### GPT-4o-mini Cross-Family Check (Optional)

**Purpose**: Validate findings on a proprietary frontier model from different family

**Configuration**:
- Items: 100 (subset of T2 items)
- Conditions: 5 (factorial only)
- k_runs: 1
- Models: 1 (GPT-4o-mini)

**Call count**: 100 × 5 × 1 = 500 calls

**Token calculation**:
- 500 calls × (1,500 input + 500 output) = 750K input + 250K output

**Cost estimate** (OpenAI pricing):
- Input: 750K × ($0.15 / 1M) = $0.1125
- Output: 250K × ($0.60 / 1M) = $0.15
- **Total**: $0.26

**Adjusted estimate** (with safety margin): **$0.30**

---

### TOTAL COST SUMMARY

| Phase | Configuration | Estimated Cost |
|-------|--------------|----------------|
| **Pilot** | 24 items, 5 conditions, 1 run, 1 model | **$0.42** |
| **Confirmatory** | 350 items, 8 conditions, 3 runs, 2 models | **$81.00** |
| **Cross-family check** | 100 items, 5 conditions, 1 run, GPT-4o-mini | **$0.30** |
| **GRAND TOTAL** | All phases | **$81.72** |

---

## Recommendations

### Phased Execution Plan

**Phase 1: Pilot Study** (~$0.42)
- Use Llama-3.3-70B-Instruct via Together AI
- Run 24 T2 items across 5 factorial conditions, k=1
- **Success criteria**:
  - Parse rate >80% across all conditions
  - Harness executes without errors
  - Accuracy variance across conditions is non-zero
- **Timeline**: 1-2 days
- **Decision point**: CONTINUE / MODIFY / KILL (see protocol for criteria)

**Phase 2: Confirmatory Study** (~$81)
- Use both Llama-3.3-70B and Qwen2.5-72B via Together AI
- Run full 350 items (100 T2 + 250 MuSR) across 8 conditions, k=3
- **Success criteria**:
  - Complete all conditions without infrastructure failures
  - Parse rates remain >80%
  - Collect all outcome variables (accuracy, calibration, parse rates, etc.)
- **Timeline**: 3-5 days (depending on rate limits)

**Phase 3: Cross-Family Validation** (~$0.30)
- Use GPT-4o-mini for subset validation
- Run 100 T2 items across 5 factorial conditions, k=1
- **Purpose**: Check if main effects replicate on proprietary frontier model
- **Timeline**: 1 day

### Cost Management

1. **Verify pricing before each phase**: All estimates based on August 2026 pricing; API pricing changes frequently
2. **Set spend limits**: Configure Together AI and OpenAI account limits to prevent runaway costs
3. **Monitor parse rates early**: If pilot shows low parse rates, halt and revise prompts before scaling up
4. **Use API key rotation**: Store keys in environment variables, never commit to version control

### Model Selection Justification

**Why these models?**
- **Llama-3.3-70B**: Strong open-weight baseline, widely available, good cost/performance ratio
- **Qwen2.5-72B**: Genuinely different architecture/family, validates generalizability
- **GPT-4o-mini**: Low-cost proprietary check, different training paradigm

**Why not larger models?** (e.g., Llama-3.1-405B, GPT-4o)
- Marginal gains in reasoning performance do not justify 5-10× cost increase
- 70B-class models already exceed human performance on many reasoning benchmarks
- Reproducibility and cost-effectiveness prioritized over maximum capability

**Why not more models?**
- Diminishing returns: 2-3 models sufficient to detect model-family effects
- Cost scales linearly with model count
- Analysis complexity increases with more models

---

## Data Privacy and Ethics

### API Data Retention
- **Together AI**: Does not train on customer API data (per ToS; verify current policy)
- **OpenAI**: API data not used for training by default (verify current policy)
- **Recommendation**: Review each provider's data usage policy before running experiments

### Dataset Privacy
- T2 dataset: Synthetic, no privacy concerns
- MuSR dataset: Fictional murder mysteries, publicly available, no privacy concerns
- No human subjects data involved

### Responsible Use
- All models used within their intended use cases (reasoning/QA tasks)
- No attempts to elicit harmful content or bypass safety guidelines
- Outputs used solely for research purposes

---

## Appendix: Pricing Source URLs

**Together AI Pricing** (verified August 2026):
- https://www.together.ai/pricing
- Llama-3.3-70B-Instruct-Turbo: $0.88/M tokens (input and output)
- Qwen2.5-72B-Instruct-Turbo: $0.88/M tokens (input and output)

**OpenAI Pricing** (verified August 2026):
- https://openai.com/api/pricing/
- GPT-4o-mini: $0.15/M input tokens, $0.60/M output tokens

**CRITICAL**: These prices are subject to change. Always verify current pricing at provider URLs before executing spending plans.

---

## Version History

- **v1.0** (2026-08-22): Initial model access plan for ETD-ACH factorial experiment
