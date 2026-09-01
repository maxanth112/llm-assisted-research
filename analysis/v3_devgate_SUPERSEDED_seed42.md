# SUPERSEDED: v3 Development-Gate Leakage Evaluation (seed=42)

**STATUS: SUPERSEDED by v3.1 dev-gate evaluation.**
**This file is retained as an audit-trail artifact. Do not treat these results as current.**

## Configuration

| Parameter | Value |
|---|---|
| Generator | `T2V3Generator` (v3.0.0) from `datasets/t2_generator/generator_v3.py` |
| n_per_regime | 50 |
| Master seed | 42 |
| Total items | 200 (4 regimes x 50) |
| Chance level | 0.25000 (all items have exactly 4 options) |
| Threshold | 0.30000 (chance + 0.05) |
| Evaluator | `analysis/run_leakage_eval.py` (template-held-out eval) |
| Evaluation type | Template-held-out on full corpus (no separate audit split) |

## Corpus Manifest

| Property | Value |
|---|---|
| SHA-256 of JSONL | `9f8cd8c741977ec569ba4975ffb21cf6d1628678bc1f68dd454009066585149c` |
| Corpus file | `analysis/t2v3_dev_corpus_SUPERSEDED.jsonl` (deterministically reproducible) |
| Regime distribution | CLEAN=50, DECOY=50, CONFLICT=50, INSUFFICIENT=50 |

## Reproduction Command

```bash
cd llm-assisted-research
python3 -c "
import sys, json
sys.path.insert(0, '.')
from datasets.t2_generator.generator_v3 import T2V3Generator
from dataclasses import asdict

gen = T2V3Generator(seed=42)
items = gen.generate_dataset(n_per_regime=50, seed=42)
with open('analysis/t2v3_dev_corpus_SUPERSEDED.jsonl', 'w') as f:
    for item in items:
        f.write(json.dumps(asdict(item), ensure_ascii=False) + '\n')
"
# Then run the leakage eval:
python3 -c "
import sys, json
sys.path.insert(0, '.')
from analysis.run_leakage_eval import (
    load_jsonl, template_held_out_eval, run_surface_form_checks,
    chance_level_correct, BASELINE_NAMES
)
items = load_jsonl('analysis/t2v3_dev_corpus_SUPERSEDED.jsonl')
sf = run_surface_form_checks(items, label='v3_dev')
ho = template_held_out_eval(items, alpha=0.05)
for name in BASELINE_NAMES:
    r = ho[name]
    print(f'{name:30s}  acc={r[\"accuracy\"]:.4f}  CI=[{r[\"ci_lower\"]:.4f},{r[\"ci_upper\"]:.4f}]  {r[\"verdict\"]}')
"
```

## Surface-Form Check Results

| Check | Result | Detail |
|---|---|---|
| S1 option count | PASS | All 200 items have exactly 4 options |
| S2 abstention position | PASS | Max diff <= 1 in every regime |
| S3 abstention presence | PASS | All items contain abstention option |
| S4 evidence count | **FAIL** | Evidence-count multisets differ: CLEAN vs CONFLICT (7 vs 8), CLEAN vs DECOY (7 vs 10) |
| S5 option text length | PASS | Per-regime means within 20% relative band |
| S6 gold position | PASS | Max diff <= 1 in every regime |

### S4 Failure Root Cause

The v2 generator produces different numbers of evidence items per regime:
- CLEAN: 7 items (3 incriminating + 3 exonerating + 1 cross-reference)
- DECOY: 10 items (CLEAN + 3 decoy items, one per suspect)
- CONFLICT: 8 items (3 incriminating + 3 exonerating + 2 conflicting sources)
- INSUFFICIENT: 7 items (3 access + 3 unverifiable + 1 ambiguity)

The v3 generator wraps v2 without modifying evidence structure, so this discrepancy is inherited.

## Baseline Leakage Results (Template-Held-Out)

### Aggregate

| Baseline | Accuracy | Wilson 95% CI | Verdict |
|---|---|---|---|
| 1_majority_class | 0.2600 | [0.2041, 0.3249] | FAIL |
| 2_label_position | 0.2600 | [0.2041, 0.3249] | FAIL |
| 3_mention_count | 0.2600 | [0.2041, 0.3249] | FAIL |
| 4_evidence_count | 0.2600 | [0.2041, 0.3249] | FAIL |
| 5_lexical_overlap | 0.3250 | [0.2639, 0.3927] | FAIL |
| 6_tfidf_word | 0.4800 | [0.4118, 0.5490] | **FAIL** |
| 7_tfidf_char | 0.2600 | [0.2041, 0.3249] | FAIL |
| 8_length_feature | 0.2700 | [0.2132, 0.3354] | FAIL |
| 9_mention_evidence | 0.2600 | [0.2041, 0.3249] | FAIL |
| 10_first_mention_order | 0.2550 | [0.1996, 0.3196] | FAIL |
| 11_combined_shallow | 0.3500 | [0.2873, 0.4184] | FAIL |

### Per-Regime: Baseline 6 (TF-IDF word) — Key Diagnostic

| Regime | Accuracy | Wilson 95% CI | Chance | Verdict |
|---|---|---|---|---|
| CLEAN | 0.2000 | [0.1124, 0.3304] | 0.2500 | FAIL |
| DECOY | 0.3600 | [0.2414, 0.4986] | 0.2500 | FAIL |
| CONFLICT | 0.3600 | [0.2414, 0.4986] | 0.2500 | FAIL |
| INSUFFICIENT | **1.0000** | [0.9286, 1.0000] | 0.2500 | **FAIL** |

## Overall Verdict: FAIL

### Failure Analysis

**Two categories of failures:**

1. **Wide-CI artifact (baselines 1-4, 7-10):** Accuracies 0.255-0.270 are essentially at
   chance (0.25), but with n=200 items and 4 options, the Wilson 95% CI upper bound
   (0.3196-0.3354) exceeds the threshold (0.30). A larger corpus would resolve these.

2. **Structural abstention signal (baseline 6, real):** TF-IDF word achieves **100% accuracy
   on INSUFFICIENT items**. The abstention hypothesis "Cannot be determined from available
   evidence" has structurally different text from "[Name] is responsible" hypotheses. The
   TF-IDF classifier trivially learns to distinguish the abstention candidate from suspect
   candidates. Since gold=abstention for all INSUFFICIENT items, it achieves perfect accuracy
   on that regime, driving overall accuracy to 0.48 (well above threshold).

   Baseline 11 (combined shallow features) also elevated at 0.35 overall, with 0.38 on
   INSUFFICIENT, likely due to the same structural asymmetry.

### Why This Result Is Informative (Not Just Wrong)

The v3 design successfully enforces:
- Universal 4-option format (S1 PASS)
- Exact gold-position balance (S2, S6 PASS)
- Exact abstention-position balance (S3 PASS, novel v3 guarantee)
- Option text length matching (S5 PASS)

But it fails on:
- **S4:** Evidence count normalization across regimes (inherited from v2's regime-specific evidence generation)
- **Abstention text asymmetry:** The abstention option is lexically distinct from suspect options, creating a trivially-exploitable surface-form signal for TF-IDF classifiers

### Corrective Actions (Implemented in v3.1)

1. **Lexically-parallel hypothesis text:** Restructure all hypothesis options to share the same syntactic frame, making bag-of-words indistinguishable between suspect and abstention options.
2. **Evidence-slot normalization:** Standardize evidence count across all regimes, varying only logical relationships (not slot count or surface length).
3. **Paired answerable/insufficient items:** Construct minimal pairs that share surface structure, differing only in evidentiary relations.
4. **Diagnostic baselines:** Add option-only, context-only, and full-pipeline baselines to disentangle "recognizing abstention text" from "detecting insufficiency from evidence".

---

*This artifact is part of the v3 -> v3.1 audit trail. Generated from commit `0166044` (v3 generator).*
