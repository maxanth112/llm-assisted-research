# v3.1 Development-Gate Leakage Evaluation Report

Generated: 2026-09-01 04:59:16 UTC

## Configuration

| Parameter | Value |
|---|---|
| Generator | T2V31Generator v3.1.0 |
| n_per_regime | 500 |
| Seed | 42 |
| Total items | 2000 |
| Evidence per item | 10 |
| Hypotheses per item | 4 |
| Chance level | 0.25000 |
| Threshold | 0.30000 |
| Corpus SHA-256 | `0512b93775ae2f1d80f74f68f1ca3fcbc6516ea843b4c9441e5c0e6f923460b3` |

## Reproduction Command

```bash
cd llm-assisted-research
python3 analysis/run_v31_dev_gate.py
```

## Surface-Form Check Results

| Check | Result |
|---|---|
| S1_option_count | PASS |
| S2_abstention_position | PASS |
| S3_abstention_presence | PASS |
| S4_evidence_count | PASS |
| S5_option_text_length | PASS |
| S6_gold_position | PASS |

## Template-Held-Out Baseline Results (Aggregate)

| Baseline | Accuracy | Wilson 95% CI | Verdict |
|---|---|---|---|
| 1_majority_class | 0.2500 | [0.2315, 0.2694] | PASS |
| 2_label_position | 0.2500 | [0.2315, 0.2694] | PASS |
| 3_mention_count | 0.2495 | [0.2310, 0.2689] | PASS |
| 4_evidence_count | 0.2495 | [0.2310, 0.2689] | PASS |
| 5_lexical_overlap | 0.2845 | [0.2651, 0.3047] | **FAIL** |
| 6_tfidf_word | 0.2450 | [0.2266, 0.2643] | PASS |
| 7_tfidf_char | 0.2495 | [0.2310, 0.2689] | PASS |
| 8_length_feature | 0.2500 | [0.2315, 0.2694] | PASS |
| 9_mention_evidence | 0.2495 | [0.2310, 0.2689] | PASS |
| 10_first_mention_order | 0.2570 | [0.2383, 0.2766] | PASS |
| 11_combined_shallow | 0.2360 | [0.2179, 0.2551] | PASS |

### Per-Regime: 6_tfidf_word

| Regime | Accuracy | Wilson 95% CI | Chance | Verdict |
|---|---|---|---|---|
| CLEAN | 0.3300 | [0.2902, 0.3724] | 0.2500 | **FAIL** |
| DECOY | 0.3080 | [0.2691, 0.3498] | 0.2500 | **FAIL** |
| CONFLICT | 0.3420 | [0.3018, 0.3846] | 0.2500 | **FAIL** |
| INSUFFICIENT | 0.0000 | [0.0000, 0.0076] | 0.2500 | PASS |

### Per-Regime: 11_combined_shallow

| Regime | Accuracy | Wilson 95% CI | Chance | Verdict |
|---|---|---|---|---|
| CLEAN | 0.2640 | [0.2273, 0.3043] | 0.2500 | **FAIL** |
| DECOY | 0.2960 | [0.2577, 0.3375] | 0.2500 | **FAIL** |
| CONFLICT | 0.3440 | [0.3037, 0.3867] | 0.2500 | **FAIL** |
| INSUFFICIENT | 0.0400 | [0.0260, 0.0610] | 0.2500 | PASS |

## Diagnostic Baseline Results

### D1: Option-Only Baseline

- **Aggregate**: acc=0.2645, CI=[0.2456, 0.2843], verdict=PASS
- Measures: whether hypothesis text alone leaks the answer

| Regime | Accuracy | Wilson 95% CI | Verdict |
|---|---|---|---|
| CLEAN | 0.3180 | [0.2787, 0.3601] | **FAIL** |
| DECOY | 0.2860 | [0.2481, 0.3271] | **FAIL** |
| CONFLICT | 0.3480 | [0.3075, 0.3908] | **FAIL** |
| INSUFFICIENT | 0.1060 | [0.0820, 0.1361] | PASS |

### D2: Context-Regime Classifier

- **Aggregate**: acc=1.0000, CI=[0.9981, 1.0000], chance=0.5000, verdict=FAIL
- Measures: whether narrative+evidence alone reveals the regime

### D3: Full Candidate-Aware TF-IDF

- **Aggregate**: acc=0.2450, CI=[0.2266, 0.2643], verdict=PASS

| Regime | Accuracy | Wilson 95% CI | Verdict |
|---|---|---|---|
| CLEAN | 0.3300 | [0.2902, 0.3724] | **FAIL** |
| DECOY | 0.3080 | [0.2691, 0.3498] | **FAIL** |
| CONFLICT | 0.3420 | [0.3018, 0.3846] | **FAIL** |
| INSUFFICIENT | 0.0000 | [0.0000, 0.0076] | PASS |

## Overall Verdict: FAIL

Failed baselines: ['5_lexical_overlap']

Elapsed: 320.2s

