#!/usr/bin/env python3
"""
Two-stage clustered bootstrap for one-sided 95% upper bound
on orientation-invariant separability statistic.

S = max(BA, 1-BA) = 0.5 + abs(BA - 0.5)

Correction 2: Uses orientation-invariant S instead of raw BA.
A raw BA of 0.0 (perfect reversal) gives S=1.0 (maximal leakage), not 0.0.
Chance = 0.5 for binary. Pass requires upper_95 of S <= 0.55.

Stage 1: resample template families with replacement
Stage 2: within each drawn family, resample pairs with replacement
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict


def two_stage_bootstrap(predictions: List[Dict],
                        n_bootstrap: int = 10000,
                        seed: int = 42) -> Dict:
    """
    Two-stage clustered bootstrap on saved OOF predictions.

    Bootstraps the ORIENTATION-INVARIANT separability statistic:
    S = 0.5 + abs(BA - 0.5)

    This ensures that perfectly reversed predictions (BA=0.0) correctly
    produce S=1.0 (maximal leakage), not S=0.0 (passing).

    Args:
        predictions: list of dicts with keys:
            item_id, family, true_label, predicted_label
        n_bootstrap: number of bootstrap resamples
        seed: random seed

    Returns dict with:
        raw_balanced_accuracy, separability, upper_95, gate_passes, etc.
    """
    rng = np.random.RandomState(seed)

    family_to_pairs = defaultdict(lambda: defaultdict(list))
    for p in predictions:
        fam = p.get("family", "unknown")
        item_id = p["item_id"]
        pair_id = p.get("pair_id",
                        item_id.rsplit("_", 1)[0]
                        if "_clean" in item_id or "_insuf" in item_id
                        else item_id)
        family_to_pairs[fam][pair_id].append(p)

    unique_families = sorted(family_to_pairs.keys())
    F = len(unique_families)

    if F == 0:
        return {"raw_balanced_accuracy": 0.5, "separability": 0.5,
                "upper_95": 1.0, "gate_passes": False,
                "n_bootstrap": n_bootstrap, "n_families": 0, "n_items": 0,
                "bootstrap_distribution": []}

    def balanced_accuracy(preds_list):
        if not preds_list:
            return 0.5
        true = np.array([p["true_label"] for p in preds_list])
        pred = np.array([p["predicted_label"] for p in preds_list])
        pos = true == 1
        neg = true == 0
        tpr = np.mean(pred[pos] == 1) if np.any(pos) else 0.5
        tnr = np.mean(pred[neg] == 0) if np.any(neg) else 0.5
        return 0.5 * (tpr + tnr)

    def separability(preds_list):
        ba = balanced_accuracy(preds_list)
        return 0.5 + abs(ba - 0.5)

    # Point estimates
    point_ba = balanced_accuracy(predictions)
    point_sep = separability(predictions)

    # Bootstrap the SEPARABILITY statistic (not raw BA)
    boot_stats = []
    for _ in range(n_bootstrap):
        sampled_fams = rng.choice(unique_families, size=F, replace=True)
        boot_preds = []
        for fam in sampled_fams:
            pairs = list(family_to_pairs[fam].keys())
            n_pairs = len(pairs)
            if n_pairs == 0:
                continue
            sampled_pairs = rng.choice(pairs, size=n_pairs, replace=True)
            for sp in sampled_pairs:
                boot_preds.extend(family_to_pairs[fam][sp])
        boot_stats.append(separability(boot_preds))

    boot_stats = np.array(boot_stats)
    upper_95 = float(np.percentile(boot_stats, 95))

    return {
        "raw_balanced_accuracy": float(point_ba),
        "separability": float(point_sep),
        "upper_95": upper_95,
        "gate_passes": upper_95 <= 0.55,
        "n_bootstrap": n_bootstrap,
        "n_families": F,
        "n_items": len(predictions),
        "bootstrap_mean": float(np.mean(boot_stats)),
        "bootstrap_std": float(np.std(boot_stats)),
        "bootstrap_5th": float(np.percentile(boot_stats, 5)),
        "bootstrap_50th": float(np.percentile(boot_stats, 50)),
        "bootstrap_95th": float(np.percentile(boot_stats, 95)),
        "sample_scale_note": "N=16, illustrative only. Real gate at >=500/regime.",
    }
