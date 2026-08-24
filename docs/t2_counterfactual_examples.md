# T2 v2 Counterfactual Minimal Pair Examples

These examples demonstrate the counterfactual minimal pair mechanism in T2 v2.
Each pair shares the same narrative, evidence vocabulary, and entity names,
but a small relational change (e.g., swapping which alibi is invalidated)
flips the correct answer to a different suspect.

**Purpose:** If tokens match but the answer differs, no bag-of-words classifier
can succeed. This is the primary anti-leakage device in T2 v2.

**Specification:** AMENDMENT-001 section 3.3. Acceptance threshold: Jaccard >= 0.85.

---

## Pair 1: CLEAN / t2v2_clean_theft_alibi_0000

- **Regime:** CLEAN
- **Token Jaccard similarity:** 1.000
- **Original gold answer:** Grey Martinez is responsible
- **Counterfactual gold answer:** Morgan Walsh is responsible

### Narrative (shared)

> At BioResearch Labs, a valuable access key was stolen from Building A - East Wing on May 8, 2024 between 8:00 PM and 9:00 PM. Investigation has identified 3 employees with potential access.

### Hypotheses

| # | Original | Counterfactual |
|---|----------|----------------|
| 1 | Grey Martinez is responsible **(GOLD)** | Grey Martinez is responsible |
| 2 | Finley Brooks is responsible | Finley Brooks is responsible |
| 3 | Morgan Walsh is responsible | Morgan Walsh is responsible **(GOLD)** |

### Evidence (first 3 items)

**E001:**
- Original: Finley Brooks claims to have been at Warehouse District during the incident, confirmed by Lab Supervisor Dr. Taylor Kim
- Counterfactual: Finley Brooks claims to have been at Warehouse District during the incident, confirmed by Lab Supervisor Dr. Taylor Kim
- **Changed:** No (identical)

**E002:**
- Original: Access logs show Morgan Walsh was present near the scene at 8:15 PM, as per company policy, in compliance with regulations
- Counterfactual: Access logs show Morgan Walsh was present near the scene at 8:15 PM, as per company policy, in compliance with regulations
- **Changed:** No (identical)

**E003:**
- Original: Custodian Mike Thompson was actually on approved leave and not present at work on May 8, 2024, according to HR records
- Counterfactual: Training Coordinator Casey Moore was actually on approved leave and not present at work on May 8, 2024, according to HR records
- **Changed:** Yes (relational swap)

### Key relational difference

The original item's gold answer is **Grey Martinez is responsible** while the
counterfactual's gold answer is **Morgan Walsh is responsible**. The swap is
achieved by changing which suspect's alibi is invalidated (or which
timeline is inconsistent), while keeping all surface-level text identical.

---

## Pair 2: CLEAN / t2v2_clean_theft_timeline_0001

- **Regime:** CLEAN
- **Token Jaccard similarity:** 1.000
- **Original gold answer:** Grey Martinez is responsible
- **Counterfactual gold answer:** Finley Brooks is responsible

### Narrative (shared)

> At Global Logistics Co, a data backup drive was removed from Building B - Research Lab on March 15, 2024. The theft occurred during a specific time window between 7:00 PM and 8:00 PM. 3 employees are under investigation.

### Hypotheses

| # | Original | Counterfactual |
|---|----------|----------------|
| 1 | Grey Martinez is responsible **(GOLD)** | Grey Martinez is responsible |
| 2 | Dana Patel is responsible | Dana Patel is responsible |
| 3 | Finley Brooks is responsible | Finley Brooks is responsible **(GOLD)** |

### Evidence (first 3 items)

**E001:**
- Original: Production Floor - Zone 4's security logs show no entries during the incident window on March 15, 2024, per the automated access system
- Counterfactual: Research Campus - North Wing's security logs show no entries during the incident window on March 15, 2024, per the automated access system
- **Changed:** Yes (relational swap)

**E002:**
- Original: Dana Patel reported being at Warehouse District at the time, verified by Shift Manager Alex Park, as documented in the official records
- Counterfactual: Dana Patel reported being at Warehouse District at the time, verified by Shift Manager Alex Park, as documented in the official records
- **Changed:** No (identical)

**E003:**
- Original: Badge records indicate Grey Martinez entered the area at 8:25 PM, according to standard procedures, following routine protocols
- Counterfactual: Badge records indicate Grey Martinez entered the area at 8:25 PM, according to standard procedures, following routine protocols
- **Changed:** No (identical)

### Key relational difference

The original item's gold answer is **Grey Martinez is responsible** while the
counterfactual's gold answer is **Finley Brooks is responsible**. The swap is
achieved by changing which suspect's alibi is invalidated (or which
timeline is inconsistent), while keeping all surface-level text identical.

---

## Pair 3: CLEAN / t2v2_clean_sabotage_alibi_0002

- **Regime:** CLEAN
- **Token Jaccard similarity:** 1.000
- **Original gold answer:** Alex Chen is responsible
- **Counterfactual gold answer:** Casey Morgan is responsible

### Narrative (shared)

> At Manufacturing Solutions Inc's Production Floor - Zone 4, critical equipment was sabotaged on March 15, 2024, causing a production shutdown. The sabotage occurred between 7:00 PM and 8:00 PM. 3 employees had the technical knowledge and access.

### Hypotheses

| # | Original | Counterfactual |
|---|----------|----------------|
| 1 | Alex Chen is responsible **(GOLD)** | Alex Chen is responsible |
| 2 | Harper Quinn is responsible | Harper Quinn is responsible |
| 3 | Casey Morgan is responsible | Casey Morgan is responsible **(GOLD)** |

### Evidence (first 3 items)

**E001:**
- Original: Harper Quinn claims to have been at Security Office Complex during the incident, confirmed by Shift Manager Alex Park
- Counterfactual: Harper Quinn claims to have been at Security Office Complex during the incident, confirmed by Shift Manager Alex Park
- **Changed:** No (identical)

**E002:**
- Original: Alex Chen claims to have been at Testing Laboratory during the incident, confirmed by Training Coordinator Casey Moore
- Counterfactual: Alex Chen claims to have been at Testing Laboratory during the incident, confirmed by Training Coordinator Casey Moore
- **Changed:** No (identical)

**E003:**
- Original: Access logs show Alex Chen was present near the scene at 8:15 PM, in accordance with protocol, in accordance with protocol
- Counterfactual: Access logs show Alex Chen was present near the scene at 8:15 PM, in accordance with protocol, in accordance with protocol
- **Changed:** No (identical)

### Key relational difference

The original item's gold answer is **Alex Chen is responsible** while the
counterfactual's gold answer is **Casey Morgan is responsible**. The swap is
achieved by changing which suspect's alibi is invalidated (or which
timeline is inconsistent), while keeping all surface-level text identical.

---

## Pair 4: CLEAN / t2v2_clean_sabotage_timeline_0003

- **Regime:** CLEAN
- **Token Jaccard similarity:** 1.000
- **Original gold answer:** Blake Rivera is responsible
- **Counterfactual gold answer:** Kelly Zhao is responsible

### Narrative (shared)

> Critical systems at TechCorp Industries's Downtown Branch Office were deliberately disabled on May 8, 2024. System logs indicate the sabotage occurred during a narrow window between 6:30 PM and 7:30 PM. 3 individuals had access.

### Hypotheses

| # | Original | Counterfactual |
|---|----------|----------------|
| 1 | Kelly Zhao is responsible | Kelly Zhao is responsible **(GOLD)** |
| 2 | Casey Morgan is responsible | Casey Morgan is responsible |
| 3 | Blake Rivera is responsible **(GOLD)** | Blake Rivera is responsible |

### Evidence (first 3 items)

**E001:**
- Original: Kelly Zhao reported being at Security Office Complex at the time, verified by IT Administrator Pat Williams, according to standard procedures
- Counterfactual: Kelly Zhao reported being at Security Office Complex at the time, verified by IT Administrator Pat Williams, according to standard procedures
- **Changed:** No (identical)

**E002:**
- Original: Blake Rivera reported being at Research Campus - North Wing at the time, verified by Operations Manager Drew Bennett, consistent with established guidelines
- Counterfactual: Blake Rivera reported being at Research Campus - North Wing at the time, verified by Operations Manager Drew Bennett, consistent with established guidelines
- **Changed:** No (identical)

**E003:**
- Original: Badge records indicate Casey Morgan entered the area at 8:15 PM, in compliance with regulations, in compliance with regulations
- Counterfactual: Badge records indicate Casey Morgan entered the area at 8:15 PM, in compliance with regulations, in compliance with regulations
- **Changed:** No (identical)

### Key relational difference

The original item's gold answer is **Blake Rivera is responsible** while the
counterfactual's gold answer is **Kelly Zhao is responsible**. The swap is
achieved by changing which suspect's alibi is invalidated (or which
timeline is inconsistent), while keeping all surface-level text identical.

---
