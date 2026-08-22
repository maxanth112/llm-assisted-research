"""
Sample Generator for T2 Diagnostic Items

Generates a sample of 16 items (4 per regime) and exports to:
- JSONL format for machine consumption
- Markdown format for human readability
"""

import sys
from pathlib import Path

# Add parent directory to path to import generator
sys.path.insert(0, str(Path(__file__).parent))

from generator import T2Generator, export_jsonl, T2Item
from typing import List


def generate_sample(n_per_regime: int = 4, seed: int = 42) -> List[T2Item]:
    """Generate a sample dataset."""
    generator = T2Generator(seed=seed)
    items = generator.generate_dataset(n_per_regime=n_per_regime, seed=seed)
    return items


def export_readable_markdown(items: List[T2Item], path: str):
    """Export items to human-readable markdown format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# T2 Diagnostic Test Items - Sample Dataset\n\n")
        f.write(f"Generated {len(items)} items across 4 regimes.\n\n")

        # Group by regime
        regimes = {}
        for item in items:
            if item.regime not in regimes:
                regimes[item.regime] = []
            regimes[item.regime].append(item)

        # Write items grouped by regime
        for regime in ["CLEAN", "DECOY", "CONFLICT", "INSUFFICIENT"]:
            if regime not in regimes:
                continue

            f.write(f"## {regime} Regime\n\n")

            if regime == "CLEAN":
                f.write("Evidence uniquely identifies one hypothesis. Multiple evidence items all point to the guilty suspect.\n\n")
            elif regime == "DECOY":
                f.write("Clean evidence + salient-but-non-diagnostic decoy evidence pointing at innocent suspects.\n\n")
            elif regime == "CONFLICT":
                f.write("Sources genuinely disagree. Source-precedence rule determines the gold answer.\n\n")
            elif regime == "INSUFFICIENT":
                f.write("Evidence is genuinely ambiguous. All suspects have equal evidence.\n\n")

            f.write("---\n\n")

            for idx, item in enumerate(regimes[regime], 1):
                f.write(f"### Item {idx}: {item.id}\n\n")

                # Metadata
                f.write(f"**Template:** {item.metadata.get('template', 'N/A')}\n\n")

                # Narrative
                f.write(f"**Narrative:**\n\n{item.narrative}\n\n")

                # Question
                f.write(f"**Question:**\n\n{item.question}\n\n")

                # Hypotheses
                f.write(f"**Hypotheses:**\n\n")
                for i, hyp in enumerate(item.hypotheses, 1):
                    marker = " ✓" if hyp == item.gold_answer else ""
                    f.write(f"{i}. {hyp}{marker}\n")
                f.write("\n")

                # Evidence
                f.write(f"**Evidence ({len(item.evidence)} items):**\n\n")
                for ev in item.evidence:
                    f.write(f"- **{ev['id']}** [{ev['diagnostic_value']}]: {ev['content']}\n")
                    if ev['supports']:
                        f.write(f"  - *Supports:* {', '.join(ev['supports'])}\n")
                    if ev['contradicts']:
                        f.write(f"  - *Contradicts:* {', '.join(ev['contradicts'])}\n")
                f.write("\n")

                # Source precedence rule (for CONFLICT items)
                if item.source_precedence_rule:
                    f.write(f"**Source Precedence Rule:**\n\n{item.source_precedence_rule}\n\n")

                # Gold answer
                f.write(f"**Gold Answer:**\n\n{item.gold_answer}\n\n")

                # Gold reasoning
                f.write(f"**Gold Reasoning:**\n\n{item.gold_reasoning}\n\n")

                f.write("---\n\n")

            f.write("\n")

    print(f"Exported readable markdown to {path}")


def print_statistics(items: List[T2Item]):
    """Print dataset statistics."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    print(f"\nTotal items: {len(items)}")

    # Regime distribution
    regime_counts = {}
    for item in items:
        regime_counts[item.regime] = regime_counts.get(item.regime, 0) + 1

    print("\nRegime distribution:")
    for regime in ["CLEAN", "DECOY", "CONFLICT", "INSUFFICIENT"]:
        count = regime_counts.get(regime, 0)
        print(f"  {regime:15s}: {count:2d} items")

    # Template distribution
    template_counts = {}
    for item in items:
        template = item.metadata.get('template', 'unknown')
        template_counts[template] = template_counts.get(template, 0) + 1

    print("\nTemplate distribution:")
    for template, count in sorted(template_counts.items()):
        print(f"  {template:15s}: {count:2d} items")

    # Evidence statistics
    evidence_counts = [len(item.evidence) for item in items]
    print(f"\nEvidence per item:")
    print(f"  Min: {min(evidence_counts)}")
    print(f"  Max: {max(evidence_counts)}")
    print(f"  Avg: {sum(evidence_counts) / len(evidence_counts):.1f}")

    # Hypothesis statistics
    hypothesis_counts = [len(item.hypotheses) for item in items]
    print(f"\nHypotheses per item:")
    print(f"  Min: {min(hypothesis_counts)}")
    print(f"  Max: {max(hypothesis_counts)}")
    print(f"  Avg: {sum(hypothesis_counts) / len(hypothesis_counts):.1f}")

    # Gold answer distribution
    cannot_determine_count = sum(1 for item in items if item.gold_answer == "Cannot be determined from available evidence")
    print(f"\nGold answers:")
    print(f"  Specific suspect: {len(items) - cannot_determine_count} items")
    print(f"  Cannot determine: {cannot_determine_count} items")

    # Conflict items with precedence rules
    conflict_with_rules = sum(1 for item in items if item.regime == "CONFLICT" and item.source_precedence_rule)
    print(f"\nConflict items with precedence rules: {conflict_with_rules}")

    print("\n" + "="*60)


def main():
    """Main execution."""
    print("Generating T2 sample dataset...")

    # Generate sample
    items = generate_sample(n_per_regime=4, seed=42)

    # Create output directory
    output_dir = Path(__file__).parent / "sample"
    output_dir.mkdir(exist_ok=True)

    # Export to JSONL
    jsonl_path = output_dir / "sample_items.jsonl"
    export_jsonl(items, str(jsonl_path))

    # Export to readable markdown
    markdown_path = output_dir / "sample_readable.md"
    export_readable_markdown(items, str(markdown_path))

    # Print statistics
    print_statistics(items)

    print(f"\nSample generation complete!")
    print(f"  JSONL output: {jsonl_path}")
    print(f"  Markdown output: {markdown_path}")


if __name__ == "__main__":
    main()
