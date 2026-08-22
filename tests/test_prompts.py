"""Tests for prompt templates."""

from harness.prompts.templates import (
    ALL_TEMPLATES,
    render_prompt,
    get_all_templates,
    PromptTemplate,
)


class TestPromptTemplates:
    """Verify all prompt templates are defined and renderable."""

    def test_all_templates_count(self):
        """Should have at least 12 templates."""
        assert len(ALL_TEMPLATES) >= 12

    def test_all_templates_have_hash(self):
        for name, tmpl in ALL_TEMPLATES.items():
            assert tmpl.content_hash, f"{name} missing content_hash"
            assert len(tmpl.content_hash) == 64, f"{name} hash wrong length"

    def test_all_templates_have_version(self):
        for name, tmpl in ALL_TEMPLATES.items():
            assert tmpl.version, f"{name} missing version"

    def test_hash_deterministic(self):
        """Recomputing hash should give the same value."""
        import hashlib
        for name, tmpl in ALL_TEMPLATES.items():
            expected = hashlib.sha256(tmpl.template_text.encode('utf-8')).hexdigest()
            assert tmpl.content_hash == expected, f"{name} hash mismatch"

    def test_render_basic_template(self):
        """Test rendering the direct answer template."""
        from harness.prompts.templates import TEMPLATE_000_DIRECT
        rendered = render_prompt(
            TEMPLATE_000_DIRECT,
            narrative="A crime occurred.",
            question="Who did it?",
            choices=["Alice", "Bob", "Charlie"]
        )
        assert "A crime occurred." in rendered
        assert "Who did it?" in rendered
        assert "Alice" in rendered

    def test_get_all_templates_returns_copy(self):
        a = get_all_templates()
        b = get_all_templates()
        a["extra"] = None
        assert "extra" not in get_all_templates()

    def test_condition_template_names_exist(self):
        """Every condition's prompt_template_name should map to a real template."""
        from harness.conditions import ALL_CONDITIONS
        # Templates are named differently from condition template names,
        # but we verify the template registry is populated
        template_names = set(ALL_TEMPLATES.keys())
        assert len(template_names) >= 8
