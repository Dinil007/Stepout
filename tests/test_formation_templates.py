from __future__ import annotations

import pytest

from app.analytics.formation_templates import (
    FormationTemplateRegistry,
    default_registry,
    from_dict,
    to_dict,
)
from app.analytics.formation_types import FormationTemplate


def test_registry_initializes_with_ten_templates():
    registry = FormationTemplateRegistry()
    assert registry.count_templates() == 10


def test_expected_formations_exist():
    expected = {
        "4-3-3",
        "4-2-3-1",
        "4-4-2",
        "4-1-4-1",
        "4-5-1",
        "3-5-2",
        "3-4-3",
        "3-4-2-1",
        "5-3-2",
        "5-4-1",
    }
    assert set(default_registry.list_templates()) == expected


def test_get_template_returns_correct_template():
    template = default_registry.get_template("4-3-3")
    assert template.formation_name == "4-3-3"
    assert template.defenders == 4
    assert template.midfielders == 3
    assert template.forwards == 3
    assert len(template.normalized_positions) == 10


def test_duplicate_registration_raises():
    registry = FormationTemplateRegistry()
    with pytest.raises(ValueError):
        registry.register_template(registry.get_template("4-3-3"))


def test_invalid_template_registration_rejected():
    registry = FormationTemplateRegistry()
    bad = FormationTemplate(
        formation_name="bad",
        defenders=-1,
        midfielders=0,
        forwards=0,
    )
    with pytest.raises(ValueError):
        registry.register_template(bad)


def test_remove_template_success():
    registry = FormationTemplateRegistry()
    registry.remove_template("4-3-3")
    assert registry.template_exists("4-3-3") is False


def test_remove_template_missing_raises():
    registry = FormationTemplateRegistry()
    with pytest.raises(KeyError):
        registry.remove_template("nonexistent")


def test_count_templates_matches_registry():
    registry = FormationTemplateRegistry()
    assert registry.count_templates() == len(registry.templates)


def test_serialization_helpers_preserve_data():
    template = default_registry.get_template("4-3-3")
    data = to_dict(template)
    recovered = from_dict(data)
    assert recovered.formation_name == template.formation_name
    assert recovered.defenders == template.defenders
    assert recovered.midfielders == template.midfielders
    assert recovered.forwards == template.forwards
    assert recovered.normalized_positions == list(template.normalized_positions)
    assert recovered.description == template.description


def test_template_exists_true():
    registry = FormationTemplateRegistry()
    assert registry.template_exists("4-4-2") is True


def test_template_exists_false():
    registry = FormationTemplateRegistry()
    assert registry.template_exists("unknown") is False