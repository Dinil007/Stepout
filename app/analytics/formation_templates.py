from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.analytics.formation_types import FormationTemplate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical formation templates
# ---------------------------------------------------------------------------

FOUR_THREE_THREE: FormationTemplate = FormationTemplate(
    formation_name="4-3-3",
    defenders=4,
    midfielders=3,
    forwards=3,
    normalized_positions=[
        # Defenders
        (0.2, 0.25),
        (0.4, 0.25),
        (0.6, 0.25),
        (0.8, 0.25),
        # Midfielders
        (0.3, 0.5),
        (0.5, 0.5),
        (0.7, 0.5),
        # Forwards
        (0.25, 0.75),
        (0.5, 0.75),
        (0.75, 0.75),
    ],
    description="Four defenders, three midfielders, and three forwards. Balanced attack with wide wingers.",
)

FOUR_TWO_THREE_ONE: FormationTemplate = FormationTemplate(
    formation_name="4-2-3-1",
    defenders=4,
    midfielders=5,
    forwards=1,
    normalized_positions=[
        # Defenders
        (0.2, 0.25),
        (0.4, 0.25),
        (0.6, 0.25),
        (0.8, 0.25),
        # Holding midfielders
        (0.35, 0.4),
        (0.65, 0.4),
        # Attacking midfielders
        (0.25, 0.58),
        (0.5, 0.58),
        (0.75, 0.58),
        # Striker
        (0.5, 0.78),
    ],
    description="Four defenders, two holding midfielders, three attacking midfielders, and one striker. Flexible midfield-heavy shape.",
)

FOUR_FOUR_TWO: FormationTemplate = FormationTemplate(
    formation_name="4-4-2",
    defenders=4,
    midfielders=4,
    forwards=2,
    normalized_positions=[
        # Defenders
        (0.2, 0.25),
        (0.4, 0.25),
        (0.6, 0.25),
        (0.8, 0.25),
        # Midfielders
        (0.2, 0.5),
        (0.4, 0.5),
        (0.6, 0.5),
        (0.8, 0.5),
        # Forwards
        (0.35, 0.78),
        (0.65, 0.78),
    ],
    description="Four defenders, four midfielders, and two forwards. Classic flat midfield with striker partnership.",
)

FOUR_ONE_FOUR_ONE: FormationTemplate = FormationTemplate(
    formation_name="4-1-4-1",
    defenders=4,
    midfielders=5,
    forwards=1,
    normalized_positions=[
        # Defenders
        (0.2, 0.25),
        (0.4, 0.25),
        (0.6, 0.25),
        (0.8, 0.25),
        # Defensive midfielder
        (0.5, 0.38),
        # Midfielders
        (0.2, 0.52),
        (0.4, 0.52),
        (0.6, 0.52),
        (0.8, 0.52),
        # Striker
        (0.5, 0.78),
    ],
    description="Four defenders, one defensive midfielder, four central midfielders, and one lone striker. Defensively solid.",
)

FOUR_FIVE_ONE: FormationTemplate = FormationTemplate(
    formation_name="4-5-1",
    defenders=4,
    midfielders=5,
    forwards=1,
    normalized_positions=[
        # Defenders
        (0.2, 0.25),
        (0.4, 0.25),
        (0.6, 0.25),
        (0.8, 0.25),
        # Midfielders
        (0.2, 0.48),
        (0.35, 0.48),
        (0.5, 0.48),
        (0.65, 0.48),
        (0.8, 0.48),
        # Striker
        (0.5, 0.78),
    ],
    description="Four defenders, five midfielders, and one striker. Compact midfield with lone striker.",
)

THREE_FIVE_TWO: FormationTemplate = FormationTemplate(
    formation_name="3-5-2",
    defenders=3,
    midfielders=5,
    forwards=2,
    normalized_positions=[
        # Defenders
        (0.3, 0.25),
        (0.5, 0.25),
        (0.7, 0.25),
        # Wing-backs
        (0.15, 0.5),
        (0.85, 0.5),
        # Central midfielders
        (0.35, 0.5),
        (0.5, 0.5),
        (0.65, 0.5),
        # Forwards
        (0.35, 0.78),
        (0.65, 0.78),
    ],
    description="Three central defenders, five midfielders including wing-backs, and two forwards. Balanced width and central presence.",
)

THREE_FOUR_THREE: FormationTemplate = FormationTemplate(
    formation_name="3-4-3",
    defenders=3,
    midfielders=4,
    forwards=3,
    normalized_positions=[
        # Defenders
        (0.3, 0.25),
        (0.5, 0.25),
        (0.7, 0.25),
        # Midfielders
        (0.2, 0.5),
        (0.4, 0.5),
        (0.6, 0.5),
        (0.8, 0.5),
        # Forwards
        (0.25, 0.78),
        (0.5, 0.78),
        (0.75, 0.78),
    ],
    description="Three central defenders, four midfielders, and three forwards. Attacking with wide midfielders.",
)

THREE_FOUR_TWO_ONE: FormationTemplate = FormationTemplate(
    formation_name="3-4-2-1",
    defenders=3,
    midfielders=6,
    forwards=1,
    normalized_positions=[
        # Defenders
        (0.3, 0.25),
        (0.5, 0.25),
        (0.7, 0.25),
        # Midfielders
        (0.2, 0.45),
        (0.4, 0.45),
        (0.6, 0.45),
        (0.8, 0.45),
        # Attacking midfielders
        (0.35, 0.65),
        (0.65, 0.65),
        # Striker
        (0.5, 0.82),
    ],
    description="Three central defenders, four midfielders, two attacking midfielders, and one striker. Solid defensive block with creative freedom.",
)

FIVE_THREE_TWO: FormationTemplate = FormationTemplate(
    formation_name="5-3-2",
    defenders=5,
    midfielders=3,
    forwards=2,
    normalized_positions=[
        # Defenders
        (0.15, 0.25),
        (0.35, 0.25),
        (0.5, 0.25),
        (0.65, 0.25),
        (0.85, 0.25),
        # Midfielders
        (0.35, 0.5),
        (0.5, 0.5),
        (0.65, 0.5),
        # Forwards
        (0.35, 0.78),
        (0.65, 0.78),
    ],
    description="Five defenders, three midfielders, and two forwards. Highly defensive with counter-attacking focus.",
)

FIVE_FOUR_ONE: FormationTemplate = FormationTemplate(
    formation_name="5-4-1",
    defenders=5,
    midfielders=4,
    forwards=1,
    normalized_positions=[
        # Defenders
        (0.15, 0.25),
        (0.35, 0.25),
        (0.5, 0.25),
        (0.65, 0.25),
        (0.85, 0.25),
        # Midfielders
        (0.2, 0.5),
        (0.4, 0.5),
        (0.6, 0.5),
        (0.8, 0.5),
        # Striker
        (0.5, 0.78),
    ],
    description="Five defenders, four midfielders, and one striker. Ultra-defensive shape suited for protecting leads.",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class FormationTemplateRegistry:
    """Reusable registry of formation templates.

    Prevents duplicate formation names and provides lightweight
    registration, lookup, and validation helpers.
    """

    templates: dict[str, FormationTemplate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Register canonical templates on initialization."""
        canonical_templates = [
            FOUR_THREE_THREE,
            FOUR_TWO_THREE_ONE,
            FOUR_FOUR_TWO,
            FOUR_ONE_FOUR_ONE,
            FOUR_FIVE_ONE,
            THREE_FIVE_TWO,
            THREE_FOUR_THREE,
            THREE_FOUR_TWO_ONE,
            FIVE_THREE_TWO,
            FIVE_FOUR_ONE,
        ]
        for template in canonical_templates:
            try:
                self.register_template(template)
            except ValueError:
                # Skip duplicates during initialization
                pass
        logger.info(
            "FormationTemplateRegistry initialized with %d templates", self.count_templates()
        )

    def register_template(self, template: FormationTemplate) -> None:
        """Register a new formation template.

        Args:
            template: FormationTemplate instance to register.

        Raises:
            ValueError: If the template is invalid or the name already exists.
        """
        if not template.is_valid():
            raise ValueError(
                f"Invalid template '{template.formation_name}': validation failed."
            )
        if self.template_exists(template.formation_name):
            logger.warning("Duplicate formation template registration: %s", template.formation_name)
            raise ValueError(
                f"Formation template '{template.formation_name}' is already registered."
            )
        self.templates[template.formation_name] = template
        logger.info("Registered formation template: %s", template.formation_name)

    def remove_template(self, name: str) -> None:
        """Remove a formation template by name.

        Args:
            name: Formation name to remove.

        Raises:
            KeyError: If the template does not exist.
        """
        if not self.template_exists(name):
            raise KeyError(f"Formation template '{name}' does not exist.")
        del self.templates[name]
        logger.info("Removed formation template: %s", name)

    def get_template(self, name: str) -> FormationTemplate:
        """Retrieve a formation template by name.

        Args:
            name: Formation name to retrieve.

        Returns:
            FormationTemplate instance.

        Raises:
            KeyError: If the template does not exist.
        """
        if not self.template_exists(name):
            raise KeyError(f"Formation template '{name}' does not exist.")
        return self.templates[name]

    def list_templates(self) -> list[str]:
        """List all registered formation names.

        Returns:
            Sorted list of formation names.
        """
        return sorted(self.templates.keys())

    def template_exists(self, name: str) -> bool:
        """Check whether a formation template is registered.

        Args:
            name: Formation name to check.

        Returns:
            True if the template is registered, False otherwise.
        """
        return name in self.templates

    def count_templates(self) -> int:
        """Count the number of registered templates.

        Returns:
            Number of registered FormationTemplate instances.
        """
        return len(self.templates)

    def validate_template(self, template: FormationTemplate) -> bool:
        """Validate a formation template without registering it.

        Args:
            template: FormationTemplate instance to validate.

        Returns:
            True if the template is valid, False otherwise.
        """
        return template.is_valid()


# Module-level default registry
default_registry = FormationTemplateRegistry()


def to_dict(template: FormationTemplate) -> dict[str, Any]:
    """Convert a FormationTemplate to a dictionary.

    Args:
        template: FormationTemplate instance to serialize.

    Returns:
        Dictionary representation of the template.
    """
    return {
        "formation_name": template.formation_name,
        "defenders": template.defenders,
        "midfielders": template.midfielders,
        "forwards": template.forwards,
        "normalized_positions": list(template.normalized_positions),
        "description": template.description,
    }


def from_dict(data: dict[str, Any]) -> FormationTemplate:
    """Create a FormationTemplate from a dictionary.

    Args:
        data: Dictionary containing template data.

    Returns:
        FormationTemplate instance initialized from the dictionary.
    """
    return FormationTemplate(
        formation_name=data["formation_name"],
        defenders=data["defenders"],
        midfielders=data["midfielders"],
        forwards=data["forwards"],
        normalized_positions=list(data.get("normalized_positions", [])),
        description=data.get("description", ""),
    )