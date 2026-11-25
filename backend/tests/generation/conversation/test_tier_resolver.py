# tests/generation/conversation/test_tier_resolver.py

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.generation.conversation.models import ProjectMetadata
from aie_estimator.generation.conversation.tier_resolver import Tier, resolve_tier


@pytest.mark.parametrize(
    ("caller_tier", "override", "transcript", "metadata", "expected_tier", "expected_rule"),
    [
        (
            Tier.EXECUTIVE,
            Tier.DEVELOPER,
            "Build a REST API using kubernetes and kafka.",
            ProjectMetadata(),
            Tier.EXECUTIVE,
            "caller_context",
        ),
        (
            None,
            Tier.DEVELOPER,
            "Confidential NDA project under GDPR.",
            ProjectMetadata(),
            Tier.DEVELOPER,
            "explicit_override",
        ),
        (
            None,
            None,
            "This project is under strict NDA and embargo.",
            ProjectMetadata(),
            Tier.EXECUTIVE,
            "nda_detected",
        ),
        (
            None,
            None,
            "Build an internal portal.",
            ProjectMetadata(agreed_scope="Covered by non-disclosure agreement."),
            Tier.EXECUTIVE,
            "nda_detected",
        ),
        (
            None,
            None,
            "Must comply with HIPAA and GDPR audit rules.",
            ProjectMetadata(),
            Tier.EXECUTIVE,
            "regulatory_context",
        ),
        (
            None,
            None,
            "Build a reporting dashboard.",
            ProjectMetadata(mentioned_technologies=["SOC2", "PostgreSQL"]),
            Tier.EXECUTIVE,
            "regulatory_context",
        ),
        (
            None,
            None,
            "Deploy microservices on Kubernetes with Kafka and Terraform.",
            ProjectMetadata(),
            Tier.DEVELOPER,
            "technical_audience",
        ),
        (
            None,
            None,
            "We run on Docker and need a web portal.",
            ProjectMetadata(assumed_team_size=2),
            Tier.PM,
            "small_team_pm",
        ),
        (
            None,
            None,
            "Build a customer onboarding portal with Stripe billing.",
            ProjectMetadata(),
            Tier.PM,
            "default",
        ),
    ],
)
def test_resolve_tier_precedence_and_rules(
    caller_tier: Tier | None,
    override: Tier | None,
    transcript: str,
    metadata: ProjectMetadata,
    expected_tier: Tier,
    expected_rule: str,
) -> None:
    """
    GIVEN various combinations of caller_tier, override, transcript, and metadata
    WHEN resolve_tier is invoked
    THEN it returns the highest-precedence matching Tier and rule_name.
    """
    tier, rule = resolve_tier(
        transcript=transcript,
        metadata=metadata,
        caller_tier=caller_tier,
        override=override,
    )

    assert tier is expected_tier
    assert rule == expected_rule

