# evals/judge.py
"""
Family 3 selective LLM-as-Judge evaluation for the estimation harness.

Some quality properties cannot be checked with schema assertions, numeric ranges, or run-to-run
variance: whether the estimate is logically consistent with the project description, whether its
deliverables and risks are concrete enough to act on, and whether its depth and framing suit the
stakeholder tier. This module covers those properties with a secondary LLM call that scores the
primary output against a rubric. It is the most expensive layer of the evaluation pyramid and
complements the deterministic and statistical metrics in ``evals.metrics``. It does not replace them.

Rubric
------
The judge applies a pointwise rubric and returns a ``JudgeVerdict`` with three 1-5 Likert ratings
and a short ``reasoning`` string:

- ``coherence``: logical consistency between the input description and the estimate.
- ``actionability``: concreteness and usefulness of deliverables, risks, or guidance.
- ``tier_appropriateness``: alignment of depth and framing with the resolved stakeholder tier.

The three ratings are normalized to ``[0.0, 1.0]`` as ``(rating - 1) / 4``, then averaged into a
single score. ``evaluate_with_judge`` compares that score with ``DEFAULT_JUDGE_PASS_THRESHOLD``.

Execution flow
--------------
1. ``render_judge_prompt`` renders the ``judge.j2`` template under ``StrictUndefined``, so a missing
   variable fails loudly instead of producing a silently degraded prompt. The prompt includes the
   resolved tier, request parameters, the original description, and the serialized estimation result.
2. ``evaluate_with_judge`` calls the provider dispatcher with ``JudgeVerdict`` as the structured
   response model. It uses ``reasoning_effort="low"`` and a bounded token budget to keep verdicts as
   reproducible as possible. ``judge_model`` can override the model or profile.
3. The verdict is converted into the same immutable ``MetricResult`` used by Family 1 and Family 2,
   with the ratings and reasoning in ``details``.

Failure handling
----------------
Judge failures (provider errors, schema violations) never abort an evaluation run. They are logged
and reported as a failed ``MetricResult`` with ``score=0.0`` and the error in ``details``, so they
stay visible in the report.

Caveats
-------
The judge is itself an LLM and can show systematic bias, such as favoring verbose outputs. The pass
threshold is a starting point and should be calibrated against human-scored samples. Because every
call adds cost and latency, run this family selectively, only for properties that deterministic
checks cannot validate. The live runner (``evals.run``) enables it with the ``--llm-judge`` flag.
"""

# --- Python Standard Library ---
from pathlib import Path
from typing import Final

# --- Third Party Libraries ---
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from pydantic import BaseModel, Field
import structlog

# --- App modules ---
from aie_estimator.schemas.estimations import EstimationResponse
from aie_estimator.services.providers.dispatcher import call_llm
from aie_estimator.services.providers.generation_options import GenerationOptions
from evals.dataset import GoldenCase
from evals.metrics import MetricResult

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

DEFAULT_JUDGE_PASS_THRESHOLD: Final[float] = 0.7
_JUDGE_TEMPLATE_DIR: Final[Path] = Path(__file__).resolve().parent
_JUDGE_TEMPLATE_NAME: Final[str] = "judge.j2"
_JUDGE_SYSTEM_PROMPT: Final[str] = (
    "You are a strict software engineering estimation auditor. "
    "Score the provided estimation objectively using the 1-5 rubric and return structured JSON."
)
_JUDGE_MAX_TOKENS: Final[int] = 800
_LIKERT_MIN: Final[int] = 1
_LIKERT_SPAN: Final[int] = 4

_JINJA_ENV: Final[Environment] = Environment(
    loader=FileSystemLoader(str(_JUDGE_TEMPLATE_DIR)),
    undefined=StrictUndefined,
    autoescape=select_autoescape(["html", "xml"]),
)


class JudgeVerdict(BaseModel):
    """Structured LLM-as-Judge rubric verdict using 1-5 Likert ratings."""

    coherence: int = Field(ge=1, le=5, description="Logical consistency between input description and estimate (1-5).")
    actionability: int = Field(ge=1, le=5, description="Concreteness and usefulness of deliverables, risks, or guidance (1-5).")
    tier_appropriateness: int = Field(ge=1, le=5, description="Alignment of depth and framing with the resolved stakeholder tier (1-5).")
    reasoning: str = Field(min_length=10, description="Concise justification supporting the assigned ratings.")

    @property
    def normalized_score(self) -> float:
        """
        Normalize the three 1-5 Likert scores into a single ``[0.0, 1.0]`` score.

        :return: Mean normalized score rounded to 4 decimal places.
        """
        scores = (self.coherence, self.actionability, self.tier_appropriateness)
        normalized = [(rating - _LIKERT_MIN) / _LIKERT_SPAN for rating in scores]
        return round(sum(normalized) / len(normalized), 4)


def render_judge_prompt(case: GoldenCase, response: EstimationResponse) -> str:
    """
    Render the ``judge.j2`` prompt template under ``StrictUndefined``.

    :param case: Golden Dataset scenario.
    :param response: API estimation response under evaluation.

    :return: Rendered user prompt string for the judge LLM.
    """
    template = _JINJA_ENV.get_template(_JUDGE_TEMPLATE_NAME)
    return template.render(
        resolved_tier=response.tier.value,
        project_type=case.project_type.value,
        detail_level=case.detail_level.value,
        expected_out_of_scope=case.expected_out_of_scope,
        description=case.description,
        result_json=response.result.model_dump_json(indent=2),
    )


async def evaluate_with_judge(
    case: GoldenCase,
    response: EstimationResponse,
    *,
    judge_model: str | None = None,
    pass_threshold: float = DEFAULT_JUDGE_PASS_THRESHOLD,
) -> MetricResult:
    """
    Run Family 3 subjective quality evaluation via ``call_llm(..., response_model=JudgeVerdict)``.

    :param case: Golden Dataset scenario.
    :param response: API estimation response to evaluate.
    :param judge_model: Optional model/profile override for the judge LLM.
    :param pass_threshold: Minimum normalized score in ``[0.0, 1.0]`` required to pass.

    :return: ``MetricResult`` for ``llm_judge`` containing normalized score and reasoning.
    """
    user_prompt = render_judge_prompt(case, response)
    opts = GenerationOptions(
        model=judge_model,
        max_tokens=_JUDGE_MAX_TOKENS,
        reasoning_effort="low",
    )

    try:
        raw = await call_llm(
            user_prompt=user_prompt,
            system_prompt=_JUDGE_SYSTEM_PROMPT,
            opts=opts,
            response_model=JudgeVerdict,
        )
        verdict: JudgeVerdict = raw["result"]
    except Exception as exc:
        log.error("eval_judge_failed", case_id=case.id, error=str(exc))
        return MetricResult(
            name="llm_judge",
            score=0.0,
            passed=False,
            details=f"judge error: {exc}",
        )

    score = verdict.normalized_score
    passed = score >= pass_threshold
    log.info(
        "eval_judge_completed",
        case_id=case.id,
        tier=response.tier.value,
        score=score,
        passed=passed,
        coherence=verdict.coherence,
        actionability=verdict.actionability,
        tier_appropriateness=verdict.tier_appropriateness,
    )
    return MetricResult(
        name="llm_judge",
        score=score,
        passed=passed,
        details=(
            f"coherence={verdict.coherence}/5, "
            f"actionability={verdict.actionability}/5, "
            f"tier={verdict.tier_appropriateness}/5 | {verdict.reasoning}"
        ),
    )
