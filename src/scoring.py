"""Transparent hybrid resume-to-job scoring."""

from dataclasses import asdict, dataclass
from typing import Iterable

from config import CONFIG, AppConfig
from src.lexical import LexicalScorer
from src.semantic import SemanticScorer
from src.skills import SkillExtractor


@dataclass(frozen=True)
class MatchResult:
    role_id: str
    role_title: str
    overall_score: float
    component_scores: dict[str, float | None]
    configured_weights: dict[str, float]
    effective_weights: dict[str, float]
    matched_skills: tuple[str, ...]
    missing_skills: tuple[str, ...]
    resume_skills: tuple[str, ...]
    job_skills: tuple[str, ...]
    required_skills: tuple[str, ...]
    preferred_skills: tuple[str, ...]
    missing_required_skills: tuple[str, ...]
    missing_preferred_skills: tuple[str, ...]
    resume_evidence: dict[str, tuple[str, ...]]
    job_evidence: dict[str, tuple[str, ...]]
    semantic_message: str
    semantic_chunks: dict[str, int]
    semantic_truncated: dict[str, bool]

    def to_dict(self) -> dict:
        return asdict(self)


class HybridScorer:
    """Weighted score whose unavailable components are explicitly renormalized.

    Scores measure textual alignment only. They are not hiring probabilities and
    the default weights are documented design choices, not learned parameters.
    """

    def __init__(
        self,
        config: AppConfig = CONFIG,
        semantic: SemanticScorer | None = None,
        lexical: LexicalScorer | None = None,
        skills: SkillExtractor | None = None,
    ):
        self.config = config
        self.semantic = semantic or SemanticScorer(config)
        self.lexical = lexical or LexicalScorer()
        self.skills = skills or SkillExtractor()
        self.configured_weights = {
            "semantic": config.weight_semantic,
            "lexical": config.weight_lexical,
            "skills": config.weight_skills,
        }

    @staticmethod
    def _effective_weights(
        scores: dict[str, float | None], configured: dict[str, float]
    ) -> dict[str, float]:
        active_total = sum(configured[name] for name, score in scores.items() if score is not None)
        if active_total <= 0:
            raise ValueError("No scoring component is available")
        return {
            name: (configured[name] / active_total if score is not None else 0.0)
            for name, score in scores.items()
        }

    def evaluate(self, resume_text: str, job_text: str, role_title: str, role_id: str) -> MatchResult:
        if not resume_text.strip() or not job_text.strip():
            raise ValueError("Resume and job-description text must be non-empty")

        semantic = self.semantic.score(resume_text, job_text)
        lexical = self.lexical.score(resume_text, job_text)
        skill_match = self.skills.compare(resume_text, job_text)
        scores = {
            "semantic": semantic.score,
            "lexical": lexical,
            "skills": skill_match.score,
        }
        effective = self._effective_weights(scores, self.configured_weights)
        overall = sum(
            effective[name] * score for name, score in scores.items() if score is not None
        )
        return MatchResult(
            role_id=role_id,
            role_title=role_title,
            overall_score=round(overall * 100, 2),
            component_scores={
                name: (round(score * 100, 2) if score is not None else None)
                for name, score in scores.items()
            },
            configured_weights=self.configured_weights.copy(),
            effective_weights={name: round(weight, 6) for name, weight in effective.items()},
            matched_skills=skill_match.matched_skills,
            missing_skills=skill_match.missing_skills,
            resume_skills=skill_match.resume_skills,
            job_skills=skill_match.job_skills,
            required_skills=skill_match.required_skills,
            preferred_skills=skill_match.preferred_skills,
            missing_required_skills=skill_match.missing_required_skills,
            missing_preferred_skills=skill_match.missing_preferred_skills,
            resume_evidence=skill_match.resume_evidence,
            job_evidence=skill_match.job_evidence,
            semantic_message=semantic.message,
            semantic_chunks={"resume": semantic.resume_chunks, "job": semantic.job_chunks},
            semantic_truncated={
                "resume": semantic.resume_truncated,
                "job": semantic.job_truncated,
            },
        )

    def evaluate_many(self, resume_text: str, jobs: Iterable[dict[str, str]]) -> list[MatchResult]:
        results = [
            self.evaluate(
                resume_text,
                job["text"],
                job.get("title", f"Role {index + 1}"),
                job.get("id", str(index + 1)),
            )
            for index, job in enumerate(jobs)
        ]
        return sorted(results, key=lambda result: result.overall_score, reverse=True)
