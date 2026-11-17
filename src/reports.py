"""Downloadable JSON, CSV, and Markdown reports."""

import csv
import io
import json
from typing import Iterable

from src.scoring import MatchResult


def json_report(results: Iterable[MatchResult]) -> str:
    return json.dumps([result.to_dict() for result in results], indent=2)


def csv_report(results: Iterable[MatchResult]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "role",
            "overall",
            "semantic",
            "lexical",
            "skills",
            "matched_skills",
            "missing_required_skills",
            "missing_preferred_skills",
        ]
    )
    for result in results:
        writer.writerow(
            [
                result.role_title,
                result.overall_score,
                result.component_scores["semantic"],
                result.component_scores["lexical"],
                result.component_scores["skills"],
                "; ".join(result.matched_skills),
                "; ".join(result.missing_required_skills),
                "; ".join(result.missing_preferred_skills),
            ]
        )
    return output.getvalue()


def markdown_report(results: Iterable[MatchResult]) -> str:
    lines = [
        "# ZeeCruiter Match Report",
        "",
        "> Scores measure textual alignment and are not hiring probabilities.",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result.role_title} — {result.overall_score:.2f}/100",
                "",
                f"- Semantic: {result.component_scores['semantic'] if result.component_scores['semantic'] is not None else 'Unavailable'}",
                f"- Lexical: {result.component_scores['lexical'] if result.component_scores['lexical'] is not None else 'Unavailable'}",
                f"- Skill coverage: {result.component_scores['skills'] if result.component_scores['skills'] is not None else 'Unavailable'}",
                f"- Matched skills: {', '.join(result.matched_skills) or 'None detected'}",
                f"- Missing required skills: {', '.join(result.missing_required_skills) or 'None detected'}",
                f"- Missing preferred skills: {', '.join(result.missing_preferred_skills) or 'None detected'}",
                f"- Semantic status: {result.semantic_message}",
                "",
            ]
        )
        if result.matched_skills:
            lines.extend(["### Match evidence", ""])
            for skill in result.matched_skills:
                lines.extend(
                    [
                        f"- **{skill}**",
                        f"  - Résumé: {result.resume_evidence[skill][0]}",
                        f"  - Job: {result.job_evidence[skill][0]}",
                    ]
                )
            lines.append("")
    return "\n".join(lines)
