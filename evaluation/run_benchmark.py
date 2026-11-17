"""Run the disclosed, developer-curated role-ranking benchmark."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scoring import HybridScorer
from src.semantic import SemanticResult


class DisabledSemantic:
    """Keep this benchmark reproducible without downloads or model caches."""

    def score(self, resume_text: str, job_text: str) -> SemanticResult:
        return SemanticResult(
            None,
            False,
            "Semantic scoring intentionally disabled for the offline benchmark",
        )


@dataclass(frozen=True)
class BenchmarkMetrics:
    benchmark_type: str
    profile_count: int
    role_count: int
    comparisons: int
    top_1_accuracy: float
    top_3_recall: float
    mean_reciprocal_rank: float
    median_profile_latency_ms: float
    p95_profile_latency_ms: float


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def run() -> tuple[BenchmarkMetrics, list[dict]]:
    payload = json.loads((ROOT / "evaluation" / "benchmark.json").read_text(encoding="utf-8"))
    scorer = HybridScorer(semantic=DisabledSemantic())
    details = []
    ranks = []
    latencies = []

    for profile in payload["profiles"]:
        started = time.perf_counter()
        ranked = scorer.evaluate_many(profile["text"], payload["roles"])
        latencies.append((time.perf_counter() - started) * 1000)
        ids = [result.role_id for result in ranked]
        rank = ids.index(profile["expected_role"]) + 1
        ranks.append(rank)
        details.append(
            {
                "profile_id": profile["id"],
                "expected_role": profile["expected_role"],
                "predicted_role": ids[0],
                "expected_role_rank": rank,
                "ranked_roles": ids,
            }
        )

    metrics = BenchmarkMetrics(
        benchmark_type="developer-curated deterministic functional benchmark; MiniLM disabled",
        profile_count=len(payload["profiles"]),
        role_count=len(payload["roles"]),
        comparisons=len(payload["profiles"]) * len(payload["roles"]),
        top_1_accuracy=round(sum(rank == 1 for rank in ranks) / len(ranks), 4),
        top_3_recall=round(sum(rank <= 3 for rank in ranks) / len(ranks), 4),
        mean_reciprocal_rank=round(statistics.mean(1 / rank for rank in ranks), 4),
        median_profile_latency_ms=round(statistics.median(latencies), 3),
        p95_profile_latency_ms=round(percentile(latencies, 0.95), 3),
    )
    return metrics, details


if __name__ == "__main__":
    metrics, details = run()
    output = {
        "limitations": [
            "The examples and labels were created by the developer, not recruiters.",
            "Profiles are concise synthetic summaries and are easier than real resumes.",
            "MiniLM is disabled so the result evaluates deterministic lexical and skill components only.",
            "These metrics must not be described as hiring accuracy or real-world model performance.",
        ],
        "metrics": asdict(metrics),
        "details": details,
    }
    destination = ROOT / "evaluation" / "results.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["metrics"], indent=2))
    print(f"Saved {destination}")

