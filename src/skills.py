"""Deterministic technical-skill extraction with guarded aliases."""

from dataclasses import dataclass
import re
from typing import Mapping, Sequence


SKILL_ALIASES: Mapping[str, Sequence[str]] = {
    "Python": (r"python(?:3)?",),
    "Java": (r"java",),
    "C": (r"c",),
    "C++": (r"c\+\+", r"cpp"),
    "C#": (r"c#", r"csharp"),
    ".NET": (r"\.net", r"dotnet"),
    "JavaScript": (r"javascript", r"java script", r"js"),
    "TypeScript": (r"typescript", r"type script"),
    "React": (r"react(?:\.js)?", r"reactjs"),
    "Angular": (r"angular(?:\.js)?", r"angularjs"),
    "Node.js": (r"node(?:\.js)?", r"nodejs"),
    "Express": (r"express(?:\.js)?", r"expressjs"),
    "Flask": (r"flask",),
    "Django": (r"django",),
    "FastAPI": (r"fastapi", r"fast api"),
    "Spring Boot": (r"spring boot", r"spring framework"),
    "HTML": (r"html(?:5)?",),
    "CSS": (r"css(?:3)?",),
    "Tailwind CSS": (r"tailwind(?: css)?",),
    "SQL": (r"sql",),
    "PostgreSQL": (r"postgresql", r"postgres"),
    "MySQL": (r"mysql",),
    "MongoDB": (r"mongodb",),
    "Redis": (r"redis",),
    "Firebase": (r"firebase",),
    "REST API": (r"rest(?:ful)? apis?", r"restful web services?"),
    "GraphQL": (r"graphql",),
    "Git": (r"git",),
    "GitHub": (r"github",),
    "Docker": (r"docker",),
    "Kubernetes": (r"kubernetes", r"k8s"),
    "Linux": (r"linux",),
    "Bash": (r"bash", r"shell scripting"),
    "AWS": (r"aws", r"amazon web services"),
    "Azure": (r"azure",),
    "Google Cloud": (r"gcp", r"google cloud(?: platform)?"),
    "CI/CD": (r"ci\s*[/\-]\s*cd", r"continuous integration"),
    "Jenkins": (r"jenkins",),
    "Kafka": (r"apache kafka", r"kafka"),
    "Airflow": (r"apache airflow", r"airflow"),
    "Machine Learning": (r"machine[ -]+learning",),
    "Deep Learning": (r"deep[ -]+learning",),
    "NLP": (r"natural language processing", r"nlp"),
    "LLM": (r"large language models?", r"llms?"),
    "PyTorch": (r"pytorch",),
    "TensorFlow": (r"tensorflow",),
    "scikit-learn": (r"scikit[ -]learn", r"sklearn"),
    "Pandas": (r"pandas",),
    "NumPy": (r"numpy",),
    "FAISS": (r"faiss",),
    "Plotly": (r"plotly",),
    "Streamlit": (r"streamlit",),
    "Power BI": (r"power\s*bi", r"powerbi"),
    "Tableau": (r"tableau",),
    "Excel": (r"microsoft excel", r"excel"),
    "Selenium": (r"selenium",),
    "Agile": (r"agile",),
    "Scrum": (r"scrum",),
    "Data Structures": (r"data structures?",),
    "Algorithms": (r"algorithms?",),
    "Swift": (r"swift",),
    "SwiftUI": (r"swiftui", r"swift ui"),
    "Kotlin": (r"kotlin",),
    "Flutter": (r"flutter",),
}


def _guard(alias: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![A-Za-z0-9_+#])(?:{alias})(?![A-Za-z0-9_+#])",
        re.IGNORECASE,
    )


@dataclass(frozen=True)
class SkillMatch:
    resume_skills: tuple[str, ...]
    job_skills: tuple[str, ...]
    matched_skills: tuple[str, ...]
    missing_skills: tuple[str, ...]
    score: float | None
    required_skills: tuple[str, ...]
    preferred_skills: tuple[str, ...]
    missing_required_skills: tuple[str, ...]
    missing_preferred_skills: tuple[str, ...]
    resume_evidence: dict[str, tuple[str, ...]]
    job_evidence: dict[str, tuple[str, ...]]


class SkillExtractor:
    def __init__(self, aliases: Mapping[str, Sequence[str]] = SKILL_ALIASES):
        self.patterns = {
            skill: tuple(_guard(alias) for alias in skill_aliases)
            for skill, skill_aliases in aliases.items()
        }

    @staticmethod
    def _without_contacts(text: str) -> str:
        return re.sub(r"https?://\S+|www\.\S+|\b\S+@\S+\b", " ", text)

    @staticmethod
    def _snippet(text: str, start: int, end: int, limit: int = 240) -> str:
        left = max(text.rfind(mark, 0, start) for mark in ("\n", ".", "!", "?")) + 1
        endings = [
            position
            for mark in ("\n", ".", "!", "?")
            if (position := text.find(mark, end)) >= 0
        ]
        right = min(endings) + 1 if endings else len(text)
        snippet = " ".join(text[left:right].split())
        if len(snippet) <= limit:
            return snippet
        match_offset = max(0, start - left)
        window_start = max(0, match_offset - limit // 2)
        return snippet[window_start : window_start + limit].strip()

    def evidence(self, text: str, max_snippets: int = 2) -> dict[str, tuple[str, ...]]:
        """Return short source excerpts for every detected canonical skill."""
        if not text:
            return {}
        searchable = self._without_contacts(text)
        output: dict[str, tuple[str, ...]] = {}
        for skill, patterns in self.patterns.items():
            snippets: list[str] = []
            for pattern in patterns:
                for match in pattern.finditer(searchable):
                    snippet = self._snippet(searchable, match.start(), match.end())
                    if snippet and snippet not in snippets:
                        snippets.append(snippet)
                    if len(snippets) >= max_snippets:
                        break
                if len(snippets) >= max_snippets:
                    break
            if snippets:
                output[skill] = tuple(snippets)
        return output

    def extract(self, text: str) -> set[str]:
        if not text:
            return set()
        return set(self.evidence(text))

    @staticmethod
    def _classify_job_skills(
        job_evidence: dict[str, tuple[str, ...]]
    ) -> tuple[set[str], set[str]]:
        markers = (
            "preferred",
            "nice to have",
            "nice-to-have",
            "bonus",
            "a plus",
            "valuable",
            "desired",
        )
        preferred = {
            skill
            for skill, snippets in job_evidence.items()
            if snippets and all(any(marker in snippet.lower() for marker in markers) for snippet in snippets)
        }
        return set(job_evidence) - preferred, preferred

    def compare(self, resume_text: str, job_text: str) -> SkillMatch:
        resume_evidence = self.evidence(resume_text)
        job_evidence = self.evidence(job_text)
        resume = set(resume_evidence)
        job = set(job_evidence)
        required, preferred = self._classify_job_skills(job_evidence)
        matched = resume & job
        missing = job - resume
        score = len(matched) / len(job) if job else None
        return SkillMatch(
            tuple(sorted(resume)),
            tuple(sorted(job)),
            tuple(sorted(matched)),
            tuple(sorted(missing)),
            score,
            tuple(sorted(required)),
            tuple(sorted(preferred)),
            tuple(sorted(required - resume)),
            tuple(sorted(preferred - resume)),
            resume_evidence,
            job_evidence,
        )
