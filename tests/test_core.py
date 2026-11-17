import io
import json
import unittest

import fitz
import numpy as np

from config import AppConfig
from src.lexical import LexicalScorer
from src.ollama import OllamaClient
from src.parsers import DocumentParsingError, clean_text, parse_pdf, parse_text
from src.reports import csv_report, json_report, markdown_report
from src.scoring import HybridScorer
from src.semantic import SemanticScorer
from src.skills import SkillExtractor
from tests.conftest import FakeEmbeddingModel, UnavailableSemantic


def make_pdf(*pages: str) -> bytes:
    document = fitz.open()
    for content in pages:
        page = document.new_page()
        page.insert_text((50, 72), content)
    payload = document.tobytes()
    document.close()
    return payload


class ConfigTests(unittest.TestCase):
    def test_valid_config(self):
        self.assertEqual(AppConfig().weight_semantic, 0.5)

    def test_invalid_configs(self):
        cases = [
            {"weight_semantic": -0.1, "weight_lexical": 0.6, "weight_skills": 0.5},
            {"weight_semantic": 0.5, "weight_lexical": 0.5, "weight_skills": 0.5},
            {"semantic_chunk_tokens": 20, "semantic_chunk_overlap": 20},
            {"semantic_max_chunks": 0},
            {"max_pdf_pages": 0},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AppConfig(**kwargs)


class ParserTests(unittest.TestCase):
    def test_parse_pdf_and_restore_position(self):
        stream = io.BytesIO(make_pdf("Python engineer", "Docker experience"))
        stream.seek(3)
        result = parse_pdf(stream)
        self.assertEqual(stream.tell(), 3)
        self.assertEqual(result.page_count, 2)
        self.assertIn("Python engineer", result.text)

    def test_page_limit_is_disclosed(self):
        result = parse_pdf(make_pdf("page one", "page two"), AppConfig(max_pdf_pages=1))
        self.assertEqual(result.processed_pages, 1)
        self.assertTrue(result.warnings)
        self.assertNotIn("page two", result.text)

    def test_invalid_pdfs_rejected(self):
        for payload in (b"", b"not a pdf"):
            with self.subTest(payload=payload), self.assertRaises(DocumentParsingError):
                parse_pdf(payload)

    def test_size_limit(self):
        with self.assertRaises(DocumentParsingError):
            parse_pdf(make_pdf("small"), AppConfig(max_pdf_size_mb=0.00001))

    def test_text_parsing(self):
        self.assertEqual(parse_text(b"Python\r\n\r\n  Docker   AWS"), "Python\nDocker AWS")
        self.assertEqual(clean_text(" a\x00  b \n\n c "), "a b\nc")


class SkillTests(unittest.TestCase):
    def setUp(self):
        self.extractor = SkillExtractor()

    def test_symbol_languages_and_aliases(self):
        found = self.extractor.extract("Built systems in C, C++, cpp, C#, csharp, and .NET.")
        self.assertTrue({"C", "C++", "C#", ".NET"} <= found)

    def test_c_boundaries(self):
        self.assertNotIn("C", self.extractor.extract("cloud computing capabilities clean code"))
        self.assertEqual(self.extractor.extract("C++"), {"C++"})
        self.assertEqual(self.extractor.extract("C#"), {"C#"})

    def test_common_word_false_positives(self):
        found = self.extractor.extract("During spring break, I took a rest and reacted calmly.")
        self.assertFalse({"Spring Boot", "REST API", "React"} & found)

    def test_explicit_framework_phrases(self):
        found = self.extractor.extract("Spring Boot service exposing REST APIs with React.js")
        self.assertTrue({"Spring Boot", "REST API", "React"} <= found)

    def test_urls_are_not_skills_and_hyphenated_ml_matches(self):
        found = self.extractor.extract(
            "Portfolio: https://github.com/candidate. Built a machine-learning pipeline."
        )
        self.assertNotIn("GitHub", found)
        self.assertIn("Machine Learning", found)

    def test_match_evidence_contains_source_text(self):
        comparison = self.extractor.compare(
            "Built Python APIs for production.",
            "We require Python and Docker experience.",
        )
        self.assertIn("Built Python APIs", comparison.resume_evidence["Python"][0])
        self.assertIn("require Python", comparison.job_evidence["Python"][0])

    def test_required_and_preferred_skills_are_separated(self):
        comparison = self.extractor.compare(
            "Python developer",
            "Python is required. Docker and AWS are preferred qualifications.",
        )
        self.assertIn("Python", comparison.required_skills)
        self.assertEqual(comparison.preferred_skills, ("AWS", "Docker"))
        self.assertEqual(comparison.missing_preferred_skills, ("AWS", "Docker"))

    def test_empty_job_skills_are_unavailable(self):
        result = self.extractor.compare("Python Docker", "Lead teams and communicate clearly")
        self.assertIsNone(result.score)

    def test_skill_coverage(self):
        result = self.extractor.compare("Python and Docker", "Python, Docker, AWS")
        self.assertAlmostEqual(result.score, 2 / 3)
        self.assertEqual(result.missing_skills, ("AWS",))


class SemanticTests(unittest.TestCase):
    def test_long_documents_include_late_content(self):
        model = FakeEmbeddingModel()
        config = AppConfig(semantic_chunk_tokens=8, semantic_chunk_overlap=2, semantic_max_chunks=10)
        result = SemanticScorer(config, model).score(
            "one two three four five six seven eight nine ten eleven python", "python backend"
        )
        self.assertTrue(result.available)
        self.assertGreater(result.resume_chunks, 1)
        self.assertTrue(any("python" in chunk for chunk in model.seen_chunks))

    def test_max_chunks_and_normalization(self):
        model = FakeEmbeddingModel()
        config = AppConfig(semantic_chunk_tokens=8, semantic_chunk_overlap=2, semantic_max_chunks=2)
        embedding, count, truncated = SemanticScorer(config, model)._document_embedding(
            " ".join(str(index) for index in range(100)), model
        )
        self.assertEqual(count, 2)
        self.assertTrue(truncated)
        self.assertTrue(np.isclose(np.linalg.norm(embedding), 1.0))

    def test_chunk_overlap(self):
        model = FakeEmbeddingModel()
        config = AppConfig(semantic_chunk_tokens=8, semantic_chunk_overlap=2)
        chunks, truncated = SemanticScorer(config, model)._chunk("a b c d e f g h i j", model)
        self.assertEqual(chunks[0].split()[-2:], chunks[1].split()[:2])
        self.assertFalse(truncated)


class ScoringTests(unittest.TestCase):
    def test_lexical_stability(self):
        scorer = LexicalScorer()
        resume = "Python FastAPI PostgreSQL backend engineer"
        job = "Python backend engineer using PostgreSQL"
        first = scorer.score(resume, job)
        scorer.score(resume, "React TypeScript frontend")
        self.assertEqual(scorer.score(resume, job), first)

    def test_missing_components_renormalize(self):
        scorer = HybridScorer(semantic=UnavailableSemantic())
        no_skills = scorer.evaluate(
            "Python engineer", "Strong communicator and team leader", "Manager", "1"
        )
        self.assertEqual(
            no_skills.effective_weights,
            {"semantic": 0.0, "lexical": 1.0, "skills": 0.0},
        )
        with_skills = scorer.evaluate("Python Docker", "Python Docker AWS", "Backend", "2")
        self.assertAlmostEqual(with_skills.effective_weights["lexical"], 0.4)
        self.assertAlmostEqual(with_skills.effective_weights["skills"], 0.6)

    def test_hybrid_bounds_and_ranking(self):
        scorer = HybridScorer(semantic=SemanticScorer(model=FakeEmbeddingModel()))
        results = scorer.evaluate_many(
            "Python backend engineer Docker AWS",
            [
                {"id": "front", "title": "Frontend", "text": "React TypeScript CSS"},
                {"id": "back", "title": "Backend", "text": "Python backend Docker AWS"},
            ],
        )
        self.assertEqual(results[0].role_id, "back")
        self.assertTrue(all(0 <= result.overall_score <= 100 for result in results))


class OllamaAndReportTests(unittest.TestCase):
    def test_exact_model_matching(self):
        self.assertTrue(OllamaClient._model_matches("mistral", "mistral:latest"))
        self.assertFalse(OllamaClient._model_matches("mistral", "custom-mistral:latest"))
        self.assertFalse(OllamaClient._model_matches("mistral:7b", "mistral:latest"))

    def test_reports(self):
        result = HybridScorer(semantic=UnavailableSemantic()).evaluate(
            "Python Docker", "Python Docker AWS", "Backend", "backend"
        )
        self.assertEqual(json.loads(json_report([result]))[0]["role_title"], "Backend")
        self.assertIn("Backend", csv_report([result]))
        self.assertIn("not hiring probabilities", markdown_report([result]))


if __name__ == "__main__":
    unittest.main()
