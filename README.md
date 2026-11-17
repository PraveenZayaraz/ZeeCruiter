# ZeeCruiter

ZeeCruiter is an explainable resume-to-role matching application. It compares one
resume with up to five job descriptions, exposes each scoring component, identifies
matched and missing technical skills, and exports the results. Optional feedback runs
through a local Ollama model and never changes the deterministic score.

## Why this rebuild matters

The original student prototype demonstrated an end-to-end idea but mixed semantic
similarity, resume completeness, arbitrary score boosts, and LLM sentiment. That made
the final number difficult to defend. This rebuild makes the system smaller and more
rigorous:

- PDF and UTF-8 text parsing is bounded, in-memory, and explicit about truncation.
- 64 canonical technologies are recognized through 91 guarded aliases.
- MiniLM processes long documents in overlapping token-aware chunks instead of
  silently using only the beginning of a document.
- Lexical similarity uses a fixed hashing feature space, so the same resume/job pair
  retains the same lexical score when other roles are added.
- Skill coverage measures requirements found in the job description, with strict
  boundaries for ambiguous names such as C, C++, C#, React, and REST API.
- Every matched skill includes the exact résumé and job-description evidence, while
  missing skills are separated into required and preferred groups.
- Missing components are marked unavailable and remaining weights are renormalized.
- Ollama/Mistral feedback is optional, local, bounded by timeouts, and separated from
  scoring.
- JSON, CSV, and Markdown reports disclose scores, weights, skills, and limitations.

The score measures textual alignment. It is not a hiring probability, ATS score, or
prediction of recruiter decisions.

## Architecture

```text
PDF/TXT input
    -> defensive parser
    -> semantic + lexical + skill components
    -> transparent weight normalization
    -> ranked multi-role results
    -> charts, optional Ollama feedback, downloadable reports
```

No candidate document is written to disk by the application.

## Windows setup

Use Python 3.11 or 3.12. From PowerShell:

```powershell
cd path\to\zecruiter_rebuild
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
streamlit run app.py
```

The first semantic analysis downloads the configured MiniLM model. After it is cached,
the scoring path can run locally. If the package or model is unavailable, ZeeCruiter
continues with lexical and skill scoring and shows the effective weights.

## Optional Ollama feedback

Install Ollama separately, then run:

```powershell
ollama pull mistral
ollama serve
```

Enable the Ollama checkbox in the sidebar. The deterministic score is identical whether
Ollama is enabled or disabled.

Configuration can be supplied through environment variables documented in
`.env.example`. The application does not automatically load `.env`; set variables in
PowerShell or through your deployment environment.

## Verification and running tests

```powershell
python -m compileall -q .
python -m unittest discover -s tests -v
python evaluation\run_benchmark.py
```

The ordinary test suite uses a deterministic fake embedding model and performs no
network requests. It covers parsing limits, stream behavior, skill boundaries,
false-positive prevention, chunk overlap, long-document handling, embedding
normalization, weight renormalization, ranking, Ollama model matching, and report
generation.

## Evaluation

`evaluation/benchmark.json` contains 16 disclosed synthetic profiles and eight role
descriptions. `evaluation/run_benchmark.py` performs 128 comparisons with semantic
scoring intentionally disabled, making it reproducible without model downloads.

The committed `evaluation/results.json` records the most recent run and its limitations.
This benchmark is a functional engineering check. Because its examples and labels were
created by the developer, its perfect top-1 result must not be presented as real-world
model accuracy or hiring performance.

## Portfolio description

**Resume-ready bullets**

- Served as primary developer for ZeeCruiter, building an explainable resume-to-role
  matcher with chunked MiniLM embeddings, stable lexical similarity, and deterministic
  coverage of 64 technical skills through 91 guarded aliases.
- Delivered multi-role comparison, optional local Mistral feedback, interactive Plotly
  visualizations, and downloadable JSON/CSV/Markdown reports; validated core behavior
  with 24 automated tests and 128 documented benchmark comparisons.

Only use the second bullet after you have personally run the included commands and can
explain what the tests and curated benchmark do.

## Known limitations

- Default hybrid weights are design choices and have not been learned from recruiter
  or hiring-outcome labels.
- Regex-based skills cannot infer proficiency, duration, recency, or context.
- PDF extraction does not perform OCR.
- The bundled benchmark is synthetic and intentionally small.
- Optional LLM feedback can still be influenced by adversarial document text and must
  be treated as qualitative advice.
