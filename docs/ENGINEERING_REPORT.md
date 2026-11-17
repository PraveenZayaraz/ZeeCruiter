# Engineering report

## Scope

This version is a separate rebuild. The uploaded source archive was not modified.

## Material changes

| Original behavior | Rebuilt behavior |
|---|---|
| MiniLM encoded the entire document once, silently truncating long input | Token-aware overlapping chunks are embedded in batches, pooled, normalized, capped, and reported |
| Substring matching confused short technologies and ordinary words | 62 canonical skills use 89 explicit, boundary-guarded aliases |
| Resume completeness affected job relevance | Every score component compares the resume with the target job |
| Arbitrary +5, +8, and +10 boosts changed the final score | No boosts; the weighted equation and effective weights are returned |
| LLM sentiment changed the deterministic result | Ollama feedback is optional and cannot modify the score |
| Missing components silently became zero | Unavailable components are identified and remaining weights renormalize |
| Candidate uploads were saved locally | Uploaded documents remain in memory |
| FAISS was initialized but unused | Unused vector-store complexity was removed |
| One job was evaluated at a time | Up to five roles can be ranked and compared together |
| PDF report replaced unsupported characters | JSON, CSV, and Markdown preserve Unicode text |
| Broad exceptions hid errors | User-facing failures disclose the failure class without exposing document contents |

## Score definition

With all components available:

```text
alignment = 0.50 * semantic + 0.20 * lexical + 0.30 * skill coverage
```

These are configurable design weights. If MiniLM is unavailable, semantic weight is
removed and lexical/skills become 0.40/0.60. If the job contains no recognized skills,
the skills component is unavailable rather than zero. All configured and effective
weights are included in the report.

## Verification performed in the build environment

- `python -m compileall -q .`
- `python -m unittest discover -s tests -v`
- `python evaluation/run_benchmark.py`

Verified result: 24/24 offline tests passed. The curated functional benchmark completed
128 comparisons across 16 profiles and eight roles. Its latest measured latency and
ranking output are stored in `evaluation/results.json` because timing varies by machine.

## What remains for local verification

This build environment could not access the Python package index, so Streamlit,
sentence-transformers, Plotly, and Ollama were not installed here. The core used by the
test suite ran against the installed Python 3.12, NumPy, scikit-learn, and PyMuPDF.
Run the Windows commands in `README.md` to verify the complete UI and live MiniLM/Ollama
integrations on the target computer.
