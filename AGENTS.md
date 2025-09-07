# Repository Guidelines

## Project Structure & Module Organization
- `zonos/`: Core TTS library (models, sampling, conditioning, utils). Backends in `zonos/backbone/`.
- `gradio_interface.py`: Launches the local UI.
- `generate_multi_speaker_audio_new.py`: Batch/scripted generation utilities.
- Assets and data: `assets/`, `presets/`, `saved_dialogues/`, `saved_speakers.json`.
- Tooling: `pyproject.toml` (deps, Ruff), `requirements-uv.txt` (compiled lock), `Dockerfile`, `docker-compose.yml`.
- Windows helpers: `1、install-uv-qinglong.ps1`, `2、run_gradio.ps1`.

## Build, Test, and Development Commands
- Install deps (uv): `uv sync`
- Run UI: `uv run python gradio_interface.py`
- Batch TTS: `uv run python generate_multi_speaker_audio_new.py`
- Lint: `uv run ruff check .`
- Docker: `docker compose up --build`

## Coding Style & Naming Conventions
- Python 3.10+, 4‑space indent, max line length 120 (Ruff).
- Names: `snake_case` for functions/vars, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
- Use type hints on public APIs; docstrings for parameters, shapes, and sample rates.
- Keep functions focused; avoid hard‑coding paths—prefer presets in `presets/`.

## Testing Guidelines
- No formal tests yet. Add `pytest` tests under `tests/` (e.g., `tests/test_sampling.py`).
- Keep tests deterministic; place tiny audio fixtures in `tests/data/` (avoid large binaries).
- Run: `uv run pytest -q` (add `pytest` to dev deps).

## Commit & Pull Request Guidelines
- Commits: short, imperative subject, optional scope; reference issues (e.g., `fix: cache speakers (#27)`).
- PRs: description, linked issues, repro steps, screenshots/UI diffs; attach short audio samples when relevant.
- Note performance/memory impact; update docs (`README.md`, `설계 안내.md`) when behavior changes.

## Security & Configuration Tips
- Do not commit model weights, generated audio, or secrets. Use caches and `.gitignore`.
- Validate user inputs in Gradio; guard file paths and sampling rates.
- Prefer environment/config presets over code constants.

