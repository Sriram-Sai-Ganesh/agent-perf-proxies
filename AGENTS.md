
rking notes

Use plain words. Strive for standalone understanding: every name, comment, message, and document should make sense on its own, with as little surrounding context as possible. Do not invent labels for things; describe what they are. Do not use jargon that only makes sense inside this project or inside one tool.

Do exactly what was asked. Add only what was asked for, and only once it has been decided. Do not add code for situations that do not exist yet. Do not add small unrequested improvements alongside a requested change.

Ask when a request is ambiguous. If an instruction can reasonably be read more than one way, ask which was meant before acting.

## Git conventions

- Work on a branch named `feat/<topic>` in lowercase, pushed to origin and merged through the GitHub interface.
- Write commit messages in lowercase: an informative one-line summary, then a plain body saying what changed and why.
- No co-author trailers.
- Commit only the files of the requested change; leave unrelated working-tree edits out.
- Before committing, run `uv run ruff check .`, `uv run ruff format --check .` and `uv run python -m pytest -q`, which is what CI and the pre-commit hooks run.
- Do not rewrite commits that a run's provenance already records.

