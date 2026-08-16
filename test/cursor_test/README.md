# cursor_test

Cursor-driven pytest and probe scripts for **statassist-py**. Work is organised **one folder per session date** (`YYYY_MM_DD`).

## Layout

```
test/cursor_test/
├── README.md              # this file
├── conftest.py            # shared fixtures (all date folders)
├── fixtures/              # R golden JSON export (shared)
└── YYYY_MM_DD/            # one folder per working day
    └── test_*.py          # tests and probes for that session
```

## Running tests

From `statassist-py/` (pytest collects every `YYYY_MM_DD/` subfolder):

```bash
cd statassist-py
py -m pytest -v
```

Run only today's folder:

```bash
py -m pytest -v test/cursor_test/2026_08_17
```

## Convention

- **New session** → create `test/cursor_test/YYYY_MM_DD/` and add tests or probes there.
- **Shared** `conftest.py` and `fixtures/` stay at the `cursor_test/` root.
- Do not delete older date folders; they record what was verified on that day.

Current session: **`2026_08_17/`** (README transcription, render pipeline, MVP workflow tests).
