# Contributing Guide

Thanks for your interest in improving this project.

## Development Setup

1. Fork and clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .[dev]
```

## Local Checks

```bash
ruff check src tests
pytest -q
```

## Pull Requests

- Keep PRs focused and small.
- Add tests for behavior changes.
- Update docs when user-facing behavior changes.
