# Contributing to Raymember

Thank you for your interest in contributing to Raymember! Raymember is an open-source, model-agnostic persistent world-memory SDK for AI agents.

## Getting Started

1. Clone the repository.
2. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e .[dev,ml]
   ```
3. Run tests using `pytest`:
   ```bash
   pytest
   ```

## Code Style & Standards

- Write clean, type-hinted Python 3.11+ code.
- Provide descriptive docstrings for public classes and functions.
- Ensure all unit tests pass before submitting pull requests.
- Maintain deterministic core logic; optional ML extensions should degrade gracefully if dependencies are absent.

## License

By contributing to Raymember, you agree that your contributions will be licensed under the project's [Apache 2.0 License](LICENSE).
