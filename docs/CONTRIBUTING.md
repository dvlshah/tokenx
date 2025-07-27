# Contributing to tokenx

Thank you for your interest in contributing to tokenx! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/neurink-ai/tokenx.git
   cd tokenx
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   pip install -e .[dev]
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

We use `ruff` for code linting and formatting. Pre-commit hooks will automatically check your code before commits.

## Testing

Run the tests using pytest:

```bash
pytest -v
```

To run specific tests:

```bash
pytest tests/test_openai_adapter.py -v
```

## Adding Support for New Providers

To add support for a new LLM provider:

1. Create a new file in `src/tokenx/providers/` for your provider (e.g., `anthropic.py`)
2. Implement the `ProviderAdapter` interface defined in `base.py`
   - Must implement the abstract `usage_from_response` method returning a `Usage` dataclass
   - See existing providers (OpenAI, Anthropic) for implementation examples
3. Add pricing information for the provider's models to `model_prices.yaml`
4. Add tests for the new provider in the `tests/` directory
5. Update the documentation to include the new provider

## Updating Pricing Information

The pricing information is stored in `src/tokenx/model_prices.yaml`. To update prices:

1. Check the official pricing page of the provider
2. Update the corresponding section in the YAML file
3. Add a note in the changelog about the price update

## Submitting Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Commit your changes following [conventional commits](https://www.conventionalcommits.org/) format

3. Push your branch and create a pull request

4. Ensure all tests pass and the code passes all linting checks

## Reporting Issues

When reporting issues, please include:

- A clear description of the issue
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment information (Python version, OS, package versions)

## Feature Requests

For feature requests, please include:

- A clear description of the proposed feature
- Use cases for the feature
- Any relevant documentation or references

## Code of Conduct

Please be respectful and considerate of others when contributing to the project.

Thank you for your contributions!