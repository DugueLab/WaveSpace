# Contributing to WaveSpace

Thank you for your interest in contributing to **WaveSpace**, a Python package for analyzing cortical traveling waves. We welcome contributions from the community to improve and expand the project.

Repository: [WaveSpace on GitHub](https://github.com/kpetras/WaveSpace)

## How to Contribute

### Questions, Bug Reports, and Feature Requests

For questions, bug reports, or feature requests, please open an [issue on GitHub](https://github.com/kpetras/WaveSpace/issues). This allows the community and project contributors to see and discuss questions and issues openly.

### Fork and Branch

1. Fork the repository to your own GitHub account.
2. Create a new branch for your feature or bugfix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

### Make Your Changes

* Keep changes focused and concise.
* Follow the existing code style and structure where possible.
* Add or update documentation as needed.
* Add or update tests when changing or adding functionality.

### Code Style

WaveSpace enforces linting with [Ruff](https://docs.astral.sh/ruff/). Pull requests must pass the same check run by GitHub Actions:

```bash
uv run --only-group dev ruff check .
```

The rules enforce import sorting and checks for common bugs, outdated Python syntax, unnecessary complexity, and problematic comprehensions. The complete, authoritative configuration is in `pyproject.toml`. Please keep functions focused, use descriptive names, and avoid unrelated formatting changes in the same pull request.

Ruff can automatically correct many findings:

```bash
uv run --only-group dev ruff check . --fix
```

### Testing

Please ensure that the test suite passes before submitting a pull request.

For a quick local test run using Python's built-in `unittest` framework, run the following from the repository root:

```bash
python -m unittest discover -s UnitTest -p "*_test.py"
```

To run the test suite against all supported Python versions (3.9–3.14) locally, use `tox`. First install the required Python versions, for example with [uv](https://docs.astral.sh/uv), and then install the development dependencies:

```bash
uv python install 3.9 3.10 3.11 3.12 3.13 3.14
uv sync --group dev
tox
```

You can also target a single Python version, for example:

```bash
tox -e py314
```

Pull requests to the `main` branch are automatically tested using the project's GitHub Actions workflow.

### Pull Requests

1. Push your branch to your forked repository.
2. Open a pull request to the `main` branch of WaveSpace.
3. Clearly describe your changes and reference related issues where applicable.

## Contribution Guidelines

* Be respectful and collaborative in discussions.
* Write clean, well-documented code.
* Keep contributions focused and manageable.
* Include appropriate tests and documentation for new or modified functionality.

We appreciate your efforts to help improve **WaveSpace**!
