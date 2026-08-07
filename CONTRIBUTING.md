# Contributing to WaveSpace

Thank you for your interest in contributing to **WaveSpace**, a Python package for analyzing cortical traveling waves. We welcome contributions from the community to improve and expand the project.

Repository: [WaveSpace on GitHub](https://github.com/kpetras/WaveSpace)

## How to Contribute

1. **Contact**
   If you have questions or want to discuss potential contributions, please reach out to [Kirsten Petras](https://github.com/kpetras) via GitHub.

2. **Fork and Branch**

   * Fork the repository to your own GitHub account.
   * Create a new branch for your feature or bugfix:

     ```bash
     git checkout -b feature/your-feature-name
     ```

3. **Make Your Changes**

   * Keep changes focused and concise.
   * Follow existing code style and structure where possible.
   * Add or update documentation as needed.

4. **Testing**

   * Ensure that all unit tests pass before submitting.
   * Contributions will only be accepted if the test suite runs successfully.
   * Run tests locally using Python's built-in `unittest` framework from the
     repository root:

     ```bash
     python -m unittest discover -s UnitTest -p "*_test.py"
     ```

   * To run the test suite against every supported Python version (3.9–3.14)
     locally with [tox](https://tox.wiki), first install the interpreters, e.g.
     with [uv](https://docs.astral.sh/uv), then install the `dev` dependency
     group which provides `tox` and `tox-uv`:

     ```bash
     uv python install 3.9 3.10 3.11 3.12 3.13 3.14
     uv sync --group dev
     tox
     ```

     You can target a single version with `tox -e py314`, for example.

5. **Pull Request**

   * Push your branch to your forked repository.
   * Open a pull request (PR) to the main branch of WaveSpace.
   * Clearly describe your changes and reference related issues if applicable.

## Contribution Guidelines

* Be respectful and collaborative in discussions.
* Write clean, well-documented code.
* Small, focused contributions are preferred over large, unfocused changes.

We appreciate your efforts to help improve **WaveSpace**!
