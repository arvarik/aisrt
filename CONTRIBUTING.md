# Contributing to Ultimate SRT Generator

First off, thank you for considering contributing to the Ultimate SRT Generator! It's people like you that make open source such a fantastic community to learn, inspire, and create.

## 🛠️ Development Setup

This project uses modern Python tooling. To get started:

1. Ensure you have Python 3.11+ installed.
2. Clone the repository:
   ```bash
   git clone https://github.com/arvarik/aisrt.git
   cd aisrt
   ```
3. Set up a virtual environment and install development dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
   *(Alternatively, if you use `poetry`, you can simply run `poetry install`)*

## 📐 Engineering Standards

We maintain strict enterprise-grade code quality. **All pull requests MUST pass the following checks before they can be merged:**

1. **Formatter & Linter (Ruff):**
   ```bash
   ruff format .
   ruff check .
   ```
2. **Type Checking (Mypy):**
   We enforce strict typing. Ensure you have fully typed your additions.
   ```bash
   mypy src/aisrt tests
   ```
3. **Unit Tests (Pytest):**
   Ensure your changes do not break existing functionality. If adding a new feature, please include corresponding `pytest-asyncio` tests.
   ```bash
   pytest tests
   ```

## 💡 Pull Request Process

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. Ensure the test suite passes.
4. Make sure your code lints and type-checks successfully.
5. Submit a descriptive Pull Request detailing the problem solved or feature added.

Thank you for contributing!
