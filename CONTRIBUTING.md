# Contributing to colab-agent-bridge

Thank you for contributing to `colab-agent-bridge`! Whether you are a human developer or an autonomous coding agent, follow these mandatory guidelines.

---

## 1. Cardinal Rules

### Rule 1: Mandatory Documentation Synchronization
> [!IMPORTANT]
> **Always update documentation whenever code changes are made.**
>
> Any PR, commit, or task that adds, modifies, or deprecates:
> - CLI commands, arguments, or flags (`colab-bridge ...`)
> - Python SDK classes, methods, or parameters (`ColabBridge(...)`)
> - Environment variables or configuration defaults (`.env`, `config.py`)
> - Backend runtime behaviors or tunnel implementations
>
> **MUST** synchronously update all corresponding documentation files in the same changeset before claiming completion:
> - `README.md` (Features, Quickstart, CLI Reference, Python SDK examples)
> - `AGENTS.md` (Autonomous agent playbook, SOPs, commands, diagnostics)
> - `docs/` (`setup_guide.md`, `hardware_and_models.md`, `coding_harness_guide.md`)
>
> Never submit or merge a pull request where documentation lags behind code changes.

### Rule 2: Compute Unit Safety & Resource Teardown
> [!CAUTION]
> Colab VMs incur billable Compute Units for every minute they remain active.
> Always ensure tests, scripts, and workflows clean up sessions upon completion, failure, or cancellation:
> ```bash
> colab-bridge down -s <session_name>
> # Or global nuclear cleanup across all accounts:
> colab-bridge down --all
> ```

---

## 2. Development & Testing Workflow

1. **Environment Setup:**
   Install the package with all development and test dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. **Automated Quality Checks:**
   Run the full test suite, linter, type checker, and packaging validator:
   ```bash
   # Run tests with coverage
   pytest tests -v --cov=colab_agent_bridge

   # Run Ruff linter and formatter check
   ruff check .
   ruff format --check colab_agent_bridge tests

   # Run Mypy static type checker
   mypy colab_agent_bridge

   # Build distribution and check packaging metadata
   python -m build --sdist --wheel
   twine check --strict dist/*
   ```

3. **Isolated Test State:**
   Never write to or mutate the developer's real `~/.config/colab-bridge/` or `~/.config/colab-cli/` during unit tests. Always use temporary directories (`tmp_path`) and pass isolated configurations.

4. **Subprocess Safety:**
   Avoid raw string shell commands with unescaped interpolation. Pass explicit argument lists to `subprocess.run` or `subprocess.Popen`.

---

## 3. Pull Request Checklist

Before submitting a PR, verify:
- [ ] Code changes adhere to project style and conventions (`ruff check .`, `ruff format`).
- [ ] All unit tests pass: `pytest tests -v`.
- [ ] Static type checking passes: `mypy colab_agent_bridge`.
- [ ] Package build passes: `python -m build && twine check --strict dist/*`.
- [ ] New tests are added covering the new functionality or bugfix.
- [ ] **All documentation is updated** (`README.md`, `AGENTS.md`, `docs/*`).
- [ ] No secrets, tokens, credentials, or personal identifiers are committed.
