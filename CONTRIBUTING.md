# Contributing

Thank you for your interest in AIDflow.


## Development Setup

```bash
git clone https://github.com/aigensys-labs/aidflow.git
cd aidflow
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Reporting Issues

- Use the [GitHub Issue Tracker](https://github.com/aigensys-labs/aidflow/issues)
- Include: Python version, AIDflow version, minimal reproducible example
- For bugs: a short script that reproduces the issue is ideal

## Submitting Pull Requests

1. Fork the repository and create a feature branch from `main`
2. Follow the existing code style (Black-formatted)
3. Add or update docstrings for any new public-facing code
4. Keep PRs focused — one feature or fix per PR
5. Open a PR against `main` with a clear description of what changed and why

> [!IMPORTANT]
> **Scope:** PRs should contain only changes related to the stated issue or feature.
> PRs mixing unrelated changes may be closed without review.
>
> **AI-generated code:** AIDflow source code is written by humans. AI may be used
> for docstrings, formatting, and test scaffolding — not for core logic.
> PRs that appear primarily LLM-generated from first-time contributors will not
> be accepted unless the author can demonstrate working knowledge of the changes
> (e.g., explaining design tradeoffs in review).

## Code Style

- **Formatter**: [Black](https://black.readthedocs.io/)
- **Docstrings**: NumPy style (consistent with the existing codebase)
- **Type hints**: Required on all public methods

## Licensing

"AIGENSYS" means **AIGENSYS s.r.o.**, a limited liability company
organized and existing under the laws of the Czech Republic, 
with its registered office at Bilkova 855/19, Prague 1, Czech Republic,
Company VAT ID Number: **CZ23450924**.

By submitting a contribution to this project, you agree that:

1. Your contribution is licensed under the project's
   [FSL-1.1-Apache-2.0](LICENSE) terms and becomes part of the Software.

2. You grant AIGENSYS a perpetual, irrevocable, worldwide,
   royalty-free license to use, modify, reproduce, and distribute
   your contribution in the Software or any other product or service
   that AIGENSYS offers.

3. To the extent your contribution would necessarily infringe any
   patents you own or control, you grant AIGENSYS a perpetual,
   irrevocable, worldwide, royalty-free license under those patents
   for the same purposes.

4. You represent and warrant that you have the right to grant the
   licenses above and that your contribution does not infringe any
   third-party intellectual property rights.

## Other Ways to Contribute

- Fix or improve documentation
- Share feedback on issues or PRs
- Write a blog post or tutorial using AIDflow and let us know

## Questions?

Open a [Discussion](https://github.com/aigensys-labs/aidflow/discussions)
or reach out via the issue tracker.