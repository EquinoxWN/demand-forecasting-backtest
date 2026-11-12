# Contributing

Thanks for helping. This is a Python project; issues and pull requests are welcome.

## Set up and check your change

Needs Python 3.11+ (NumPy and pandas, CPU only). Caches, virtual environments and downloaded data stay in git-ignored folders inside the repo.

```bash
make setup   # install dependencies
make lint    # ruff, ruff format, mypy --strict
make test    # full test suite on synthetic M5-format data
make audit   # known-vulnerability check (as in CI)
make run     # download M5 (checksum-verified) and score store CA_1
```

A pull request is ready when `make lint` and `make test` pass and CI is green.

## Guidelines

- Keep each change focused; explain the problem it solves in the pull request.
- Add or update a test for every behaviour change, and update `docs/results/` when numbers change.
- A new forecaster must be scored through `evaluate`, so it is compared with the seasonal-naive baseline.
- Features must not use data after the forecast day; extend `test_features_never_look_ahead` for new ones.
- Significant design changes need an ADR in `docs/adr/` (copy `0001` for the format).
- Keep function comments to one short phrase; let names and tests explain the rest.
- Report security problems privately, as described in [SECURITY.md](SECURITY.md).

By contributing you agree that your work is released under the [MIT License](LICENSE).
