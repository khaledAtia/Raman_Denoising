# Local packages

Two helper packages are bundled under `code/` rather than installed from PyPI.
Put both on the import path before running anything in `code/`:

- `code/RamanUtils/` — `airPLS.py` (the independent baseline estimator used as a
  cross-check throughout), `RamanHelpers.py` (`remove_laser`, file readers) and
  `RamanGenerator.py`.
- `code/JournalPlots/` — `JournalStyles.py`, the matplotlib style used by the
  manuscript figure scripts.

The simplest route is to run scripts from `code/` with that directory on
`PYTHONPATH`:

    set PYTHONPATH=<archive>/code        (Windows)
    export PYTHONPATH=<archive>/code     (POSIX)
