Release notes
=============

The CellScanner version is set in `cellscanner/scripts/__init__.py`. It is shown in the GUI title and
by `CellscannerCLI.py --version`, and it is recorded in the `run_parameters.yml` and
`training_parameters.yml` files of every run, so you can always tell which version produced a result.

Earlier releases are listed on [GitHub](https://github.com/msysbio/CellScanner/tags).


## 2.1.0


### Results change compared to 2.0

- **Gated training used the wrong events.** With gating on, training kept debris and dead events
  instead of live cells. Models trained with gating on 2.0.x should be retrained.
- **Suggested uncertainty threshold.** It is now computed with the natural log (matching the entropy
  used for predictions) and chosen as the most accurate threshold that keeps at least 80% of the
  validation events; the previous search always returned the strictest cut-off.
  The full curve is saved as `model/uncertainty_threshold_curve.csv`.
- **Random seed.** Runs are now reproducible (`seed` in `config.yml`, "Random seed" in the GUI,
  default 42), so results differ from earlier unseeded runs.

### Fixed

- Only one stain (e.g. SYBR green only) can be used for gating, in the GUI and the CLI.
- A gate that does not split a file is a warning instead of an error; gating that removes every
  event of a file still stops the run.
- Extra stains accept `greater_than` / `less_than` as documented (they were treated as `<`).
- The CLI runs without PyQt5 installed.
- CLI: the requested 3D plot axes are used; `0` / `false` settings are respected; `umap_min_dist`
  is read (its `config.yml` entry now uses `value:`).
- Co-culture channels are matched to the ones the model was trained on.
- Prediction counts store `0` instead of empty values; `Blank` is reported as a single count (#29).
- Merged heterogeneity results are aligned by species.
- A class left with fewer than 2 events after nearest-neighbour filtering stops training with a
  clear message (it was silently dropped, or failed with an unclear error).
- `.fcs` files without a `Time` column are no longer skipped in the GUI.
- Training statistics: fold count and best fold were swapped.

### Added

- `run_parameters.yml` in each Prediction folder, and `training_parameters.yml` in the model folder,
  recording the settings of each run; the CLI also keeps a copy of its config as `config_used.yml`.
- Version tracking (`cellscanner/scripts/__init__.py`, `--version`, GUI title, docs).
- Installation docs for Windows from the source code and with Docker, and a CLI tutorial.

### Removed

- The Windows `.exe` (PyInstaller) build instructions and the code that supported it;
  on Windows, run CellScanner from the source code or with Docker.

### Docker

- Based on `python:3.12-slim`, matching the tested Python version (the code now needs Python 3.11
  or newer); the image is tagged and labelled with the CellScanner version, runs the CLI as well as
  the GUI, and works as a non-root user.
- `requirements.txt`: fixed the invalid `PyQt5>==5.15.11` specifier that broke `pip install`.
