
Command Line Interface
============

## Introduction

Everything you can do in the [GUI](./gui.md) you can also do from the command line, with `CellscannerCLI.py`.
The command line interface (CLI) reads all its settings from a single configuration file (`config.yml`), which makes it
convenient for servers and clusters without a display, for running many analyses in a row,
and for keeping an exact record of the settings behind a result.
The CLI does not need PyQt5.

We use the same data as in the [GUI tutorial](./gui.md): *Roseburia intestinalis* (RI) and
*Bacteroides thetaiotaomicron* (BT) grown in mono- and co-culture, stained with SYBR-Green and propidium iodide (PI),
at the 50-hour time point. Please have a look there for the biological background.
Download the [tutorial files](http://msysbiology.com/documents/CellScanner/CS2TutorialFiles.zip)
and unzip them in the `Testfiles` folder of the repository, so that you get `Testfiles/CS2TutorialFiles`.

For installing CellScanner, see [Installation](./install.md).
To check that everything works, run from the root folder of the repository:

```bash
python cellscanner/CellscannerCLI.py --version
```


## The configuration file

A template, [`config.yml`](https://github.com/msysbio/CellScanner/blob/main/config.yml), is in the root folder of the repository.
It is already set up for the tutorial files and every parameter in it comes with a description.
Make a copy for your analysis and edit that:

```bash
cp config.yml my_config.yml
```

Each parameter is an entry with its setting under `value` (or `path` for files and folders, and `name` for channels).
If you leave a setting empty, CellScanner uses the entry's `default`.
Paths can be relative to the folder you run CellScanner from, or start with `~` for your home directory.

### Input files

**Blanks** are `.fcs` files of the cell-free medium, and **monocultures** are `.fcs` files of a single species.
For the monocultures, give the species name of every file under `species_names`, in the same order as the files:

```yaml
blank_files:
  directories:
    - directory:
        path: Testfiles/CS2TutorialFiles
        filenames:
          - 01-t12_d50_wc_btri_control-H7.fcs
          - 01-t12_d50_wc_btri_control-H8.fcs

species_files:
  directories:
    - directory:
        path: Testfiles/CS2TutorialFiles
        filenames:
          - 01-t12_d50_wc_btA-D1.fcs
          - 01-t12_d50_wc_btB-D2.fcs
          - 01-t12_d50_wc_btC-D3.fcs
          - 01-t12_d50_wc_riA-D4.fcs
          - 01-t12_d50_wc_riB-D5.fcs
          - 01-t12_d50_wc_riC-D6.fcs
        species_names: [BT, BT, BT, RI, RI, RI]
```

The **co-cultures** are the samples you want to count:

```yaml
coculture_files:
  directories:
    - directory:
        path: Testfiles/CS2TutorialFiles
        filenames:
          - 01-t12_d50_wc_btriA-D7.fcs
          - 01-t12_d50_wc_btriB-D8.fcs
          - 01-t12_d50_wc_btriC-D9.fcs
```

If you leave out `filenames`, all files in `path` are used.
Files can come from several folders: add one `- directory:` entry per folder.

Results are written to `output_directory`:

```yaml
output_directory:
  path: Testfiles/cs_output
```

### Training a new model or using an existing one

To train a new model, leave `prev_trained_model` empty:

```yaml
prev_trained_model:
  path:
```

To reuse a model trained earlier (in the CLI or the GUI), give the folder that contains
`trained_model.keras`, `scaler.pkl` and `label_encoder.pkl`, e.g. `Testfiles/cs_output/model`.
Blank and monoculture files are then ignored. Make sure `scaling_constant` is the one used to train that model;
you can find it in the model folder's `training_parameters.yml`.

### Training settings

These are the settings of the **Train Model** panel of the GUI, and the
[GUI tutorial](./gui.md) (section *Train Model*) explains what they do. The defaults work well for the tutorial files:

| Parameter | Default | What it sets |
|---|---|---|
| `umap_events` | 1000 | events sampled from each file for training |
| `n_neighbors` | 50 | size of the neighbourhood UMAP looks at |
| `umap_min_dist` | 0.0 | how tightly UMAP packs points together |
| `nn_non_blank` / `nn_blank` | 25 / 20 | how many of an event's 50 nearest neighbours must have its label for the event to be kept |
| `scaling_constant` | 150 | constant of the arcsinh transformation |
| `folds` | 0 | number of cross-validation folds; 0 trains on 80% of the events and validates on 20% |
| `epochs`, `batch_size`, `early_stopping_patience` | 50, 32, 10 | neural network training |
| `seed` | 42 | random seed, see *Reproducible results* below |

### Prediction settings

The three channels for the 3D plots (if empty or not found in the files, the first three channels are used):

```yaml
x_axis:
  name: FSC-A
y_axis:
  name: SSC-A
z_axis:
  name: FITC-A
```

Uncertainty filtering labels events the model cannot assign with confidence as `Unknown`.
Uncertainty is measured as entropy, which ranges from 0 to ln(number of classes):

```yaml
filter_out_uncertain:
  value: true
  threshold:
```

With `threshold` empty, CellScanner uses the threshold it suggested when training the model (also listed in
`model_statistics.csv`). You can set your own value instead, or `-1` for half of the maximum entropy.
When you reuse a model, give a threshold or `-1`.

### Gating

Gating uses stains to tell cells from debris and live from dead cells.
Set `gating` to `true` and describe each stain with its `channel` (the column name in the `.fcs` files),
a `sign` (`greater_than` or `less_than`) and a threshold `value`, in raw (untransformed) intensities:

- `stain1_*` stains all cells (e.g. SYBR-Green): events meeting the threshold are cells, the rest are debris.
- `stain2_*` stains dead cells (e.g. PI): events meeting the threshold are dead, the rest are live.

The `_train` stains are applied to the monocultures before training, so that the model only learns from live cells;
the `_predict` stains are applied to the co-cultures. Usually both use the same thresholds.
You can use one stain only: leave the other one empty.

```yaml
gating:
  value: true

stain1_train:
  channel: FITC-A
  sign: greater_than
  value: 500000

stain2_train:
  channel: PerCP-H
  sign: greater_than
  value: 2000000

stain1_predict:
  channel: FITC-A
  sign: greater_than
  value: 500000

stain2_predict:
  channel: PerCP-H
  sign: greater_than
  value: 2000000
```

Further stains can be added for the prediction step under `extra_stains`, each with a `label`
that names the events meeting its threshold.


## Running CellScanner

From the root folder of the repository:

```bash
python cellscanner/CellscannerCLI.py -c my_config.yml
```

CellScanner first checks that the gating channels exist in your files, then trains the model
(unless you gave `prev_trained_model`) and finally predicts every co-culture.
Training on the tutorial files takes a few minutes; progress is printed as it goes, including how many
events of each file pass the gating and how many of each class are kept after the nearest-neighbour filtering.

If a gating threshold leaves no events in a file, or the filtering leaves a class with almost no events,
CellScanner stops and tells you which file or class is affected. Revisit the thresholds in that case.


## Output

In `output_directory` you will find:

- **`model/`** with the trained model (`trained_model.keras`, `scaler.pkl`, `label_encoder.pkl`) and
  - `model_statistics.csv`: accuracy, the suggested uncertainty threshold, the confusion matrix and the classification report;
  - `uncertainty_threshold_curve.csv`: accuracy and share of events kept for each candidate uncertainty threshold;
  - `umap_Before_filtering.html` and `umap_After_filtering.html`: the UMAP embedding before and after filtering;
  - `training_parameters.yml`: all training settings;
  - `gating_input_data.txt`: events per file before and after gating (when gating is used).
- **`Prediction_<date>_<time>/`** with the co-culture results. These are the same files as described in the
  [GUI tutorial](./gui.md) (section *Run prediction*): per sample `prediction_counts.csv`, `raw_predictions.csv` and a 3D plot,
  the `gated/`, `heterogeneity_results/` and `uncertainty_counts/` subfolders and, for several samples,
  **`merged_prediction_counts.csv`**. In addition:
  - `run_parameters.yml` lists every setting of the run, including the CellScanner version and the training settings of the model used;
  - `config_used.yml` is a copy of the configuration file you ran with.

With both stains, `prediction_counts.csv` has `<species>_live`, `<species>_dead` and `<species>_debris` rows for every species,
plus `Blank` (events resembling the cell-free medium) and `Unknown` (uncertain events).
The counts of a sample add up to the number of events in its `.fcs` file.


## Reproducible results

UMAP, the sampling of events and the training of the neural network involve randomness.
CellScanner fixes it with the `seed` parameter, so the same files, settings and seed give the same counts on the same machine.
To see how much your results depend on this randomness, run the analysis a few times with different seeds and compare the counts.
Together with `run_parameters.yml` and `config_used.yml`, this lets you rerun any earlier analysis exactly.


## Using Docker

The CellScanner Docker image includes the CLI. Put your `.fcs` files and configuration file in one folder and mount it to `/media`;
in the configuration file, refer to the files as `/media/...`, and leave `output_directory` empty, as results go to `/media`:

```bash
docker run --rm --user $(id -u):$(id -g) -v ./Testfiles:/media hariszaf/cell_scanner \
    python CellscannerCLI.py -c /media/my_config.yml
```

`--user` makes the result files belong to you rather than to root.
