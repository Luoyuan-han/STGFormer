# Data layout

The Git repository contains code, metadata and small smoke fixtures only. The
full dataset trees are intentionally kept outside Git because they are several
gigabytes and contain individual files larger than GitHub's 100 MB limit.

When using the provided local archives, see [fine-tuning data layout](../fintune_datas/README.md)
and [pretraining data layout](../train_data/README.md) for extraction locations and
the expected directory trees.

## Fine-tuning dataset directories

Fine-tuning datasets use CSV as the preferred input format. Each split CSV has
`smiles` and `tags` columns; alternatively, one aggregate CSV may contain
`smiles`, `tags`, and `split` columns. Multi-task labels may be JSON-style
arrays in `tags` or separate `task_0`, `task_1`, ... columns. The three
external roots use consistent task-first layouts and are passed directly
through `--scaffold_root`:

```text
fintune_datas/moleculenet_random_scaffold/<task>/seed42/{train,valid,test}.csv
fintune_datas/moleculenet_random_scaffold/<task>/seed52/{train,valid,test}.csv
fintune_datas/moleculenet_scaffold/<task>/{train,valid,test}.csv
fintune_datas/pytdc_scaffold/<task>/{train,valid,test}.csv
```

`fintune_datas/moleculenet_scaffold` contains one fixed scaffold split reused across model
seeds. `fintune_datas/moleculenet_random_scaffold` contains one randomized
scaffold partition per seed. `*_enforced.csv` files are randomized-SMILES data;
they do not define additional scaffold partitions. In `main.py`,
`--random_smiles` uses the enforced training split while keeping
validation/test canonical, whereas `--enforced` selects enforced files for all
available splits.

For the official-scaffold comparisons in the paper, use
`fintune_datas/moleculenet_scaffold/` for Table 3 and
`fintune_datas/pytdc_scaffold/` for Table 4. Do not substitute the per-seed
randomized partitions in `fintune_datas/moleculenet_random_scaffold/` for
these tables.

Both MoleculeNet roots are limited to the eight classification tasks reported
in the paper: `bace`, `bbbp`, `hiv`, `clintox`, `tox21`, `sider`, `toxcast`,
and `muv`. Historical regression, controlled-isomer, and unreported benchmark
task directories are not part of the release dataset.

The two roots are deliberately separate. Do not combine their files into one
task directory or compare results without recording the protocol and exact
file checksums.

## Local placement

For a local checkout, place downloaded data wherever convenient and pass the
absolute path explicitly:

```powershell
python pipeline/run_seeds.py `
  --task bace `
  --scaffold-root fintune_datas/moleculenet_random_scaffold `
  --seeds 42,52,62,72,82,92,102,112,122,132
```

On Windows, an external absolute path can be used instead, for example
`E:\\datasets\\moleculenet_random_scaffold`.

Run `python convert_datasets_to_csv.py <dataset-root> [...]` to export legacy
split JSON files to corresponding CSV files. Add `--delete-json` to remove those
sources only after their CSV replacements pass a non-empty check.

Tiny model and loader fixtures are kept under `data/smoke/`.

## PyTDC ADMET data

The local `fintune_datas/pytdc_scaffold/<task>/` tree contains single-task PyTDC scaffold
splits with the same three files per task:

```text
fintune_datas/pytdc_scaffold/<task>/{train,valid,test}.csv
```

The local tree is intentionally limited to the six PyTDC classification tasks
reported in the paper: `ames`, `herg`, `hia_hou`, `pgp_broccatelli`,
`bioavailability_ma`, and `DILI`.

Example invocation:

```powershell
python pipeline/run_seeds.py `
  --task ames `
  --scaffold-root fintune_datas/pytdc_scaffold `
  --seeds 42,52,62,72,82,92,102,112,122,132
```

## Pretraining corpus

The pretraining corpora are also external to Git and should be placed under:

```text
train_data/train_1000k.csv
train_data/valid_1000k.csv
train_data/train_all.csv
train_data/valid_all.csv
train_data/gpt_train/train_1000k.csv
train_data/gpt_train/valid_1000k.csv
train_data/gpt_train/train_100k.csv
train_data/gpt_train/valid_100k.csv
train_data/gpt_train/train_75k.csv
train_data/gpt_train/valid_75k.csv
train_data/gpt_train/train_50k.csv
train_data/gpt_train/valid_50k.csv
train_data/gpt_train/train_25k.csv
train_data/gpt_train/valid_25k.csv
```

Every file is a CSV with one `smiles` column. `main.py --mode
pretrain` streams the in-code `BertTokenizer` over the root BERT train/valid
pair. `main.py --mode gpt` uses the `gpt_train` pair and streams the in-code
`GPTTokenizer` by default. The `*_all.csv` files are available for a larger
BERT run; pass them explicitly with `--pretrain_train_csv` and
`--pretrain_val_csv`. The 25k/50k/75k GPT pairs can be selected by overriding
`--gpt_train_csv` and `--gpt_val_csv`.

The complete directory remains ignored and is not published to GitHub.

## Publishing full data

Use an external release (Zenodo, Hugging Face, institutional object storage,
or Git LFS) for the full trees. Add a versioned manifest under
`data/manifests/` containing the download URL, source version, preprocessing
parameters, number of rows per split, and SHA256 checksums before publishing.
