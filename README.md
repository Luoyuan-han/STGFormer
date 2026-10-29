# STGFormer

Topology-guided attention for SMILES-based molecular property prediction.

The model couples a standard SMILES transformer with **STG-Attn**, a
graph-guided attention layer that injects a token-level shortest-path
distance matrix (the *structural topology graph*, STG) derived from the
molecular graph, so that attention can depend on molecular connectivity in
addition to token composition and token order.

This repository contains the model code, the training / evaluation entry
points and small smoke fixtures for checking the model path end to end.

The Git checkout is intentionally lightweight. The complete benchmark trees
(`fintune_datas/moleculenet_random_scaffold/`, `fintune_datas/moleculenet_scaffold/` and
`fintune_datas/pytdc_scaffold/`) are
external data artefacts and are ignored by Git; see [`data/README.md`](data/README.md)
for the three fine-tuning datasets and their required layouts. Checkpoints
are external as well and are documented in [`checkpoints/README.md`](checkpoints/README.md).

## Contents

```
main.py                     training / evaluation / pretraining / generation CLI
config.py                   model + training configuration
modell.py                   transformer blocks, GraphSelfAttention (STG-Attn), RoPE
Tokenizer.py                SMILES tokenizers (BERT and GPT variants)
tool.py                     SMILES -> molecular graph and shortest-path matrices
Dataset.py                  datasets and collation used by every entry point
Bert_trainer.py             STGFormerTrainer: fine-tuning, evaluation, CLS readout
GPT generation support is kept inside `modell.py`, `Dataset.py` and
`Bert_trainer.py`.

pipeline/                   model training, evaluation and multi-seed drivers
data/                       external-data layout, manifests and smoke fixtures
configs/                    configuration guidance
configs/paper_hyperparameters.md
                            table-specific paper splits and protocols
docs/                       reproducibility checklist
```

## Requirements

Python 3.9+ and the packages in `requirements.txt`:

```powershell
pip install -r requirements.txt
# install a torch build that matches your CUDA driver, e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

RDKit is required for all SMILES processing. A GPU is strongly recommended for
training; inference and evaluation can run on CPU as well.

## Quick start

```powershell
python pipeline/run_pipeline.py
```

This runs the whole workflow on real data with the real code (no
re-implementation) and writes everything to `runs/run_<timestamp>/`:

| stage | what it does |
| --- | --- |
| `env` | interpreter, dependency and device audit |
| `sanity` | real tokenizer, real STG matrices and real `BertDataset` preprocessing |
| `data` | resolves the selected scaffold split and builds a small run-local training subset |
| `train` | fine-tunes one variant through `pipeline/train_variant.py` (a thin wrapper around the project `STGFormerTrainer`) |
| `eval` | reloads and evaluates the freshly trained checkpoint |
| `report` | aggregates everything into `run_report.md` and `metrics_summary.csv` |

```powershell
python pipeline/run_pipeline.py --stages env,sanity,data     # single stages
python pipeline/run_pipeline.py --full-eval                  # also evaluate the full valid/test splits
```

### Multi-seed runs

Two multi-seed protocols are supported through `pipeline/run_seeds.py`, which
detects the layout automatically.

`run_seeds.py` always trains on the complete selected split. With batch size
`32`, the progress-bar upper bound is the number of training batches (for
example, BACE's 1210 training rows produce 38 steps per epoch), not `1`.

Ten randomized scaffold splits
(`fintune_datas/moleculenet_random_scaffold/<task>/seed42 ... seed132`), one split per seed:

```powershell
python pipeline/run_seeds.py --task bace --scaffold-root fintune_datas/moleculenet_random_scaffold `
    --seeds 42,52,62,72,82,92,102,112,122,132 --seq-len 256 --epochs 50
```

One fixed scaffold split with ten random seeds (`fintune_datas/moleculenet_scaffold/<task>/`,
where `train/valid/test.csv` sit directly in the task directory):

```powershell
python pipeline/run_seeds.py --task bace --scaffold-root fintune_datas/moleculenet_scaffold `
    --seeds 42,52,62,72,82,92,102,112,122,132 --seq-len 256 --epochs 50
```

`--seeds` accepts the confirmed ten-run list above, an explicit range, or
`random10` (the ten `seed*` directories when present, otherwise the confirmed
list), matching `main.py`. Each protocol writes
per-seed run directories plus `seed_summary.csv` (mean/SD/min/max),
`seed_metrics_long.csv`, `<task>_seed_results.json` and `seed_report.md`
under `runs/seeds_<task>_<timestamp>/`.

Downstream fine-tuning uses learning rate `3e-5`, batch size `32`, and
validation early stopping patience `10`. The eight reported MoleculeNet tasks
are `bace`, `bbbp`, `hiv`, `clintox`, `tox21`, `sider`, `toxcast`, and `muv`.
The six reported PyTDC tasks are
`ames`, `herg`, `hia_hou`, `pgp_broccatelli`, `bioavailability_ma`, and `DILI`.
Use `--random-smiles` when `*_enforced.csv` should augment training only; use
`--enforced` when enforced randomized-SMILES files should be used for every
available split.

For the paper's official-scaffold comparisons, Table 3 uses the MoleculeNet
splits under `fintune_datas/moleculenet_scaffold/`, and Table 4 uses the PyTDC
splits under `fintune_datas/pytdc_scaffold/`. The randomized MoleculeNet splits
under `fintune_datas/moleculenet_random_scaffold/` are a separate protocol.

## Model variants

`pipeline/train_variant.py` exposes the two architecture switches directly:

| variant | `STG` (STG-Attn active when `True`) | `learnable_pos` (learned positional encoding) |
| --- | --- | --- |
| Base | False | False |
| PE | False | True |
| STG | True | False |
| STG+PE | True | True |

`Base` therefore has neither a positional signal nor a topology bias and is
approximately invariant to permutations of the non-special input tokens;
`PE` adds token order; `STG` adds connectivity-sensitive token interactions;
`STG+PE` combines both.

## Checkpoints

Model weights are not stored in Git. Use the separately released BERT
pretraining checkpoint to initialize fine-tuning:

```powershell
$env:STGFORMER_PRETRAINED = "checkpoints/pretraining/Bert_STG+PE_1000k.pt"
python pipeline/run_pipeline.py
```

The same path can be supplied with `--pretrained-checkpoint`. Fine-tuned
checkpoints are produced by each run and are not part of the release package.

## Data

`data/smoke/` contains a tiny loader/training fixture. The eight reported
MoleculeNet tasks, six reported PyTDC tasks, and pretraining data are external;
see
[`data/README.md`](data/README.md) for their layouts and randomized-SMILES
(`*_enforced.csv`) selection rules.

## CLI reference

```powershell
python main.py --mode finetune --task bace --scaffold_root fintune_datas/moleculenet_random_scaffold
python main.py --mode pretrain `
  --pretrain_variant STG+PE `
  --pretrain_train_csv train_data/train_1000k.csv `
  --pretrain_val_csv train_data/valid_1000k.csv `
  --pretrain_seq_len 128 --pretrain_batch_size 512 `
  --pretrain_lr 3e-4 --epochs 50
python main.py --mode gpt `
  --gpt_variant STG --gpt_stream `
  --gpt_train_csv train_data/gpt_train/train_1000k.csv `
  --gpt_val_csv train_data/gpt_train/valid_1000k.csv `
  --gpt_seq_len 128 --gpt_batch_size 256 --gpt_lr 3e-4 --epochs 50
python main.py --mode sample   --gpt_ckpt <checkpoint> --sample_num 100
```

The BERT and GPT pretraining defaults point to the local data layout under
`train_data/`. BERT uses the root `train_1000k.csv` /
`valid_1000k.csv` pair; GPT uses the matching files under
`train_data/gpt_train/`. The GPT loader is streaming by default so the
900k-molecule corpus is not materialized in memory. Use `--no-gpt_stream`
only for small datasets.

`python main.py --help` lists every option (splits, seeds, early stopping,
pooling, learning rate, sampling temperature, ...).

For the ten paper runs, use seeds `42,52,62,72,82,92,102,112,122,132`.
The random-scaffold driver accepts this explicitly:

```powershell
python pipeline/run_seeds.py --task bace `
  --scaffold-root fintune_datas/moleculenet_random_scaffold `
  --seeds 42,52,62,72,82,92,102,112,122,132 --epochs 50
```

## Model entry points

```powershell
python main.py --mode finetune --task bace --scaffold_root fintune_datas/moleculenet_random_scaffold
```

`main.py` also accepts `--run_dir <path>`; without it, outputs go to a
timestamped `runs/<mode>_<timestamp>/` directory.

```powershell
python pipeline/run_seeds.py --task bace --scaffold-root fintune_datas/moleculenet_random_scaffold `
  --seeds 42,52,62,72,82,92,102,112,122,132
```

For one classification task benchmarked across all ten seeds with both
validation-loss and validation-AUC early stopping:

```powershell
python pipeline/run_benchmark.py `
  --task bace --scaffold-root fintune_datas/moleculenet_random_scaffold `
  --pretrained-checkpoint checkpoints/pretraining/Bert_STG+PE_1000k.pt
```

The script reloads each best validation checkpoint, evaluates the same test
split, and writes `benchmark_results.csv`, `benchmark_results.json`, and
`benchmark_report.md`.

## Representation analysis

COMPAS and QM9 representation analysis is available as a standalone script. It accepts either an MLM-pretrained or downstream fine-tuned checkpoint and selects extraction, cosine-distance, or both metrics:

```powershell
python analysis/representation_analysis.py `
  --checkpoint checkpoints/pretraining/Bert_STG+PE_1000k.pt `
  --dataset qm9 --task gap --split test --metric both
```

For fine-tuned checkpoints, the stage, sequence length, architecture variant, and output dimension are detected automatically. The representation readout defaults to the paper's final-layer CLS state. For example, a COMPAS HOMO checkpoint can be analyzed with the same command shape:

```powershell
python analysis/representation_analysis.py `
  --checkpoint checkpoints/finetuned/Bert_best_model42.pt `
  --dataset compas --task homo --split test --metric both
```

For fine-tuned scalar regression, the pair output also reports the true property difference, predicted property difference, and the paper's pairwise `Delta-MAE`; standardized predictions are inverse-transformed using the training split automatically.

See [`analysis/README.md`](analysis/README.md) for the output files and grouping options.

## Citation

If you use this code, please cite the accompanying manuscript. Full citation
details will be added once the paper is published.

## Licence

Not yet specified by the authors.
