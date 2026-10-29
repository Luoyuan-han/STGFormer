# STGFormer runnable pipeline

One command runs the model workflow on real data with the project's own
code (tokenizer, STG construction, preprocessing, model, trainer). Nothing is
re-implemented for the pipeline.

```powershell
python pipeline/run_pipeline.py                     # quick run (~10 min)
python pipeline/run_pipeline.py --full-eval         # + full valid/test splits (~30 min)
python pipeline/run_pipeline.py --stages env,sanity # single stages
```

## Stages

| stage | what it does |
| --- | --- |
| `env` | interpreter, dependency and device audit |
| `sanity` | real tokenizer + real STG matrices + real `BertDataset` preprocessing |
| `data` | resolves the selected scaffold split and writes a run-local smoke subset |
| `train` | fine-tunes one variant through `pipeline/train_variant.py`, a thin wrapper around the project `STGFormerTrainer` |
| `eval` | reloads and evaluates the freshly trained checkpoint |
| `report` | aggregates everything into `run_report.md` + `metrics_summary.csv` |

## Variants

`pipeline/train_variant.py` exposes the two architecture switches that
`main.py` does not put on the command line (values taken from `config.py`):

| variant | `STG` (STG-Attn active when `True`) | `learnable_pos` (PE active) |
| --- | --- | --- |
| Base | False | False |
| PE | False | True |
| STG | True | False |
| STG+PE | True | True |

## Configuration

Every path comes from `pipeline/pipeline_config.json`, a CLI flag, or an
environment variable - no machine-specific path is compiled into the code:

```powershell
$env:STGFORMER_PRETRAINED = "checkpoints/pretraining/Bert_STG+PE_1000k.pt"
python pipeline/run_pipeline.py
```

Fine-tuned checkpoints are written into the run directory and immediately
reloaded for test evaluation.

## Outputs

Everything lands in `runs/run_<timestamp>/`:

```
01_env.json                  02_code_sanity.json        03_data.json
train/train_STGpPE.json      train/checkpoints/Bert_best_model42.pt
eval/<source>/<variant>/...  05_eval.json
run_report.md                metrics_summary.csv        run_manifest.json
logs/*.log                   train/logs/*.jsonl
```

## Multi-seed runs

For the normal single-task paper benchmark, `run_benchmark.py` runs all ten
seeds twice: once with validation-loss early stopping and once with
validation-AUC early stopping. Both branches are evaluated on the same test
split:

```powershell
python pipeline/run_benchmark.py `
    --task bace --scaffold-root fintune_datas/moleculenet_random_scaffold `
    --pretrained-checkpoint checkpoints/pretraining/Bert_STG+PE_1000k.pt
```

The repository ships two different multi-seed protocols. Both are handled by
`pipeline/run_seeds.py`, which detects the layout automatically and logs it
(`layout=per-seed splits` or `layout=single split reused for every seed`).
Multi-seed runs read the complete selected split directly; the small smoke
subset is only used by `run_pipeline.py` when `--full-data` is omitted.

### 1. Ten randomized scaffold splits - `fintune_datas/moleculenet_random_scaffold/<task>/seed42 ... seed132`

Each `seed*` directory holds its own `train/valid/test.csv`, so every seed is
a different scaffold split:

```powershell
python pipeline/run_seeds.py --task bace --scaffold-root fintune_datas/moleculenet_random_scaffold `
    --seeds 42,52,62,72,82,92,102,112,122,132 --seq-len 256 --epochs 50 `
    --pretrained-checkpoint <external-pretrained-checkpoint.pt>
```

### 2. One scaffold split, ten random seeds - `fintune_datas/moleculenet_scaffold/<task>/`

`train.csv`, `valid.csv` and `test.csv` sit directly in the task directory.
The same split is reused for every seed and the seed only controls
initialisation and batch order:

```powershell
python pipeline/run_seeds.py --task bace `
    --scaffold-root fintune_datas/moleculenet_scaffold `
    --seeds 42,52,62,72,82,92,102,112,122,132 --seq-len 256 --epochs 50 `
    --pretrained-checkpoint <external-pretrained-checkpoint.pt>
```

`--seeds` accepts the confirmed ten-run list, an explicit range or `random10`
(the ten `seed*` directories when they exist, otherwise the confirmed list),
matching `main.py`

Add `--random-smiles` to select `train_enforced.csv` while keeping canonical
validation/test files, or `--enforced` to select enforced files for all splits.

The downstream defaults are learning rate `3e-5`, batch size `32`, and
validation early stopping patience `10`.
semantics. Both protocols write `runs/seeds_<task>_<timestamp>/`
containing one sub-directory per seed plus:

```
seed_summary.csv             mean / SD / min / max per metric, across seeds
seed_metrics_long.csv        every (seed, model, split, metric) value
<task>_seed_results.json     one record per seed (project convention)
seed_report.md               readable summary table
```

## Notes

* `pipeline/train_variant.py` imports the repository modules directly; run it
  from the repository root so `sys.path` resolves (`run_pipeline.py` does this
  for you).
* The pipeline raises on any failed stage instead of continuing.
