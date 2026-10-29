# Representation analysis

`representation_analysis.py` loads one BERT/STGFormer checkpoint, extracts an encoder representation for each molecule, and optionally computes within-group cosine distances for graph-distinct molecules sharing the same tokenizer token multiset. The checkpoint stage is detected automatically: MLM checkpoints use the final encoder CLS state, while fine-tuned regression checkpoints use the configured pooled representation.

The script accepts the local COMPAS and QM9 layouts under `fintune_datas/compas_qm9/`. It also accepts any CSV or JSON/metadata JSON with SMILES and optional target values.

```powershell
python analysis/representation_analysis.py `
  --checkpoint checkpoints/pretraining/Bert_STG+PE_1000k.pt `
  --dataset qm9 `
  --task gap `
  --split test `
  --metric both `
  --seq-len 128 `
  --output runs/analysis_qm9_stgpe
```

For a fine-tuned COMPAS checkpoint, the stage, sequence length, architecture variant, and output dimension are read from the checkpoint and its path. The default readout is the paper's final-layer CLS representation:

```powershell
python analysis/representation_analysis.py `
  --checkpoint checkpoints/finetuned/Bert_best_model42.pt `
  --dataset compas --task homo --split test --metric both
```

Use `--stage pretrained` or `--stage finetuned` to override stage detection, `--variant base|pe|stg|stg+pe` when the checkpoint path does not identify the architecture, and `--pooling mean` only when reproducing an older/custom run that used mean pooling. `--seq-len` is inferred from learnable positional embeddings when they are present; pass it explicitly for checkpoints without them.

If Windows raises `PermissionError` while loading a checkpoint on another drive, verify that the drive is online and healthy, then copy the file to a local NTFS directory such as `D:\STGFormer\checkpoints\finetuned\` and pass the copied path. The script reports the checkpoint path and the underlying Windows error.

Use `--metric extract` to write only molecule representations, or `--metric cosine` to write the pairwise analysis. `--group-key formula_tokens` is the strict QM9/COMPAS grouping rule; `--group-key tokens` omits the molecular-formula condition. If the checkpoint filename follows the release names, the variant is inferred automatically; otherwise pass `--variant base`, `--variant pe`, `--variant stg`, or `--variant stg+pe`.

For a fine-tuned scalar-regression checkpoint, the pair table also includes the true property difference, predicted property difference, and pairwise property error:

\[
\Delta\mathrm{-MAE}_{ij}=\left|\,|y_i-y_j|-|\hat y_i-\hat y_j|\,\right|.
\]

The default `--prediction-scale auto` follows the paper protocol: model outputs are treated as training-set z-scores and inverse-transformed with the mean and standard deviation from `train_meta.json`. Use `--prediction-scale raw` for a checkpoint trained directly on physical-unit targets. Outputs include `molecules.csv`, `embeddings.npy`, `controlled_groups.csv`, `controlled_molecules.csv`, `cosine_pairs.csv`, `pair_property_errors.csv`, and `analysis.json`. `controlled_molecules.csv` contains the extracted molecules participating in at least one controlled pair.
