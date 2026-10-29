# Checkpoints

All model weights live under `checkpoints/pretraining/` locally and are ignored
by Git. The source repository publishes code and the manifest only; distribute
the weight files as a separate release asset.

The current local set contains 10 BERT checkpoints and 2 GPT checkpoints. The
manifest records each filename, size, and SHA256 hash. Verify the local files
from the repository root with:

```powershell
python checkpoints/verify_manifest.py
```

For the default `STG+PE` fine-tuning run, use:

```text
checkpoints/pretraining/Bert_STG+PE_1000k.pt
```

The 1000k variant matches the documented pretraining corpus. The GPT
checkpoints are `GPT_STG_all.pt` and `GPT_STG+PE_all.pt`; the GPT CLI defaults
to the STG model without learned positional embeddings.

Example fine-tuning command:

```powershell
python pipeline/run_benchmark.py `
  --task ames `
  --scaffold-root fintune_datas/pytdc_scaffold `
  --pretrained-checkpoint checkpoints/pretraining/Bert_STG+PE_1000k.pt
```

The pipeline, multi-seed, benchmark, and legacy CLI defaults point to the
checkpoint names above. Override them with `--pretrained-checkpoint`,
`--ckpt`, `--gpt_ckpt`, or the `STGFORMER_PRETRAINED` environment variable as
needed. Never force-add model weight files to Git.
