# Fine-tuning and Analysis Data

From the repository root, extract `fintune_data.zip` into this directory. The
archive already contains the four data roots shown below, so do not extract it
into an extra `fintune_data/` subdirectory.

```powershell
Expand-Archive -Path .\fintune_datas\fintune_data.zip -DestinationPath .\fintune_datas -Force
```

Expected layout:

```text
fintune_datas/
|-- moleculenet_random_scaffold/
|   |-- <task>/
|   |   |-- cleaning/                  # preprocessing audit files
|   |   |-- seed42/
|   |   |   |-- train.csv
|   |   |   |-- valid.csv
|   |   |   |-- test.csv
|   |   |   `-- *_enforced.csv         # randomized-SMILES variants
|   |   `-- seed52/ ... seed132/
|-- moleculenet_scaffold/
|   `-- <task>/{train,valid,test}.csv
|-- pytdc_scaffold/
|   `-- <task>/{train,valid,test}.csv
`-- compas_qm9/
    |-- compas_homo/{train,valid,test}.json
    |-- compas_lumo/{train,valid,test}.json
    `-- qm9_gap/{train,valid,test}.json
```

`<task>` is one of the reported task names:

- MoleculeNet: `bace`, `bbbp`, `hiv`, `clintox`, `tox21`, `sider`, `toxcast`, `muv`
- PyTDC: `ames`, `herg`, `hia_hou`, `pgp_broccatelli`, `bioavailability_ma`, `DILI`
- Analysis: `compas_homo`, `compas_lumo`, `qm9_gap`

The randomized MoleculeNet root has one split directory per seed (`seed42`,
`seed52`, ..., `seed132`). The fixed MoleculeNet and PyTDC roots have one
`train.csv`, `valid.csv`, and `test.csv` directly inside each task directory.
Table 3 uses `moleculenet_scaffold`; Table 4 uses `pytdc_scaffold`.

The COMPAS/QM9 archive entries also include `*_meta.json` files and a
`manifest.json` in each analysis task directory. Keep those files alongside
the corresponding split files. These analysis inputs are JSON in the supplied
archive; the classification benchmark splits are CSV.

Pass a data root to the training scripts, for example:

```powershell
python pipeline/run_seeds.py `
  --task bace `
  --scaffold-root fintune_datas/moleculenet_random_scaffold `
  --seeds 42,52,62,72,82,92,102,112,122,132
```

The archive and extracted datasets are local-only and are not tracked by Git.
