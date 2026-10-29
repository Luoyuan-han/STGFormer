# Paper Table Protocols

This note records table-specific data splits and experimental conditions stated
in the manuscript. It does not infer settings that the paper leaves to
model-specific configurations.

| Table | Dataset or analysis | Split and table-specific protocol |
| --- | --- | --- |
| 1 | MoleculeNet classification | Randomized scaffold split; matched pretraining scale and fine-tuning protocols. Fine-tuning is limited to 50 epochs with validation-metric-based early stopping. |
| 2 | PyTDC classification | Official PyTDC scaffold split; matched pretraining scale and fine-tuning protocols. The same 50-epoch maximum and validation-metric-based early stopping as Table 1 apply. |
| 3 | MoleculeNet reference comparison with externally pretrained SMILES models | Official MoleculeNet scaffold split. External models have different pretraining corpora, objectives, capacities, and fine-tuning protocols, so the comparison is contextual rather than architecture-only. |
| 4 | PyTDC reference comparison with externally pretrained SMILES models | Official PyTDC scaffold split. The external-model comparison has the same protocol caveat as Table 3. |
| 5 | MoleculeNet serialization robustness | Reuses Table 1 molecule partitions; train on canonical SMILES without augmentation; evaluate on canonical and 10x enumerated SMILES for the same test molecules. |
| 6 | MoleculeNet serialization robustness | Reuses Table 1 molecule partitions; train on randomized SMILES with validation augmentation; evaluate on canonical and 10x enumerated SMILES for the same test molecules. |
| 7 | MoleculeNet serialization robustness | Reuses Table 1 molecule partitions; use 10x SMILES training augmentation with validation augmentation; evaluate on canonical and 10x enumerated SMILES for the same test molecules. |
| 8 | Pretraining-validation serialization stability | Retain 4,915 molecules with five valid, non-truncated serializations each. Average the ten pairwise cosine distances per molecule, then report mean and sample SD across molecules. |
| 9 | QM9 controlled structural-isomer analysis | 4,521 molecules in 285 groups, yielding 92,371 unordered within-group pairs; target is the HOMO-LUMO gap. Pairwise Delta-MAE is reported in Ha, alongside full test-set MAE across ten training runs. |
| 10 | COMPAS controlled structural-isomer analysis | 692 molecules in 158 groups, yielding 2,252 unordered within-group pairs; targets are HOMO and LUMO. Pairwise Delta-MAE is reported in eV, alongside full test-set MAE across ten training runs. |
| 11 | BRD4 molecular generation | Scaffold split of 6,367 training, 796 validation, and 796 test molecules. Generate 30,000 strings per pretrained model before fine-tuning and 5,000 per model after BRD4 fine-tuning; post-fine-tuning outputs supply the non-validity metrics. |

## Supporting tables

Tables S1-S4 use the BRD4 generation runs described for Table 11. Table S1
reports output counts, Table S2 scaffold exploration, Table S3 predictor-derived
statistics for the top 10% ranked by predicted pIC50, and Table S4 training-set
similarity and held-out coverage. Table S4 uses radius-2, 2,048-bit Morgan
fingerprints and reports Coverage@0.5.

## Repository data roots

The local directories corresponding to the paper's classification protocols
are:

| Paper tables | Repository data root |
| --- | --- |
| Table 1 | `fintune_datas/moleculenet_random_scaffold/` |
| Table 2 and Table 4 | `fintune_datas/pytdc_scaffold/` |
| Table 3 | `fintune_datas/moleculenet_scaffold/` |

Table 3 uses the official MoleculeNet scaffold split. It is distinct from the
randomized per-seed MoleculeNet partitions used by Table 1.
