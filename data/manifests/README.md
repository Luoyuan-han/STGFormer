# Data manifests

Put versioned, reviewable manifests here. A manifest should record at least:

- dataset/task name and protocol (`fixed_scaffold` or `random_scaffold`);
- source URL and source release/version;
- RDKit and preprocessing versions;
- split seed(s), split ratio and cleaning rules;
- relative paths to `train.csv`, `valid.csv`, `test.csv`;
- row counts and SHA256 checksums;
- whether a file is an original split or a `*_enforced` random-SMILES expansion.

Large CSV data files themselves are ignored by the root `.gitignore`. Legacy
JSON files may be listed as migration provenance, but CSV is the runtime input.
