# Smoke fixtures

Keep tiny, redistributable fixtures here for CI and for checking that the
tokenizer, STG construction, dataset loader and CLI wiring work without
downloading the full datasets. Fixtures use the public CSV schema with
`smiles` and `tags` columns; labels are stored as JSON-style arrays in `tags`.

The full benchmark data is not copied into this directory.
