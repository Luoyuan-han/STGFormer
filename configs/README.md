# Configuration

All model and training defaults live in the root `config.py`.

- `BertConfig`: BERT/STGFormer MLM and downstream prediction.
- `GPTConfig`: autoregressive molecular generation and sampling.
- `BaseConfig`: shared defaults used by low-level collate helpers.

Use the shared helpers instead of assigning a long list of fields in a new
entry point:

```python
from config import BertConfig, configure_finetune

config = configure_finetune(
    BertConfig(), seq_len=256, batch_size=32, epochs=50, lr=3e-5,
    pooling="cls", early_stop_metric="loss", use_lr_scheduler=True,
    label_dim=1,
)
```

`gen_trian_epoch` is retained as a historical field because the existing
trainer reads that name. `configure_pretraining` and `configure_generation`
apply the MLM and molecular-generation protocols.
