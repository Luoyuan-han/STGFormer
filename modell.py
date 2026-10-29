import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
def rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device=None, dtype=None):
    assert head_dim % 2 == 0, "head_dim must be even"
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    ang = torch.outer(t, inv_freq).repeat_interleave(2, dim=-1)
    cos = ang.cos()[None, None, :, :]
    sin = ang.sin()[None, None, :, :]
    if dtype: cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin

def rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):

    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)

    while cos.dim() < x.dim(): cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    return x * cos + x_rot * sin

def get_rope_cache(module: nn.Module, seq_len: int, head_dim: int, device, dtype):

    cos = getattr(module, "rope_cos", None)
    sin = getattr(module, "rope_sin", None)
    if cos is None or sin is None or cos.size(-2) < seq_len or cos.size(-1) != head_dim:
        cos, sin = rope_cache(seq_len, head_dim, device=device, dtype=dtype)
        return cos, sin
    return cos[:, :, :seq_len, :].to(device=device, dtype=dtype), sin[:, :, :seq_len, :].to(device=device, dtype=dtype)

def generate_padding_mask(input_ids, pad_token_id=0):

    padding_mask = (input_ids == pad_token_id).to(torch.long)

    return padding_mask

class GraphSelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads,
                 dropout,config):
        super().__init__()
        assert input_size % num_heads == 0

        self.config = config
        self.key = nn.Linear(input_size, hidden_size)
        self.query = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.proj = nn.Linear(input_size, hidden_size)

        self.n_head = num_heads
        if self.config.rope:

            rope_len = getattr(config, 'seq_len', 256)
            head_dim = hidden_size // num_heads
            cos, sin = rope_cache(rope_len, head_dim, device=None, dtype=None)
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)

        self.a=nn.Parameter(torch.tensor([0.5 for _ in range(config.num_heads)]), requires_grad=True)

    def forward(self, Q, K, V,seq_lens,g_attn, mask=None):
        B, T, C = Q.size()

        k = self.key(K).view(B, K.size(1), self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(Q).view(B, Q.size(1), self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(V).view(B, V.size(1), self.n_head, C // self.n_head).transpose(1, 2)
        if self.config.rope:
            cos, sin = get_rope_cache(self, T, C // self.n_head, q.device, q.dtype)
            q = rope_apply(q, cos, sin)
            k = rope_apply(k, cos, sin)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att = att.masked_fill(mask[:, :, :T, :T] == 1, -1e4)
        att_with_G=att.clone()

        if self.config.stg:

            a=F.softplus(self.a.view(1,-1,1,1))
            g_attn = g_attn[:, :, :T, :T].to(device=att_with_G.device, dtype=att_with_G.dtype)
            inf_mask = torch.isinf(g_attn)

            if inf_mask.any():

                finite_g = torch.where(inf_mask, torch.zeros_like(g_attn), g_attn)
                finite_g = torch.nan_to_num(finite_g, nan=0.0, posinf=0.0, neginf=0.0)
                att_with_G = att_with_G - finite_g * a
                att_with_G = att_with_G.masked_fill(inf_mask, -1e4)
            else:
                g_attn = torch.nan_to_num(g_attn, nan=0.0, posinf=0.0, neginf=0.0)

                att_with_G = att_with_G-g_attn*a

        att = F.softmax(att_with_G, dim=-1)

        att = self.attn_drop(att)

        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))

        return y, att
class GraphSelfAttentionGPT(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads,
                 dropout,config):
        super().__init__()
        assert input_size % num_heads == 0

        self.config = config
        self.key = nn.Linear(input_size, hidden_size)
        self.query = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.proj = nn.Linear(input_size, hidden_size)

        self.n_head = num_heads

        self.a=nn.Parameter(torch.tensor([0.5 for _ in range(config.num_heads)]), requires_grad=True)

    def forward(self, Q, K, V,g_attn,seq_lens=None, mask=None):
        B, T, C = Q.size()

        k = self.key(K).view(B, K.size(1), self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(Q).view(B, Q.size(1), self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(V).view(B, V.size(1), self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if mask is not None:
            att = att.masked_fill(mask[:, :, :T, :T] == 1, -1e4)

        att_with_G=att.clone()
        if self.config.stg:

            a=F.softplus(self.a.view(1,-1,1,1))
            g_attn = g_attn[:, :, :T, :T].to(device=att_with_G.device, dtype=att_with_G.dtype)
            inf_mask = torch.isinf(g_attn)
            if inf_mask.any():

                finite_g = torch.where(inf_mask, torch.zeros_like(g_attn), g_attn)
                finite_g = torch.nan_to_num(finite_g, nan=0.0, posinf=0.0, neginf=0.0)
                att_with_G = att_with_G - finite_g * a
                att_with_G = att_with_G.masked_fill(inf_mask, -1e4)
            else:
                g_attn = torch.nan_to_num(g_attn, nan=0.0, posinf=0.0, neginf=0.0)
                att_with_G = att_with_G-g_attn*a

        att = F.softmax(att_with_G, dim=-1)

        att = self.attn_drop(att)

        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))

        return y, att
class PositionwiseFeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.d_model, config.d_model*config.ffn, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.d_model*config.ffn, config.d_model, bias=config.bias))

    def forward(self, inputs):
        return self.fc(inputs)

class GraphDecoderLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.dec_attn = GraphSelfAttentionGPT(config.d_model, config.d_model, config.num_heads,config.dropout,config)
        self.mlp = PositionwiseFeedForward(config)

    def forward(self, x,s_g,seq_lens,mask):

        ln_x=self.ln1(x)
        y, dec_attn = self.dec_attn(ln_x,ln_x,ln_x,s_g,seq_lens, mask)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, dec_attn

class Decoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.tgt_emb = nn.Embedding(config.vocab_size, config.d_model)

        if config.scaffold_lead:
            self.prompt_emb=nn.LSTM(input_size=768, hidden_size=768, num_layers=2,
               batch_first=True, bidirectional=False, dropout=0.2)

        self.layers = nn.ModuleList([GraphDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.fc_out = nn.Linear(config.d_model, config.vocab_size)

        if self.config.learnable_pos:
            self.pos=nn.Parameter(torch.normal(0,0.2,[1,config.seq_len,config.d_model]))
    def forward(self, dec_inputs, s_g=None, seq_lens=None, prompt=None):

        device = dec_inputs.device
        dec_outputs = self.tgt_emb(dec_inputs)

        if self.config.scaffold_lead:
            prompt=self.tgt_emb(prompt)
            max_len = prompt.size(1)
            lengths_cpu = torch.as_tensor(seq_lens, dtype=torch.long, device='cpu')
            prompt = pack_padded_sequence(
                prompt,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False
            )
            prompt,(_,_)=self.prompt_emb(prompt)
            prompt, _ = pad_packed_sequence(
                prompt,
                batch_first=True,
                total_length=max_len
            )
            batch_size = len(seq_lens)
            index = torch.LongTensor(seq_lens).cpu() - 1
            last_hidden_states = prompt[torch.arange(batch_size), index, :]

            dec_outputs=torch.cat([last_hidden_states.unsqueeze(1),dec_outputs],dim=1)
            if self.config.learnable_pos:
                dec_outputs = dec_outputs + self.pos[:, :dec_outputs.size(1), :]

            seq_len = dec_outputs.size(1)
            batch_size = dec_inputs.size(0)

            attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), 1)[
                None, None]
            seq_mask = generate_padding_mask(torch.cat([torch.ones([batch_size,1],device=device),dec_inputs.detach()],dim=-1))[:, None, None, :]
        else:
            if self.config.learnable_pos:
                dec_outputs = dec_outputs + self.pos[:, :dec_outputs.size(1), :]

            seq_len = dec_outputs.size(1)
            batch_size = dec_inputs.size(0)

            attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), 1)[
                None, None]
            seq_mask = generate_padding_mask(dec_inputs)[:, None, None, :]

        mask = attn_mask | seq_mask

        attn_save=[]
        for layer in self.layers:

            dec_outputs, attn_score = layer(dec_outputs, s_g,seq_lens,mask)

            attn_save.append(attn_score)

        logits = self.fc_out(dec_outputs)

        return logits,attn_save
class EncoderLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.dec_attn = GraphSelfAttention(config.d_model, config.d_model, config.num_heads,config.dropout,config)
        self.mlp = PositionwiseFeedForward(config)

    def forward(self, x,s_g,seq_lens,mask):

        ln_x=self.ln1(x)
        y, dec_attn = self.dec_attn(ln_x,ln_x,ln_x,seq_lens,s_g, mask)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, dec_attn

def pool_sequence_outputs(dec_outputs, enc_inputs, pooling='cls'):
    if pooling == 'mean':
        valid_mask = ((enc_inputs != 0) & (enc_inputs != 1)).unsqueeze(-1).float()
        denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (dec_outputs * valid_mask).sum(dim=1, keepdim=True) / denom
    return dec_outputs[:, :1, :]

class Encoder(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.config = config
        self.tgt_emb = nn.Embedding(config.vocab_size, config.d_model)
        if config.learnable_pos:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.seq_len, config.d_model))
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.decoder_layers)])
        if not config.MLM:
            self.fc_out = nn.Linear(config.d_model, self.config.tgt_size)
        else:
            self.tokens=nn.Linear(config.d_model,config.tgt_size)
        self.mlp = PositionwiseFeedForward(config)
    def forward(self, enc_inputs,seq_lens,s_g,enc_mask=None):
        enc_outputs = self.tgt_emb(enc_inputs)

        seq_len = enc_outputs.size(1)
        if self.config.learnable_pos:
            enc_outputs=enc_outputs+self.pos_emb[:,:seq_len,:]
        seq_mask = generate_padding_mask(enc_inputs).unsqueeze(1).unsqueeze(-2)

        dec_outputs = self.mlp(enc_outputs)
        attn_layers=[]

        for layer in self.layers:
            dec_outputs, attn_score = layer(dec_outputs, s_g, seq_lens, seq_mask)
            attn_layers.append(attn_score)

        if not self.config.MLM:
            pooled = pool_sequence_outputs(
                dec_outputs,
                enc_inputs,
                getattr(self.config, 'pooling', 'cls')
            )
            logits = self.fc_out(pooled)
            return logits, pooled,attn_layers
        else:
            logits = self.tokens(dec_outputs)
            return logits,None

class BertModel(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.Encoder = Encoder(config)

    def forward(self, enc_input, seq_lens, s_g):

        if ((torch.is_tensor(seq_lens) and not torch.is_tensor(s_g)) or
                (torch.is_tensor(seq_lens) and torch.is_tensor(s_g) and
                 seq_lens.ndim >= 3 and s_g.ndim <= 1)):
            seq_lens, s_g = s_g, seq_lens
        return self.Encoder(enc_input, seq_lens, s_g)

    def encode(self, enc_input, seq_lens, s_g):

        if ((torch.is_tensor(seq_lens) and not torch.is_tensor(s_g)) or
                (torch.is_tensor(seq_lens) and torch.is_tensor(s_g) and
                 seq_lens.ndim >= 3 and s_g.ndim <= 1)):
            seq_lens, s_g = s_g, seq_lens
        return self.Encoder(enc_input, seq_lens, s_g)[1]

class GPTModel(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.Decoder = Decoder(config)
    def forward(self, dec_inputs,s_g=None,seq_lens=None,prompt=None):

        dec_outputs,attn = self.Decoder(dec_inputs,s_g,seq_lens,prompt)

        return dec_outputs,attn
