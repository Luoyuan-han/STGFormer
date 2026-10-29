import csv
import os

from torch.utils.data import Dataset, IterableDataset, get_worker_info
from Tokenizer import GPTTokenizer,BertTokenizer
import torch
import ijson
from tqdm import tqdm
import torch.nn.functional as F
import random
from tool import shortest_path_matrix, sm2graph
from rdkit import Chem
def _safe_len(obj):
    try:
        return len(obj)
    except TypeError:
        return None

def _csv_value(row, key=None):
    if not row:
        return None
    if isinstance(row, dict):
        for name in (key, "smiles", "SMILES", "text", "input", "sm", "smi"):
            if name and name in row and row[name] is not None:
                return row[name]
        return next(iter(row.values()), None)
    return row[0]


def iter_data_file(path, key=None, limit=None):
    """Stream a CSV SMILES column or a legacy top-level JSON array."""
    path = str(path)
    if path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                first = next(reader)
            except StopIteration:
                return
            known_headers = {"smiles", "text", "input", "sm", "smi"}
            has_header = any(str(value).strip().lower() in known_headers for value in first)
            rows = reader if has_header else iter([first], reader.__next__)
            count = 0
            for row in rows:
                if limit is not None and count >= limit:
                    break
                value = _csv_value(row, key=key)
                if value is None or not str(value).strip():
                    continue
                count += 1
                yield value
        return

    with open(path, "r", encoding="utf-8") as handle:
        for idx, item in enumerate(ijson.items(handle, "item")):
            if limit is not None and idx >= limit:
                break
            if key is not None and isinstance(item, dict):
                yield item.get(key, item)
            else:
                yield item


def count_data_rows(path):
    return sum(1 for _ in iter_data_file(path))


def iter_json_array(path, key=None, limit=None):
    """Backward-compatible alias for the data-file stream reader."""
    yield from iter_data_file(path, key=key, limit=limit)

def tar_padding(seq,max_len):

    seq.append(2)
    seq.extend([0] * (max_len - len(seq)+1))

    return seq[1:max_len+1]
def prompt_tar(seq,max_len):

    seq.append(2)
    seq.extend([0]* (max_len - len(seq)))

    return seq[:max_len]
def dec_input_padding(seq,max_len):

    seq.append(2)
    seq.extend([0] * (max_len - len(seq)+1))

    return seq[:max_len]
def pro_dec_input_padding(seq,max_len):

    seq.append(2)
    seq.extend([0] * (max_len - len(seq)+1))

    return seq[:max_len-1]
def enc_input_padding(seq,max_len):

    seq.insert(0, 1)

    seq.extend([0] * (max_len - len(seq)))

    return seq[:max_len]

def _normalize_bert_item(item):
    if isinstance(item, dict):
        for key in ("text", "smiles", "sm", "smi", "input"):
            if key in item:
                return item[key], item.get("label"), item
        return item, item.get("label"), item
    if isinstance(item, (list, tuple)):
        if len(item) >= 2:
            meta = item[2] if len(item) > 2 and isinstance(item[2], dict) else None
            return item[0], item[1], meta
        if len(item) == 1:
            return item[0], None, None
    return item, None, None

def normalize_solvent_smiles(smiles):
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

def build_bert_pair_token_text(sm, meta, config, tokenizer):
    token_text = tokenizer.tokenize(sm)
    if token_text is None:
        return None, 0

    if getattr(config, "use_solvent", False) and isinstance(meta, dict) and meta.get("solvent"):
        solvent = normalize_solvent_smiles(meta.get("solvent"))
        solvent_tokens = tokenizer.tokenize(solvent) if solvent else None
        if solvent_tokens is None:
            return None, 0

        max_len = max(0, config.seq_len - 1)
        if len(token_text) + len(solvent_tokens) <= max_len:
            solute_tokens = token_text
            token_text = solute_tokens + solvent_tokens
        else:

            solvent_keep = min(len(solvent_tokens), max(1, max_len // 4))
            solute_keep = max(0, max_len - solvent_keep)
            solute_tokens = token_text[:solute_keep]
            token_text = solute_tokens + solvent_tokens[:solvent_keep]
        return token_text, len(solute_tokens)

    token_text = token_text[:max(0, config.seq_len - 1)]
    return token_text, len(token_text)

def build_bert_token_text(sm, meta, config, tokenizer):
    return build_bert_pair_token_text(sm, meta, config, tokenizer)[0]

def build_bert_graph_matrix(token_text, solute_len, config):
    def normalized_shortest_path(tokens):
        matrix = torch.tensor(shortest_path_matrix(sm2graph(tokens)), dtype=torch.float)
        return matrix / max(float(torch.max(matrix).item()), 1.0)

    if not getattr(config, "use_solvent", False) or solute_len >= len(token_text):
        return normalized_shortest_path(token_text)

    solute_tokens = token_text[:solute_len]
    solvent_tokens = token_text[solute_len:]
    n = len(token_text)
    matrix = torch.full((n, n), float("inf"), dtype=torch.float)
    if solute_tokens:
        matrix[:solute_len, :solute_len] = normalized_shortest_path(solute_tokens)
    if solvent_tokens:
        matrix[solute_len:n, solute_len:n] = normalized_shortest_path(solvent_tokens)
    return matrix

def _normalize_gpt_item(item, key="text"):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in (key, "text", "smiles", "SMILES", "sm", "smi", "input"):
            if k in item:
                return item[k]
        return None
    if isinstance(item, (list, tuple)) and len(item) > 0:
        return item[0]
    return None

def smiles_to_2d_canonical(smiles):
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

def _process_bert_sample(sample, config, tokenizer):
    sm, t, meta = _normalize_bert_item(sample)
    if sm is None:
        return None
    token_text, solute_len = build_bert_pair_token_text(sm, meta, config, tokenizer)
    if token_text is None:
        return None

    if len(token_text) == 0:
        return None
    tokens = tokenizer.sm2index(token_text)
    enc = enc_input_padding(tokens[:], config.seq_len)

    if not config.MLM:
        if t is None:
            return None
        if not config.reg:
            tar = torch.tensor([int(i) for i in t[:]], dtype=torch.long)
        elif config.mutcls:
            tar = torch.tensor([i for i in t[:]], dtype=torch.float)
        else:
            tar = torch.tensor([i for i in t[:]], dtype=torch.float)
    else:
        tar = enc[:]

    seq_len = len(tokens)
    s_g = build_bert_graph_matrix(token_text, solute_len, config)

    if not config.MLM:
        return enc, tar, sm, s_g, seq_len

    input_tensor = torch.tensor([i for i in enc], dtype=torch.long)
    mask_token_id = 3
    input_tensor_mlm = input_tensor.clone()
    probability_matrix = torch.full(input_tensor.shape, 0.15, device=input_tensor.device)
    special_tokens_mask = ((input_tensor == 0) | (input_tensor == 1))
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    if not masked_indices.any():
        candidates = torch.nonzero(~special_tokens_mask, as_tuple=False).flatten()
        if candidates.numel() > 0:
            choice = candidates[torch.randint(candidates.numel(), (1,)).item()]
            masked_indices[choice] = True
    input_tensor_mlm[masked_indices] = mask_token_id
    input_tensor[~masked_indices] = 0

    s_g = F.pad(
        s_g,
        [1, config.seq_len - seq_len - 1, 1, config.seq_len - 1 - seq_len],
        mode="constant",
        value=0,
    )
    s_g = s_g.expand([config.num_heads, s_g.size(-1), s_g.size(-1)])

    return input_tensor_mlm, seq_len, s_g, input_tensor
from concurrent.futures import ThreadPoolExecutor

class MoleculeDataset(Dataset):
    def __init__(self,smiles_iter,config,train=False):
        self.tokenizer = GPTTokenizer()
        self.input_sm = []
        self.tar_tensor = []
        self.sm = []
        self.config = config
        self.s_g = []
        self.seq_lens = []

        def process_one(sm):

            if self.tokenizer.tokenize(sm) is not None:

                tokens = self.tokenizer.sm2index(self.tokenizer.tokenize(sm))

                if max(tokens) >= self.config.vocab_size:
                    return None, None, None, None, None

                enc = dec_input_padding(tokens[:], self.config.seq_len)
                if self.config.scaffold_lead:
                    tar = torch.tensor(prompt_tar(tokens[:], self.config.seq_len))
                else:
                    tar = torch.tensor(tar_padding(tokens[:], self.config.seq_len))
                seq_len = len(tokens)

                s_gs=[]

                tokens=self.tokenizer.tokenize(sm)
                n=len(tokens)
                max_value=1
                for t in range(1,n+1):
                    s_g_raw = shortest_path_matrix(sm2graph(tokens[:t]), t - 1)
                    if s_g_raw is None:
                        return None, None, None, None, None
                    s_g_=torch.tensor(s_g_raw)

                    max_value=max(torch.max(s_g_).item(), max_value)

                    s_g_ = s_g_ * (1 /max_value)
                    if self.config.scaffold_lead:
                        s_g_ = F.pad(s_g_, [1, n - t-1],
                                     mode='constant',
                                     value=0)
                    else:
                        s_g_ = F.pad(s_g_, [0, n - t],
                                     mode='constant',
                                     value=0)
                    s_gs.append(s_g_)

                s_g = torch.stack(s_gs)

                return enc, tar, sm, s_g, seq_len
            else:
                return None, None, None, None, None

        with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2)//2)) as executor:
            results = executor.map(process_one, smiles_iter, chunksize=500)

            for enc, tar, sm, s_g, seq_len in tqdm(results, total=len(smiles_iter)):
                if enc is not None:
                    self.input_sm.append(enc)
                    self.tar_tensor.append(tar)
                    self.sm.append(sm)
                    self.seq_lens.append(seq_len)
                    self.s_g.append(s_g)
                else:
                    pass

    def get_dict(self):
        return self.tokenizer.get_dict()

    def max_len(self):

        return max(len(item) for item in self.input_sm)

    def get_smiles(self, ind):
        return self.input_sm[ind]

    def __getitem__(self, idx):
        input_batch, tar_tensor, sm, seq_len, s_g =\
            self.input_sm[idx], self.tar_tensor[idx], self.sm[idx], self.seq_lens[idx], self.s_g[idx]
        if self.config.scaffold_lead:
            s_g = F.pad(s_g, [1, self.config.seq_len - seq_len, 1, self.config.seq_len - seq_len], mode='constant',
                        value=0)
        else:
            s_g = F.pad(s_g, [0, self.config.seq_len - seq_len, 0, self.config.seq_len - seq_len], mode='constant',
                        value=0)

        s_g = s_g.expand([self.config.num_heads, s_g.size(-1), s_g.size(-1)])
        return input_batch, tar_tensor, sm, seq_len, s_g

    def __len__(self):
        return len(self.input_sm)

class GPTStreamDataset(IterableDataset):
    def __init__(self, source, config, key="text", limit=None):
        super().__init__()
        self.source = source
        self.config = config
        self.key = key
        self.limit = limit
        self.tokenizer = GPTTokenizer()
        self._length = None

    def _iter_source(self, worker_id=0, num_workers=1):
        if isinstance(self.source, str):
            for idx, item in enumerate(iter_data_file(self.source, key=self.key, limit=self.limit)):
                if idx % num_workers == worker_id:
                    yield item
            return
        else:
            iterator = iter(self.source)

        for idx, item in enumerate(iterator):
            if self.limit is not None and idx >= self.limit:
                break
            if idx % num_workers == worker_id:
                yield item

    def _process_one(self, item):
        sm = _normalize_gpt_item(item, key=self.key)
        if not isinstance(sm, str) or len(sm) == 0:
            return None
        if getattr(self.config, "gpt_to_2d", False):
            sm = smiles_to_2d_canonical(sm)
            if sm is None:
                return None

        token_text = self.tokenizer.tokenize(sm)
        if token_text is None:
            return None

        token_ids = self.tokenizer.sm2index(token_text[:])
        if len(token_ids) == 0 or max(token_ids) >= self.config.vocab_size:
            return None

        enc = dec_input_padding(token_ids[:], self.config.seq_len)
        tar = torch.tensor(tar_padding(token_ids[:], self.config.seq_len), dtype=torch.long)

        graph_tokens = token_text[:self.config.seq_len]
        seq_len = len(graph_tokens)
        if seq_len <= 0:
            return None

        s_gs = []
        max_value = 1
        for t in range(1, seq_len + 1):
            s_g_raw = shortest_path_matrix(sm2graph(graph_tokens[:t]), t - 1)
            if s_g_raw is None:
                return None
            s_g_ = torch.tensor(s_g_raw)
            max_value = max(torch.max(s_g_).item(), max_value)
            s_g_ = s_g_ * (1 / max_value)
            s_g_ = F.pad(s_g_, [0, seq_len - t], mode='constant', value=0)
            s_gs.append(s_g_)

        s_g = torch.stack(s_gs)
        s_g = F.pad(s_g, [0, self.config.seq_len - seq_len, 0, self.config.seq_len - seq_len], mode='constant', value=0)
        s_g = s_g.expand([self.config.num_heads, s_g.size(-1), s_g.size(-1)])
        return enc, tar, sm, seq_len, s_g

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        for item in self._iter_source(worker_id=worker_id, num_workers=num_workers):
            sample = self._process_one(item)
            if sample is not None:
                yield sample

    def __len__(self):
        if self._length is not None:
            return self._length
        if not isinstance(self.source, str):
            raise TypeError("Length is only available for file-backed GPTStreamDataset.")

        count = count_data_rows(self.source)
        self._length = count if self.limit is None else min(count, self.limit)
        return self._length
class GPTDataset(Dataset):
    def __init__(self,smiles_iter,config,train=False):
        self.tokenizer = GPTTokenizer()
        self.input_sm = []
        self.tar_tensor = []
        self.sm = []
        self.config = config
        self.s_g = []
        self.seq_lens = []

        def process_one(sm):

            if self.tokenizer.tokenize(sm) is not None:

                tokens = self.tokenizer.sm2index(self.tokenizer.tokenize(sm))

                enc = dec_input_padding(tokens[:], self.config.seq_len)
                tar = torch.tensor(tar_padding(tokens[:], self.config.seq_len))
                seq_len = len(tokens)

                return enc, tar, sm, None, seq_len
            else:
                return None, None, None, None, None

        with ThreadPoolExecutor(max_workers=os.cpu_count()//2) as executor:
            results = executor.map(process_one, smiles_iter, chunksize=500)

            for enc, tar, sm, s_g, seq_len in tqdm(results, total=len(smiles_iter)):
                if enc is not None:
                    self.input_sm.append(enc)
                    self.tar_tensor.append(tar)
                    self.sm.append(sm)
                    self.seq_lens.append(seq_len)
                    self.s_g.append(s_g)
                else:
                    pass

    def get_dict(self):
        return self.tokenizer.get_dict()

    def max_len(self):

        return max(len(item) for item in self.input_sm)

    def get_smiles(self, ind):
        return self.input_sm[ind]

    def __getitem__(self, idx):
        input_batch, tar_tensor, sm, seq_len, s_g =\
            self.input_sm[idx], self.tar_tensor[idx], self.sm[idx], self.seq_lens[idx], self.s_g[idx]

        return input_batch, tar_tensor, sm, seq_len, s_g

    def __len__(self):
        return len(self.input_sm)

class BertDataset(Dataset):
    def __init__(self,smiles_iter,config):
        super(BertDataset, self).__init__()
        self.tokenizer = BertTokenizer()
        self.input_sm = []
        self.tar_tensor = []
        self.sm = []
        self.config = config
        self.s_g=[]
        self.seq_lens=[]
        self.z_sc = False
        self.x_ = None
        self.seigema = None

        def process_one(sm):

            meta = None
            if not self.config.MLM:
                meta = sm[2] if len(sm) > 2 and isinstance(sm[2], dict) else None
                sm,t=sm[0],sm[1]
            token_text, solute_len = build_bert_pair_token_text(sm, meta, self.config, self.tokenizer)
            if token_text is not None:

                if len(token_text) == 0:
                    return None, None, None, None, None
                tokens = self.tokenizer.sm2index(token_text)

                enc = enc_input_padding(tokens[:], self.config.seq_len)
                if not self.config.MLM:
                    if not self.config.reg:
                        tar = torch.tensor([int(i) for i in t[:]], dtype=torch.long)
                    elif self.config.mutcls:
                        tar = torch.tensor([i for i in t[:]], dtype=torch.float)
                    else:
                        tar = torch.tensor([i for i in t[:]], dtype=torch.float)
                else:
                    tar=enc[:]
                seq_len=len(tokens)
                s_g = build_bert_graph_matrix(token_text, solute_len, self.config)

                return enc, tar, sm,s_g,seq_len
            else:
                return None,None,None,None,None
        count = 0
        total = _safe_len(smiles_iter)

        with ThreadPoolExecutor(max_workers=32) as executor:
            for enc, tar, sm, s_g, seq_len in tqdm(
                executor.map(process_one, smiles_iter, chunksize=100),
                total=total,
            ):
                if enc is not None:

                    self.input_sm.append(enc)
                    self.tar_tensor.append(tar)
                    self.sm.append(sm)
                    self.seq_lens.append(seq_len)
                    self.s_g.append(s_g)
                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} items", end="\r")
    def get_dict(self):
        return self.tokenizer.get_dict()
    def max_len(self):

        return max(len(item) for item in self.input_sm)
    def get_smiles(self,ind):

        return self.input_sm[ind]

    def z_score(self):
        self.z_sc = True
        if self.x_ is None or self.seigema is None:
            tars = torch.stack(self.tar_tensor).float()
            finite = torch.isfinite(tars)
            safe = torch.where(finite, tars, torch.zeros_like(tars))
            count = finite.sum(dim=0).clamp(min=1)
            self.x_ = safe.sum(dim=0) / count
            var = torch.where(finite, (tars - self.x_) ** 2, torch.zeros_like(tars)).sum(dim=0) / count
            self.seigema = torch.sqrt(var)
            self.seigema = torch.where(self.seigema < 1e-12, torch.ones_like(self.seigema), self.seigema)
        return self.x_, self.seigema

    def set_z_score(self, x_, seigema):
        self.z_sc = True
        self.x_ = x_.detach().cpu().float() if torch.is_tensor(x_) else torch.tensor(x_, dtype=torch.float)
        self.seigema = seigema.detach().cpu().float() if torch.is_tensor(seigema) else torch.tensor(seigema, dtype=torch.float)
        self.seigema = torch.where(self.seigema < 1e-12, torch.ones_like(self.seigema), self.seigema)
        return self.x_, self.seigema

    def __getitem__(self, idx):
        input_batch,tar_tensor,sm,seq_len,s_g=\
            self.input_sm[idx], self.tar_tensor[idx], self.sm[idx], self.seq_lens[idx], self.s_g[idx]
        if self.z_sc:
            tar_tensor = (tar_tensor - self.x_) / self.seigema
        input_tensor = torch.tensor([i for i in input_batch], dtype=torch.long)
        s_g = F.pad(s_g, [1, self.config.seq_len - seq_len - 1, 1, self.config.seq_len - 1 - seq_len], mode='constant',
                    value=0)
        s_g = s_g.expand([self.config.num_heads, s_g.size(-1), s_g.size(-1)])

        if not self.config.MLM:
            return input_batch,tar_tensor,sm,seq_len,s_g

        mask_token_id = 3

        input_tensor_mlm = input_tensor.clone()

        probability_matrix = torch.full(input_tensor.shape, 0.15, device=input_tensor.device)
        special_tokens_mask = ((input_tensor == 0) | (input_tensor == 1))
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        if not masked_indices.any():
            candidates = torch.nonzero(~special_tokens_mask, as_tuple=False).flatten()
            if candidates.numel() > 0:
                choice = candidates[torch.randint(candidates.numel(), (1,)).item()]
                masked_indices[choice] = True

        input_tensor_mlm[masked_indices] = mask_token_id

        input_tensor[~masked_indices] = 0
        return input_tensor_mlm, seq_len, s_g,input_tensor
    def __len__(self):
        return len(self.input_sm)

class BertStreamDataset(IterableDataset):
    def __init__(self, source, config, key="text", limit=None):
        super().__init__()
        self.source = source
        self.config = config
        self.key = key
        self.limit = limit
        self._length = None

    def _iter_source(self, worker_id=0, num_workers=1):
        if isinstance(self.source, str):
            for idx, item in enumerate(iter_data_file(self.source, key=self.key, limit=self.limit)):
                if idx % num_workers == worker_id:
                    yield item
            return

        for idx, item in enumerate(self.source):
            if self.limit is not None and idx >= self.limit:
                break
            if idx % num_workers != worker_id:
                continue
            if self.key is not None and isinstance(item, dict):
                yield item.get(self.key, item)
            else:
                yield item

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        tokenizer = BertTokenizer()
        for sample in self._iter_source(worker_id, num_workers):
            out = _process_bert_sample(sample, self.config, tokenizer)
            if out is not None:
                yield out

    def __len__(self):
        if self.limit is not None:
            return self.limit
        if self._length is not None:
            return self._length
        if not isinstance(self.source, str):
            raise TypeError("Length is only available for file-backed BertStreamDataset.")

        self._length = count_data_rows(self.source)
        return self._length

class RandomBertDataset(Dataset):
    def __init__(self, smiles_iter, config):
        super(RandomBertDataset, self).__init__()

        self.tokenizer = BertTokenizer()
        self.input_sm = {}
        self.tar_tensor = {}
        self.sm = {}
        self.config = config
        self.s_g = {}
        self.seq_lens = {}
        con_smiles=set()

        self.z_sc=False
        self.x_ = None
        self.seigema = None
        def process_one(sm):

            meta = None
            if not self.config.MLM:
                meta = sm[2] if len(sm) > 2 and isinstance(sm[2], dict) else None
                sm, t = sm[0], sm[1]

            token_text, solute_len = build_bert_pair_token_text(sm, meta, self.config, self.tokenizer)
            if token_text is not None:

                if len(token_text) == 0:
                    return None, None, None, None, None
                tokens = self.tokenizer.sm2index(token_text)

                enc = enc_input_padding(tokens[:], self.config.seq_len)
                if not self.config.MLM:
                    if not self.config.reg:
                        tar = torch.tensor([int(i) for i in t[:]], dtype=torch.long)
                    elif self.config.mutcls:
                        tar = torch.tensor([i for i in t[:]], dtype=torch.float)
                    else:
                        tar = torch.tensor([i for i in t[:]], dtype=torch.float)
                else:
                    tar = enc[:]
                seq_len = len(tokens)
                s_g = build_bert_graph_matrix(token_text, solute_len, self.config)

                return enc, tar, sm, s_g, seq_len
            else:
                return None, None, None, None, None

        count = 0

        with ThreadPoolExecutor(max_workers=32) as executor:

            for enc, tar, sm, s_g, seq_len in executor.map(process_one, smiles_iter, chunksize=100):
                if enc is not None:
                    mol = Chem.MolFromSmiles(sm)
                    if mol is None:
                        continue

                    c_x = Chem.MolToSmiles(mol)
                    if c_x in con_smiles:
                        self.input_sm[c_x].append(enc)
                        self.tar_tensor[c_x].append(tar)
                        self.sm[c_x].append(sm)
                        self.seq_lens[c_x].append(seq_len)
                        self.s_g[c_x].append(s_g)
                    else:
                        con_smiles.add(c_x)
                        self.input_sm[c_x] = [enc]
                        self.tar_tensor[c_x] = [tar]
                        self.sm[c_x] = [sm]
                        self.seq_lens[c_x] = [seq_len]
                        self.s_g[c_x] = [s_g]

                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} items", end="\r")
                else:
                    pass
            self.iter=list(self.input_sm.keys())
    def get_dict(self):
        return self.tokenizer.get_dict()

    def max_len(self):
        return
        return max(len(item) for item in self.input_sm)
    def z_score(self):
        self.z_sc=True
        if self.x_ is None or self.seigema is None:
            tars = torch.stack([i[0] for i in self.tar_tensor.values()]).float()
            finite = torch.isfinite(tars)
            safe = torch.where(finite, tars, torch.zeros_like(tars))
            count = finite.sum(dim=0).clamp(min=1)
            self.x_ = safe.sum(dim=0) / count
            var = torch.where(finite, (tars - self.x_) ** 2, torch.zeros_like(tars)).sum(dim=0) / count
            self.seigema = torch.sqrt(var)
            self.seigema = torch.where(self.seigema < 1e-12, torch.ones_like(self.seigema), self.seigema)

        return self.x_,self.seigema

    def set_z_score(self, x_, seigema):
        self.z_sc = True
        self.x_ = x_.detach().cpu().float() if torch.is_tensor(x_) else torch.tensor(x_, dtype=torch.float)
        self.seigema = seigema.detach().cpu().float() if torch.is_tensor(seigema) else torch.tensor(seigema, dtype=torch.float)
        self.seigema = torch.where(self.seigema < 1e-12, torch.ones_like(self.seigema), self.seigema)
        return self.x_, self.seigema
    def get_smiles(self, ind):

        return self.input_sm[ind]

    def __getitem__(self, idx):

        id_key = self.iter[idx]
        random_number = random.randint(0, len(self.input_sm[id_key])-1)
        input_batch, tar_tensor, sm, seq_len, s_g=self.input_sm[id_key][random_number], self.tar_tensor[id_key][random_number], self.sm[id_key][random_number], self.seq_lens[id_key][random_number], self.s_g[id_key][random_number]
        if self.z_sc:

            tar_tensor=(tar_tensor-self.x_)/self.seigema
        input_tensor = torch.tensor([i for i in input_batch], dtype=torch.long)
        s_g = F.pad(s_g, [1, self.config.seq_len - seq_len - 1, 1, self.config.seq_len - 1 - seq_len], mode='constant',
                    value=0)
        s_g = s_g.expand([self.config.num_heads, s_g.size(-1), s_g.size(-1)])

        if not self.config.MLM:
            return input_batch, tar_tensor, sm, seq_len, s_g

        mask_token_id = 3

        input_tensor_mlm = input_tensor.clone()

        probability_matrix = torch.full(input_tensor.shape, 0.15, device=input_tensor.device)
        special_tokens_mask = ((input_tensor == 0) | (input_tensor == 1))
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        if not masked_indices.any():
            candidates = torch.nonzero(~special_tokens_mask, as_tuple=False).flatten()
            if candidates.numel() > 0:
                choice = candidates[torch.randint(candidates.numel(), (1,)).item()]
                masked_indices[choice] = True

        input_tensor_mlm[masked_indices] = mask_token_id

        input_tensor[~masked_indices] = 0
        return input_tensor_mlm, seq_len, s_g, input_tensor

    def __len__(self):
        return len(self.input_sm)
from config import BaseConfig
torch.set_printoptions(threshold=1000000)

if __name__ == '__main__':

    config = BaseConfig()
    example_path = os.environ.get('STGFORMER_EXAMPLE_DATA', 'data/smoke/train.csv')
    if os.path.isfile(example_path):
        from data_io import read_dataset
        dataset = MoleculeDataset(read_dataset(example_path), config=config)
        print(dataset.z_score())
    else:
        print(f'Example data not found: {example_path}. Set STGFORMER_EXAMPLE_DATA to a split CSV.')
