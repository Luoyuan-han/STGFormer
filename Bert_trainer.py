import json
import random
from modell import GPTModel
from Tokenizer import GPTTokenizer, BertTokenizer
from torch.utils.data import DataLoader, IterableDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from matplotlib import pyplot as plt
from config import BaseConfig
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import jensenshannon
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import os
import math
from pathlib import Path
from Dataset import BertDataset
from sklearn.manifold import TSNE
from modell import BertModel
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tool import shortest_path_matrix, sm2graph
from functools import partial
def expand_pos_embed_1d(pos_old, L_new):

    if pos_old.dim() == 2:
        pos_old = pos_old.unsqueeze(0)

    pos = pos_old.permute(0, 2, 1)
    pos_interp = F.interpolate(pos, size=L_new, mode='linear', align_corners=False)
    pos_new = pos_interp.permute(0, 2, 1)
    return pos_new

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mae = mean_absolute_error(y_true, y_pred)

    r2 = r2_score(y_true, y_pred)

    return rmse, mae, r2
def calculate_auc_per_class(y_true, y_pred, num_classes):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true.shape) == 1 or y_true.shape[1] == 1:
        y_true = label_binarize(y_true.reshape(-1), classes=np.arange(num_classes))

    auc_scores = []
    for i in range(num_classes):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        mask = np.isfinite(true_i) & np.isfinite(pred_i)
        if mask.sum() > 0 and len(np.unique(true_i[mask])) == 2:
            auc_scores.append(roc_auc_score(true_i[mask], pred_i[mask]))

    return auc_scores

def chem_collate_fn(batch, config=None):
    config = config or BaseConfig()
    input_batch, tar_tensor, sm, seq_len, s_g = zip(*batch)

    input_tensor = torch.tensor([i for i in input_batch], dtype=torch.long)
    if config.MLM:
        tar_tensor = torch.tensor([i for i in tar_tensor], dtype=torch.long)
    elif config.reg:
        tar_tensor = torch.stack([i for i in tar_tensor]).float()
    else:
        tar_tensor = torch.stack([i for i in tar_tensor]).long()
    sm = [i for i in sm]
    seq_lens = [i for i in seq_len]
    s_g = torch.stack([i for i in s_g])
    return input_tensor,tar_tensor,sm,seq_lens,s_g
def gpt_collate_fn(batch, config=None):
    config = config or BaseConfig()

    if config.scaffold_lead:
        input_batch, tar_tensor, sm, seq_len, s_g ,prompt= zip(*batch)

        input_tensor = torch.tensor([i for i in input_batch], dtype=torch.long)

        tar_tensor = torch.stack([i for i in tar_tensor]).long()
        sm = [i for i in sm]
        seq_lens = [i for i in seq_len]
        s_g = torch.stack([i for i in s_g])
        prompt = torch.tensor([i for i in prompt]).long()
        return input_tensor,tar_tensor,sm,seq_lens,s_g,prompt
    else:
        input_batch, tar_tensor, sm, seq_len, s_g = zip(*batch)

        input_tensor = torch.tensor([i for i in input_batch], dtype=torch.long)

        tar_tensor = torch.stack([i for i in tar_tensor]).long()
        sm = [i for i in sm]
        seq_lens = [i for i in seq_len]
        s_g = torch.stack([i for i in s_g])
        return input_tensor,tar_tensor,sm,seq_lens,s_g

def MLM_c(batch, config=None):
    config = config or BaseConfig()
    input_tensor_mlm, seq_lens, s_g,labels = zip(*batch)

    input_tensor_mlm=torch.stack([i for i in input_tensor_mlm])
    labels=torch.stack([i for i in labels])

    seq_lens = [i for i in seq_lens]
    s_g = torch.stack([i for i in s_g])

    return input_tensor_mlm, seq_lens, s_g,labels
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, task=None,
                 path=None, ind=None, seed=None, checkpoint_dir=None):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf if task=='regression' else -np.inf if task=='classification' else -np.inf
        self.delta = delta
        target_dir = Path(checkpoint_dir or 'runs/default/checkpoints')
        target_dir.mkdir(parents=True, exist_ok=True)
        self.path = str(target_dir / f'Bert_best_model{seed}.pt')
        self.task = task
    def __call__(self, val_loss, model,epoch):
        score = -val_loss if self.task=='regression' else val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,epoch):

        if self.verbose:
            metric_name = 'AUC' if self.task == 'classification' else 'loss'
            print(f'Validation {metric_name} improved ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
class STGFormerTrainer:
    def __init__(self,config,state_dict=None,seed=None,ind=None):
        self.sample_K=1.0
        if config.G_GPT:
            self.model=GPTModel(config)
        else:
            self.model=BertModel(config)
        self.config=config
        self.last_eval_details = {}
        self.tokenizer = GPTTokenizer() if config.G_GPT else BertTokenizer()
        self.model.to(config.device)

        early_stop_metric = getattr(self.config, "early_stop_metric", "loss")
        early_stop_task = 'classification' if (early_stop_metric == 'auc' and not self.config.reg and not self.config.MLM) else 'regression'
        self.early_stopping = EarlyStopping(
            patience=int(getattr(self.config, "early_stopping_patience", 10)),
            verbose=True,
            task=early_stop_task,
            seed=seed,
            ind=ind,
            checkpoint_dir=getattr(config, "checkpoint_dir", None),
        )
        if state_dict is not None:

            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

            if self.config.learnable_pos and not self.config.G_GPT and 'Encoder.pos_emb' in state_dict:
                pos_old = state_dict['Encoder.pos_emb']
                L_new = self.config.seq_len
                pos_new = expand_pos_embed_1d(pos_old, L_new)
                state_dict['Encoder.pos_emb'] = nn.Parameter(pos_new.to(self.config.device))

            model_state = self.model.state_dict()
            filtered_state = {}
            skipped = []
            for k, v in state_dict.items():
                if k in model_state and hasattr(v, 'shape') and model_state[k].shape == v.shape:
                    filtered_state[k] = v
                elif k in model_state:
                    skipped.append(k)
            self.model.load_state_dict(filtered_state, strict=False)
            print(f"Loaded pretrained weights: {len(filtered_state)} tensors; skipped shape-mismatch: {len(skipped)}")
            if skipped:
                print("Skipped keys:", skipped[:8], "..." if len(skipped) > 8 else "")

        else:
            self._init_weights()
        self.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = self.model.to(self.config.device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = torch.nn.DataParallel(self.model)
    def _init_weights(self):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p is None:
                    continue

                if name.endswith(".bias"):
                    p.zero_()

                elif name.endswith(".weight"):

                    if "layernorm" in name.lower() or ".ln" in name.lower() or name.lower().endswith("norm.weight"):
                        p.fill_(1.0)
                    else:
                        p.normal_(mean=0.0, std=0.02)

    def build_optimizers(self,model):

        params_with_decay = []
        params_without_decay = []

        for name, param in model.named_parameters():
            if 'bias' in name:
                params_without_decay.append(param)
            elif 'tgt_emb' in name and self.config.MLM:
                params_without_decay.append(param)
            else:
                params_with_decay.append(param)

        optimizer = optim.AdamW([
            {'params': params_with_decay, 'weight_decay': 1e-5},
            {'params': params_without_decay, 'weight_decay': 0}
        ], lr=self.config.lr)

        return optimizer

    def build_scheduler(self, optimizer, train_loader):
        if not getattr(self.config, "use_lr_scheduler", True):
            return None

        try:
            steps_per_epoch = len(train_loader)
        except TypeError:
            steps_per_epoch = getattr(self.config, "steps_per_epoch", None)

        if steps_per_epoch is None or steps_per_epoch <= 0:
            return None

        total_steps = max(1, steps_per_epoch * self.config.gen_trian_epoch)
        warmup_ratio = float(getattr(self.config, "warmup_ratio", 0.05))
        warmup_steps = int(total_steps * warmup_ratio)
        if warmup_ratio > 0:
            warmup_steps = max(1, warmup_steps)
        min_lr_ratio = getattr(self.config, "min_lr_ratio", 0.1)

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        print(
            f"LR scheduler: warmup+cosine, "
            f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
            f"warmup_steps={warmup_steps}, min_lr_ratio={min_lr_ratio}"
        )
        return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def get_current_lr(optimizer):
        return optimizer.param_groups[0]["lr"]

    def log_epoch(self, record):
        log_dir = getattr(self.config, "log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, getattr(self.config, "loss_log_file", "bert_loss_curve.jsonl"))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def train_gpt(self, train_dataset, test_dataset=None, val_dataset=None, cls_w=None):

        def unwrap_model(model):
            return model.module if isinstance(model, torch.nn.DataParallel) else model

        def make_loader(dataset, shuffle):
            if dataset is None:
                return None
            num_workers = getattr(self.config, "gpt_num_workers", 0)
            if isinstance(dataset, IterableDataset):
                shuffle = False
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                collate_fn=partial(gpt_collate_fn, config=self.config),
                drop_last=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            )

        scaler = GradScaler(init_scale=float(getattr(self.config, "amp_init_scale", 1024.0)))
        optimizer = self.build_optimizers(self.model)
        train_loader = make_loader(train_dataset, True)
        val_loader = make_loader(val_dataset, False)
        test_loader = make_loader(test_dataset, False)
        scheduler = self.build_scheduler(optimizer, train_loader)

        best_monitor = float("inf")
        for epoch in range(self.config.gen_trian_epoch):
            self.model.train()
            train_losses = []
            pbar = tqdm(train_loader, desc='GPTModel Train', colour='green')

            for batch in pbar:
                if self.config.scaffold_lead:
                    input_tensor, tar_tensor, _, seq_lens, s_g, prompt = batch
                    prompt = prompt.to(self.config.device, non_blocking=True)
                else:
                    input_tensor, tar_tensor, _, seq_lens, s_g = batch
                    prompt = None

                input_tensor = input_tensor.to(self.config.device, non_blocking=True)
                tar_tensor = tar_tensor.to(self.config.device, non_blocking=True)
                s_g = s_g.to(self.config.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    output, _ = self.model(input_tensor, s_g, seq_lens, prompt=prompt)
                    loss = F.cross_entropy(
                        output.view(-1, self.config.tgt_size),
                        tar_tensor.reshape(-1),
                        ignore_index=0
                    )

                scaler.scale(loss).backward()
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_updated = scaler.get_scale() >= scale_before
                if scheduler is not None and optimizer_updated:
                    scheduler.step()

                train_losses.append(loss.item())
                pbar.set_postfix(loss=loss.item(), lr=self.get_current_lr(optimizer))

            avg_train_loss = sum(train_losses) / max(1, len(train_losses))
            print(epoch, avg_train_loss)

            eval_loader = val_loader if val_loader is not None else train_loader
            eval_name = 'valid' if val_loader is not None else 'train'
            self.model.eval()
            eval_losses = []
            with torch.no_grad():
                pbar = tqdm(eval_loader, desc=f'GPTModel {eval_name}', colour='green')
                for batch in pbar:
                    if self.config.scaffold_lead:
                        input_tensor, tar_tensor, _, seq_lens, s_g, prompt = batch
                        prompt = prompt.to(self.config.device, non_blocking=True)
                    else:
                        input_tensor, tar_tensor, _, seq_lens, s_g = batch
                        prompt = None

                    input_tensor = input_tensor.to(self.config.device, non_blocking=True)
                    tar_tensor = tar_tensor.to(self.config.device, non_blocking=True)
                    s_g = s_g.to(self.config.device, non_blocking=True)

                    output, _ = self.model(input_tensor, s_g, seq_lens, prompt=prompt)
                    loss = F.cross_entropy(
                        output.view(-1, self.config.tgt_size),
                        tar_tensor.reshape(-1),
                        ignore_index=0
                    )
                    eval_losses.append(loss.item())

            avg_eval_loss = sum(eval_losses) / max(1, len(eval_losses))
            best_monitor = min(best_monitor, avg_eval_loss)
            print(f'{eval_name}_loss:{avg_eval_loss}')
            self.log_epoch({
                "mode": "GPTModel",
                "epoch": epoch,
                "train_loss": avg_train_loss,
                f"{eval_name}_loss": avg_eval_loss,
                "lr": self.get_current_lr(optimizer),
            })

            self.early_stopping(avg_eval_loss, unwrap_model(self.model), epoch)
            if self.early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        if test_loader is not None and os.path.exists(self.early_stopping.path):
            state_dict = torch.load(self.early_stopping.path, map_location=self.config.device)
            unwrap_model(self.model).load_state_dict(state_dict)
            self.model.eval()
            test_losses = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='GPTModel Test', colour='green'):
                    if self.config.scaffold_lead:
                        input_tensor, tar_tensor, _, seq_lens, s_g, prompt = batch
                        prompt = prompt.to(self.config.device, non_blocking=True)
                    else:
                        input_tensor, tar_tensor, _, seq_lens, s_g = batch
                        prompt = None
                    input_tensor = input_tensor.to(self.config.device, non_blocking=True)
                    tar_tensor = tar_tensor.to(self.config.device, non_blocking=True)
                    s_g = s_g.to(self.config.device, non_blocking=True)
                    output, _ = self.model(input_tensor, s_g, seq_lens, prompt=prompt)
                    loss = F.cross_entropy(output.view(-1, self.config.tgt_size), tar_tensor.reshape(-1), ignore_index=0)
                    test_losses.append(loss.item())
            test_loss = sum(test_losses) / max(1, len(test_losses))
            print(f'test_loss:{test_loss}')
            return test_loss

        return best_monitor

    def evaluate_downstream(self, dataset, split_name="test"):

        def move_seq_lens(seq_lens):
            if torch.is_tensor(seq_lens):
                return seq_lens.to(self.config.device, non_blocking=True)
            return torch.tensor(seq_lens, device=self.config.device)

        if dataset is None:
            raise ValueError("evaluate_downstream requires a non-empty dataset.")

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=chem_collate_fn,
            drop_last=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        self.model.eval()
        x_, seigema = None, None
        x_device, seigema_device = None, None
        if self.config.reg and self.config.z:
            x_ = getattr(dataset, "x_", None)
            seigema = getattr(dataset, "seigema", None)
            if x_ is not None and seigema is not None:
                x_ = x_.detach().cpu().float() if torch.is_tensor(x_) else torch.tensor(x_, dtype=torch.float)
                seigema = seigema.detach().cpu().float() if torch.is_tensor(seigema) else torch.tensor(seigema, dtype=torch.float)
                x_device = x_.to(self.config.device)
                seigema_device = seigema.to(self.config.device)
            else:
                print('WARNING: regression z-score is enabled, but eval dataset has no normalization stats.')
        if (self.config.tgt_size == 2 or (self.config.reg and self.config.tgt_size == 1)) and not self.config.mutcls:
            y_true = np.zeros([1])
            y_pred = np.zeros([1, self.config.tgt_size])
        else:
            y_true = np.zeros([1, self.config.tgt_size])
            y_pred = np.zeros([1, self.config.tgt_size])

        with torch.no_grad():
            for batch in tqdm(loader, desc=f'Eval {split_name}', colour='green'):
                input_tensor, tar_tensor, _, seq_lens, s_g = batch
                input_tensor = input_tensor.to(self.config.device, non_blocking=True)
                s_g = s_g.to(self.config.device, non_blocking=True)
                seq_lens = move_seq_lens(seq_lens)

                output, _, _ = self.model(input_tensor, seq_lens, s_g)
                if self.config.reg:
                    if self.config.z and x_device is not None and seigema_device is not None:
                        output = output * seigema_device + x_device
                elif self.config.tgt_size == 1:
                    output = torch.sigmoid(output)
                elif self.config.tgt_size == 2 and not self.config.mutcls:
                    output = F.softmax(output, dim=-1)
                else:
                    output = torch.sigmoid(output)

                true_tensor = tar_tensor.cpu().clone().detach()
                if self.config.reg and self.config.z and x_ is not None and seigema is not None:
                    true_tensor = true_tensor * seigema + x_
                y_true = np.vstack((y_true, true_tensor.numpy()))
                y_pred = np.vstack((y_pred, output.view(-1, self.config.tgt_size).cpu().detach().numpy()))

        details = {"split": split_name}
        if self.config.reg:
            true = y_true[1:].reshape(-1)
            pred = y_pred[1:].reshape(-1)
            mask = np.isfinite(true) & np.isfinite(pred)
            rmse, mae, r2 = calculate_metrics(true[mask], pred[mask])
            details.update({"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)})
            score = mae if self.config.MAE else rmse
        elif self.config.tgt_size == 1:
            true = y_true[1:].reshape(-1)
            pred = y_pred[1:].reshape(-1)
            mask = np.isfinite(true) & np.isfinite(pred)
            score = roc_auc_score(true[mask], pred[mask]) if len(np.unique(true[mask])) == 2 else float('nan')
            details.update({"AUC": float(score)})
        else:
            if self.config.tgt_size == 2 and not self.config.mutcls:
                true = y_true.reshape(-1)[1:]
                score = roc_auc_score(true, y_pred[1:, 1]) if len(np.unique(true)) == 2 else float('nan')
                details.update({"AUC": float(score)})
            else:
                aucs = calculate_auc_per_class(y_true[1:], y_pred[1:], self.config.tgt_size)
                score = sum(aucs) / len(aucs) if len(aucs) > 0 else float('nan')
                details.update({"AUC": float(score), "per_task_auc": [float(x) for x in aucs]})

        print(f'{split_name}_score:{score}')
        print(details)
        return float(score), details

    def train_and_test(self, train_dataset, test_dataset, val_dataset=None, cls_w=None):
        def unwrap_model(model):
            return model.module if isinstance(model, torch.nn.DataParallel) else model

        def move_seq_lens(seq_lens):

            if torch.is_tensor(seq_lens):
                return seq_lens.to(self.config.device, non_blocking=True)
            return torch.tensor(seq_lens, device=self.config.device)

        def make_loader(dataset, shuffle, collate_fn):
            if dataset is None:
                return None

            num_workers = 6
            if isinstance(dataset, IterableDataset):
                shuffle = False
                num_workers = getattr(self.config, "stream_num_workers", 4)

                try:
                    if len(dataset) < self.config.batch_size * max(1, num_workers):
                        num_workers = 0
                except TypeError:
                    pass

            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0
            )

        print(f'Model parameters: {sum(p.numel() for p in self.model.parameters())}')

        scaler = GradScaler(init_scale=float(getattr(self.config, "amp_init_scale", 1024.0)))
        self.model.train()

        if self.config.z and self.config.reg and not self.config.MLM:
            x_, seigema = train_dataset.z_score()
            if val_dataset is not None and hasattr(val_dataset, "set_z_score"):
                val_dataset.set_z_score(x_, seigema)
            if test_dataset is not None and hasattr(test_dataset, "set_z_score"):
                test_dataset.set_z_score(x_, seigema)
            x_device = x_.to(self.config.device)
            seigema_device = seigema.to(self.config.device)
            print(f'Regression z-score enabled: mean={x_.tolist()}, std={seigema.tolist()}')
        else:
            x_, seigema = None, None
            x_device, seigema_device = None, None

        optimizer = self.build_optimizers(self.model)

        if not self.config.MLM:
            train_Dataset = make_loader(train_dataset, True, chem_collate_fn)
            test_Dataset = make_loader(test_dataset, False, chem_collate_fn)
            val_Dataset = make_loader(val_dataset, False, chem_collate_fn)
        else:
            train_Dataset = make_loader(train_dataset, True, MLM_c)
            test_Dataset = make_loader(test_dataset, False, MLM_c)
            val_Dataset = make_loader(val_dataset, False, MLM_c)

        scheduler = self.build_scheduler(optimizer, train_Dataset)

        max_roc = float('-inf')
        RMSE = float('inf')
        best_reg_details = {}
        best_mlm_val = float('inf')

        for epoch in range(self.config.gen_trian_epoch):
            self.model.train()
            losses = []

            pbar = tqdm(train_Dataset, desc='Iteration', colour='green')

            for batch in pbar:
                optimizer.zero_grad(set_to_none=True)

                if self.config.MLM:
                    input_tensor_mlm, seq_lens, s_g, labels = batch

                    input_tensor_mlm = input_tensor_mlm.to(self.config.device, non_blocking=True)
                    s_g = s_g.to(self.config.device, non_blocking=True)
                    labels = labels.to(self.config.device, non_blocking=True)
                    seq_lens = move_seq_lens(seq_lens)

                else:
                    input_tensor, tar_tensor, _, seq_lens, s_g = batch

                    input_tensor = input_tensor.to(self.config.device, non_blocking=True)
                    tar_tensor = tar_tensor.to(self.config.device, non_blocking=True)
                    s_g = s_g.to(self.config.device, non_blocking=True)
                    seq_lens = move_seq_lens(seq_lens)

                with autocast():
                    if not self.config.MLM and self.config.reg:
                        output, _, _ = self.model(input_tensor, seq_lens, s_g)
                        pred = output.view(-1, self.config.tgt_size)
                        target = tar_tensor.float().view(-1, self.config.tgt_size)
                        mask = torch.isfinite(target)
                        loss_mat = F.mse_loss(pred, torch.nan_to_num(target, nan=0.0), reduction='none')
                        loss = loss_mat[mask].mean() if mask.any() else loss_mat.mean()

                    elif not self.config.MLM and self.config.tgt_size == 1:
                        output, _, _ = self.model(input_tensor, seq_lens, s_g)
                        target = tar_tensor.view(-1).float()
                        mask = torch.isfinite(target)
                        loss_mat = F.binary_cross_entropy_with_logits(
                            output.reshape(-1),
                            torch.nan_to_num(target, nan=0.0),
                            reduction='none'
                        )
                        loss = loss_mat[mask].mean() if mask.any() else loss_mat.mean()

                    elif self.config.MLM:
                        output, _ = self.model(input_tensor_mlm, seq_lens, s_g)

                        loss = F.cross_entropy(
                            output.view(-1, self.config.tgt_size),
                            labels.view(-1),
                            ignore_index=0
                        )

                    elif self.config.mutcls:
                        output, _, _ = self.model(input_tensor, seq_lens, s_g)
                        pred = output.reshape(-1, output.size(-1))
                        target = tar_tensor.float().view_as(pred)
                        mask = torch.isfinite(target)
                        loss_mat = F.binary_cross_entropy_with_logits(
                            pred,
                            torch.nan_to_num(target, nan=0.0),
                            reduction='none'
                        )
                        loss = loss_mat[mask].mean() if mask.any() else loss_mat.mean()

                    else:
                        output, _, _ = self.model(input_tensor, seq_lens, s_g)

                        loss = F.cross_entropy(
                            output.reshape(-1, output.size(-1)),
                            tar_tensor.reshape(-1).long()
                        )

                scaler.scale(loss).backward()
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_updated = scaler.get_scale() >= scale_before
                if scheduler is not None and optimizer_updated:
                    scheduler.step()

                losses.append(loss.item())
                pbar.set_postfix(loss=loss.item(), lr=self.get_current_lr(optimizer))

            avg_loss = sum(losses) / len(losses)
            print(epoch, avg_loss)

            self.model.eval()

            if not self.config.MLM:
                if val_Dataset is None:
                    continue

                if not self.config.mutcls and (self.config.tgt_size == 2 or (self.config.reg and self.config.tgt_size == 1)):
                    y_true = np.zeros([1])
                    y_pred = np.zeros([1, self.config.tgt_size])
                else:
                    y_true = np.zeros([1, self.config.tgt_size])
                    y_pred = np.zeros([1, self.config.tgt_size])

                val_losses = []
                pbar = tqdm(val_Dataset, desc='Iteration', colour='green')

                with torch.no_grad():
                    for batch in pbar:
                        input_tensor, tar_tensor, _, seq_lens, s_g = batch

                        input_tensor = input_tensor.to(self.config.device, non_blocking=True)
                        tar_tensor_device = tar_tensor.to(self.config.device, non_blocking=True)
                        s_g = s_g.to(self.config.device, non_blocking=True)
                        seq_lens = move_seq_lens(seq_lens)

                        raw_output, _, _ = self.model(input_tensor, seq_lens, s_g)

                        if self.config.reg:
                            pred_loss = raw_output.view(-1, self.config.tgt_size)
                            target = tar_tensor_device.float().view(-1, self.config.tgt_size)
                            mask = torch.isfinite(target)
                            loss_mat = F.mse_loss(pred_loss, torch.nan_to_num(target, nan=0.0), reduction='none')
                            val_loss = loss_mat[mask].mean() if mask.any() else loss_mat.mean()
                        elif self.config.tgt_size == 1:
                            target = tar_tensor_device.view(-1).float()
                            mask = torch.isfinite(target)
                            loss_mat = F.binary_cross_entropy_with_logits(
                                raw_output.reshape(-1),
                                torch.nan_to_num(target, nan=0.0),
                                reduction='none'
                            )
                            val_loss = loss_mat[mask].mean() if mask.any() else loss_mat.mean()
                        elif self.config.mutcls:
                            pred_loss = raw_output.reshape(-1, raw_output.size(-1))
                            target = tar_tensor_device.float().view_as(pred_loss)
                            mask = torch.isfinite(target)
                            loss_mat = F.binary_cross_entropy_with_logits(
                                pred_loss,
                                torch.nan_to_num(target, nan=0.0),
                                reduction='none'
                            )
                            val_loss = loss_mat[mask].mean() if mask.any() else loss_mat.mean()
                        else:
                            val_loss = F.cross_entropy(
                                raw_output.reshape(-1, raw_output.size(-1)),
                                tar_tensor_device.reshape(-1).long()
                            )
                        val_losses.append(val_loss.item())

                        output = raw_output
                        if self.config.reg:
                            if self.config.z:
                                output = output * seigema_device + x_device
                        elif self.config.tgt_size == 1:
                            output = torch.sigmoid(output)
                        elif self.config.tgt_size == 2 and not self.config.mutcls:
                            output = F.softmax(output, dim=-1)
                        else:
                            output = torch.sigmoid(output)

                        true_tensor = tar_tensor.cpu().clone().detach()
                        if self.config.reg and self.config.z:
                            true_tensor = true_tensor * seigema + x_
                        y_true = np.vstack((y_true, true_tensor.numpy()))

                        y_pred = np.vstack(
                            (y_pred, output.view(-1, self.config.tgt_size).cpu().detach().numpy())
                        )

                val_loss_avg = sum(val_losses) / len(val_losses)
                print(f'valid_loss:{val_loss_avg}')

                if self.config.reg:
                    true = y_true[1:].reshape(-1)
                    pred = y_pred[1:].reshape(-1)
                    mask = np.isfinite(true) & np.isfinite(pred)
                    n_RMSE, n_MAE, n_R2 = calculate_metrics(true[mask], pred[mask])

                    current_reg_score = n_MAE if self.config.MAE else n_RMSE
                    if current_reg_score < RMSE:
                        RMSE = current_reg_score
                        best_reg_details = {
                            "split": "valid",
                            "RMSE": float(n_RMSE),
                            "MAE": float(n_MAE),
                            "R2": float(n_R2),
                            "score": float(current_reg_score),
                        }
                    print(f'RMSE:{n_RMSE},MAE:{n_MAE},R2:{n_R2}')

                elif self.config.tgt_size == 1:
                    true = y_true[1:].reshape(-1)
                    pred = y_pred[1:].reshape(-1)
                    mask = np.isfinite(true) & np.isfinite(pred)
                    rocs = roc_auc_score(true[mask], pred[mask]) if len(np.unique(true[mask])) == 2 else float('nan')

                    if np.isfinite(rocs):
                        max_roc = max(max_roc, rocs)
                    print(max_roc)

                else:
                    if self.config.tgt_size == 2 and not self.config.mutcls:
                        true = y_true.reshape(-1)[1:]
                        rocs = roc_auc_score(true, y_pred[1:, 1]) if len(np.unique(true)) == 2 else float('nan')
                    else:
                        rocs = calculate_auc_per_class(
                            y_true[1:],
                            y_pred[1:],
                            self.config.tgt_size
                        )

                    if isinstance(rocs, float):
                        print(rocs)
                    else:
                        rocs = sum(rocs) / len(rocs) if len(rocs) > 0 else float('-inf')
                        print(rocs)

                    if np.isfinite(rocs):
                        max_roc = max(max_roc, rocs)

                early_stop_metric = getattr(self.config, "early_stop_metric", "loss")
                if early_stop_metric == "auc" and not self.config.reg:
                    monitor_value = rocs if np.isfinite(rocs) else float("-inf")
                    print(f'early_stop_auc:{monitor_value}')
                else:
                    monitor_value = val_loss_avg
                    print(f'early_stop_loss:{monitor_value}')

                self.early_stopping(monitor_value, unwrap_model(self.model), epoch)

                if self.early_stopping.early_stop:
                    print("Early stopping triggered!")

                    if test_Dataset is None:
                        if self.config.reg:
                            self.last_eval_details = best_reg_details
                        return RMSE if self.config.reg else max_roc

                    state_dict = torch.load(
                        self.early_stopping.path,
                        map_location=self.config.device
                    )
                    unwrap_model(self.model).load_state_dict(state_dict)

                    if (self.config.tgt_size == 2 or (self.config.reg and self.config.tgt_size == 1)) and not self.config.mutcls:
                        y_true = np.zeros([1])
                        y_pred = np.zeros([1, self.config.tgt_size])
                    else:
                        y_true = np.zeros([1, self.config.tgt_size])
                        y_pred = np.zeros([1, self.config.tgt_size])

                    pbar = tqdm(test_Dataset, desc='Iteration', colour='green')

                    with torch.no_grad():
                        for batch in pbar:
                            input_tensor, tar_tensor, _, seq_lens, s_g = batch

                            input_tensor = input_tensor.to(self.config.device, non_blocking=True)
                            s_g = s_g.to(self.config.device, non_blocking=True)
                            seq_lens = move_seq_lens(seq_lens)

                            output, _, _ = self.model(input_tensor, seq_lens, s_g)

                            if self.config.reg:
                                if self.config.z:
                                    output = output * seigema_device + x_device
                            elif self.config.tgt_size == 1:
                                output = torch.sigmoid(output)

                            elif self.config.tgt_size == 2 and not self.config.mutcls:
                                output = F.softmax(output, dim=-1)

                            else:
                                output = torch.sigmoid(output)

                            true_tensor = tar_tensor.cpu().clone().detach()
                            if self.config.reg and self.config.z:
                                true_tensor = true_tensor * seigema + x_
                            y_true = np.vstack((y_true, true_tensor.numpy()))

                            y_pred = np.vstack(
                                (y_pred, output.view(-1, self.config.tgt_size).cpu().detach().numpy())
                            )

                    if self.config.reg:
                        true = y_true[1:].reshape(-1)
                        pred = y_pred[1:].reshape(-1)
                        mask = np.isfinite(true) & np.isfinite(pred)
                        n_RMSE, n_MAE, n_R2 = calculate_metrics(true[mask], pred[mask])
                        reg_score = n_MAE if self.config.MAE else n_RMSE
                        self.last_eval_details = {
                            "split": "test",
                            "RMSE": float(n_RMSE),
                            "MAE": float(n_MAE),
                            "R2": float(n_R2),
                            "score": float(reg_score),
                        }
                        print(f'test_RMSE:{n_RMSE},test_MAE:{n_MAE},test_R2:{n_R2}')
                        return reg_score

                    elif self.config.tgt_size == 1:
                        true = y_true[1:].reshape(-1)
                        pred = y_pred[1:].reshape(-1)
                        mask = np.isfinite(true) & np.isfinite(pred)
                        rocs = roc_auc_score(true[mask], pred[mask])
                        return rocs

                    else:
                        if self.config.tgt_size == 2 and not self.config.mutcls:
                            rocs = roc_auc_score(
                                y_true.reshape(-1)[1:],
                                y_pred[1:, 1]
                            )
                        else:
                            rocs = calculate_auc_per_class(
                                y_true[1:],
                                y_pred[1:],
                                self.config.tgt_size
                            )

                        if isinstance(rocs, float):
                            print(rocs)
                        else:
                            rocs = sum(rocs) / len(rocs) if len(rocs) > 0 else float('nan')

                        return rocs

            else:
                if val_Dataset is None:
                    continue

                losses = []
                pbar = tqdm(val_Dataset, desc='Iteration', colour='green')

                with torch.no_grad():
                    for batch in pbar:
                        input_tensor_mlm, seq_lens, s_g, labels = batch

                        input_tensor_mlm = input_tensor_mlm.to(self.config.device, non_blocking=True)
                        s_g = s_g.to(self.config.device, non_blocking=True)
                        labels = labels.to(self.config.device, non_blocking=True)
                        seq_lens = move_seq_lens(seq_lens)

                        output, _ = self.model(input_tensor_mlm, seq_lens, s_g)

                        loss = F.cross_entropy(
                            output.view(-1, self.config.tgt_size),
                            labels.view(-1),
                            ignore_index=0
                        )

                        losses.append(loss.item())

                avg_val_loss = sum(losses) / len(losses)
                best_mlm_val = min(best_mlm_val, avg_val_loss)
                print(f'averge_loss:{avg_val_loss}')

                self.log_epoch({
                    "epoch": epoch,
                    "mode": "MLM",
                    "train_loss": float(avg_loss),
                    "val_loss": float(avg_val_loss),
                    "lr": float(self.get_current_lr(optimizer)),
                })

                self.early_stopping(
                    avg_val_loss,
                    unwrap_model(self.model),
                    epoch
                )

                if self.early_stopping.early_stop:
                    print("Early stopping triggered!")

                    if test_Dataset is None:
                        return best_mlm_val

                    state_dict = torch.load(
                        self.early_stopping.path,
                        map_location=self.config.device
                    )
                    unwrap_model(self.model).load_state_dict(state_dict)

                    losses = []
                    pbar = tqdm(test_Dataset, desc='Iteration', colour='green')

                    with torch.no_grad():
                        for batch in pbar:
                            input_tensor_mlm, seq_lens, s_g, labels = batch

                            input_tensor_mlm = input_tensor_mlm.to(self.config.device, non_blocking=True)
                            s_g = s_g.to(self.config.device, non_blocking=True)
                            labels = labels.to(self.config.device, non_blocking=True)
                            seq_lens = move_seq_lens(seq_lens)

                            output, _ = self.model(input_tensor_mlm, seq_lens, s_g)

                            loss = F.cross_entropy(
                                output.view(-1, self.config.tgt_size),
                                labels.view(-1),
                                ignore_index=0
                            )

                            losses.append(loss.item())

                    best_test_loss = sum(losses) / len(losses)
                    print(f'best_test_loss:{best_test_loss}')

                    json.dump(
                        best_test_loss,
                        open('best_loss.json', 'w', encoding='utf-8')
                    )

                    break

        if self.config.MLM:
            return best_mlm_val
        if self.config.reg:
            self.last_eval_details = best_reg_details
            return RMSE
        else:
            return max_roc

    def sample_(self,sample_batch_size):
        K = self.sample_K
        batch_size = sample_batch_size
        max_len = int(getattr(self.config, "sample_max_len", getattr(self.config, "seq_len", 128)))

        input_vector = torch.full(
            (batch_size, 1), 8, dtype=torch.long, device=self.config.device
        )
        for step in tqdm(range(max_len - 1)):
            tokens = self.tokenizer.index2sm(input_vector)
            s_g = torch.zeros([batch_size, 1, step + 1, step + 1])
            max_value = 1
            for batch in range(batch_size):
                for t in range(1, step + 2):
                    s_g_ = shortest_path_matrix(sm2graph(tokens[batch][:t]), t - 1)
                    if s_g_ is None:
                        break
                    s_g_ = torch.tensor(s_g_)
                    max_value = max(torch.max(s_g_).item(), max_value)
                    s_g_ = s_g_ * (1 / max_value)
                    s_g_ = F.pad(s_g_, [0, step + 1 - t], mode='constant', value=0)
                    s_g[batch, 0, t - 1] = s_g_

            seq_lens = torch.tensor([step + 1 for _ in range(batch_size)], device='cpu')
            logits, _ = self.model(input_vector.to(self.config.device), s_g.to(self.config.device), seq_lens)
            logits = logits[:, -1, :].expand(batch_size, -1)

            mu_logits = logits.detach().clone()

            for bad_id in (0, 1, 3):
                if bad_id < mu_logits.size(-1):
                    mu_logits[:, bad_id] = -float('inf')
            if K:
                mu_logits = mu_logits / K

            probabilities = mu_logits.softmax(dim=1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            input_vector = torch.cat((input_vector, next_token), dim=-1)

        return input_vector, None

    def sample_scaffold_topk(self, prompt, top_k=10):
        K = self.sample_K
        max_len = int(getattr(self.config, "sample_max_len", getattr(self.config, "seq_len", 128)))
        prompt = torch.tensor(
            self.tokenizer.sm2index(self.tokenizer.tokenize(prompt)), dtype=torch.long, device=self.config.device
        ).expand(self.config.batch_size, -1)

        input_vector = torch.full(
            (self.config.batch_size, 1), 8, dtype=torch.long, device=self.config.device
        )

        seq_lens = torch.tensor([prompt.shape[-1] for _ in range(self.config.batch_size)], device='cpu')

        for _ in tqdm(range(max_len - 1)):
            tokens = self.tokenizer.index2sm(input_vector)
            s_g = torch.zeros([self.config.batch_size, 1, _ + 1, _ + 1])

            max_value = 1
            for batch in range(self.config.batch_size):
                for t in range(1, _):
                    s_g_ = shortest_path_matrix(sm2graph(tokens[batch][:t]), t - 1)
                    if s_g_ is None:
                        break
                    s_g_ = torch.tensor(s_g_)

                    max_value = max(torch.max(s_g_).item(), max_value)

                    s_g_ = s_g_ * (1 / max_value)

                    s_g_ = F.pad(s_g_, [1, _ - t], mode='constant', value=0)

                    s_g[batch, 0, t] = s_g_

            s_g = torch.cat([torch.zeros([s_g.shape[0], s_g.shape[-3], 1, s_g.shape[-1]]), s_g], dim=-2)
            s_g = torch.cat([torch.zeros([s_g.shape[0], s_g.shape[-3], s_g.shape[-2], 1]), s_g], dim=-1)
            s_g = s_g.to(self.config.device)

            logits, _ = self.model(input_vector.to(self.config.device), s_g, seq_lens, prompt=prompt)
            logits = logits[:, -1, :].expand(self.config.batch_size, -1)

            mu_logits = logits.detach().clone()
            if K:
                mu_logits = mu_logits / K

            probabilities = mu_logits.softmax(dim=1)

            top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=1)

            next_token = torch.gather(top_k_indices, 1, torch.multinomial(top_k_probs, num_samples=1))

            input_vector = torch.cat((input_vector, next_token), dim=-1)

        return input_vector, None

    @torch.no_grad()
    def sample(self,sample_batch_size,prompt=None):
        self.model.eval()
        mols=[]
        while len(mols)<sample_batch_size:
            cur_batch_size = min(getattr(self.config, "sample_batch_size", sample_batch_size), sample_batch_size - len(mols))
            if self.config.scaffold_lead:

                samples=self.sample_scaffold_topk(prompt)
            else:
                samples=self.sample_(cur_batch_size)

            mols+=[''.join(i) for i in self.tokenizer.index2sm(samples[0])]
        return mols,None
