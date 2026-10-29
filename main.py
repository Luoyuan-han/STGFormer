import csv
import random
import re

import numpy as np
from Bert_trainer import STGFormerTrainer
import json
import torch
from data_io import find_task_dir, has_aggregate_csv, has_split_file, read_split
import os
from rdkit import RDLogger, Chem
from Dataset import (
    BertDataset,
    BertStreamDataset,
    GPTStreamDataset,
    GPTDataset,
    MoleculeDataset,
    RandomBertDataset,
    iter_data_file,
)
from config import (
    BertConfig,
    GPTConfig,
    configure_finetune,
    configure_generation,
    configure_pretraining,
)
RDLogger.DisableLog('rdApp.*')

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    import argparse
    from datetime import datetime
    from pathlib import Path
    from benchmark_tasks import PAPER_TASKS

    def parse_seeds(seed_text):
        seed_text = str(seed_text).strip().lower()

        if seed_text in {'random10', 'ten', 'baseline10'}:
            return [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
        if '-' in seed_text and ',' not in seed_text:
            left, right = seed_text.split('-', 1)
            return list(range(int(left), int(right) + 1))
        return [int(x.strip()) for x in seed_text.split(',') if x.strip()]

    def get_task_root(root, task):
        task_root = Path(root) / task
        if not task_root.exists() and (Path(root) / 'scaffold' / task).exists():
            task_root = Path(root) / 'scaffold' / task
        return task_root

    def available_scaffold_seeds(root, task):
        task_root = get_task_root(root, task)
        seeds = []
        if task_root.exists():
            for p in task_root.glob('seed*'):
                if p.is_dir():
                    try:
                        seeds.append(int(p.name.replace('seed', '')))
                    except ValueError:
                        pass
        return sorted(seeds)

    def resolve_seeds(args):
        seed_text = str(args.seeds).strip().lower()

        if seed_text in {'random10', 'ten', 'baseline10', 'scaffold10', 'split10'}:
            seeds = available_scaffold_seeds(args.scaffold_root, args.task)
            if len(seeds) >= 10:
                return seeds[:10]
            return parse_seeds(seed_text)
        return parse_seeds(seed_text)

    def normalize_scaffold_rows(rows):

        out = []
        for row in rows:
            if len(row) < 2:
                continue
            smiles, label = row[0], row[1]
            if label is None:
                label = [float('nan')]
            elif not isinstance(label, list):
                label = [label]
            label = [float('nan') if x is None else x for x in label]
            if len(row) > 2 and isinstance(row[2], dict):
                out.append([smiles, label, row[2]])
            else:
                out.append([smiles, label])
        return out

    def find_scaffold_task_dir(root, task, seed):
        task_root = find_task_dir(Path(root), task)
        seed_dir = task_root / f'seed{seed}'
        if has_split_file(seed_dir, 'train') or has_aggregate_csv(seed_dir):
            return seed_dir
        if has_split_file(task_root, 'train') or has_aggregate_csv(task_root):
            return task_root
        candidates = sorted([p for p in task_root.glob('seed*') if p.is_dir()])
        for p in candidates:
            if has_split_file(p, 'train') or has_aggregate_csv(p):
                print(f'WARNING: seed{seed} not found for {task}, use {p.name} instead.')
                return p
        raise FileNotFoundError(f'Cannot find scaffold split files for task={task}, seed={seed}')

    def load_scaffold_split(root, task, seed, enforced=False, train_enforced=False):
        split_dir = find_scaffold_task_dir(root, task, seed)

        def read_split(name):

            rows, fp = read_split(
                split_dir, name, enforced=enforced,
                random_smiles=(train_enforced and name == 'train'),
            )
            print(f'Loaded {len(rows)} {name} rows from {fp}')
            return normalize_scaffold_rows(rows)

        train_data = read_split('train')
        val_data = read_split('valid')
        test_data = read_split('test')
        print(
            f'Loaded scaffold/{task}/{split_dir.name}: '
            f'train={len(train_data)}, valid={len(val_data)}, test={len(test_data)}, '
            f'train_enforced={enforced or train_enforced}, '
            f'valid_enforced={enforced}, test_enforced={enforced}'
        )
        print('sample:', train_data[0] if train_data else None)
        return train_data, test_data, val_data

    def infer_label_dim(data):
        for row in data:
            label = row[1]
            if isinstance(label, list):
                return len(label)
        return 1

    def configure_for_scaffold(config, task, train_data, args):
        label_dim = infer_label_dim(train_data)
        has_solvent = any(
            len(row) > 2 and isinstance(row[2], dict) and row[2].get('solvent')
            for row in train_data[: min(len(train_data), 2000)]
        )

        configure_finetune(
            config,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            pooling=args.pooling,
            early_stop_metric=args.early_stop_metric,
            use_lr_scheduler=not args.no_scheduler,
            label_dim=label_dim,
            use_solvent=has_solvent,
            loss_log_file=f'{task}_finetune_loss.jsonl',
        )

        print(
            f'Fine-tune config: task_type=classification, label_dim={label_dim}, '
            f'tgt_size={config.tgt_size}, mutcls={config.mutcls}, reg={config.reg}, '
            f'z_score={getattr(config, "z", False)}, '
            f'use_solvent={getattr(config, "use_solvent", False)}, '
            f'pooling={config.pooling}, rope={config.rope}, '
            f'random_smiles={getattr(args, "random_smiles", False)}, '
            f'early_stop_metric={config.early_stop_metric}, lr={config.lr}, '
            f'batch_size={config.batch_size}, seq_len={config.seq_len}'
        )
        return config

    def load_state_dict_if_exists(path):
        if path is None or str(path).strip() == '':
            print('No pretrained checkpoint is used.')
            return None
        if not os.path.exists(path):
            print(f'WARNING: checkpoint not found: {path}; train from scratch.')
            return None
        print(f'Load checkpoint: {path}')
        return torch.load(path, map_location='cpu')

    def to_2d_canonical_smiles(smiles):
        if not isinstance(smiles, str):
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

    def normalize_smiles_list(rows, to_2d=False):
        smiles = []
        for item in rows:
            if isinstance(item, str):
                value = item
            elif isinstance(item, dict):
                value = item.get('text', item.get('smiles', item.get('SMILES', None)))
            elif isinstance(item, (list, tuple)) and len(item) > 0:
                value = item[0]
            else:
                value = None
            if not isinstance(value, str) or len(value) == 0:
                continue
            if to_2d:
                value = to_2d_canonical_smiles(value)
            if isinstance(value, str) and len(value) > 0:
                smiles.append(value)
        return [s for s in smiles if isinstance(s, str) and len(s) > 0]

    def load_gpt_data(args):
        if args.gpt_train_csv:
            train_data = normalize_smiles_list(iter_data_file(args.gpt_train_csv), to_2d=args.gpt_to_2d)
            val_data = normalize_smiles_list(iter_data_file(args.gpt_val_csv), to_2d=args.gpt_to_2d) if args.gpt_val_csv else None
            test_data = normalize_smiles_list(iter_data_file(args.gpt_test_csv), to_2d=args.gpt_to_2d) if args.gpt_test_csv else None
        else:
            data = normalize_smiles_list(iter_data_file(args.gpt_csv), to_2d=args.gpt_to_2d)
            random.shuffle(data)
            n_train = int(len(data) * args.gpt_train_ratio)
            n_val = int(len(data) * args.gpt_val_ratio)
            train_data = data[:n_train]
            val_data = data[n_train:n_train + n_val]
            test_data = data[n_train + n_val:] if args.gpt_use_test_split else None
        print(
            f'Loaded GPTModel data: train={len(train_data)}, '
            f'valid={0 if val_data is None else len(val_data)}, '
            f'test={0 if test_data is None else len(test_data)}'
        )
        print('sample:', train_data[0] if train_data else None)
        return train_data, test_data, val_data

    def iter_smiles_from_file(path):
        if not path or not os.path.exists(path):
            return
        yield from iter_data_file(path, key='smiles')

    def canonical_smiles(smiles):
        if not isinstance(smiles, str):
            return None
        smiles = smiles.replace('&', '').replace('<', '').replace('[MASK]', '').strip()
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

    def load_reference_canonical_set(path, limit=None):
        refs = set()
        if not path:
            return refs
        for idx, smi in enumerate(iter_smiles_from_file(path)):
            if limit is not None and idx >= limit:
                break
            can = canonical_smiles(smi)
            if can is not None:
                refs.add(can)
        print(f'Loaded reference molecules for novelty: {len(refs)} from {path}')
        return refs

    def evaluate_generated_smiles(samples, reference_set=None):
        canonical_valid = []
        invalid = []
        for smi in samples:
            can = canonical_smiles(smi)
            if can is None:
                invalid.append(smi)
            else:
                canonical_valid.append(can)

        total = max(1, len(samples))
        unique_valid = set(canonical_valid)
        metrics = {
            'num_samples': len(samples),
            'num_valid': len(canonical_valid),
            'num_unique_valid': len(unique_valid),
            'validity': len(canonical_valid) / total,
            'uniqueness': len(unique_valid) / max(1, len(canonical_valid)),
        }
        if reference_set is not None and len(reference_set) > 0:
            novel = [s for s in unique_valid if s not in reference_set]
            metrics['num_novel_unique'] = len(novel)
            metrics['novelty'] = len(novel) / max(1, len(unique_valid))
        return metrics, canonical_valid, invalid

    def build_gpt_config(args):
        config = GPTConfig()
        variant = str(args.gpt_variant).lower()
        if variant not in {'base', 'pe', 'stg', 'stg+pe'}:
            raise ValueError(f'unknown GPT variant: {args.gpt_variant}')
        config.stg = variant in {'stg', 'stg+pe'}
        config.learnable_pos = variant in {'pe', 'stg+pe'}
        return configure_generation(
            config,
            seq_len=args.gpt_seq_len,
            batch_size=args.gpt_batch_size,
            epochs=args.epochs,
            lr=args.gpt_lr,
            sample_batch_size=(args.sample_batch_size
                               if args.sample_batch_size > 0 else args.gpt_batch_size),
            sample_max_len=(args.sample_max_len
                            if args.sample_max_len > 0 else args.gpt_seq_len),
            use_lr_scheduler=not args.no_scheduler,
            gpt_num_workers=args.gpt_num_workers,
            gpt_to_2d=args.gpt_to_2d,
        )

    def format_ckpt_path(template, seed, ind, task):
        if template is None or str(template).strip() == '':
            return ''
        path = str(template)
        if '{' in path and '}' in path:
            path = path.format(seed=seed, ind=ind, task=task)
        elif seed != 42 and '42' in os.path.basename(path):

            dirname, filename = os.path.split(path)
            filename = re.sub(r'42(?=\.pt$)', str(seed), filename)
            if filename == os.path.split(path)[1]:
                filename = filename.replace('42', str(seed))
            path = os.path.join(dirname, filename)
        return path

    def configure_run_paths(config, run_dir, seed):

        run_root = Path(run_dir)
        config.checkpoint_dir = str(run_root / 'checkpoints' / f'seed{seed}')
        config.log_dir = str(run_root / 'logs' / f'seed{seed}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['finetune', 'pretrain', 'gpt', 'sample', 'eval'], default='finetune')
    parser.add_argument('--task', choices=PAPER_TASKS, default='bace')
    parser.add_argument('--scaffold_root', default='fintune_datas/moleculenet_random_scaffold')
    parser.add_argument('--seeds', default='42')
    parser.add_argument('--enforced', action='store_true')
    parser.add_argument('--random_smiles', action='store_true')
    parser.add_argument('--ckpt', default='checkpoints/pretraining/Bert_STG+PE_1000k.pt', help='pretrained BERT checkpoint')
    parser.add_argument('--eval_ckpt_template', default='')
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--pooling', choices=['cls', 'mean'], default='cls')
    parser.add_argument('--early_stop_metric', choices=['loss', 'auc'], default='loss')
    parser.add_argument('--no_scheduler', action='store_true')
    parser.add_argument('--pretrain_csv', default='train_data/train_1000k.csv')
    parser.add_argument('--pretrain_train_csv', default='train_data/train_1000k.csv')
    parser.add_argument('--pretrain_val_csv', default='train_data/valid_1000k.csv')
    parser.add_argument('--pretrain_cache_dir', default='train_data/cache')
    parser.add_argument('--pretrain_seq_len', type=int, default=128)
    parser.add_argument('--pretrain_batch_size', type=int, default=512)
    parser.add_argument('--pretrain_lr', type=float, default=3e-4)
    parser.add_argument('--pretrain_variant', choices=['Base', 'PE', 'STG', 'STG+PE'], default='STG+PE')
    parser.add_argument('--gpt_csv', default='train_data/gpt_train/train_1000k.csv')
    parser.add_argument('--gpt_ckpt', default='', help='optional GPT checkpoint to resume or use for sampling')
    parser.add_argument('--gpt_stream', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--gpt_to_2d', action='store_true')
    parser.add_argument('--gpt_num_workers', type=int, default=0)
    parser.add_argument('--gpt_variant', choices=['Base', 'PE', 'STG', 'STG+PE'], default='STG')
    parser.add_argument('--gpt_seq_len', type=int, default=128)
    parser.add_argument('--gpt_batch_size', type=int, default=256)
    parser.add_argument('--gpt_lr', type=float, default=3e-4)
    parser.add_argument('--gpt_train_csv', default='train_data/gpt_train/train_1000k.csv')
    parser.add_argument('--gpt_val_csv', default='train_data/gpt_train/valid_1000k.csv')
    parser.add_argument('--gpt_test_csv', default='')
    parser.add_argument('--gpt_train_ratio', type=float, default=0.9)
    parser.add_argument('--gpt_val_ratio', type=float, default=0.1)
    parser.add_argument('--gpt_use_test_split', action='store_true')
    parser.add_argument('--sample_after_train', type=int, default=0)
    parser.add_argument('--sample_num', type=int, default=100)
    parser.add_argument('--sample_batch_size', type=int, default=0)
    parser.add_argument('--sample_max_len', type=int, default=0)
    parser.add_argument('--sample_temperature', type=float, default=1.0)
    parser.add_argument('--sample_output', default='')
    parser.add_argument('--sample_ref_json', default='')
    parser.add_argument('--sample_ref_limit', type=int, default=0)
    parser.add_argument('--no_sample_eval', action='store_true')
    parser.add_argument('--run_dir', default='',
                        help='run output root (default: runs/<mode>_<timestamp>)')
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent

    def resolve_repo_path(value):
        if value is None or str(value).strip() == '':
            return value
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else (repo_root / path).resolve())

    scaffold_root = Path(args.scaffold_root).expanduser()
    args.scaffold_root = str(scaffold_root if scaffold_root.is_absolute() else (repo_root / scaffold_root).resolve())
    for path_arg in (
        'ckpt', 'eval_ckpt_template', 'pretrain_csv', 'pretrain_train_csv',
        'pretrain_val_csv', 'pretrain_cache_dir', 'gpt_ckpt', 'gpt_csv',
        'gpt_train_csv', 'gpt_val_csv', 'gpt_test_csv', 'sample_output',
        'sample_ref_json',
    ):
        setattr(args, path_arg, resolve_repo_path(getattr(args, path_arg)))
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        repo_root / 'runs' / f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = str(run_dir)
    print(f'Run directory: {run_dir}')
    seeds = resolve_seeds(args)
    print('Seeds:', seeds)

    rocs = []
    result_records = []
    for ind in range(1):
        for seed in seeds:
            set_seed(seed)
            config = BertConfig()

            if args.mode == 'sample':
                config = build_gpt_config(args)
                configure_run_paths(config, run_dir, seed)
                state_dict = load_state_dict_if_exists(args.gpt_ckpt)
                if state_dict is None:
                    print('WARNING: sample mode without --gpt_ckpt will sample from random initialized GPTModel.')
                trainer = STGFormerTrainer(config, state_dict=state_dict, seed=seed, ind=ind)
                trainer.sample_K = args.sample_temperature

                samples, _ = trainer.sample(args.sample_num)
                samples = samples[:args.sample_num]
                sample_path = args.sample_output or str(run_dir / 'samples' / f'gpt_samples_seed{seed}.json')
                Path(sample_path).parent.mkdir(parents=True, exist_ok=True)

                ref_path = args.sample_ref_json or args.gpt_train_csv or args.gpt_csv
                reference_set = None
                if not args.no_sample_eval:
                    reference_set = load_reference_canonical_set(
                        ref_path,
                        limit=args.sample_ref_limit if args.sample_ref_limit > 0 else None
                    )
                    metrics, canonical_valid, invalid = evaluate_generated_smiles(samples, reference_set)
                else:
                    metrics, canonical_valid, invalid = {}, [], []

                output_obj = {
                    'seed': seed,
                    'checkpoint': args.gpt_ckpt,
                    'num_requested': args.sample_num,
                    'metrics': metrics,
                    'samples': samples,
                    'canonical_valid': canonical_valid,
                    'invalid': invalid,
                }
                json.dump(output_obj, open(sample_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
                print(f'Saved samples to {sample_path}')
                if metrics:
                    print('Sample metrics:', metrics)
                rocs.append(float(metrics.get('validity', 0.0)) if metrics else 0.0)
                result_records.append({'task': 'GPT_SAMPLE', 'seed': seed, 'score': float(rocs[-1]), 'metrics': metrics})
                continue

            if args.mode == 'eval':
                train_data, test_data, val_data = load_scaffold_split(
                    args.scaffold_root,
                    args.task,
                    seed,
                    enforced=args.enforced,
                    train_enforced=args.random_smiles
                )
                config = configure_for_scaffold(config, args.task, train_data, args)
                configure_run_paths(config, run_dir, seed)
                te = BertDataset(test_data, config)
                ckpt_template = args.eval_ckpt_template if args.eval_ckpt_template else args.ckpt
                ckpt_path = format_ckpt_path(ckpt_template, seed=seed, ind=ind, task=args.task)
                state_dict = load_state_dict_if_exists(ckpt_path)
                if state_dict is None:
                    raise FileNotFoundError(f'Eval checkpoint not found for seed={seed}: {ckpt_path}')

                trainer = STGFormerTrainer(config, state_dict=state_dict, seed=seed, ind=ind)
                score, details = trainer.evaluate_downstream(te, split_name='test')
                rocs.append(score)
                record = {
                    'task': args.task,
                    'seed': seed,
                    'score': float(score),
                    'checkpoint': ckpt_path,
                    'details': details,
                }
                result_records.append(record)
                print(rocs)
                print(f'{args.task}, seed={seed}, ckpt={ckpt_path}')
                continue

            if args.mode == 'pretrain':

                pretrain_variant = args.pretrain_variant.lower()
                config.stg = pretrain_variant in {'stg', 'stg+pe'}
                config.learnable_pos = pretrain_variant in {'pe', 'stg+pe'}
                config.rope = False

                configure_pretraining(
                    config,
                    seq_len=args.pretrain_seq_len,
                    batch_size=args.pretrain_batch_size,
                    epochs=args.epochs,
                    lr=args.pretrain_lr,
                )

                pretrain_train_csv = args.pretrain_train_csv
                pretrain_val_csv = args.pretrain_val_csv
                if pretrain_train_csv and not Path(pretrain_train_csv).is_file():
                    raise FileNotFoundError(f'BERT pretraining train split not found: {pretrain_train_csv}')
                if pretrain_val_csv and not Path(pretrain_val_csv).is_file():
                    raise FileNotFoundError(f'BERT pretraining validation split not found: {pretrain_val_csv}')
                if not (pretrain_train_csv and pretrain_val_csv):
                    source_path = Path(args.pretrain_csv)
                    if not source_path.is_file():
                        raise FileNotFoundError(f'BERT pretraining source not found: {source_path}')
                    cache_dir = Path(args.pretrain_cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    stem = source_path.stem
                    pretrain_train_csv = str(cache_dir / f'{stem}.seed{seed}.train.csv')
                    pretrain_val_csv = str(cache_dir / f'{stem}.seed{seed}.valid.csv')

                if os.path.exists(pretrain_train_csv) and os.path.exists(pretrain_val_csv):
                    tr = BertStreamDataset(pretrain_train_csv, config)
                    te = None
                    va = BertStreamDataset(pretrain_val_csv, config)
                else:
                    smiles = normalize_smiles_list(iter_data_file(args.pretrain_csv))
                    random.shuffle(smiles)
                    train_data, val_data = smiles[:int(len(smiles) * 0.9)], smiles[int(len(smiles) * 0.9):]
                    for cache_path, rows in ((pretrain_train_csv, train_data), (pretrain_val_csv, val_data)):
                        with open(cache_path, 'w', encoding='utf-8', newline='') as handle:
                            writer = csv.writer(handle, lineterminator='\n')
                            writer.writerow(['smiles'])
                            writer.writerows((row,) for row in rows)
                    tr = BertStreamDataset(pretrain_train_csv, config)
                    te = None
                    va = BertStreamDataset(pretrain_val_csv, config)
                print(
                    f'Pretraining data: train={pretrain_train_csv}, '
                    f'valid={pretrain_val_csv}, seq_len={config.seq_len}, '
                    f'variant={args.pretrain_variant}, batch_size={config.batch_size}, '
                    f'lr={config.lr}, scheduler=cosine, '
                    f'early_stopping_patience={config.early_stopping_patience}'
                )
                state_dict = None
                task_name = 'MLM'
            elif args.mode == 'gpt':
                config = build_gpt_config(args)
                configure_run_paths(config, run_dir, seed)

                if args.gpt_stream:
                    train_path = args.gpt_train_csv if args.gpt_train_csv else args.gpt_csv
                    val_path = args.gpt_val_csv if args.gpt_val_csv else ''
                    test_path = args.gpt_test_csv if args.gpt_test_csv else ''
                    if not train_path or not Path(train_path).is_file():
                        raise FileNotFoundError(f'GPT pretraining train split not found: {train_path}')
                    if val_path and not Path(val_path).is_file():
                        raise FileNotFoundError(f'GPT pretraining validation split not found: {val_path}')
                    if test_path and not Path(test_path).is_file():
                        raise FileNotFoundError(f'GPT pretraining test split not found: {test_path}')
                    print(
                        f'GPTModel stream mode: train={train_path}, '
                        f'valid={val_path if val_path else None}, test={test_path if test_path else None}, '
                        f'variant={args.gpt_variant}, seq_len={config.seq_len}, '
                        f'batch_size={config.batch_size}, lr={config.lr}, '
                        f'workers={config.gpt_num_workers}'
                    )
                    tr = GPTStreamDataset(train_path, config)
                    va = GPTStreamDataset(val_path, config) if val_path else None
                    te = GPTStreamDataset(test_path, config) if test_path else None
                else:
                    if args.gpt_train_csv and not Path(args.gpt_train_csv).is_file():
                        raise FileNotFoundError(f'GPT training split not found: {args.gpt_train_csv}')
                    if args.gpt_val_csv and not Path(args.gpt_val_csv).is_file():
                        raise FileNotFoundError(f'GPT validation split not found: {args.gpt_val_csv}')
                    train_data, test_data, val_data = load_gpt_data(args)
                    tr = MoleculeDataset(train_data, config)
                    te = MoleculeDataset(test_data, config) if test_data else None
                    va = MoleculeDataset(val_data, config) if val_data else None
                state_dict = load_state_dict_if_exists(args.gpt_ckpt)
                task_name = 'GPTModel'
            else:
                train_data, test_data, val_data = load_scaffold_split(
                    args.scaffold_root,
                    args.task,
                    seed,
                    enforced=args.enforced,
                    train_enforced=args.random_smiles
                )
                config = configure_for_scaffold(config, args.task, train_data, args)
                configure_run_paths(config, run_dir, seed)
                if args.random_smiles:

                    tr = RandomBertDataset(train_data, config)
                else:
                    tr = BertDataset(train_data, config)
                te, va = BertDataset(test_data, config), BertDataset(val_data, config)
                state_dict = load_state_dict_if_exists(args.ckpt)
                task_name = args.task

            configure_run_paths(config, run_dir, seed)
            trainer = STGFormerTrainer(config, state_dict=state_dict, seed=seed, ind=ind)
            if args.mode == 'gpt':
                score = trainer.train_gpt(tr, te, va, cls_w=None)
                if args.sample_after_train > 0:
                    trainer.sample_K = args.sample_temperature
                    samples, _ = trainer.sample(args.sample_after_train)
                    sample_path = str(run_dir / 'samples' / f'gpt_samples_seed{seed}.json')
                    Path(sample_path).parent.mkdir(parents=True, exist_ok=True)
                    samples = samples[:args.sample_after_train]
                    if not args.no_sample_eval:
                        ref_path = args.sample_ref_json or args.gpt_train_csv or args.gpt_csv
                        reference_set = load_reference_canonical_set(
                            ref_path,
                            limit=args.sample_ref_limit if args.sample_ref_limit > 0 else None
                        )
                        metrics, canonical_valid, invalid = evaluate_generated_smiles(samples, reference_set)
                        output_obj = {
                            'seed': seed,
                            'metrics': metrics,
                            'samples': samples,
                            'canonical_valid': canonical_valid,
                            'invalid': invalid,
                        }
                    else:
                        output_obj = samples
                    json.dump(output_obj, open(sample_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
                    print(f'Saved samples to {sample_path}')
            else:
                score = trainer.train_and_test(tr, te, va, cls_w=None)
            rocs.append(score)
            record = {'task': task_name, 'seed': seed, 'score': float(score)}
            if getattr(trainer, "last_eval_details", None):
                record['details'] = trainer.last_eval_details
            result_records.append(record)
            print(rocs)
            print(f'{task_name}, seed={seed}')

    out_name = (
        f'{args.task}_results.json' if args.mode == 'finetune'
        else f'{args.task}_eval_results.json' if args.mode == 'eval'
        else 'gpt_results.json' if args.mode == 'gpt'
        else 'gpt_sample_results.json' if args.mode == 'sample'
        else 'pretrain_results.json'
    )
    json.dump([float(i) for i in rocs], open(run_dir / out_name, 'w'))
    if args.mode in {'finetune', 'gpt', 'sample', 'eval'}:
        detail_name = (
            f'{args.task}_seed_results.json' if args.mode == 'finetune'
            else f'{args.task}_eval_seed_results.json' if args.mode == 'eval'
            else 'gpt_seed_results.json' if args.mode == 'gpt'
            else 'gpt_sample_seed_results.json'
        )
        json.dump(result_records, open(run_dir / detail_name, 'w'), indent=2)
    print(rocs, args.mode)
    print(np.mean(rocs))
    print(np.std(rocs))
