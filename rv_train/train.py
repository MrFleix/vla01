# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].
import argparse
import gc
import os
import pickle as pkl
import pprint
import random
import shutil
from contextlib import redirect_stdout
from datetime import datetime
from time import time

import datasets.features as dsf
from datasets import Sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, distributed
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

if not hasattr(dsf, "List"):
    dsf.List = Sequence
import roboverse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from rv_train import models
from rv_train.configs import get_cfg_defaults
from rv_train.models.qwen.dataset import QwenSFTDataset
from rv_train.models.qwen.model import QwenActor
from rv_train.utils import train_utils as utils

DEVICE = ""

START_TIME = time()

import os
import pickle as pkl

from transformers import TrainerCallback

"""
Training script usage:

# Resume training on single GPU with a LoRA adapter
python -m rv_train.train \
  --exp-config ./configs/vla0.yaml \
  --resume \
  --model-path /home/felix/vla0/runs/vla0/checkpoint-150 \
  --devices 0

# Train from scratch
python -m rv_train.train --exp-config ./configs/vla0.yaml

Notes:
- The --resume flag allows loading either a full model or a LoRA adapter.
- Dataset statistics are saved automatically:
    dataset_stats_path = os.path.join(log_dir, "dataset_stats.pkl")
    with open(dataset_stats_path, "wb") as f:
        pkl.dump(train_dataset.stats, f)
    print(f"Dataset stats saved to {dataset_stats_path}")
- The final model is saved via trainer.save_model().
- Multi-GPU training is supported via --devices argument and mp.spawn.
"""


class DatasetStatsCallback(TrainerCallback):
    def __init__(self, model, log_dir):
        self.model = model
        self.log_dir = log_dir

    def on_save(self, args, state, control, **kwargs):
        # nur auf Rank 0 speichern
        if not dist.is_initialized() or dist.get_rank() == 0:
            stats_path = os.path.join(self.log_dir, "dataset_stats.pkl")
            with open(stats_path, "wb") as f:
                pkl.dump(self.model.original_dataset_stats, f)
            print(f"Dataset stats saved to {stats_path}")


def get_pretrained_model(
    model_path: str,
    device=0,
):
    """
    Baut exakt das gleiche QwenActor-Modell wie im Training
    und lädt Checkpoint + Dataset-Stats korrekt.
    """

    device = f"cuda:{device}" if isinstance(device, int) else device

    # -------------------------------------------------
    # 1️⃣ Config laden
    # -------------------------------------------------
    model_folder = os.path.dirname(
        model_path
    )  # .../full_model/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit
    version_folder = os.path.basename(
        model_path
    )  # train_steps10_lora16_Batch8_GradAccu2_Horizon8_v0
    parent_model_name = os.path.basename(
        model_folder
    )  # Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit

    cfg_path = os.path.join(model_folder, f"{parent_model_name}_config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found at {cfg_path}")

    cfg = get_cfg(cfg_path, cfg_opts="")

    # -------------------------------------------------
    # 2️⃣ Modell bauen (Inference-Modus)
    # -------------------------------------------------
    model = get_model(
        cfg,
        calculate_dataset_stats=False,
        for_training=False,
    ).to(device)
    stats_path = os.path.join(os.path.dirname(model_path), "dataset_stats.pkl")
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            dataset_stats = pkl.load(f)
            model.set_dataset_stats(dataset_stats)
    model.eval()

    print("Pretrained model ready for eval & inference")

    return model, cfg


def get_cfg(cfg_path="", cfg_opts=""):
    """
    Loads config with optional overrides.

    Automatically switches to full_model config if cfg_path points
    to a model folder under runs/full_model.

    :param cfg_path: Path to config.yaml or model folder
    :param cfg_opts: Overrides from command line
    """
    cfg = get_cfg_defaults()

    if cfg_path and os.path.isdir(cfg_path) and "full_model" in cfg_path:
        model_name = os.path.basename(cfg_path)
        cfg_path_candidate = os.path.join(cfg_path, f"{model_name}_config.yaml")
        if os.path.exists(cfg_path_candidate):
            cfg_path = cfg_path_candidate
        else:
            raise FileNotFoundError(
                f"Config for inference not found at {cfg_path_candidate}"
            )

    if cfg_path != "":
        cfg.merge_from_file(cfg_path)

    if cfg_opts != "":
        cfg.merge_from_list(cfg_opts.split(" "))
        cfg.EXP.EXP_ID += f"_{utils.short_name(cfg_opts)}"

    cfg.freeze()
    print(cfg)
    return cfg


def get_inp(cfg, data_batch):
    """
    Constructs the input for the model using the batched data.
    :param cfg: config object
    :param data_batch: contains the batched data provided by the dataloader
    """

    inp = data_batch
    return inp


def get_model(
    cfg,
    calculate_dataset_stats: bool = True,
    for_training: bool = True,
):
    """
    Returns model based on the config.
    for_training:
        True  -> Training (Unsloth training mode)
        False -> Inference / Evaluation
    """

    if cfg.EXP.MODEL == "unsloth":
        model = models.QwenActor(
            **cfg.MODEL.UNSLOTH,
            for_training=for_training,
        )
    else:
        raise AssertionError(f"Invalid model: {cfg.EXP.MODEL}")

    # -------------------------------------------------
    # Dataset stats NUR im Training berechnen
    # -------------------------------------------------
    if calculate_dataset_stats and for_training:
        temp_dataset = get_dataloader(
            split="train",
            cfg=cfg,
            get_dataset=True,
        )
        model.set_dataset_stats(temp_dataset.stats)
        del temp_dataset

    return model


def default_batch_proc(data_batch, device):
    for x in data_batch:
        if isinstance(data_batch[x], dict):
            for y in data_batch[x]:
                data_batch[x][y] = data_batch[x][y].to(device).float()
        else:
            if isinstance(data_batch[x], torch.Tensor):
                data_batch[x] = data_batch[x].to(device).float()
            else:
                data_batch[x] = data_batch[x]
    return data_batch


def get_dataloader(split, cfg, get_dataset=False):
    """
    Returns dataloader based on the config and split
    :param get_dataset: whether to return the dataset or the dataloader
    """
    num_workers = cfg.DATALOADER.num_workers
    dataset_args = {"split": split}

    if cfg.EXP.DATASET == "roboverse":
        print("WARNING: split is ignored for roboverse dataset.")
        dataset_args = dict(**cfg.DATALOADER.ROBOVERSE)
        dataset = roboverse.get_unified_dataset(**dataset_args)
    else:
        raise NotImplementedError

    if "batch_proc" not in dir(dataset):
        dataset.batch_proc = default_batch_proc

    if get_dataset:
        return dataset
    else:
        return DataLoader(
            dataset,
            num_workers=num_workers,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            pin_memory=(torch.cuda.is_available()) and (not num_workers),
            persistent_workers=(num_workers > 0),
        )


def check_grad(model, loss):
    bad_grad = False
    if loss.ne(loss).any():
        bad_grad = True
        print("WARNING: nan in the loss")
    else:
        for x in model.parameters():
            if x.grad is not None:
                if x.grad.ne(x.grad).any():
                    print("WARNING: nan in a gradient")
                    bad_grad = True
                    break
                if ((x.grad == float("inf")) | (x.grad == float("-inf"))).any():
                    print("WARNING: inf in a gradient")
                    bad_grad = True
                    break
    return bad_grad


def print_model_stats(model):
    """Print model statistics including parameter counts."""
    # Get model module if using DDP
    model_module = model.module if isinstance(model, DDP) else model

    # Count total parameters
    total_params = sum(p.numel() for p in model_module.parameters())

    # Count trainable parameters
    trainable_params = sum(
        p.numel() for p in model_module.parameters() if p.requires_grad
    )

    # Count non-trainable parameters
    non_trainable_params = total_params - trainable_params

    print("=" * 50)
    print("Model Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("=" * 50)


def get_log_dir(cfg, logdir_with_time=False):
    if logdir_with_time:
        log_dir = (
            f"./runs/{cfg.EXP.EXP_ID}/{str(datetime.now())[:-7].replace(' ', '-')}"
        )
    else:
        log_dir = f"./runs/{cfg.EXP.EXP_ID}"
    return log_dir


def entry_train(
    rank,
    cfg,
    logdir_with_time=False,
    resume=False,
    model_path="",
    devices=[0],
    port=12345,
):
    multi_gpu = len(devices) > 1
    device = f"cuda:{devices[rank]}" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Multi-GPU Setup
    # -------------------------
    if multi_gpu:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=len(devices),
            rank=rank,
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # -------------------------
    # Log-Ordner
    # -------------------------
    log_dir = get_log_dir(cfg, logdir_with_time)
    os.makedirs(log_dir, exist_ok=True)

    # -------------------------
    # Dataset
    # -------------------------
    loader_train = get_dataloader(split="train", cfg=cfg, get_dataset=True)
    train_dataset = models.QwenSFTDataset(
        loader_train, None
    )  # model wird später gesetzt

    # -------------------------
    # Model laden (Unsloth/FastVisionModel)
    # -------------------------
    model = get_model(cfg, calculate_dataset_stats=not resume, for_training=True).to(
        device
    )

    if resume and model_path:
        # Dataset-Stats laden
        stats_path = os.path.join(os.path.dirname(model_path), "dataset_stats.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                model.set_dataset_stats(pkl.load(f))

    # Multi-GPU: DDP auf model.model (FastVisionModel)
    if multi_gpu:
        model.model = DDP(model.model, device_ids=[device], output_device=device)

    # Model im Dataset Wrapper setzen
    train_dataset.model = model

    # -------------------------
    # Trainer
    # -------------------------
    stats_callback = DatasetStatsCallback(model, log_dir)

    trainer = SFTTrainer(
        model=model.model,
        tokenizer=model.tokenizer,
        train_dataset=train_dataset,
        data_collator=models.MaskedVisionDataCollator(
            model.model,
            model.tokenizer,
            train_on_responses_only=True,
            instruction_part="user",
            response_part="assistant",
            completion_only_loss=True,
        ),
        args=SFTConfig(
            per_device_train_batch_size=cfg.TRAIN.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.TRAIN.gradient_accumulation_steps,
            warmup_steps=cfg.TRAIN.warmup_steps,
            max_steps=cfg.TRAIN.max_steps,
            learning_rate=cfg.TRAIN.learning_rate,
            logging_steps=cfg.TRAIN.logging_steps,
            optim=cfg.TRAIN.optim,
            weight_decay=cfg.TRAIN.weight_decay,
            lr_scheduler_type=cfg.TRAIN.lr_scheduler_type,
            seed=cfg.TRAIN.seed,
            output_dir=log_dir,
            report_to=cfg.TRAIN.report_to,
            remove_unused_columns=cfg.TRAIN.remove_unused_columns,
            dataset_text_field=cfg.TRAIN.dataset_text_field,
            dataset_kwargs=dict(cfg.TRAIN.dataset_kwargs),
            max_length=cfg.TRAIN.max_length,
            save_steps=cfg.TRAIN.save_steps,
            save_total_limit=cfg.TRAIN.save_total_limit,
        ),
    )
    trainer.add_callback(stats_callback)

    # -------------------------
    # Training starten
    # -------------------------
    trainer.train(resume_from_checkpoint=resume and bool(model_path))

    # -------------------------
    # Speichern nur auf Rank 0
    # -------------------------
    if not multi_gpu or (multi_gpu and dist.get_rank() == 0):
        trainer.save_model()
        stats_path = os.path.join(log_dir, "dataset_stats.pkl")
        with open(stats_path, "wb") as f:
            pkl.dump(train_dataset.stats, f)
        print(f"Dataset stats saved to {stats_path}")

    # Multi-GPU: Process Group zerstören
    if multi_gpu:
        dist.destroy_process_group()


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", type=str, default="train")
    parser.add_argument("--exp-config", type=str, default="")
    parser.add_argument("--exp-cfg-opts", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--logdir-with-time", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0")

    cmd_args = parser.parse_args()

    # from rv_train.configs import get_cfg

    _cfg = get_cfg(cmd_args.exp_config, cmd_args.exp_cfg_opts)
    devices = [int(x) for x in cmd_args.devices.split(",")]

    if len(devices) > 1:
        mp.spawn(
            entry_train,
            args=(
                _cfg,
                cmd_args.logdir_with_time,
                cmd_args.resume,
                cmd_args.model_path,
                devices,
                27000 + random.randint(0, 3000),
            ),
            nprocs=len(devices),
            join=True,
        )
    else:
        entry_train(
            rank=0,
            cfg=_cfg,
            logdir_with_time=cmd_args.logdir_with_time,
            resume=cmd_args.resume,
            model_path=cmd_args.model_path,
            devices=devices,
        )
