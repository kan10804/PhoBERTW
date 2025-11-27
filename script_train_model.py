# script_train_model.py
# Training script for BertAbsSum (single-GPU or DDP)
# This script builds config, trains and saves a FULL checkpoint:
# {"model": state_dict, "model_config": config, "model_arguments": {...}, "optimizer_state_dict": ...}

import os
import sys
import json
import shutil
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.nn.init import xavier_normal_
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModel
from transformer_model import BertAbsSum
from transformer_preprocess import DataProcessor
from transformer_utils import *
from params_helper import Constants, Params

set_seed(0)
logger = get_logger(__name__)

# CUDA / visible GPUs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if Params.visible_gpus == "-1":
    Params.visible_gpus = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = Params.visible_gpus
num_gpus = torch.cuda.device_count()

# load tokenizer / bert model for config
tokenizer = AutoTokenizer.from_pretrained(Params.bert_model)
bert_model = AutoModel.from_pretrained(Params.bert_model)


def init_process(rank, world_size):
    if world_size <= 1:
        return
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = Params.ddp_master_port
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def get_model(rank, device, checkpoint, output_dir):
    logger.info(f"*** Getting model at rank {rank} ***")

    # build config (or load from resume checkpoint dir)
    if Params.resume_from_epoch > 0 and Params.resume_checkpoint_dir:
        config_path = os.path.join(Params.resume_checkpoint_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        bert_cfg = bert_model.config
        config = {
            "bert_model": Params.bert_model,
            "bert_config": bert_cfg.__dict__,
            "decoder_config": {
                "vocab_size": bert_cfg.vocab_size,
                "d_word_vec": bert_cfg.hidden_size,
                "n_layers": Params.decoder_layers_num,
                "n_head": bert_cfg.num_attention_heads,
                "d_k": Params.decoder_attention_dim,
                "d_v": Params.decoder_attention_dim,
                "d_model": bert_cfg.hidden_size,
                "d_inner": bert_cfg.intermediate_size
            },
            "freeze_encoder": Params.freeze_encoder
        }

    if rank == 0:
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4, default=str)

    model = BertAbsSum(config=config, constants=Constants, device=device)

    # checkpoint can be dict with "model" or "model_state_dict" or direct state_dict
    if checkpoint:
        if "model" in checkpoint:
            state = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint
        # load tolerant
        load_res = model.load_state_dict(state, strict=False)
        missing = getattr(load_res, "missing_keys", None) or getattr(load_res, "missing", None) or []
        unexpected = getattr(load_res, "unexpected_keys", None) or getattr(load_res, "unexpected", None) or []
        logger.info("Loaded checkpoint: missing keys=%s unexpected keys=%s", missing, unexpected)

    model.to(device)

    if num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    return model


def get_optimizer(model, checkpoint):
    params_iter = model.module.named_parameters() if isinstance(model, DistributedDataParallel) else model.named_parameters()
    model_params = list(params_iter)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n, p in model_params if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    opt = torch.optim.AdamW(groups, lr=Params.learning_rate)

    if checkpoint and "optimizer_state_dict" in checkpoint:
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
    return opt


def get_train_dataloader(rank, world_size):
    logger.info(f"Loading train data at {Params.train_data_path}")
    data = torch.load(Params.train_data_path)
    proc = DataProcessor()
    loader = proc.create_distributed_dataloader(rank, world_size, data, Params.train_batch_size) if world_size > 1 else proc.create_dataloader(data, Params.train_batch_size)
    check_data(loader)
    return loader


def get_valid_dataloader(rank, world_size):
    logger.info(f"Loading valid data at {Params.valid_data_path}")
    data = torch.load(Params.valid_data_path)
    loader = DataProcessor().create_dataloader(data, Params.valid_batch_size, shuffle=False)
    check_data(loader)
    return loader


def check_data(dataloader):
    logger.info("*** Checking data ***")
    batch = next(iter(dataloader))
    src_ids = batch[1]
    tgt_ids = batch[3]
    logger.info(f"Source (sample): {tokenizer.decode(src_ids[0], skip_special_tokens=True)}")
    logger.info(f"Target (sample): {tokenizer.decode(tgt_ids[0], skip_special_tokens=True)}")


def cal_performance(logits, ground):
    # logits: (B, T, V), ground: (B, T)
    ground = ground[:, 1:]
    logits = logits.view(-1, logits.size(-1))
    ground = ground.contiguous().view(-1)
    loss = F.cross_entropy(logits, ground, ignore_index=Constants.PAD, label_smoothing=Params.label_smoothing_factor)
    pad = ground.ne(Constants.PAD)
    pred = logits.max(-1)[1]
    correct = pred.eq(ground).masked_select(pad).sum().item()
    n_tokens = pad.sum().item()
    return loss, correct, n_tokens


def init_parameters(model):
    for n, p in model.named_parameters():
        # do not init encoder params (we use pretrained bert) but init others
        if "encoder" not in n and p.dim() > 1:
            xavier_normal_(p)


def train(rank, world_size, output_dir):
    if world_size > 1:
        init_process(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Rank {rank}/{world_size} process initialized.")

    train_loader = get_train_dataloader(rank, world_size)
    valid_loader = get_valid_dataloader(rank, world_size)

    checkpoint = None
    if num_gpus > 1:
        dist.barrier()

    model = get_model(rank, device, checkpoint, output_dir)
    init_parameters(model)
    optimizer = get_optimizer(model, checkpoint)

    model.train()
    global_step = 0

    for epoch in range(1, Params.num_train_epochs + 1):
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}", ascii=True)
        for step, batch in enumerate(train_iter, start=1):
            global_step += 1

            batch_src = batch[1].to(device)
            batch_mask = batch[2].to(device)
            batch_tgt = batch[3].to(device)
            tgt_mask = batch[4].to(device)

            logits = model(batch_src_seq=batch_src, batch_src_mask=batch_mask, batch_tgt_seq=batch_tgt, batch_tgt_mask=tgt_mask)

            loss, correct, total = cal_performance(logits, batch_tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                train_iter.set_postfix({"Loss": loss.item()})

    # Save full checkpoint (main process only)
    if rank == 0:
        save_path = os.path.join(output_dir, "Best_Checkpoint.pt")
        model_to_save = model.module if isinstance(model, DistributedDataParallel) else model

        checkpoint_to_save = {
            "model": model_to_save.state_dict(),
            "model_config": model_to_save.config,
            "model_arguments": {
                "max_src_len": getattr(Params, "max_src_len", None),
                "max_tgt_len": getattr(Params, "max_tgt_len", None),
                "decoder_layers": getattr(Params, "decoder_layers_num", None)
            },
            "optimizer_state_dict": optimizer.state_dict()
        }

        torch.save(checkpoint_to_save, save_path)
        logger.info(f"Saved FULL checkpoint -> {save_path}")

    if world_size > 1:
        dist.destroy_process_group()


def cleanup_on_error(out):
    if os.path.isdir(out) and len(os.listdir(out)) < 2:
        shutil.rmtree(out)


if __name__ == "__main__":
    WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 1
    normalized = Params.bert_model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    model_dir = f"model_{normalized}_{Params.decoder_layers_num}layers_{timestamp}"
    output_dir = os.path.join(Params.output_dir, model_dir)
    os.makedirs(output_dir, exist_ok=True)

    try:
        if WORLD_SIZE == 1:
            train(0, 1, output_dir)
        else:
            mp.spawn(train, args=(WORLD_SIZE, output_dir), nprocs=WORLD_SIZE)
    except Exception as e:
        logger.error("Training failed!")
        cleanup_on_error(output_dir)
        raise e
