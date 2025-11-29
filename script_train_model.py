import os
import sys
import random
import numpy as np
import torch
import json
import shutil
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from datetime import datetime
from tqdm.auto import tqdm
from torch.nn.parallel import DistributedDataParallel
from torch.nn.init import xavier_normal_
from transformers import AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup
from transformer_model import BertAbsSum
from transformer_preprocess import DataProcessor
from transformer_utils import *
from params_helper import Constants, Params

set_seed(0)

logger = get_logger(__name__)

# Force enable GPU in Colab
if Params.visible_gpus == "-1":
	Params.visible_gpus = "0"

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = Params.visible_gpus
num_gpus = torch.cuda.device_count()

tokenizer = AutoTokenizer.from_pretrained(Params.bert_model, local_files_only=False)
bert_model = AutoModel.from_pretrained(Params.bert_model, local_files_only=False)


def init_process(rank, world_size):
	"""Initialize process group only when multi-GPU"""
	if world_size <= 1:
		return

	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = Params.ddp_master_port

	backend = "nccl" if torch.cuda.is_available() else "gloo"

	dist.init_process_group(
		backend=backend,
		rank=rank,
		world_size=world_size
	)


def get_model(rank, device, checkpoint, output_dir):
	logger.info(f"*** Getting model at rank {rank} ***")

	# Load or create config
	if Params.resume_from_epoch > 0 and Params.resume_checkpoint_dir:
		config_path = os.path.join(Params.resume_checkpoint_dir, "config.json")
		with open(config_path, "r") as f:
			config = json.load(f)
	else:
		bert_config = bert_model.config
		config = {
			"bert_model": Params.bert_model,
			"bert_config": bert_config.__dict__,
			"decoder_config": {
				"vocab_size": bert_config.vocab_size,
				"d_word_vec": bert_config.hidden_size,
				"n_layers": Params.decoder_layers_num,
				"n_head": bert_config.num_attention_heads,
				"d_k": Params.decoder_attention_dim,
				"d_v": Params.decoder_attention_dim,
				"d_model": bert_config.hidden_size,
				"d_inner": bert_config.intermediate_size,
			},
			"freeze_encoder": Params.freeze_encoder
		}

	# Save config only on main process
	if rank == 0:
		with open(os.path.join(output_dir, "config.json"), "w") as f:
			json.dump(config, f, indent=4, default=str)

	model = BertAbsSum(config=config, device=device)

	if checkpoint:
		# checkpoint is expected to be a dict with "model_state_dict"
		if "model_state_dict" in checkpoint:
			model.load_state_dict(checkpoint["model_state_dict"])
		else:
			# backward compatibility
			model.load_state_dict(checkpoint)

	model.to(device)

	# Wrap with DDP only when multi-GPU
	if num_gpus > 1:
		model = DistributedDataParallel(
			model,
			device_ids=[rank],
			output_device=rank,
			find_unused_parameters=True
		)

	return model


def get_optimizer(model, checkpoint):
	params_iter = model.module.named_parameters() if isinstance(model, DistributedDataParallel) else model.named_parameters()
	model_params = list(params_iter)

	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

	groups = [
		{'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
	]

	opt = torch.optim.AdamW(groups, lr=Params.learning_rate)

	if checkpoint and "optimizer_state_dict" in checkpoint:
		opt.load_state_dict(checkpoint["optimizer_state_dict"])

	return opt


def get_train_dataloader(rank, world_size):
	logger.info(f"Loading train data at {Params.train_data_path}")
	data = torch.load(Params.train_data_path)
	proc = DataProcessor()

	if world_size > 1:
		loader = proc.create_distributed_dataloader(rank, world_size, data, Params.train_batch_size)
	else:
		loader = proc.create_dataloader(data, Params.train_batch_size)

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
	logger.info(f"Source: {tokenizer.decode(src_ids[0], skip_special_tokens=True)}")
	logger.info(f"Target: {tokenizer.decode(tgt_ids[0], skip_special_tokens=True)}")


def cal_performance(logits, ground):
	ground = ground[:, 1:]
	logits = logits.view(-1, logits.size(-1))
	ground = ground.contiguous().view(-1)
	loss = F.cross_entropy(
		logits, ground,
		ignore_index=Constants.PAD,
		label_smoothing=Params.label_smoothing_factor
	)
	pad = ground.ne(Constants.PAD)
	pred = logits.max(-1)[1]
	correct = pred.eq(ground).masked_select(pad).sum().item()
	n_tokens = pad.sum().item()
	return loss, correct, n_tokens


def init_parameters(model):
	# Support DDP-wrapped model
	params_iter = model.module.named_parameters() if isinstance(model, DistributedDataParallel) else model.named_parameters()
	for n, p in params_iter:
		if "encoder" not in n and "tgt_embed" not in n and p.dim() > 1:
			xavier_normal_(p)


# New: evaluate function
def evaluate(rank, model, valid_loader, device):
	model_eval = model.module if isinstance(model, DistributedDataParallel) else model
	model_eval.eval()
	total_loss = 0.0
	total_tokens = 0
	with torch.no_grad():
		for batch in valid_loader:
			batch_src = batch[1].to(device)
			batch_mask = batch[2].to(device)
			batch_tgt = batch[3].to(device)
			tgt_mask = batch[4].to(device)

			logits = model_eval(
				batch_src_seq=batch_src,
				batch_src_mask=batch_mask,
				batch_tgt_seq=batch_tgt,
				batch_tgt_mask=tgt_mask
			)

			loss, correct, n_tokens = cal_performance(logits, batch_tgt)
			total_loss += loss.item() * n_tokens
			total_tokens += n_tokens

	if total_tokens == 0:
		return float("inf")
	return total_loss / total_tokens


def train(rank, world_size, output_dir):
	if world_size > 1:
		init_process(rank, world_size)

	device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

	logger.info(f"Rank {rank}/{world_size} process initialized.")

	train_loader = get_train_dataloader(rank, world_size)
	valid_loader = get_valid_dataloader(rank, world_size)

	# Try to load resume checkpoint if requested
	checkpoint = None
	if getattr(Params, "resume_from_epoch", 0) > 0 and getattr(Params, "resume_checkpoint_dir", None):
		# prefer explicit checkpoint file name pattern, fallback to Best_Checkpoint.pt
		cp_path = os.path.join(Params.resume_checkpoint_dir, f"checkpoint_epoch_{Params.resume_from_epoch}.pt")
		if not os.path.exists(cp_path):
			cp_path = os.path.join(Params.resume_checkpoint_dir, "Best_Checkpoint.pt")
		if os.path.exists(cp_path):
			logger.info(f"Loading checkpoint from {cp_path}")
			checkpoint = torch.load(cp_path, map_location=device)
		else:
			logger.info("Requested resume checkpoint not found, starting fresh.")

	# synchronize before model init when multi-gpu
	if num_gpus > 1:
		dist.barrier()

	model = get_model(rank, device, checkpoint, output_dir)
	init_parameters(model)

	optimizer = get_optimizer(model, checkpoint)

	# Scheduler setup
	num_training_steps = len(train_loader) * Params.num_train_epochs if len(train_loader) > 0 else 0
	warmup_steps = getattr(Params, "warmup_steps", 0)
	scheduler = None
	if num_training_steps > 0:
		scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

	# ---- Main training ---- #
	model.train()
	global_step = 0
	best_val_loss = float("inf")

	for epoch in range(1, Params.num_train_epochs + 1):
		train_iter = tqdm(train_loader, desc=f"Epoch {epoch}", ascii=True)

		for step, batch in enumerate(train_iter, start=1):
			global_step += 1

			batch_src = batch[1].to(device)
			batch_mask = batch[2].to(device)
			batch_tgt = batch[3].to(device)
			tgt_mask = batch[4].to(device)

			# Use model.module for forward if DDP wrapper is present
			model_forward = model.module if isinstance(model, DistributedDataParallel) else model

			logits = model_forward(
				batch_src_seq=batch_src,
				batch_src_mask=batch_mask,
				batch_tgt_seq=batch_tgt,
				batch_tgt_mask=tgt_mask
			)

			loss, correct, total = cal_performance(logits, batch_tgt)

			optimizer.zero_grad()
			loss.backward()

			# gradient clipping
			max_norm = getattr(Params, "max_grad_norm", 1.0)
			params_to_clip = model.module.parameters() if isinstance(model, DistributedDataParallel) else model.parameters()
			torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm)

			optimizer.step()
			if scheduler:
				scheduler.step()

			if rank == 0:
				train_iter.set_postfix({"Loss": loss.item()})

		# After each epoch evaluate on validation set (main rank only)
		if rank == 0:
			val_loss = evaluate(rank, model, valid_loader, device)
			logger.info(f"Epoch {epoch} validation loss: {val_loss:.6f}")

			# Save checkpoint for this epoch
			state = {
				"epoch": epoch,
				"model_state_dict": model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None
			}
			cp_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
			torch.save(state, cp_path)
			logger.info(f"Saved checkpoint: {cp_path}")

			# Save best
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				best_path = os.path.join(output_dir, "Best_Checkpoint.pt")
				torch.save(state, best_path)
				logger.info(f"Saved best checkpoint: {best_path}")

	# Cleanup and finalize
	if world_size > 1:
		dist.destroy_process_group()


def cleanup_on_error(out):
	if os.path.isdir(out) and len(os.listdir(out)) < 2:
		shutil.rmtree(out)


if __name__ == "__main__":
	if torch.cuda.is_available():
		WORLD_SIZE = torch.cuda.device_count()
	else:
		WORLD_SIZE = 1

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
		logger.info("Program error!")
		cleanup_on_error(output_dir)
		raise e