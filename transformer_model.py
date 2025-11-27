# transformer_model.py
# BertAbsSum model (PhoBERT encoder + Transformer decoder)
# Includes BertDecoder, BertPositionEmbedding, and BertAbsSum.
# Ensure this file is placed in your src/ (or update imports accordingly).

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings

from transformer_utils import timing, get_logger

# Các hàm lớp DecoderLayer, Models2... bạn vẫn dùng từ repo gốc (src/transformer)
# đảm bảo đường dẫn import đúng (bạn đã có file Layers.py và Models2.py)
from src.transformer.Layers import DecoderLayer
from src.transformer.Models2 import (
    get_non_pad_mask,
    get_sinusoid_encoding_table,
    get_attn_key_pad_mask,
    get_subsequent_mask
)

logger = get_logger(__name__)


class BertPositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, hidden_size):
        super().__init__()
        table = get_sinusoid_encoding_table(
            n_position=max_seq_len + 1,
            d_hid=hidden_size,
            padding_idx=0
        )
        self.embedding = nn.Embedding.from_pretrained(table, freeze=True)

    def forward(self, x):
        return self.embedding(x)


class BertDecoder(nn.Module):
    def __init__(self, config, constants, device):
        super().__init__()
        self.device = device
        self.constants = constants

        bert_conf = BertConfig.from_dict(config["bert_config"])
        dec = config["decoder_config"]

        # Use BertEmbeddings for token/segment/position embeddings (wordpiece compat)
        self.seq_embedding = BertEmbeddings(bert_conf)
        # position embedding is sinusoidal table (wrapped as embedding)
        self.pos_embedding = BertPositionEmbedding(
            max_seq_len=constants.MAX_TGT_SEQ_LEN,
            hidden_size=dec["d_model"]
        )

        # Transformer decoder layers (your Layers.DecoderLayer)
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=dec["d_model"],
                d_inner=dec["d_inner"],
                n_head=dec["n_head"],
                d_k=dec["d_k"],
                d_v=dec["d_v"]
            ) for _ in range(dec["n_layers"])
        ])

        # final linear projection to vocab
        self.linear = nn.Linear(dec["d_model"], dec["vocab_size"])

    @timing
    def forward(self, src_seq, enc_output, tgt_seq):
        """
        src_seq: (B, S) - needed to create enc_mask (attention pad mask)
        enc_output: (B, S_enc, D)
        tgt_seq: (B, T)
        returns: logits (B, T, V), None
        """
        tgt_seq = tgt_seq.to(self.device)

        batch_size, length = tgt_seq.size()
        pos = torch.arange(1, length + 1).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        dec_mask = (get_attn_key_pad_mask(tgt_seq, tgt_seq) +
                    get_subsequent_mask(tgt_seq)).gt(0)
        non_pad = get_non_pad_mask(tgt_seq)
        enc_mask = get_attn_key_pad_mask(src_seq, tgt_seq)

        out = self.seq_embedding(tgt_seq) + self.pos_embedding(pos)

        for layer in self.layers:
            out, _, _ = layer(out, enc_output, non_pad, dec_mask, enc_mask)

        return self.linear(out), None


class BertAbsSum(nn.Module):
    """
    PhoBERT (or other BERT) encoder + Transformer decoder for abstractive summarization.
    config: dict with "bert_model", "bert_config", "decoder_config", "freeze_encoder" keys
    constants: object with PAD/BOS/EOS/MAX_TGT_SEQ_LEN
    device: 'cpu' or 'cuda'
    """
    def __init__(self, config, constants, device):
        super().__init__()
        # store config so we can save it in checkpoint and use at inference
        self.config = config
        self.device = device
        self.constants = constants

        # encoder from transformers (PhoBERT)
        self.encoder = AutoModel.from_pretrained(config["bert_model"])

        if config.get("freeze_encoder", False):
            for p in self.encoder.parameters():
                p.requires_grad = False

        # decoder instance
        self.decoder = BertDecoder(config, constants, device)

    def batch_encode_src_seq(self, src, mask):
        # return last_hidden_state
        return self.encoder(input_ids=src, attention_mask=mask)[0]

    def forward(self, batch_src_seq, batch_src_mask, batch_tgt_seq, batch_tgt_mask):
        # used during training: returns logits (B, T, V)
        enc_out = self.batch_encode_src_seq(batch_src_seq, batch_src_mask)
        logits, _ = self.decoder(batch_src_seq, enc_out, batch_tgt_seq)
        return logits

    @timing
    def greedy_decode(self, batch):
        """
        batch: tuple like (None, ids, mask, None, None) used by web.py
        returns: dec_seq (tensor) (B, T_generated)
        """
        src = batch[1].to(self.device)
        mask = batch[2].to(self.device)

        enc_out = self.batch_encode_src_seq(src, mask)

        dec_seq = torch.full((src.size(0), 1),
                             self.constants.BOS,
                             dtype=torch.long).to(self.device)

        for _ in range(self.constants.MAX_TGT_SEQ_LEN):
            logits, _ = self.decoder(src, enc_out, dec_seq)
            next_tok = logits[:, -1].argmax(-1).unsqueeze(1)
            dec_seq = torch.cat([dec_seq, next_tok], dim=1)

        return dec_seq
