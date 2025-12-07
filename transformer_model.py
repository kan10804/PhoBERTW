import torch.nn as nn
import torch
import operator
from torch.nn.functional import log_softmax
from transformer.Layers import DecoderLayer
from transformer.Models2 import (
    get_non_pad_mask,
    get_sinusoid_encoding_table,
    get_attn_key_pad_mask,
    get_subsequent_mask,
)
from transformers import AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformer_utils import *
from params_helper import Params, Constants

logger = get_logger(__name__)

class BertPositionEmbedding(nn.Module):
    """
    Sinh positional encoding dạng sinusoidal cho decoder.
    """
    def __init__(self, max_seq_len, hidden_size, padding_idx=0):
        super().__init__()

        sinusoid_encoding = get_sinusoid_encoding_table(
            n_position=max_seq_len + 1,
            d_hid=hidden_size,
            padding_idx=padding_idx,
        )

        self.embedding = nn.Embedding.from_pretrained(
            embeddings=sinusoid_encoding, freeze=True
        )

    def forward(self, x):
        return self.embedding(x)


class BertDecoder(nn.Module):
    """
    Decoder Transformer: tự xây dựng gồm nhiều DecoderLayer.
    Nhận: output từ encoder BERT và chuỗi target.
    """
    def __init__(self, config, device, dropout=0.1):
        super().__init__()
        self.device = device

        bert_config = BertConfig.from_dict(config['bert_config'])
        decoder_config = config['decoder_config']

        self.sequence_embedding = BertEmbeddings(config=bert_config)
        self.position_embedding = BertPositionEmbedding(
            max_seq_len=Constants.MAX_TGT_SEQ_LEN,
            hidden_size=decoder_config['d_model'],
        )

        self.layer_stack = nn.ModuleList([
            DecoderLayer(
                decoder_config['d_model'],
                decoder_config['d_inner'],
                decoder_config['n_head'],
                decoder_config['d_k'],
                decoder_config['d_v'],
                dropout=dropout,
            )
            for _ in range(decoder_config['n_layers'])
        ])

        self.last_linear = nn.Linear(
            in_features=decoder_config['d_model'],
            out_features=decoder_config['vocab_size'],
        )

    @timing
    def forward(self, batch_src_seq, batch_enc_output, batch_tgt_seq):
        # Đảm bảo dữ liệu nằm trên đúng device
        batch_tgt_seq = batch_tgt_seq.to(self.device)
        if batch_enc_output is not None:
            batch_enc_output = batch_enc_output.to(self.device)

        # Tạo mask cho decoder
        dec_non_pad_mask = get_non_pad_mask(batch_tgt_seq)
        slf_attn_mask_subseq = get_subsequent_mask(batch_tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(batch_tgt_seq, batch_tgt_seq)
        dec_slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(batch_src_seq, batch_tgt_seq)

        # Tạo positional id
        batch_size, tgt_len = batch_tgt_seq.size()
        batch_tgt_pos = (
            torch.arange(1, tgt_len + 1)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to(self.device)
        )

        # Embedding = token embed + positional embed
        dec_output = self.sequence_embedding(batch_tgt_seq) + self.position_embedding(batch_tgt_pos)

        # Qua từng lớp decoder
        dec_slf_attn_list, dec_enc_attn_list = [], []
        for layer in self.layer_stack:
            dec_output, self_att, enc_att = layer(
                dec_output,
                batch_enc_output,
                non_pad_mask=dec_non_pad_mask,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask,
            )
            dec_slf_attn_list.append(self_att)
            dec_enc_attn_list.append(enc_att)

        # Dự đoán vocab
        batch_logits = self.last_linear(dec_output)
        return batch_logits, dec_enc_attn_list


class BertAbsSum(nn.Module):
    """
    Kiến trúc Encoder–Decoder:
    - Encoder: BERT hoặc PhoBERT
    - Decoder: Transformer Decoder tự xây dựng
    """
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config

        # Load BERT encoder
        self.encoder = AutoModel.from_pretrained(config['bert_model'])

        # Tùy chọn đóng băng encoder
        if config.get('freeze_encoder', False):
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Decoder transformer
        self.decoder = BertDecoder(config=config, device=device)

        # Log thống kê số params
        stats = self.get_model_stats()
        logger.info(f"Encoder parameters: {stats['enc_params']:,}")
        logger.info(f"Decoder parameters: {stats['dec_params']:,}")
        logger.info(f"Total parameters: {stats['total_params']:,}")

    def get_model_stats(self):
        """Đếm tổng số tham số của encoder và decoder."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        enc_train = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        dec_train = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        return {
            'enc_params': enc_params,
            'dec_params': dec_params,
            'total_params': enc_params + dec_params,
            'enc_trainable_params': enc_train,
            'dec_trainable_params': dec_train,
            'total_trainable_params': enc_train + dec_train,
        }

    def forward(self, batch_src_seq, batch_src_mask, batch_tgt_seq, batch_tgt_mask):
        # Dịch phải target trong chế độ teacher forcing
        batch_tgt_seq = batch_tgt_seq[:, :-1]

        # Encode
        batch_enc_output = self.batch_encode_src_seq(batch_src_seq, batch_src_mask)

        # Decode
        logits, _ = self.decoder(batch_src_seq, batch_enc_output, batch_tgt_seq)
        return logits

    def batch_encode_src_seq(self, batch_src_seq, batch_src_mask):
        
        batch_src_seq = batch_src_seq.to(self.device)
        batch_src_mask = batch_src_mask.to(self.device)

        bert_name = Params.bert_model.lower() if isinstance(Params.bert_model, str) else ''

        # PhoBERT: Win
        if 'phobert' in bert_name and Params.max_src_len > 256:
            window = 256
            seq1, mask1 = batch_src_seq[:, :window], batch_src_mask[:, :window]
            seq2, mask2 = batch_src_seq[:, window:], batch_src_mask[:, window:]

            out1 = self.encoder(input_ids=seq1, attention_mask=mask1)[0]

            if seq2.size(1) > 0:
                out2 = self.encoder(input_ids=seq2, attention_mask=mask2)[0]
                return torch.cat([out1, out2], dim=1)
            return out1

        # BERT chuẩn
        return self.encoder(input_ids=batch_src_seq, attention_mask=batch_src_mask)[0]

    @timing
    def greedy_decode(self, batch):
       # Giải mã chuỗi bằng greedy search.
        batch_src_seq = batch[1].to(self.device)
        batch_src_mask = batch[2].to(self.device)

        enc_output = self.batch_encode_src_seq(batch_src_seq, batch_src_mask)
        dec_seq = torch.full((batch_src_seq.size(0), 1), Constants.BOS, dtype=torch.long, device=self.device)

        for _ in range(Constants.MAX_TGT_SEQ_LEN):
            logits, _ = self.decoder(batch_src_seq, enc_output, dec_seq)
            next_token = logits[:, -1].argmax(-1)
            dec_seq = torch.cat([dec_seq, next_token.unsqueeze(-1)], dim=1)

        return dec_seq

    @timing
    def beam_decode(self, batch_guids, batch_src_seq, batch_src_mask, beam_size, n_best):
        # Beam search để sinh chuỗi tốt hơn greedy.
        batch_size = len(batch_guids)
        batch_src_seq = batch_src_seq.to(self.device)
        batch_src_mask = batch_src_mask.to(self.device)
        enc_output = self.batch_encode_src_seq(batch_src_seq, batch_src_mask)

        decoded_batch = []

        for idx in range(batch_size):
            gui = batch_guids[idx]
            src = batch_src_seq[idx].unsqueeze(0)
            enc = enc_output[idx].unsqueeze(0)

            beams = []
            start = BeamSearchNode(None, Constants.BOS, 0)
            beams.append((start.eval(), start))
            end_nodes = []

            for _ in range(Constants.MAX_TGT_SEQ_LEN):
                candidates = []
                for score, node in beams:
                    dec_seq = torch.LongTensor(node.seq_tokens).unsqueeze(0).to(self.device)
                    logits, _ = self.decoder(src, enc, dec_seq)

                    log_probs = log_softmax(logits[:, -1][0], dim=-1)
                    top_log_probs, top_idx = torch.sort(log_probs, descending=True)

                    count = 0
                    i = 0
                    while count < beam_size and i < top_idx.size(0):
                        token = top_idx[i].item()
                        prob = top_log_probs[i].item()
                        i += 1

                        new_node = BeamSearchNode(node, token, node.log_prob + prob)

                        if token == Constants.EOS:
                            end_nodes.append((new_node.eval(), new_node))
                        else:
                            candidates.append((new_node.eval(), new_node))
                            count += 1

                if len(end_nodes) >= n_best:
                    break

                if not candidates:
                    break

                candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_size]

            if len(end_nodes) < n_best:
                beams = sorted(beams, key=lambda x: x[0], reverse=True)
                end_nodes.extend(beams[: n_best - len(end_nodes)])

            end_nodes = sorted(end_nodes, key=lambda x: x[0], reverse=True)
            decoded_batch.append([(score, node.seq_tokens) for score, node in end_nodes[:n_best]])

        return decoded_batch


class BeamSearchNode:
 
    # Node trong beam search, lưu trạng thái chuỗi.
   
    def __init__(self, prev_node, token_id, log_prob):
        self.prev_node = prev_node
        self.token_id = token_id
        self.log_prob = log_prob

        self.seq_tokens = [token_id] if prev_node is None else prev_node.seq_tokens + [token_id]
        self.seq_len = len(self.seq_tokens)
        self.finished = token_id == Constants.EOS

    def set_log_prob(self, log_prob):
        self.log_prob = log_prob

    def eval(self):
        norm = 5
        length_norm = ((norm + self.seq_len) / (norm + 1)) ** getattr(Params, 'len_norm_factor', 0)
        return self.log_prob / length_norm

    def __lt__(self, other):
        return self.seq_len < other.seq_len

    def __gt__(self, other):
        return self.seq_len > other.seq_len