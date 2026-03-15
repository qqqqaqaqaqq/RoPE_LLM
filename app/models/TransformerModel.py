import torch
import torch.nn as nn
from app.models.RoPE_EncoderLayer import CustomEncoderLayer

class TranslateModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float, pad_token_id: int, cls_token_id: int, eos_token_id: int, mask_token_id: int, **kwargs):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id

        self.embedding = nn.Embedding(vocab_size, d_model)

        # RoPE 이론 적용
        self.encoder_layers = nn.ModuleList([
            CustomEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Translate Decoder
        translate_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.translate_transformer_decoder = nn.TransformerDecoder(
            translate_decoder_layer,
            num_layers=num_layers
        )

        self.translate_decoder = nn.Linear(d_model, vocab_size)

    # 순전파
    def forward(self, x:torch.Tensor, tgt:torch.Tensor):
        device = x.device

        # tgt 내부에 eos 있으면 pad로 전체 치환
        # tgt : ["CLS", "나는", "바보", "다", ".", "PAD", "PAD", "PAD", "PAD"]
        tgt[tgt == self.eos_token_id] = self.pad_token_id

        seq_len_tgt = tgt.size(1)

        # PAD 처리
        src_key_padding_mask = (x == self.pad_token_id)
        tgt_key_padding_mask = (tgt == self.pad_token_id)

        x_emb = self.embedding(x)

        for layer in self.encoder_layers:
            x_emb = layer(
                x_emb,
                src_key_padding_mask=src_key_padding_mask
          )

        encoded = x_emb

        tgt_emb = self.embedding(tgt)
        tgt_mask = torch.triu(torch.ones(seq_len_tgt, seq_len_tgt), diagonal=1).bool().to(device)

        translate_decoded_out = self.translate_transformer_decoder(
            tgt=tgt_emb,
            memory=encoded,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        logits = self.translate_decoder(translate_decoded_out)

        return logits
    

class RestoreModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float, pad_token_id: int, cls_token_id: int, eos_token_id: int, mask_token_id: int, **kwargs):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id

        self.embedding = nn.Embedding(vocab_size, d_model)

        # RoPE 이론 적용
        self.encoder_layers = nn.ModuleList([
            CustomEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Translate Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.decoder = nn.Linear(d_model, vocab_size)

    # 순전파
    def forward(self, x:torch.Tensor, tgt:torch.Tensor):
        device = x.device

        # tgt 내부에 eos 있으면 pad로 전체 치환
        # tgt : ["CLS", "나는", "바보", "다", ".", "PAD", "PAD", "PAD", "PAD"]
        tgt[tgt == self.eos_token_id] = self.pad_token_id

        seq_len_tgt = tgt.size(1)

        # PAD 처리
        src_key_padding_mask = (x == self.pad_token_id)
        tgt_key_padding_mask = (tgt == self.pad_token_id)

        x_emb = self.embedding(x)

        for layer in self.encoder_layers:
            x_emb = layer(
                x_emb,
                src_key_padding_mask=src_key_padding_mask
          )

        encoded = x_emb

        tgt_emb = self.embedding(tgt)
        tgt_mask = torch.triu(torch.ones(seq_len_tgt, seq_len_tgt), diagonal=1).bool().to(device)

        translate_decoded_out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=encoded,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        logits = self.decoder(translate_decoded_out)

        return logits