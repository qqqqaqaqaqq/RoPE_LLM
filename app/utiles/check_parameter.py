from app.core.config import GlobalConfig

# 파라미터 계산 식
def count_transformer_params(config:GlobalConfig, tie_embedding=False):
    vocab = config.vocab_size
    d = config.d_model
    ff = config.dim_feedforward
    L = config.num_layers
    max_len = config.max_len

    # -----------------------
    # Embedding
    # -----------------------
    token_embedding = vocab * d
    positional_embedding = max_len * d
    embedding_total = token_embedding + positional_embedding

    # -----------------------
    # Encoder (1 layer)
    # -----------------------
    enc_self_attn = 4 * (d * d + d)  # Q,K,V,O weight+bias
    enc_ffn = d * ff + ff + ff * d + d
    enc_layernorm = 2 * (d + d)

    encoder_layer = enc_self_attn + enc_ffn + enc_layernorm
    encoder_total = encoder_layer * L

    # -----------------------
    # Decoder (1 layer)
    # -----------------------
    dec_self_attn = 4 * (d * d + d)
    dec_cross_attn = 4 * (d * d + d)
    dec_ffn = d * ff + ff + ff * d + d
    dec_layernorm = 3 * (d + d)

    decoder_layer = dec_self_attn + dec_cross_attn + dec_ffn + dec_layernorm
    decoder_total = decoder_layer * L

    # -----------------------
    # Output projection
    # -----------------------
    if tie_embedding: # embedding weight와 output projection weight를 공유하는지
        output_projection = 0
    else:
        output_projection = d * vocab + vocab

    # -----------------------
    # Total
    # -----------------------
    total = embedding_total + encoder_total + decoder_total + output_projection

    return {
        "embedding": embedding_total,
        "encoder": encoder_total,
        "decoder": decoder_total,
        "output_projection": output_projection,
        "total": total
    }