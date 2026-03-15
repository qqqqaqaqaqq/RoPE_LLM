# Insert Pad

def insert_pad(sequences, max_len: int, pad_id: int) -> list:
    for i in range(len(sequences)):
        seq = sequences[i]
        if len(seq) < max_len:
            sequences[i] = seq + [pad_id] * (max_len - len(seq))
        else:
            sequences[i] = seq[:max_len]
    return sequences