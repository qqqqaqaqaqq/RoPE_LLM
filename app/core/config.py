from dataclasses import dataclass
import torch

@dataclass
class GlobalConfig:
    vocab_size: int = 50000

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size: int = 1024
    epochs: int = 100
    lr: float = 3e-4
    accum_steps: int = 4
    weight_decay: float = 0.005
    patience: int = 10

    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.2
    dim_feedforward: int = 2048

    max_len: int = 48
    train_val_ratio : float = 0.99

    temperature:float = 0.3
    repetition_penalty:float = 1.2
    top_p: float = 0.85
    top_k: int = 1

    # --- A100 전용 가속 설정 ---
    use_compile: bool = True     # torch.compile 사용 여부, 이진탐색으로 변경되면서 False
    use_amp: bool = True         # bfloat16 mixed precision 사용
    tf32: bool = True            # TensorFloat-32 가속 사용 여부

    # --- [추가] 체크포인팅 여부 (오타 방지용) ---
    # 글카 점유율 낮춰줌
    use_checkpointing: bool = False

# 실행
config = GlobalConfig()