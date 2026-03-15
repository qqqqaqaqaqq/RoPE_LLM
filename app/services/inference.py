import os

import torch
import torch.nn.functional as F

from tokenizers import Tokenizer
from dataclasses import asdict
from app.utiles.load_Tokenizer import load_Tokenizer
from app.core.config import GlobalConfig
from app.models.TransformerModel import TranslateModel, RestoreModel

class Inference():
    def __init__(self, config:GlobalConfig, mode, BASE_DIR):
        self.config = config
        self.device = config.device
        
        self.pre_token:Tokenizer = load_Tokenizer(BASE_DIR=BASE_DIR)
        self.pad_token_id = self.pre_token.token_to_id("[PAD]")
        self.cls_token_id = self.pre_token.token_to_id("[CLS]")
        self.mask_token_id = self.pre_token.token_to_id("[MASK]")
        self.eos_token_id = self.pre_token.token_to_id("[EOS]")

        params = asdict(self.config)
        special_tokens = {
            'pad_token_id': self.pad_token_id,
            'cls_token_id': self.cls_token_id,
            'eos_token_id': self.eos_token_id,
            'mask_token_id': self.mask_token_id
        }

                
        if mode == "1":
            self.model = RestoreModel(
                **params,
                **special_tokens
            )
            model_path = os.path.join(BASE_DIR, "weights", "restore_model.pth")
        elif mode == "2":
            self.model = TranslateModel(
                **params,
                **special_tokens
            )            
            model_path = os.path.join(BASE_DIR, "weights", "translate_model.pth")

        print(f"📦 모델 로딩 중... {model_path}")
        checkpoint:dict = torch.load(model_path, map_location=self.device,  weights_only=False)
        self.model.load_state_dict(checkpoint.get("model_state_dict"), strict=False)
        self.model.to(self.device)
        self.model.eval()
        print("✅ 모델 로딩 완료!")
        print(f"model loss : {checkpoint.get('loss')}")
        print(f"model epoch : {checkpoint.get('epoch')}")

    def apply_repetition_penalty(self, logits, generated_ids, penalty):
        """양수/음수 logit을 구분하여 올바르게 반복 페널티 적용"""
        for token_id in set(generated_ids):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
        return logits

    def apply_no_repeat_ngram(self, logits, generated_ids, ngram_size=3):
        """최근 n-1개 토큰과 동일한 n-gram을 형성하는 토큰 차단"""
        if len(generated_ids) < ngram_size - 1:
            return logits

        # 마지막 ngram_size-1 토큰
        tail = tuple(generated_ids[-(ngram_size - 1):])

        # 과거에서 같은 tail 이후 등장했던 토큰 수집
        banned = set()
        for i in range(len(generated_ids) - ngram_size + 1):
            if tuple(generated_ids[i:i + ngram_size - 1]) == tail:
                banned.add(generated_ids[i + ngram_size - 1])

        for token_id in banned:
            logits[token_id] = float('-inf')

        return logits

    def top_p_sampling(self, logits, top_p=0.9, temperature=1.0):
        # 1. Temperature 스케일링
        logits = logits / max(temperature, 1e-8)

        # 2. 소프트맥스 → 확률
        probs = F.softmax(logits, dim=-1)

        # 3. 누적 확률 계산
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 4. Top-P 마스킹
        sorted_indices_to_remove = cumulative_probs > top_p
        # 첫 번째 토큰은 무조건 유지
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # indices_to_remove를 원래 logits 인덱스로 복구
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        logits[indices_to_remove] = float('-inf')

        # 5. 재샘플링
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        return next_token

    def run(self, input_text):
        self.model.eval()

        try:
            encoded = self.pre_token.encode(input_text)
            source_sequences = [self.cls_token_id] + encoded.ids
            source_tensor = torch.tensor([source_sequences], dtype=torch.long, device=self.device)
            tgt = [self.cls_token_id]

            # 최대 생성 길이
            max_len = 256

            for _ in range(max_len):
                tgt_tensor = torch.tensor([tgt], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    # forward 선언부
                    logits = self.model(x=source_tensor, tgt=tgt_tensor)
                    next_token_logits = logits[0, -1, :].clone()

                # 반복 페널티 (수정된 버전)
                next_token_logits = self.apply_repetition_penalty(
                    next_token_logits,
                    tgt,
                    self.config.repetition_penalty
                )

                # No-repeat n-gram 필터
                next_token_logits = self.apply_no_repeat_ngram(
                    next_token_logits,
                    tgt,
                    ngram_size=3
                )

                # Top k 확률 출력
                top_probs, top_indices = torch.topk(torch.softmax(next_token_logits, dim=-1), k=3)
                step = len(tgt)
                print(f"  step {step:2d} │ " + " │ ".join(
                    [f"{self.pre_token.id_to_token(idx.item()):>12s}: {p:.3f}"
                    for idx, p in zip(top_indices, top_probs)]
                ))

                # Top-P 샘플링
                next_token = self.top_p_sampling(
                    next_token_logits.unsqueeze(0),
                    top_p=self.config.top_p,
                    temperature=self.config.temperature
                )

                if next_token == self.eos_token_id:
                    break
                tgt.append(next_token)

            text_out = self.pre_token.decode(tgt[1:])
            print(f"\n{'='*40}\n📝 입력: {input_text}\n복원 text: {text_out}\n{'='*40}")
            
        except Exception as e:
            print(e)