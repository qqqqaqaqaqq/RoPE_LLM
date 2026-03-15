import os
import torch
import math
import traceback
import time
import gc
import bitsandbytes as bnb

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataclasses import asdict
from tokenizers import Tokenizer

from app.utiles.timer import timer
from app.utiles.load_Tokenizer import load_Tokenizer
from app.utiles.data_loader import DataLoader_Setting
from app.models.TransformerModel import RestoreModel, TranslateModel
from app.core.config import GlobalConfig
from app.utiles.h5_load_data import load_pre_dataset

class Train():
    def __init__(self, config:GlobalConfig, mode, BASE_DIR):
        self.config = config
        self.device = config.device
        self.BASE_DIR = BASE_DIR

        pre_token:Tokenizer = load_Tokenizer(BASE_DIR=BASE_DIR)
        self.pad_token_id = pre_token.token_to_id("[PAD]")
        self.cls_token_id = pre_token.token_to_id("[CLS]")
        self.mask_token_id = pre_token.token_to_id("[MASK]")
        self.eos_token_id = pre_token.token_to_id("[EOS]")

        params = asdict(self.config)
        special_tokens = {
            'pad_token_id': self.pad_token_id,
            'cls_token_id': self.cls_token_id,
            'eos_token_id': self.eos_token_id,
            'mask_token_id': self.mask_token_id
        }
        self.mode = mode
        save_dir = BASE_DIR

        if mode == "1":
            # 모델 생성
            self.model = RestoreModel(
                **params,
                **special_tokens
            ).to(self.device)
            self.model_path = os.path.join(save_dir, "weights", "restore_model.pth")
            self.epoch_save_dir = os.path.join(save_dir, "checkpoints", "restore")
            self.logs_save_dir = os.path.join(save_dir, "logs", "restore")
            
        elif mode == "2":
            # 모델 생성
            self.model = TranslateModel(
                **params,
                **special_tokens
            ).to(self.device)    
            self.model_path = os.path.join(save_dir, "weights", "translate_model.pth")
            self.epoch_save_dir = os.path.join(save_dir, "checkpoints", "translate")
            self.logs_save_dir = os.path.join(save_dir, "logs", "translate")

        self.start_epoch = 0
        self.best_loss = float('inf')

        if os.path.exists(self.model_path):
            checkpoint:dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
            model_to_load = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
            model_to_load.load_state_dict(checkpoint.get('model_state_dict'), strict=False)
            self.start_epoch = checkpoint.get('epoch', 0)
            self.best_loss = checkpoint.get('loss', float('inf'))

        # 폴더 없으면 생성
        os.makedirs(self.epoch_save_dir, exist_ok=True)
        os.makedirs(self.logs_save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.BASE_DIR, "weights"), exist_ok=True)

        if hasattr(self.config, 'use_checkpointing') and self.config.use_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            print("🛡️ Gradient Checkpointing 준비 완료")

        if self.config.use_compile and hasattr(torch, 'compile'):
            print("⚡ A100 고속 컴파일 완료 (Mode: default)")
            torch._dynamo.config.capture_scalar_outputs = True
            self.model = torch.compile(self.model, mode="default")

    def _save_checkpoint(self, path, epoch, loss, optimizer, scheduler):
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'config': asdict(self.config),
        }
        torch.save(checkpoint, path)

    def validate(self, model, criterion, val_loader):
        model.eval()
        total_loss = 0
        total_tokens = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                src = batch_x.to(self.device)
                tgt_in  = batch_y[:, :-1].to(self.device)
                tgt_out = batch_y[:, 1:].to(self.device)


                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.config.use_amp):
                    # forward 선언 부
                    logits = model(x=src, tgt=tgt_in)

                    l_ce = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

                    total_loss += l_ce.item()
                    total_tokens += tgt_out.numel()

        return total_loss / len(val_loader)

    def get_warmup_cosine_scheduler(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def generate_batch(self, batch_x, batch_y):
        src = batch_x.to(self.device, non_blocking=True)
        tgt_in = batch_y[:, :-1].to(self.device, non_blocking=True)
        tgt_out = batch_y[:, 1:].to(self.device, non_blocking=True)

        return src, tgt_in, tgt_out

    def run(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        data_setting = DataLoader_Setting(self.config)

        # 데이터셋 넘버
        number = 0

        # 데이터셋 불러오기
        with timer("데이터셋 로드"):
          pre_dataset = load_pre_dataset(number=number, base_dir=self.BASE_DIR)

        with timer("DataLoader 로드"):
          train_loader, val_loader = data_setting.preprocessing(pre_dataset=pre_dataset)

        # 모델 선언
        self.model = self.model.to(torch.bfloat16)

        total = sum(p.numel() for p in self.model.parameters())
        print(f"파라미터 수: {total/1e6:.1f}M")
        print(f"모델 메모리: {total * 2 / 1024**3:.2f} GB")

        print("🔥 CUDA 워밍업 중...")
        dummy_x = torch.zeros(1, 48, dtype=torch.long).to(self.device)
        dummy_tgt = torch.zeros(1, 47, dtype=torch.long).to(self.device)
        with torch.no_grad():
            _ = self.model(x=dummy_x, tgt=dummy_tgt)
        torch.cuda.synchronize()
        print("✅ 워밍업 완료")

        # 옵티마이저
        optimizer = bnb.optim.AdamW8bit(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, label_smoothing=0.1)

        total_steps = len(train_loader) * self.config.epochs

        # 스케쥴러 셋팅
        warmup_steps = total_steps // 10
        scheduler = self.get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

        writer = SummaryWriter(log_dir=self.logs_save_dir)

        print(f"Model precision: {next(self.model.parameters()).dtype}")

        global_step = 0

        try:
            print(f"🚀 Train Start")
            for epoch_idx in range(self.start_epoch, self.config.epochs):
                start_time = time.time()
                epoch_start = time.time()
                step_start_time = time.time()

                total_tokens = 0
                
                self.model.train()
                print(f"DataLoader Length : {len(train_loader)}")

                train_iter = iter(train_loader)
                losses = []

                for step in range(len(train_loader)):
                    batch_x, batch_y = next(train_iter)

                    src, tgt_in, tgt_out = self.generate_batch(batch_x, batch_y)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.config.use_amp):
                        logits = self.model(x=src, tgt=tgt_in)
                        raw_loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
                        loss = raw_loss / self.config.accum_steps

                    if not torch.isfinite(loss):
                        print("🚨 NaN 발견! 학습을 중단하거나 스킵합니다.")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    loss.backward()
                    torch.cuda.synchronize()

                    if (step + 1) % self.config.accum_steps == 0 or (step + 1) == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    losses.append(raw_loss.detach())
                    total_tokens += tgt_out.numel()

                    global_step += 1

                    if (step + 1) % 5000 == 0:
                        step_end_time = time.time()
                        print(f"{step + 1} / {len(train_loader)} step | time {step_end_time - step_start_time} sec")
                        step_start_time = time.time()

                print(f"✅ EP : {epoch_idx + 1} 종료")

                avg_train_loss = torch.stack(losses).mean().item() if losses else float('inf')

                with timer("Validate 걸린 시간"):
                    avg_val_loss = self.validate(model=self.model, criterion=criterion, val_loader=val_loader)

                writer.add_scalar('Loss/val', avg_val_loss, epoch_idx)

                elapsed_time = time.time() - epoch_start
                tokens_per_sec = total_tokens / elapsed_time

                print(f"📊 Tokens: {total_tokens:,} | Speed: {tokens_per_sec:.2f} tps")
                print(f"🚀 Ep {epoch_idx+1} | Loss T: {avg_train_loss:.4f} V: {avg_val_loss:.4f}")

                if avg_val_loss < self.best_loss:
                    self.best_loss = avg_val_loss
                    self._save_checkpoint(self.model_path, epoch_idx+1, self.best_loss, optimizer, scheduler)
                    print(f"⭐ Best Model Updated!")

                epoch_save_path = os.path.join(self.epoch_save_dir, f"ep{epoch_idx+1}_loss{int(avg_val_loss*1000)}.pth")
                self._save_checkpoint(epoch_save_path, epoch_idx+1, self.best_loss, optimizer, scheduler)

                end_time = time.time()
                elips = end_time - start_time
                print(f"✨ 1 EPOCH에 총 걸린 시간 {elips:.2f} sec")

                print(f"🚀 새로운 데이터 셋 생성 중")
                del train_loader
                del val_loader
                del pre_dataset
                gc.collect()

                if epoch_idx + 1 < self.config.epochs:
                    number += 1
                    pre_daatset = load_pre_dataset(number=number, base_dir=self.BASE_DIR)
                    train_loader, val_loader = data_setting.preprocessing(pre_daatset)

        except Exception:
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

        return