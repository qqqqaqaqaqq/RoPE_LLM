from torch.utils.data import random_split, DataLoader
from tokenizers import Tokenizer

from app.core.config import GlobalConfig
from app.utiles.load_Tokenizer import load_Tokenizer
from app.utiles.h5_load_data import H5TranslationDataset

class DataLoader_Setting():
    def __init__(self, config):
        self.config: GlobalConfig = config
        self.pre_token:Tokenizer = load_Tokenizer()
        self.pad_id = self.pre_token.token_to_id("[PAD]")

    def preprocessing(self, pre_dataset: H5TranslationDataset):
        vocab_size = self.pre_token.get_vocab_size()
        self.config.vocab_size = vocab_size
        print(f"✅ 크기 확인 : {vocab_size} | {len(pre_dataset)}")

        train_size = int(self.config.train_val_ratio * len(pre_dataset))
        val_size = len(pre_dataset) - train_size
        train_dataset, val_dataset = random_split(pre_dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )

        return train_loader, val_loader