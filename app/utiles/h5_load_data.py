import os
import h5py
import torch
import os

def load_pre_dataset(number, base_dir):
    pre_dataset_path = os.path.join(base_dir, "data", f"pre_dataset_{number}.h5")
    pre_dataset = H5TranslationDataset(pre_dataset_path)
    return pre_dataset

# HDF5 기반 Dataset  (램 메모리 대체용)
class H5TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.length = f['source'].shape[0]
        self._file = None

    def _get_file(self):
        pid = os.getpid()
        if not hasattr(self, '_pid') or self._pid != pid:
            self._file = h5py.File(self.h5_path, 'r')
            self._pid = pid
        return self._file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self._get_file()
        src = torch.tensor(f['source'][idx, :], dtype=torch.long)
        tgt = torch.tensor(f['target'][idx, :], dtype=torch.long)
        return src, tgt

    def __del__(self):
        if self._file is not None:
            self._file.close()

def check_h5(BASE_DIR):
    pre_dataset_path = os.path.join(BASE_DIR,  "dataset", f"pre_dataset_0.h5")

    with h5py.File(pre_dataset_path, 'r') as f:
        print("keys:", list(f.keys()))
        print("source dtype:", f['source'].dtype)
        print("source shape:", f['source'].shape)
        print("첫번째 행:", f['source'][0])