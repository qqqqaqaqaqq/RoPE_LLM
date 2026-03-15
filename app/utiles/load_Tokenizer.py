# Tokenizer Load
import os
import glob
from datetime import datetime
from tokenizers import Tokenizer

def load_Tokenizer(BASE_DIR:str) -> Tokenizer:
    dataset_root = os.path.join(BASE_DIR, "dataset", "tokenized")

    print(f"경로 : {dataset_root}")
    # 1. 폴더 내 모든 .json 파일 탐색
    token_files = glob.glob(os.path.join(dataset_root, "*.json"))

    if not token_files:
        # 새로 학습을 유도하거나 기본 경로를 반환
        return None

    # 수정 시간 순으로 정렬 (가장 최신 파일이 마지막에 오도록)
    token_files.sort(key=os.path.getmtime)

    for idx, path in enumerate(token_files):
        fname = os.path.basename(path)
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path) / 1024
        dt = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    # 3. 가장 최신 파일 선택 (또는 특정 파일 지정 가능)
    selected_token_path = token_files[-1]

    # 4. 실제 로드
    try:
        pre_token = Tokenizer.from_file(selected_token_path)
        return pre_token
    except Exception as e:
        print(f"❌ 로드 중 오류 발생: {e}")
        return None