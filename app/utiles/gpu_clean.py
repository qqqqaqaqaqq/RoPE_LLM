import torch
import gc

# 1. 파이썬 객체 정리 및 시스템 RAM 해제
def clear_system_ram():
    gc.collect()
    print("✅ 시스템 RAM 쓰레기 수거 완료")

# 2. GPU 메모리(VRAM) 캐시 완전히 비우기
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("✅ GPU 메모리 캐시 비우기 완료")
    else:
        print("❌ 사용 가능한 GPU가 없습니다.")

# 실행
clear_system_ram()
clear_gpu_memory()

# 현재 상태 출력
if torch.cuda.is_available():
    free_vram = torch.cuda.mem_get_info()[0] / 1024**2
    total_vram = torch.cuda.mem_get_info()[1] / 1024**2
    print(f"\n📊 현재 GPU 상태: {free_vram:.2f}MB / {total_vram:.2f}MB 사용 가능")