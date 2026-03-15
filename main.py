import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR, "app")
from app.core.config import config

def print_banner():
    print("""
╔══════════════════════════════════════╗
║         🤖 AI Language Model         ║
╚══════════════════════════════════════╝
""")

def select_mode():
    print("┌─────────────────────────────┐")
    print("│        ⚙️  MODE 선택         │")
    print("├─────────────────────────────┤")
    print("│  1️⃣   Train  (학습 모드)     │")
    print("│  2️⃣   Infer  (추론 모드)     │")
    print("└─────────────────────────────┘")
    return input("▶ 선택 : ").strip()

def select_task_mode():
    print()
    print("┌─────────────────────────────┐")
    print("│        🎯 작업 선택         │")
    print("├─────────────────────────────┤")
    print("│  1️⃣   복원 모드              │")
    print("│  2️⃣   번역 모드              │")
    print("└─────────────────────────────┘")
    return input("▶ 선택 : ").strip()

if __name__ == "__main__":
    print_banner()
    input_selection = select_mode()

    if input_selection == "1":
        mode = select_task_mode()

        if mode == "1":
            print("\n🚀 복원 Train 모드를 시작합니다...\n")
            from app.services.train import Train
            Train(config=config, mode=mode, BASE_DIR=BASE_DIR).run()

        elif mode == "2":
            print("\n🚀 번역 Train 모드를 시작합니다...\n")
            from app.services.train import Train
            Train(config=config, mode=mode, BASE_DIR=BASE_DIR).run()

        else:
            print("❌ 올바른 모드를 선택해주세요.")

    elif input_selection == "2":
        mode = select_task_mode()

        if mode == "1":
            print("\n🔧 복원 추론 모드를 시작합니다...\n")
            from app.services.inference import Inference
            infer = Inference(config=config, mode=mode, BASE_DIR=BASE_DIR)
            while True:
                input_text = input("📝 복원할 문장 입력 (종료: Ctrl+C) : ").strip()
                infer.run(input_text=input_text)

        elif mode == "2":
            print("\n🌐 번역 추론 모드를 시작합니다...\n")
            from app.services.inference import Inference
            infer = Inference(config=config, mode=mode, BASE_DIR=BASE_DIR)
            while True:
                input_text = input("📝 번역할 문장 입력 (종료: Ctrl+C) : ").strip()
                infer.run(input_text=input_text)

        else:
            print("❌ 올바른 모드를 선택해주세요.")

    else:
        print("❌ 올바른 모드를 선택해주세요.")