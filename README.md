# Model
- 복원, 번역 모델
  - 한국어, 영어
- 데이터셋 : AI HUB 구어체
- BART 구조 기반, 인코더에 RoPE 적용 (Q, K 회전), 디코더는 표준 TransformerDecoder

* **해당 모델은 프로토 타입으로**
* **토큰 길이가 48개를 초과하면 복원 및 번역 어려움이 있습니다.**

---
# Inference 방식 보완
greedy가 아닌 top-p, k 적용

--- 
예시 자료

- 복원 모델
개요 : 불완전한 문장을 완전한 문장으로 복원.
loss : 1.84
epoch : 9
현재 학습 진행 중

```
Input : Nice meet you.
Ouput : Nice to meet you.

Input : 오늘 먹었다.
Output : 오늘 점심 먹었다.

Input : The weather nice today so decided take walk park house
Output : The weather is nice today so I decided to take a walk park in the house .
```

# 번역 모델
개요 : 한국어/영어 간 번역.
loss : 3.405
epoch : 4
현재 학습 진행 중

```
Input : Nice meet you.
Ouput : 반갑습니다.
```