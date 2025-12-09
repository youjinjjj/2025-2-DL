# Transfusion Reproduction (ICLR 2025)

## 1. 프로젝트 개요

- **논문**: *TRANSFUSION: Predict the Next Token and Diffuse Images with One Multi-Modal Model* (ICLR 2025)
- **목표**
  - 하나의 Transformer가 **텍스트는 다음 토큰 예측**, **이미지는 diffusion 노이즈 예측**으로 동시에 학습되는 Transfusion 구조를,
  - MS-COCO 텍스트–이미지 실험 수준에서 **소규모로 재현**하고 동작을 확인한다.
- **원본 코드**
  - 공식 구현이 아닌, 공개 PyTorch 구현인 [`lucidrains/transfusion-pytorch`](https://github.com/lucidrains/transfusion-pytorch)를 기반으로 재현을 수행하였다.

---

## 2. 재현 스코프 및 차이점

### 2.1 이번 재현에서 한 것

- **Task**: COCO 기반 **텍스트 → 이미지 생성** 부분만 다룸
- **모델**
  - Transfusion PyTorch 구현의 single-scale 모델 사용
  - 텍스트와 이미지 패치를 하나의 시퀀스로 넣고,
    - 텍스트: causal attention (next-token prediction)
    - 이미지 패치: intra-image bidirectional attention + diffusion loss
- **데이터**
  - MS-COCO 2017 이미지–캡션 쌍 사용
  - 전체 데이터가 아닌, GPU/시간 제약에 맞춘 **subset + 제한된 스텝(예: 5,000 step)** 학습
- **검증**
  - 학습 로그(loss)를 저장하고,
  - 학습된 모델로 텍스트 프롬프트에 대한 이미지를 생성하여 **질적으로** 확인

### 2.2 논문과의 주요 차이

- 7B 모델 + 2T 토큰 large-scale 학습은 수행하지 않음
- Chameleon-style discrete 토큰 baseline은 구현하지 않음
- FID, CLIP, CIDEr 등 **정량 지표는 계산하지 못함**
- 이미지→텍스트, multi-modal text generation 등 **다른 task**는 범위 밖

이번 레포는 “논문 전체 결과” 재현이 아니라,  
**Transfusion 구조가 소규모 COCO 설정에서 실제로 동작하는지 확인하는 부분 재현**임을 전제로 한다.

---

## 3. 환경 및 디렉터리 구조

### 3.1 Environment

- Python 3.x
- PyTorch 2.x + CUDA 11.x
- 주요 라이브러리
  - `torch`, `torchvision`, `accelerate`, `diffusers`, `transformers`, `tqdm` 등

(실제 사용한 GPU/OS는 여기서 간단히 적어주면 좋음)

### 3.2 Directory

```text
2025-2-DL/
 ├─ transfusion-repro/
 │   ├─ train_coco_subset.py         # COCO subset 학습
 │   ├─ eval_coco_val.py            # train/val loss 평가 및 로그 저장
 │   ├─ inference.py                # 학습된 모델로 이미지 생성
 │   ├─ transfusion_pytorch/        # Transfusion PyTorch 구현
 │   └─ results_coco_single/
 │       ├─ train_loss_steps_5000.txt  # step별 train loss
 │       └─ val_loss_train.txt         # 평균 loss 요약
 └─ ...
