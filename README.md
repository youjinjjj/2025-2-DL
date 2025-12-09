# Transfusion Reproduction (ICLR 2025)

## 1. Project Overview

- **Paper**: *TRANSFUSION: Predict the Next Token and Diffuse Images with One Multi-Modal Model* (ICLR 2025)
- **Goal**:  
  - 하나의 Transformer가 텍스트는 **다음 토큰 예측**, 이미지는 **diffusion 노이즈 예측** 두 개의 목적함수로 동시에 학습될 수 있다는 Transfusion의 핵심 아이디어를,  
    **MS-COCO 텍스트-이미지 실험** 수준에서 재현한다.
  - 논문에서 주장하는 “하나의 모델로 텍스트와 이미지를 함께 생성할 수 있다”는 부분을,  
    **소규모 설정 + 공개 PyTorch 구현**으로 검증한다.

- **This repo**
  - `/transfusion-repro/` 폴더에 재현용 코드와 로그를 모아두었다.
  - 원본 구현이 아닌, 공개 PyTorch 구현(`lucidrains/transfusion-pytorch`)을 기반으로 재현을 진행하였다.

---

## 2. Original Paper Summary

### 2.1 Core Idea

- 텍스트는 **언어모델(Next-Token Prediction)**, 이미지는 **Diffusion (노이즈 예측 MSE)** 로 학습하면서도,
- 두 modality를 **하나의 Transformer**에 올려서 함께 학습하는 방법을 제안.
- 이미지 패치는 bidirectional attention, 텍스트 토큰은 causal attention 을 쓰고,  
  하나의 시퀀스 안에서 BOI/EOI 토큰으로 이미지 구간을 감싼 뒤 함께 처리한다.
- 텍스트-텍스트, 텍스트-이미지, 이미지-텍스트를 모두 다루는 **진짜 multi-modal generation**을 목표로 한다.

### 2.2 Main Contributions (논문 기준)

1. **통합 아키텍처**  
   - 텍스트와 이미지를 모두 처리하는 단일 Transformer + modality-specific encoder/decoder.
2. **Scaling Law 분석**  
   - 같은 데이터/연산량 조건에서, 이미지 토큰 양자화(Chameleon-style)보다 Transfusion이  
     텍스트/이미지/캡셔닝 모두에서 더 좋은 scaling 을 보인다고 보고.
3. **Ablation**  
   - intra-image bidirectional attention, patch size, U-Net encoder/decoder의 역할 분석.
4. **7B 모델 결과**  
   - 7B Transfusion + 2T multi-modal tokens 학습 시,  
     기존 diffusion/언어모델들과 비슷한 수준의 이미지·텍스트 성능을 달성.

---

## 3. Reproduction Scope & Differences

### 3.1 What We Reproduced

- **Task 포커스**:  
  - 전체 multi-modal 세팅이 아닌, **MS-COCO 텍스트 → 이미지 생성** 실험을 소규모로 재현.
- **모델**
  - 공개 PyTorch 구현에서 제공하는 **단일-스케일 Transfusion**(소형 모델)을 사용.
  - 텍스트 + 이미지 패치가 하나의 시퀀스로 들어가고,  
    텍스트에는 causal attention, 이미지 패치에는 intra-image bidirectional attention 을 사용.
- **데이터**
  - MS-COCO 이미지-캡션 쌍 사용.
  - 과제/리소스 제약 상, **논문 전체 데이터(0.5T/2T tokens)** 가 아닌  
    PyTorch 구현에서 제공하는 **COCO subset 설정**을 사용.
- **목표**
  - 학습 스크립트가 정상적으로 동작하는지 확인.
  - 학습/검증 loss를 기록하고, 샘플 이미지를 생성해 논문에서 주장하는 특성이  
    **질적으로는** 재현되는지 확인.

### 3.2 What We Did *Not* Reproduce (중요한 차이)

- 7B 모델 + 2T 토큰 전체 학습은 **리소스 부족으로 수행하지 않음**.
- Chameleon-style discrete baseline 을 직접 구현하거나 비교하지 않음.
- VAE / U-Net encoder–decoder 를 논문과 동일한 설정으로 다시 학습하지 않고,  
  **공개 구현의 사전학습 VAE/모델**을 그대로 사용.
- 논문에서 보고하는 FID, CLIP, CIDEr, GenEval 등 **정량 지표는 측정하지 못함**  
  (시간·GPU 제약 / COCO val 전체 사용 X).
- multi-modal text generation (텍스트-only, 이미지→텍스트 캡셔닝, 이미지 편집) 실험은 범위 밖.

이 README와 보고서는, 위와 같은 **스코프 제한**을 전제로 읽어야 한다.

---

## 4. Environment & Code Structure

### 4.1 Environment

- OS / Docker: (예: Ubuntu 20.04 + NVIDIA Docker)
- Python: 3.x
- PyTorch: 2.x (CUDA 11.x)
- GPU: (예: A100 1장 / 2080Ti 1장 – 실제 사용한 GPU 명시)

필요 패키지는 `requirements.txt` 또는 아래 명령으로 설치:

```bash
pip install -r requirements.txt
# or
pip install torch torchvision tqdm einops accelerate pillow
