# Transfusion Reproduction (ICLR 2025)

본 저장소는 ICLR 2025에 게재된 논문  
**“Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model”**의 핵심 아키텍처를 재현하고,  
제한된 컴퓨팅 자원(Single GPU) 및 소규모 데이터 환경에서의 **학습 역학(Training Dynamics)**을 분석한 프로젝트입니다.

---

## 1. 연구 개요 (Abstract)

Transfusion은 텍스트(이산형)와 이미지(연속형)를 **단일 Transformer 모델** 내에서 동시에 처리하는 새로운 방법론을 제안합니다.  
텍스트는 *Next Token Prediction*, 이미지는 *Diffusion* 손실 함수로 학습하는 것이 특징입니다.

본 연구의 목표는 원 논문의 대규모 학습(7B 파라미터, 2T 토큰)을 그대로 복제하는 것이 아니라,

- **경량화된 환경**(1.1B 모델, MS-COCO Subset)에서도  
- 해당 아키텍처의 손실 함수가 **수학적으로 유효하게 수렴하는지 검증**하고,
- 데이터 규모에 따른 멀티모달 생성 모델의 **한계점(Scaling Limits)**을 실험적으로 규명하는 데 있습니다.

---

## 2. 방법론 (Methodology)

### 2.1 모델 아키텍처 (Model Architecture)

논문의 *Unified Multi-Modal Model* 개념을 유지하되, 리소스 제약을 고려하여 다음과 같이 구성하였습니다.

- **Backbone**: TinyLlama-1.1B (Language Model)  
- **Image Encoder / Decoder**: Stable Diffusion VAE (Pre-trained)  
- **Attention Mechanism**
  - Text Tokens: Causal Attention (Autoregressive)
  - Image Patches: Intra-image Bidirectional Attention  
- **Loss Function**

  \[
  \mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{LM}} + \lambda \cdot \mathcal{L}_{\text{Diffusion}}
  \]

### 2.2 실험 환경 (Experimental Setup)

- **Dataset**: MS-COCO 2017 Train Subset (Image–Caption Pairs)  
- **Hardware**: Single NVIDIA GPU (RTX 3090 급, 24GB VRAM)

**Training Configuration**

- Optimizer: AdamW (LR = 1e-4)  
- Batch Size: 4  
- Total Steps: 5,000  
  → 약 20,000 샘플 학습

---

## 3. 구현 내용 (Implementation Details)

본 구현은 오픈소스 [lucidrains/transfusion-pytorch](https://github.com/lucidrains/transfusion-pytorch)를 베이스로 하되,  
**MS-COCO 데이터셋 파이프라인 구축** 및 **학습 안정화**를 위해 코드를 수정 및 보완하였습니다.

### 파일 구조 (File Structure)

```text
├── train_coco_subset.py          # 메인 학습 스크립트 (Single GPU 최적화)
├── eval_coco_val.py              # Loss 검증 및 평가 스크립트
├── inference.py                  # 텍스트-투-이미지 생성 및 디버깅 스크립트
├── transfusion_pytorch/          # 모델 아키텍처 정의 (Transfusion Core)
└── results_coco_single/          # 학습 로그, 체크포인트, 결과 그래프
```
---

## 4. 실행 방법 (Usage)

1) 환경 설정
```
pip install torch torchvision transformers diffusers accelerate vector-quantize-pytorch
```
2) 모델 학습 (Training)
```
python train_coco_subset.py \
  --coco-root ./data/coco \
  --coco-ann annotations/captions_train2017.json \
  --image-size 256 \
  --batch-size 4 \
  --max-steps 5000 \
  --results-dir ./results_coco_single
```
3) 추롬 및 테스트 (Inference)
```
python inference.py
```
## 5. 실험 결과 (Experimental Results)
4.1 정량적 분석: 학습 안정성 (Quantitative Analysis)

5,000 Step의 학습 과정에서 Total Loss는 1.01에서 0.37로 안정적으로 감소하였습니다.
이는 서로 다른 성격의 두 모달리티(텍스트, 이미지)가 단일 Transformer 내에서 충돌 없이 동시에 최적화될 수 있다는
Transfusion 방법론의 수학적 타당성을 입증합니다.

Figure 1: 5,000 Step 동안의 학습 손실 곡선.
지속적인 우하향 추세를 보이며 모델이 데이터 패턴을 학습하고 있음을 나타냄.

4.2 정성적 분석: 모드 붕괴 현상 (Qualitative Analysis)

학습된 모델로 텍스트 기반 이미지 생성을 시도하였으나,
모델은 텍스트 프롬프트 입력 후 이미지 생성 모드(Diffusion Latent Generation)로 진입하지 못하고
**텍스트 토큰만을 지속적으로 생성하는 현상(Mode Collapse)**이 관찰되었습니다.

Figure 2: 추론 로그.
모델이 이미지 텐서 대신 텍스트 토큰만 생성하고 있음을 보여줌.

## 6. References
>>[1] Zhou et al., "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model", ICLR 2025.
>>[2] Lucidrains, Transfusion-pytorch Repository. https://github.com/lucidrains/transfusion-pytorch?tab=readme-ov-file 
