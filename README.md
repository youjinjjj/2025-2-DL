# Transfusion Reproduction
Reproduction of the core architecture of  
**“Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model.”**

---

## 1. Overview

Transfusion proposes a new method for handling **text (discrete)** and **images (continuous)** **within a single Transformer model**.  
Text is trained with *next-token prediction*, while images are trained with a *diffusion* loss; this joint objective is the key idea of the paper.

Since faithfully reproducing the original large-scale setting (7B parameters, 2T tokens) is infeasible in this project, the goal here is to:

- verify, in a **lightweight setting** (1.1B model, MS-COCO subset),  
- whether the joint loss formulation **converges in a numerically stable way**, and  
- empirically explore the **scaling limits** of the multi-modal generative model with respect to data size.

---

## 2. Methodology

### 2.1 Model Architecture

We keep the paper’s *Unified Multi-Modal Model* idea, while adapting it to our resource constraints:

- **Backbone**: TinyLlama-1.1B (language model)  
- **Image encoder / decoder**: Stable Diffusion VAE (pre-trained)  
- **Attention mechanism**
  - Text tokens: causal (autoregressive) attention  
  - Image patches: intra-image bidirectional attention  
- **Loss function**

  \[
  \mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{LM}} + \lambda \cdot \mathcal{L}_{\text{Diffusion}}
  \]

> The core multi-modal logic—treating text and image tokens as a single sequence and jointly optimizing  
> LM loss + diffusion-style loss—is used **as-is** from the `Transfusion` class in `transfusion_pytorch`.

We experiment with **two configurations of the loss weight λ**:

- `sub_1.py`: baseline setting with a smaller image loss weight (λ ≈ 1.0)  
- `main.py`: image-focused setting with a larger image loss weight (λ ≈ 5.0, closer to the paper)

### 2.2 Experimental Setup

- **Dataset**: MS-COCO 2017 train subset (image–caption pairs)  
- **Hardware**: single NVIDIA GPU (RTX 3090, 24GB VRAM)

**Training configuration**

- Optimizer: AdamW (LR = 1e-4)  
- Batch size: 4  
- Total steps: 5,000  
  - ≈ 20,000 image–caption pairs seen during training

---

## 3. Implementation Details

This implementation is based on the open-source repository **`lucidrains/transfusion-pytorch`**,  
but the training script is modified for **MS-COCO data loading** and **stable training**.

- Inside `transfusion_pytorch/`, the **`Transfusion` model implementation and loss definition** are used **without changes**.  
- On top of that, we newly implement scripts that build a full pipeline combining **COCO + TinyLlama + SD-VAE**.

### File Structure

```text
transfusion-repro/
├── main.py              # Training script (higher image loss weight, λ ≈ 5)
├── sub_1.py             # Training script (baseline loss weight, λ ≈ 1)
├── inference.py         # Text-to-image generation & debugging
├── eval_coco_val.py     # Evaluation script (COCO train/val loss)
├── transfusion_pytorch/ # Core Transfusion implementation (from lucidrains)
└── results/             # Training logs, checkpoints, sample images
```

## 4. How to Run 

### 4.1 Environment Setup
```
pip install torch torchvision transformers diffusers accelerate vector-quantize-pytorch
```
### 4.2 Training – Baseline (λ ≈ 1, sub_1.py)
```
python sub_1.py \
  --coco-root ./data/coco \
  --coco-ann annotations/captions_train2017.json \
  --image-size 256 \
  --batch-size 4 \
  --max-steps 5000 \
  --results-dir ./results/lambda1
```
This run corresponds to a lower image-loss weight (λ around 1.0), which is the configuration used for the main quantitative results reported below.

### 4.3 Training – Image-focused (λ ≈ 5, main.py)
```
python main.py \
  --coco-root ./data/coco \
  --coco-ann annotations/captions_train2017.json \
  --image-size 256 \
  --batch-size 4 \
  --max-steps 5000 \
  --results-dir ./results/lambda5
```
This run uses a larger image loss weight (λ ≈ 5.0), closer to the setting described in the original paper.
It is used to qualitatively compare how strongly emphasizing the diffusion loss affects convergence and generation behaviour.

### 4.4 Inference / Testing
inference.py loads a trained checkpoint, runs text-to-image generation for a small list of prompts, and saves any image tensors it finds.
Inside inference.py, set:
- MODEL_PATH to the checkpoint you want to use (e.g. ./results/lambda1/model_step_20000.pt)
- SAVE_DIR to an output folder (e.g. ./results/lambda1_samples)
Then run:
```
python inference.py
```
The script prints detailed information about the returned objects from ```model.sample(...)``` and saves any image-shaped tensors as PNG files

## 5. Experimental Results

We trained two COCO setups (λ = 1 and λ = 5) for about one epoch each, and in both cases the total loss decreased stably from roughly 1.0 to 0.3.
However, at inference time both models only generated text: they produced the <BOI> token correctly but never entered the image latent sampling phase, even when the λ = 1 model was extended to 30,000 steps.
Overall, the joint loss converged numerically, but in this low-resource regime the model failed to learn reliable text → image modality switching and effectively behaved like a pure language model.

## 6. References
>[1] Zhou et al., "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model", ICLR 2025.

>[2] Lucidrains, Transfusion-pytorch Repository. https://github.com/lucidrains/transfusion-pytorch?tab=readme-ov-file 
