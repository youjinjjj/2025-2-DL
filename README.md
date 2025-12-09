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

### 2.2 Experimental Setup

- **Dataset**: MS-COCO 2017 train subset (image–caption pairs)  
- **Hardware**: single NVIDIA GPU (RTX 3090, 24GB VRAM)

**Training configuration**

- Optimizer: AdamW (LR = 1e-4)  
- Batch size: 4  
- Total steps: 5,000  
  → roughly 20,000 samples seen during training

---

## 3. Implementation Details

This implementation is based on the open-source repository **`lucidrains/transfusion-pytorch`**,  
but the training script is modified for **MS-COCO data loading** and **stable training**.

- Inside `transfusion_pytorch/`, the **`Transfusion` model implementation and loss definition** are used **without changes**.  
- On top of that, we newly implement scripts that build a full pipeline combining **COCO + TinyLlama + SD-VAE**.

### File Structure

```text
├── train_coco_subset.py          # main training script
├── eval_coco_val.py              # evaluation script (train/val loss)
├── inference.py                  # text-to-image generation & debugging
├── transfusion_pytorch/          # Transfusion core implementation
└── results_coco_single/          # training logs & checkpoints
```

## 4. How to Run 

1) Environment Setup
```
pip install torch torchvision transformers diffusers accelerate vector-quantize-pytorch
```
2) Training
```
python train_coco_subset.py \
  --coco-root ./data/coco \
  --coco-ann annotations/captions_train2017.json \
  --image-size 256 \
  --batch-size 4 \
  --max-steps 5000 \
  --results-dir ./results_coco_single
```
3) Inference / Testing
```
python inference.py
```
## 5. Experimental Results
###5.1 Quantitative Analysis

During 5,000 training steps, the total loss decreased from 1.01 to 0.37 in a stable manner.
This suggests that two very different modalities (text and images) can be jointly optimized in a single Transformer
without the objectives collapsing into each other, supporting the core idea of Transfusion.

###5.2 Qualitative Analysis

For qualitative evaluation, we attempted text-to-image generation.
However, after receiving a text prompt, the model:
-correctly produced the special <BOI> token, but
-failed to transition into the image generation phase (diffusion latent sampling), and
-kept generating only text tokens.

This behavior corresponds to a mode collapse in modality space:
despite the numerically converged joint loss, the model behaves like a pure language model
and does not utilize the image branch at inference time.


## 6. References
>[1] Zhou et al., "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model", ICLR 2025.

>[2] Lucidrains, Transfusion-pytorch Repository. https://github.com/lucidrains/transfusion-pytorch?tab=readme-ov-file 
