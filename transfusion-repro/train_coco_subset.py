import argparse
from pathlib import Path
from shutil import rmtree

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.datasets import CocoCaptions

from diffusers.models import AutoencoderKL
from transformers import AutoTokenizer
from accelerate import Accelerator  # ★ DDP 핵심

from transfusion_pytorch import Transfusion, print_modality_sample

# ---------------------------
# 옵션 파서
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", type=str, required=True, help="COCO root path")
    p.add_argument("--coco-ann", type=str, default="annotations/captions_train2017.json")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4) # GPU당 배치사이즈 (3장이면 총 12)
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--results-dir", type=str, default="./results_coco_ddp")
    return p.parse_args()

# ---------------------------
# 토크나이저 설정
# ---------------------------
# Llama 구조 사용 (TinyLlama)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# 패딩 토큰이 없으면 EOS 토큰을 패딩으로 지정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def decode_tokens(tokens):
    # ★ 수정됨: chr() 대신 tokenizer.decode 사용
    # tokens 범위가 0~32000이므로 chr() 쓰면 에러남
    return tokenizer.decode(tokens, skip_special_tokens=True)

# ---------------------------
# VAE Wrapper (Encoder/Decoder)
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae
    
    def forward(self, image):
        # image: [0,1] -> [-1, 1] for SD VAE
        with torch.no_grad():
            latent = self.vae.encode(image * 2 - 1).latent_dist.sample()
        return 0.18215 * latent

class Decoder(nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        return (image / 2 + 0.5).clamp(0, 1)

# ---------------------------
# 데이터셋
# ---------------------------
class CocoTextImageDataset(Dataset):
    def __init__(self, coco_root, ann_rel_path, image_size=256):
        import os
        image_root = os.path.join(coco_root, "train2017")
        ann_path = os.path.join(coco_root, ann_rel_path)
        
        self.ds = CocoCaptions(root=image_root, annFile=ann_path)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, captions = self.ds[idx]
        
        # 캡션 중 랜덤 하나 선택 or 첫번째
        caption = captions[0]
        
        # 이미지 전처리
        image = image.convert("RGB")
        image = self.transform(image)
        
        # ★ 수정됨: max_length 패딩 필수 (배치 묶기 위해)
        text_enc = tokenizer(
            caption, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=128 # 텍스트 길이 제한
        )
        text_tokens = text_enc['input_ids'].squeeze(0)
        
        return text_tokens, image

# 무한 루프 헬퍼
def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# ---------------------------
# 메인 학습 루프
# ---------------------------
def main():
    args = parse_args()
    
    # 1. Accelerator 초기화
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        rmtree(args.results_dir, ignore_errors=True)
        results_folder = Path(args.results_dir)
        results_folder.mkdir(exist_ok=True, parents=True)
        print(f"Training on GPUs: {args.results_dir} with {accelerator.num_processes} processes")

    # VAE 로드 (평가 모드 고정)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.requires_grad_(False)
    vae.eval() 
    vae.to(accelerator.device)

    encoder = Encoder(vae)
    decoder = Decoder(vae)

    # 모델 설정
    model = Transfusion(
        num_text_tokens = tokenizer.vocab_size,
        dim_latent = 4,
        channel_first_latent = True,
        modality_default_shape = (32, 32),
        modality_encoder = encoder,
        modality_decoder = decoder,
        pre_post_transformer_enc_dec = (
            nn.Conv2d(4, 768, 3, 2, 1),
            nn.ConvTranspose2d(768, 4, 3, 2, 1, output_padding=1),
        ),
        add_pos_emb = False,
        modality_num_dim = 2,
        fallback_to_default_shape_if_invalid = True,
        transformer = dict(
            dim = 768,
            depth = 12,
            dim_head = 64,
            heads = 12,
        )
    )

    dataset = CocoTextImageDataset(
        coco_root=args.coco_root, 
        ann_rel_path=args.coco_ann,
        image_size=args.image_size
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    iter_dl = cycle(dataloader)

    # EMA 설정
    ema_model = None
    if accelerator.is_main_process:
        try:
            ema_model = accelerator.unwrap_model(model).create_ema(0.9)
        except AttributeError:
            print("Warning: create_ema method not found. Skipping EMA.")
            ema_model = None

    model.train()
    
    # ★ 총 Step 50,000회
    for step in range(1, args.max_steps + 1):
        batch = next(iter_dl)
        
        # 배치 언패킹
        text_tokens, images = batch
        
        loss = model(text_tokens, images)
        
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if accelerator.is_main_process:
            if ema_model:
                ema_model.update()
            
            # 1. 로그 출력 (자주)
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")

            # 2. 이미지 생성 (500 Step마다 - 보고서용 시각화 자료 확보 중요!)
            if step % 500 == 0:
                print(f"Generating Sample at step {step}...")
                
                # ★ [수정] eval_model 정의가 빠져있던 오류 수정
                eval_model = ema_model if ema_model else accelerator.unwrap_model(model)
                eval_model.eval()
                
                with torch.no_grad():
                    try:
                        sample = eval_model.sample()
                        
                        cnt = 0
                        for item in sample:
                            if isinstance(item, torch.Tensor) and item.ndim == 4:
                                save_path = str(results_folder / f"{step}_sample_{cnt}.png")
                                save_image(item, save_path)
                                cnt += 1
                        if cnt == 0:
                            print("Warning: No image tensor found.")
                    except Exception as e:
                        print(f"Sampling Failed: {e}")

                model.train()

            # 3. 모델 저장 (5000 Step마다 - 재현성 점수 확보용)
            if step % 5000 == 0:
                save_path = results_folder / f"model_step_{step}.pt"
                accelerator.save(accelerator.unwrap_model(model).state_dict(), save_path)
                print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()