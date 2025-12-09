import torch
from torch import nn
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from transformers import AutoTokenizer
from transfusion_pytorch import Transfusion
import os

# ==========================================
# 1. 클래스 정의 (학습 코드와 동일해야 함)
# ==========================================
class Encoder(nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae
    def forward(self, image):
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

def main():
    # 설정
    MODEL_PATH = "./results_coco_single/model_step_5000.pt"
    SAVE_DIR = "./generated_results"
    
    # 프롬프트: 모델이 이미지를 그리도록 유도하는 짧고 명확한 문장
    prompts = [
        "A cute cat",
        "A pizza",
        "A sunset",
        "A dog with ball"
    ]

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ==========================================
    # 2. 모델 로드
    # ==========================================
    print("Loading VAE and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.requires_grad_(False)
    vae.eval()
    
    encoder = Encoder(vae)
    decoder = Decoder(vae)

    print("Building Transfusion Model...")
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

    # 가중치 로드
    print(f"Loading weights from {MODEL_PATH}...")
    try:
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.to(device)
    model.eval()

    # ==========================================
    # 3. 이미지 생성 및 정밀 분석
    # ==========================================
    print("Starting generation...")
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Generating for prompt: '{prompt}'")
        
        text_enc = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
        text_tokens = text_enc['input_ids'].to(device)
        
        with torch.no_grad():
            # max_length를 줄여서 빠르게 확인 (256 토큰 정도만 생성)
            output = model.sample(text_tokens, max_length=256)

        found_image = False
        print(f"Output type: {type(output)}")
        
        # 출력 결과 분석
        if isinstance(output, list) or isinstance(output, tuple):
            print(f"Output list length: {len(output)}")
            for idx, item in enumerate(output):
                if isinstance(item, torch.Tensor):
                    print(f"  Item {idx} shape: {item.shape}")
                    # 차원이 3(CHW) 또는 4(BCHW)면 이미지로 저장
                    if item.ndim == 3 or item.ndim == 4:
                        save_path = f"{SAVE_DIR}/result_{i}_{idx}.png"
                        save_image(item, save_path)
                        print(f"  -> Saved image to {save_path}")
                        found_image = True
                else:
                    print(f"  Item {idx} type: {type(item)}")
        else:
            print(f"Output is not a list. It is: {output}")

        if not found_image:
            print("  Warning: No image tensor found in this output.")
            print("  (Explanation: The model might have generated only text tokens.)")

    print("\nDone!")

if __name__ == "__main__":
    main()
