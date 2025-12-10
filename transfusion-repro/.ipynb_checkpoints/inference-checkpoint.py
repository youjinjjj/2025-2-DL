import argparse
import torch
from transformers import AutoTokenizer
from diffusers.models import AutoencoderKL
from torch import nn
from transfusion_pytorch import Transfusion
from torchvision.utils import save_image
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    return parser.parse_args()


class Encoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    def forward(self, image):
        with torch.no_grad():
            latent = self.vae.encode(image * 2 - 1).latent_dist.sample()
        return 0.18215 * latent

class Decoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    def forward(self, latents):
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        return (image / 2 + 0.5).clamp(0, 1)

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    
    encoder = Encoder(vae)
    decoder = Decoder(vae)

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
        transformer = dict(dim=768, depth=12, dim_head=64, heads=12)
    ).to(device)


    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint not found at {args.ckpt}")
        return

    print(f"Loading weights from {args.ckpt} ...")
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    prompt = "A cute cat"
    print(f"Original Prompt: '{prompt}'")
    

    text_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)


    boi_id = model.som_ids[0] 
    boi_tensor = torch.tensor([[boi_id]], device=device)


    forced_input = torch.cat([text_ids, boi_tensor], dim=1)

    print(f"Force prompting with <BOI> token injection...")

    with torch.no_grad():
        output = model.sample(prompt=forced_input, max_length=1200)

    cnt = 0
    for item in output:
        if isinstance(item, torch.Tensor) and item.ndim == 4:
            save_name = f"success_force_{cnt}.png"
            save_image(item, save_name)
            print(f"★ SUCCESS: Image Generated! Saved to {save_name}")
            cnt += 1

    if cnt == 0:
        print("Result: No image generated. (Model might need more training steps)")

if __name__ == "__main__":
    main()