import os
import sys
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from datetime import datetime
import argparse
from huggingface_hub import login

# Configurações para GitHub Actions
os.environ['TRANSFORMERS_CACHE'] = '/home/runner/.cache/huggingface'
os.environ['HF_HOME'] = '/home/runner/.cache/huggingface'

# Login no Hugging Face se token estiver disponível
if 'HF_TOKEN' in os.environ:
    login(token=os.environ['HF_TOKEN'])

class LightweightImageGenerator:
    def __init__(self, model_id="CompVis/stable-diffusion-v1-4"):
        """
        Inicializa o gerador com modelo mais leve para Actions
        """
        print(f"Carregando modelo {model_id}...")
        
        # Usa CPU no GitHub Actions
        self.device = "cpu"
        
        # Carrega o pipeline com configurações otimizadas
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True
        )
        
        # Scheduler mais rápido
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe = self.pipe.to(self.device)
        print("Modelo carregado!")
    
    def generate_image(self, prompt, negative_prompt="", steps=20, guidance=7.5, 
                      width=512, height=512, seed=None):
        """
        Gera imagem com configurações otimizadas para CPU
        """
        if seed:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        else:
            generator = None
        
        print(f"Gerando imagem: '{prompt}'")
        print(f"Isso pode levar alguns minutos no GitHub Actions...")
        
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=generator
        ).images[0]
        
        return image
    
    def save_image(self, image, prompt):
        """
        Salva a imagem gerada
        """
        os.makedirs("generated_images", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_'))
        safe_prompt = safe_prompt.replace(' ', '_')
        
        filename = f"{timestamp}_{safe_prompt}.png"
        filepath = os.path.join("generated_images", filename)
        
        image.save(filepath)
        print(f"Imagem salva: {filepath}")
        return filepath

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=str, default="")
    parser.add_argument("--batch", type=int, default=1)
    
    args = parser.parse_args()
    
    # Inicializa gerador
    generator = LightweightImageGenerator()
    
    # Gera imagens
    for i in range(args.batch):
        print(f"\nGerando imagem {i+1} de {args.batch}...")
        
        seed = int(args.seed) + i if args.seed else None
        
        image = generator.generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            steps=args.steps,
            guidance=args.guidance,
            width=args.width,
            height=args.height,
            seed=seed
        )
        
        generator.save_image(image, args.prompt)
    
    print("\nGeração concluída!")

if __name__ == "__main__":
    main()
