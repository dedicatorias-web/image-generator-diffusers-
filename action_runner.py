import os
import sys
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from datetime import datetime
import argparse
import warnings

# Suprime avisos desnecessários
warnings.filterwarnings("ignore")

# Configurações para GitHub Actions
os.environ['TRANSFORMERS_CACHE'] = '/home/runner/.cache/huggingface'
os.environ['HF_HOME'] = '/home/runner/.cache/huggingface'
os.environ['TORCH_HOME'] = '/home/runner/.cache/torch'

class LightweightImageGenerator:
    def __init__(self, model_id="hf-internal-testing/tiny-stable-diffusion-pipe"):
        """
        Inicializa o gerador com modelo tiny para testes no Actions
        """
        print(f"Carregando modelo {model_id}...")
        
        # Usa CPU no GitHub Actions
        self.device = "cpu"
        
        try:
            # Tenta carregar um modelo menor para testes
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print(f"Erro ao carregar modelo de teste: {e}")
            print("Tentando modelo CompVis...")
            # Fallback para modelo padrão
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True,
                revision="fp16" if self.device == "cuda" else "main"
            )
        
        # Scheduler mais rápido
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Otimizações para CPU
        self.pipe.enable_attention_slicing()
        
        print("Modelo carregado!")
    
    def generate_image(self, prompt, negative_prompt="", steps=15, guidance=7.5, 
                      width=256, height=256, seed=None):
        """
        Gera imagem com configurações otimizadas para CPU
        """
        if seed:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        else:
            generator = None
        
        print(f"Gerando imagem: '{prompt}'")
        print(f"Configurações: {steps} passos, {width}x{height}")
        print(f"Isso pode levar alguns minutos no GitHub Actions...")
        
        with torch.no_grad():
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
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=str, default="")
    parser.add_argument("--batch", type=int, default=1)
    
    args = parser.parse_args()
    
    # Ajusta tamanhos para Actions (limita para economizar recursos)
    args.width = min(args.width, 512)
    args.height = min(args.height, 512)
    args.steps = min(args.steps, 25)
    
    print(f"Configurações ajustadas para GitHub Actions:")
    print(f"- Dimensões: {args.width}x{args.height}")
    print(f"- Passos: {args.steps}")
    
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
