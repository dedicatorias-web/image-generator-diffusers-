import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from datetime import datetime
import argparse
from config import *

class ImageGenerator:
    def __init__(self, model_id=MODEL_ID, device=DEVICE):
        """
        Inicializa o gerador de imagens
        """
        print(f"Carregando modelo {model_id}...")
        
        # Carrega o pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True
        )
        
        # Configura o scheduler para geração mais rápida
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Move para o dispositivo apropriado
        self.pipe = self.pipe.to(device)
        
        # Habilita otimizações de memória se estiver usando GPU
        if device == "cuda":
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
        
        print("Modelo carregado com sucesso!")
    
    def generate_image(
        self,
        prompt,
        negative_prompt="",
        num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale=DEFAULT_GUIDANCE_SCALE,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        seed=None
    ):
        """
        Gera uma imagem baseada no prompt fornecido
        """
        # Define seed para reprodutibilidade (opcional)
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
        else:
            generator = None
        
        # Gera a imagem
        print(f"Gerando imagem com o prompt: '{prompt}'")
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator
        ).images[0]
        
        return image
    
    def save_image(self, image, prompt, output_dir=OUTPUT_DIR):
        """
        Salva a imagem gerada com um nome único
        """
        # Cria um nome de arquivo único baseado no timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Limita o prompt a 50 caracteres e remove caracteres especiais
        safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')
        
        filename = f"{timestamp}_{safe_prompt}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Salva a imagem
        image.save(filepath)
        print(f"Imagem salva em: {filepath}")
        
        return filepath

def main():
    parser = argparse.ArgumentParser(description="Gerador de imagens usando Diffusers")
    parser.add_argument("prompt", type=str, help="Prompt para gerar a imagem")
    parser.add_argument("--negative-prompt", type=str, default="", help="Prompt negativo (o que evitar)")
    parser.add_argument("--steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS, help="Número de passos de inferência")
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE_SCALE, help="Escala de orientação")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Altura da imagem")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Largura da imagem")
    parser.add_argument("--seed", type=int, default=None, help="Seed para reprodutibilidade")
    parser.add_argument("--batch", type=int, default=1, help="Número de imagens para gerar")
    
    args = parser.parse_args()
    
    # Inicializa o gerador
    generator = ImageGenerator()
    
    # Gera as imagens
    for i in range(args.batch):
        print(f"\nGerando imagem {i+1} de {args.batch}...")
        
        image = generator.generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            seed=args.seed + i if args.seed else None
        )
        
        generator.save_image(image, args.prompt)
    
    print("\nGeração concluída!")

if __name__ == "__main__":
    main()
