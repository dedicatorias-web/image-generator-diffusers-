import os

# Configurações do modelo
MODEL_ID = "stabilityai/stable-diffusion-2-1"  # Pode ser alterado para outros modelos
DEVICE = "cuda"  # ou "cpu" se não tiver GPU

# Configurações de geração
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512

# Diretórios
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
