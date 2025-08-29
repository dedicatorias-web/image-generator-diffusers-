# Gerador de Imagens com Diffusers - GitHub Actions

Este projeto utiliza GitHub Actions para gerar imagens com Stable Diffusion diretamente no GitHub, sem necessidade de hardware local.

## 🚀 Como Usar

### Método 1: Interface Web (Recomendado)

1. Vá para a aba **Actions** do repositório
2. Selecione **"Generate Image"** no menu lateral
3. Clique em **"Run workflow"**
4. Preencha os parâmetros:
   - **Prompt**: Descrição da imagem (obrigatório)
   - **Negative prompt**: O que evitar (opcional)
   - **Steps**: Número de passos (padrão: 30)
   - **Guidance**: Escala de orientação (padrão: 7.5)
   - **Width/Height**: Dimensões da imagem (padrão: 512x512)
   - **Seed**: Para reproduzir resultados (opcional)
   - **Batch**: Quantidade de imagens (padrão: 1)
5. Clique em **"Run workflow"**

### Método 2: Geração Automática Diária

O workflow `Scheduled Image Generation` executa automaticamente todos os dias às 12:00 UTC, gerando uma imagem com prompt aleatório.

### Método 3: Via API do GitHub

```bash
curl -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/SEU_USUARIO/image-generator-diffusers/actions/workflows/generate-image.yml/dispatches \
  -d '{
    "ref": "main",
    "inputs": {
      "prompt": "astronauta surfando em Saturno",
      "steps": "30",
      "guidance": "7.5"
    }
  }'
