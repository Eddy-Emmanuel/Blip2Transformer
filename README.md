# BLIP-2 from Scratch 🖼️→💬

A full ground-up implementation of the BLIP-2 multimodal architecture in PyTorch — built to understand every component rather than just calling a pretrained pipeline. The Q-Former is implemented entirely from scratch; the vision encoder and LLM are frozen pretrained backbones, consistent with the original paper's design.

Trained on **COCO Captions (33k)** for image captioning. Q-Former weights are published to HuggingFace.

---

## Architecture

BLIP-2 bridges a frozen vision encoder and a frozen LLM through a lightweight, trainable **Q-Former** (Querying Transformer). Only Q-Former parameters are updated during training.

```
Input Image
     │
     ▼
┌─────────────────────────────────┐
│  Vision Encoder (Frozen)        │
│  SigLIP (so400m-patch14-384)    │
│  → patch embeddings [N, 1152]   │
└────────────────┬────────────────┘
                 │ cross-attention
                 ▼
┌─────────────────────────────────┐
│  Q-Former (Trainable) ✦         │
│  32 learned query tokens        │
│  12 layers × 16 heads           │
│  dim = 1024                     │
│  → compressed visual repr.      │
│    [32, 1024]                   │
└────────────────┬────────────────┘
                 │ prepended to text embeddings
                 ▼
┌─────────────────────────────────┐
│  LLM (Frozen)                   │
│  Qwen3-0.6B (4-bit NF4 QLoRA)  │
│  → generated caption            │
└─────────────────────────────────┘
```

---

## Components

### Q-Former (from scratch)
| Component | Details |
|---|---|
| `QFormerAttention` | Multi-head self-attention with learned query tokens; cross-attends to vision patch features |
| `QFormerLayer` | Attention → cross-attention → FFN with pre-norm |
| `QFormer` | 12 stacked layers, 32 query tokens, dim 1024, 16 heads |

### Vision Encoder
- `google/siglip-so400m-patch14-384` — pretrained, fully frozen
- Patch embeddings extracted from `.last_hidden_state[:, 1:, :]` (CLS token excluded)
- Output shape: `[B, N_patches, 1152]`

### Language Model
- `Qwen/Qwen3-0.6B` — pretrained, fully frozen
- Loaded in **4-bit NF4** via `BitsAndBytesConfig`
- Q-Former output is prepended to text token embeddings before the LLM forward pass
- Custom Qwen3 chat template applied manually (ChatML format with `<|im_start|>` / `<|im_end|>`)

### BlipProcessor
- Wraps SigLIP's `AutoProcessor` for image preprocessing
- Wraps `AutoTokenizer` for text tokenization with chat template application
- Handles all preprocessing internally — `BlipDataSet` returns raw PIL images

### BLIP2Model
- Composes vision encoder + Q-Former + LLM into a single `nn.Module`
- Freezes vision and LLM parameters on init; only Q-Former is trainable
- `InferenceMode()` handles full image-to-caption generation with `model.generate()`

---

## Training

```python
# Only Q-Former parameters are trained
optimizer = torch.optim.AdamW([
    {"params": blip_model.qformer.parameters(), "lr": 1e-4}
])
```

| Setting | Value |
|---|---|
| Dataset | `astro21/coco-caption-train-split-33k` |
| Train/Val split | 70% / 30% |
| Optimizer | AdamW |
| LR Scheduler | OneCycleLR |
| Early stopping | patience=3, min_delta=1e-4 |
| Epochs | 2 |
| Trainable params | Q-Former only |
| Frozen params | SigLIP encoder + Qwen3-0.6B |

The `Blip2Trainer` class handles the training loop, validation, early stopping, and checkpoint saving (`qformer_best.pt`).

---

## Inference

```python
from PIL import Image
import requests
from io import BytesIO

# Load image
url = "https://your-image-url.jpg"
image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")

# Build prompt
prompt = [
    {"role": "system",  "content": "You are an expert image captioning assistant."},
    {"role": "user",    "content": "Write a natural, single-sentence caption describing the image."}
]

chat_temp = text_tokenizer.apply_chat_template(
    prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
)

# Generate caption
caption = InferenceMode(blip_model, image, chat_temp, Config)
print("Caption:", caption)
```

---

## Setup

```bash
pip install torch transformers peft bitsandbytes datasets pillow requests
```

**Q-Former weights** are saved at:
```
/kaggle/input/models/edifonjimmy/qformer/pytorch/default/1/checkpoints/qformer_best.pt
```

Or load directly from HuggingFace: [`Edifon`](https://huggingface.co/Edifon)

---

## Config

```python
class Config:
    device            = "cuda"
    vision_processor  = "google/siglip-so400m-patch14-384"
    llm_model         = "Qwen/Qwen3-0.6B"
    embedding_dim     = 1024        # Q-Former hidden dim
    vision_dim_model  = 1152        # SigLIP output dim
    n_queries         = 32          # Learnable query tokens
    n_layers          = 12          # Q-Former depth
    n_heads           = 16          # Attention heads
    max_length        = 1024
    dropout           = 0
    layer_bias        = True
```

---

## Design Decisions

**Why freeze the vision encoder and LLM?**  
Consistent with the original BLIP-2 paper — the Q-Former acts as a learned bridge that distills visual information into a compact token sequence the LLM can consume. This keeps compute tractable and lets the Q-Former be swapped across different backbone pairs.

**Why raw PIL images in the dataset?**  
`BlipDataSet` intentionally returns raw images with no preprocessing applied. All preprocessing (SigLIP resize/normalize, tokenization) happens inside `BlipProcessor`, which is part of `BLIP2Model`. This keeps the dataset generic and the model self-contained.

**Why exclude the CLS token from vision features?**  
The CLS token aggregates global image context; the Q-Former already learns to aggregate via its cross-attention mechanism. Patch tokens carry richer spatial detail for the cross-attending queries.

---

## Before & After Training

The same image and prompt passed through the model before any Q-Former training vs. after.

> **Prompt:** *"Write a natural, single-sentence caption describing the image."*

| | Before Training | After Training |
|---|---|---|
| **Image** | ![before]() | ![after]() |
| **Caption** | *(random / incoherent output — Q-Former weights are randomly initialized)* | *(coherent, image-grounded caption)* |

Replace the image `src` placeholders above with your actual screenshots once uploaded.

---

## Notebooks

| Notebook | Description |
|---|---|
| `blip2-transformer.ipynb` | Full pipeline: data loading, model definition, training |
| `blip2inference.ipynb` | Load saved Q-Former weights and run inference on arbitrary images |

---
## License

MIT
