# Byte-Level Modeling (BLM)

This project trains compact GPT-style transformers directly on byte sequences for compression-oriented experiments.

## Quick start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare a dataset directory**

   Place any collection of text or binary files under a directory. All files are read as raw bytes, concatenated, and chunked into fixed-length contexts during training.

3. **Launch training**

   ```bash
   python -m blm.train_bytes /path/to/dataset \
       --block-size 512 \
       --batch-size 32 \
       --epochs 10 \
       --output-dir checkpoints
   ```

   Common flags:

   - `--cpu`: force CPU training even if CUDA is available.
   - `--amp`: enable automatic mixed precision (CUDA only).
   - `--save-best`: keep only the checkpoint with the best validation loss.
   - `--stride`: control the hop length between training windows (defaults to the block size).

4. **Generate bytes**

   ```python
   import torch
   from blm.gpt import BytesGPT, BytesGPTConfig, decode

   ckpt = torch.load("checkpoints/best.pt", map_location="cpu")
   config = BytesGPTConfig(**ckpt["config"])
   model = BytesGPT(config)
   model.load_state_dict(ckpt["model_state"])
   model.eval()

   context = torch.tensor([[66, 76, 77]], dtype=torch.long)  # "BLM"
   generated = model.generate(context, max_new_tokens=64)
   print(decode(generated[0].tolist()))
   ```

## Repository layout

- `blm/gpt.py` — PyTorch implementation of the byte-level GPT model.
- `blm/data.py` — Dataset and DataLoader helpers for byte corpora.
- `blm/train_bytes.py` — Command-line training script with validation and checkpointing.
- `blm/train.py` — Torch-based cross-entropy loss helper.
- `blm/analysis.py` — Visualization utilities for probability distributions and entropy.
- `blm/compression.py` — Arithmetic coding experiments powered by trained models.

## Troubleshooting

- **"Corpus is shorter than block_size + 1"** — reduce `--block-size` or supply more data.
- **Training is slow on CPU** — add `--amp` and run on a CUDA-capable GPU, or lower `--n_layer`/`--n_embd`.
- **NaN loss** — decrease `--lr` or disable mixed precision.

Feel free to adapt the configuration parameters (layers, heads, embedding size, stride) to match your hardware and compression goals.
