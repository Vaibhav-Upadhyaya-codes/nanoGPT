# nanoGPT Chatbot

This repository contains a tiny **character‑level GPT** implementation based on Andrej Karpathy's nanoGPT tutorial, extended with a simple interactive chatbot interface.

## Features
- **Transformer architecture** with multi‑head self‑attention and feed‑forward layers.
- Configurable hyper‑parameters (embedding size, number of layers, heads, dropout, etc.).
- **Training loop** that logs training and validation loss.
- **`chat()` function** providing a REPL‑style conversation with the trained model.  It builds a prompt from the conversation history, encodes it, generates a response and prints it.  Type `quit`, `exit` or `q` to leave the chat.

## Technical Details
- **Model**: `BigramLanguageModel` – token and positional embeddings → stacked `Block`s (self‑attention + feed‑forward) → final layer‑norm and linear head.
- **Hyper‑parameters** (can be tweaked in `model.py`):
  - `batch_size = 16`
  - `block_size = 32`
  - `n_embd = 64`
  - `n_head = 4`
  - `n_layer = 4`
  - `learning_rate = 1e-3`
  - `max_iters = 5000`
- **Training data**: a text file `clalong_childhood_fantasy_stories.txt` (your custom fantasy story corpus).  The script builds a character‑level vocabulary, encodes the text, and splits it 90 %/10 % for training/validation.
- **Generation**: `model.generate` samples tokens autoregressively using softmax probabilities.
- **Chat loop** (`chat()`):
  ```python
  def chat(max_new_tokens=256):
      # interactive REPL – builds prompt from history, encodes, generates, extracts reply
  ```
  The function maintains `conversation_history` so each turn includes the full context.

## Usage
1. **Install dependencies** (PyTorch, etc.)
2. Run the script:
   ```bash
   python model.py
   ```
   Training will start, showing loss every `eval_interval` steps.
3. After training, the script automatically enters chat mode.  Interact with the model via the console.

## License
This code is provided for educational purposes.  Feel free to fork, experiment, and improve it.
