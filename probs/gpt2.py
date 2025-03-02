import logging
from math import *

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

class ByteLevelGPT2:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        # Set device to GPU if available
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model.to(self.device)

        # Map each byte to its token representation
        self.byte_to_token_map = self._map_bytes_to_tokens()

    def _map_bytes_to_tokens(self):
        """Map each byte to its corresponding token(s) in GPT-2's vocabulary."""
        byte_tokens = {}
        # GPT-2 encodes bytes 0-255 in its vocabulary
        for byte in range(256):
            # Convert byte to its UTF-8 representation
            byte_str = bytes([byte]).decode('utf-8', errors='replace')
            # Get token(s) that represent this byte
            tokens = self.tokenizer.tokenize(byte_str)

            if tokens:  # Some bytes might map to empty tokens
                byte_tokens[byte] = tokens
        return byte_tokens

    def get_next_byte_distribution(self, text):
        """Calculate probability distribution over the next byte."""
        # Tokenize the input text
        tokens = self.tokenizer.tokenize(text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_tensor = torch.tensor([input_ids], device=self.device)

        # Get next token distribution
        with torch.no_grad():
            outputs = self.model(input_tensor)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)

        # Convert token probabilities to byte probabilities
        byte_probs = np.zeros(256)

        # For each token in the vocabulary
        for token_id, token_prob in enumerate(next_token_probs):
            token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            # Get the byte representation of this token
            try:
                # This is simplified - in reality we need to handle multi-byte tokens
                token_bytes = token.replace('Ġ', ' ').encode('utf-8')
                # Distribute token probability to its first byte
                if token_bytes:
                    byte_probs[token_bytes[0]] += token_prob.item()
            except UnicodeEncodeError:
                # Some tokens may have special encoding
                continue

        # Normalize byte probabilities
        byte_probs = byte_probs / np.sum(byte_probs)
        return {i: p for i, p in enumerate(byte_probs) if p > 0}


blg = None

def load_model(model_name="gpt2"):
    global blg
    blg = ByteLevelGPT2(model_name=model_name)


def next_distribution(text,model_name="gpt2"):
    """
    Returns the probability distribution over the next byte given some text.
    """
    global blg

    if blg is None:
        load_model(model_name=model_name)

    if len(text) == 0:
        # return uniform probability for all bytes
        probs = {i: 1 / 256 for i in range(256)}
        return probs

    probs = blg.get_next_byte_distribution(text)
    return probs


if __name__ == "__main__":

    from analysis import plot_probs, autoregressive_byte_analysis

    base_text = "The "

    # First, let's look at the distribution for the next byte
    probs = next_distribution(base_text)

    # Print the top 50 most probable bytes
    top_50 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:50]
    print("Top 50 probable next bytes:")
    for byte_val, prob in top_50:
        char_repr = chr(byte_val) if 32 <= byte_val <= 126 else f"\\x{byte_val:02x}"
        print(f"{char_repr}: {prob:.6f}")

    # Calculate entropy
    entropy = -sum([p * log2(p) for t, p in probs.items()])
    print(f"Entropy at current position: {entropy:.4f} bits")

    # Plot the distribution
    plot_probs(probs)

    # Now analyze a word autoregressively
    print("\n" + "="*60)
    print("AUTOREGRESSIVE ANALYSIS")
    print("="*60)
    results = autoregressive_byte_analysis(base_text, next_distribution, target_word="quick brown fox jumps over the lazy dog")