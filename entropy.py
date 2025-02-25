import torch
import numpy as np
from math import log2
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration

import probs.gpt2
import probs.byt5

class EntropyEstimator:
    """Universal class for estimating entropy across different model types"""

    def __init__(self, model_name, level="token"):
        """
        Initialize entropy estimator

        Args:
            model_name: HuggingFace model name
            level: 'token' or 'byte' approach
        """
        self.level = level

        if level == "token":
            self.probablity_func = probs.gpt2.next_distribution
        elif level == "byte":
            self.probablity_func = probs.byt5.next_distribution
        else:
            raise ValueError("level must be 'token' or 'byte'")

        self.model.eval()

    def estimate_entropy(self, text, position):
        """Estimate entropy at position in text"""
        if self.level == "token":
            return self._estimate_token_entropy(text, position)
        else:
            return self._estimate_byte_entropy(text, position)

    def _estimate_token_entropy(self, text, position):
        """Estimate entropy for token-based models"""
        # Tokenize the whole text
        tokens = self.tokenizer.encode(text)

        # Find which token contains our position
        token_spans = []
        current_pos = 0
        for token_id in tokens:
            token_text = self.tokenizer.decode([token_id])
            token_len = len(token_text)
            token_spans.append((current_pos, current_pos + token_len, token_id))
            current_pos += token_len

        # Find which token contains our position
        target_token = None
        for start, end, token_id in token_spans:
            if start <= position < end:
                target_token = token_id
                token_position = len([t for s, e, t in token_spans if s < start])
                break

        if target_token is None:
            return 0  # Position is beyond text

        # Get context (all tokens before our target token)
        context_tokens = tokens[:token_position]
        inputs = self.tokenizer(self.tokenizer.decode(context_tokens), return_tensors="pt")

        # Convert logits to probabilities
        probs = self.probablity_func(inputs)

        # Get probability of the actual next token
        token_prob = probs[target_token].item()

        # Calculate entropy
        if token_prob > 0:
            entropy = -log2(token_prob)
        else:
            entropy = float('inf')

        return entropy

    def _estimate_byte_entropy(self, text, position):
        """Estimate entropy for byte-based models"""
        if position <= 0:
            return 0

        # Get context and target byte
        context = text[:position]
        target_byte = ord(text[position]) if position < len(text) else 0


        probs = self.probablity_func(context)

        # Get probability of the target byte
        byte_prob = probs[target_byte].item()

        # Calculate entropy
        if byte_prob > 0:
            entropy = -log2(byte_prob)
        else:
            entropy = float('inf')

        return entropy

    def calculate_text_entropy(self, text):
        """Calculate entropy for entire text"""
        total_entropy = 0
        for pos in range(len(text)):
            total_entropy += self.estimate_entropy(text, pos)
        return total_entropy

    def entropy_budget_segmentation(self, text, budget=10):
        """Segment text into chunks of approximately equal entropy"""
        segments = []
        current_segment = ""
        current_budget = 0

        for i, char in enumerate(text):
            entropy = self.estimate_entropy(text, i)
            if current_budget + entropy > budget and current_segment:
                segments.append(current_segment)
                current_segment = char
                current_budget = entropy
            else:
                current_segment += char
                current_budget += entropy

        if current_segment:
            segments.append(current_segment)

        return segments
