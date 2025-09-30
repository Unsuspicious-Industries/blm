#!/usr/bin/env python3
"""Model-based compression using trained BytesGPT for probability estimation."""

from __future__ import annotations

import argparse
import math
import sys
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn

from .gpt import BytesGPT, BytesGPTConfig, encode as bytes_encode, decode as bytes_decode

# Set high precision for arithmetic coding
getcontext().prec = 300


class ModelCompressor:
    """Arithmetic coding compressor using a trained BytesGPT model."""

    def __init__(self, model: BytesGPT, temperature: float = 1.0, context_length: int = 256):
        self.model = model
        self.model.eval()
        self.temperature = max(temperature, 1e-8)
        self.context_length = context_length
        self.device = next(model.parameters()).device

    def _get_distribution(self, context: List[int]) -> Dict[int, float]:
        """Get next-byte probability distribution given context."""
        if not context:
            # Uniform distribution for empty context
            prob = 1.0 / 256.0
            return {i: prob for i in range(256)}

        # Truncate context to model limits
        context = context[-min(len(context), self.context_length):]
        
        with torch.no_grad():
            input_tensor = torch.tensor([context], dtype=torch.long, device=self.device)
            logits, _ = self.model(input_tensor)
            
            # Get logits for the last position
            last_logits = logits[0, -1, :] / self.temperature
            probs = torch.softmax(last_logits, dim=0)
            
            # Convert to dictionary
            return {i: float(probs[i]) for i in range(256)}

    def compress(self, data: bytes, verbose: bool = False) -> Tuple[Decimal, Decimal]:
        """Compress bytes using arithmetic coding with model probabilities."""
        if not data:
            return Decimal(0), Decimal(1)

        # Adjust precision based on data length
        required_precision = max(300, len(data) * 15)
        getcontext().prec = required_precision

        byte_list = list(data)
        low, high = Decimal(0), Decimal(1)
        context = []

        for i, byte_val in enumerate(byte_list):
            if verbose:
                char_repr = chr(byte_val) if 32 <= byte_val <= 126 else f"\\x{byte_val:02x}"
                print(f"Position {i}: {char_repr} (byte {byte_val})")

            # Get probability distribution
            probs = self._get_distribution(context)
            
            # Convert to Decimal for precision
            decimal_probs = {k: Decimal(str(v)) for k, v in probs.items()}
            
            # Calculate cumulative probabilities
            sorted_bytes = sorted(decimal_probs.keys())
            cumulative = Decimal(0)
            
            # Find cumulative probability up to current byte
            for b in sorted_bytes:
                if b == byte_val:
                    break
                cumulative += decimal_probs[b]
            
            prob = decimal_probs[byte_val]
            range_size = high - low
            
            # Update interval
            new_low = low + cumulative * range_size
            new_high = low + (cumulative + prob) * range_size
            
            if verbose:
                print(f"  Prob: {float(prob):.6f}, Range: [{float(new_low):.10f}, {float(new_high):.10f}]")
            
            low, high = new_low, new_high
            
            # Update context for next prediction
            context.append(byte_val)
            if len(context) > self.context_length:
                context = context[-self.context_length:]

        return low, high

    def decompress(self, low: Decimal, high: Decimal, length: int, verbose: bool = False) -> bytes:
        """Decompress using arithmetic decoding with model probabilities."""
        if length == 0:
            return b""

        # Use midpoint of range as the encoded value
        encoded_value = (low + high) / 2
        result = []
        context = []
        
        current_low, current_high = Decimal(0), Decimal(1)

        for i in range(length):
            # Get probability distribution
            probs = self._get_distribution(context)
            decimal_probs = {k: Decimal(str(v)) for k, v in probs.items()}
            
            # Find which byte the encoded value corresponds to
            range_size = current_high - current_low
            target_value = (encoded_value - current_low) / range_size
            
            cumulative = Decimal(0)
            decoded_byte = None
            
            for byte_val in sorted(decimal_probs.keys()):
                prob = decimal_probs[byte_val]
                if cumulative <= target_value < cumulative + prob:
                    decoded_byte = byte_val
                    break
                cumulative += prob
            
            if decoded_byte is None:
                # Fallback to last byte if precision issues
                decoded_byte = 255
                cumulative = sum(decimal_probs[b] for b in range(255))
                prob = decimal_probs[255]
            else:
                prob = decimal_probs[decoded_byte]
            
            if verbose:
                char_repr = chr(decoded_byte) if 32 <= decoded_byte <= 126 else f"\\x{decoded_byte:02x}"
                print(f"Position {i}: {char_repr} (byte {decoded_byte})")
            
            result.append(decoded_byte)
            
            # Update range for next iteration
            new_low = current_low + cumulative * range_size
            new_high = current_low + (cumulative + prob) * range_size
            current_low, current_high = new_low, new_high
            
            # Update context
            context.append(decoded_byte)
            if len(context) > self.context_length:
                context = context[-self.context_length:]

        return bytes(result)

    def calculate_compression_ratio(self, data: bytes) -> float:
        """Calculate theoretical compression ratio without actually compressing."""
        if not data:
            return 1.0

        total_bits = 0.0
        context = []
        
        for byte_val in data:
            probs = self._get_distribution(context)
            prob = probs.get(byte_val, 1e-10)  # Small fallback probability
            bits = -math.log2(prob)
            total_bits += bits
            
            context.append(byte_val)
            if len(context) > self.context_length:
                context = context[-self.context_length:]
        
        original_bits = len(data) * 8
        return original_bits / total_bits if total_bits > 0 else 1.0


def load_model(checkpoint_path: str) -> BytesGPT:
    """Load a trained BytesGPT model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        config = BytesGPTConfig(**config_dict)
        model = BytesGPT(config)
        model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint["model"])
    else:
        # Fallback: try to infer config or use defaults
        print("Warning: No config found in checkpoint, using default config")
        config = BytesGPTConfig()
        model = BytesGPT(config)
        model.load_state_dict(checkpoint)
    
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Test compression with trained BytesGPT models")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--input", "-i", help="Input file to compress (default: read from stdin)")
    parser.add_argument("--text", "-t", help="Text string to compress")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--context-length", type=int, default=256, help="Context length for compression (default: 256)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--ratio-only", action="store_true", help="Only calculate compression ratio (faster)")
    parser.add_argument("--test-roundtrip", action="store_true", help="Test compression and decompression")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    try:
        model = load_model(args.checkpoint)
        print(f"Model loaded successfully on {next(model.parameters()).device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Get input data
    if args.text:
        data = args.text.encode("utf-8")
        source = f"text argument ({len(data)} bytes)"
    elif args.input:
        try:
            data = Path(args.input).read_bytes()
            source = f"file {args.input} ({len(data)} bytes)"
        except Exception as e:
            print(f"Error reading input file: {e}")
            return 1
    else:
        try:
            data = sys.stdin.buffer.read()
            source = f"stdin ({len(data)} bytes)"
        except Exception as e:
            print(f"Error reading from stdin: {e}")
            return 1
    
    if not data:
        print("No input data provided")
        return 1
    
    print(f"Input: {source}")
    
    # Create compressor
    compressor = ModelCompressor(model, args.temperature, args.context_length)
    
    if args.ratio_only:
        # Just calculate theoretical compression ratio
        print("Calculating compression ratio...")
        ratio = compressor.calculate_compression_ratio(data)
        original_bits = len(data) * 8
        compressed_bits = original_bits / ratio
        
        print(f"Original size: {len(data)} bytes ({original_bits} bits)")
        print(f"Compressed size (theoretical): {compressed_bits:.1f} bits ({compressed_bits/8:.1f} bytes)")
        print(f"Compression ratio: {ratio:.3f}x")
        print(f"Space savings: {(1 - 1/ratio)*100:.1f}%")
        
    else:
        # Perform actual compression
        print("Compressing...")
        low, high = compressor.compress(data, verbose=args.verbose)
        
        # Calculate compression statistics
        range_size = high - low
        bits_needed = -float(range_size.ln()) / math.log(2)  # Convert to log2
        compression_ratio = (len(data) * 8) / bits_needed
        
        print(f"\nCompression results:")
        print(f"Original size: {len(data)} bytes ({len(data) * 8} bits)")
        print(f"Compressed range: [{float(low):.15f}, {float(high):.15f}]")
        print(f"Range size: {float(range_size):.2e}")
        print(f"Bits needed: {bits_needed:.1f}")
        print(f"Compression ratio: {compression_ratio:.3f}x")
        print(f"Space savings: {(1 - 1/compression_ratio)*100:.1f}%")
        
        if args.test_roundtrip:
            print("\nTesting roundtrip compression...")
            decompressed = compressor.decompress(low, high, len(data), verbose=args.verbose)
            
            if decompressed == data:
                print("✅ Roundtrip test PASSED: decompressed data matches original")
            else:
                print("❌ Roundtrip test FAILED: decompressed data differs from original")
                print(f"Original: {data[:50]}..." if len(data) > 50 else f"Original: {data}")
                print(f"Decompressed: {decompressed[:50]}..." if len(decompressed) > 50 else f"Decompressed: {decompressed}")
                return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())