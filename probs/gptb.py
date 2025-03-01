import torch
from transformers import PreTrainedModel, PretrainedConfig, GPT2LMHeadModel, GPT2Config
from transformers import PreTrainedTokenizer

# Keep your custom tokenizer as is.
class ByteTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        self.byte_encoder = {i: chr(i) for i in range(256)}
        super().__init__(**kwargs)

    def _tokenize(self, text):
        # text ➡️ UTF-8 bytes ➡️ token IDs as strings
        return [str(b) for b in text.encode("utf-8")]

    def _convert_token_to_id(self, token):
        # token (string of byte) ➡️ token ID (int)
        return int(token)

    def _convert_id_to_token(self, index):
        return str(index)

    def convert_tokens_to_string(self, tokens):
        byte_values = [int(t) for t in tokens]
        return bytes(byte_values).decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return 256

    def get_vocab(self):
        return {str(i): i for i in range(256)}


# Define a simple configuration for our GPTB model.
class GPTBConfig(PretrainedConfig):
    model_type = "gptb"
    def __init__(self, **kwargs):
        # Set a default vocabulary size for byte tokens if not provided.
        if "vocab_size" not in kwargs:
            kwargs["vocab_size"] = 256
        super().__init__(**kwargs)


# Now construct a model that behaves like a Hugging Face model.
class GPTBForCausalLM(PreTrainedModel):
    config_class = GPTBConfig

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = ByteTokenizer()
        # When creating the underlying GPT2 model, merge our config with GPT2Config.
        # This ensures required parameters (like vocab_size) are set.
        gpt2_config = GPT2Config(**config.to_dict())
        self.model = GPT2LMHeadModel(gpt2_config)
        self.model

    def forward(self, input_ids, labels=None, **kwargs):
        # Forward pass using the underlying GPT2 model.
        outputs = self.model(input_ids=input_ids, labels=labels, **kwargs)
        return outputs

    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids=input_ids, **kwargs)

    # Optional: add utility methods for encoding/decoding text.
    def encode_text(self, text):
        tokens = self.tokenizer.tokenize(text)
        return torch.tensor(
            [self.tokenizer._convert_token_to_id(t) for t in tokens]
        ).unsqueeze(0)

    def decode_tokens(self, token_ids):
        tokens = [str(id) for id in token_ids]
        return self.tokenizer.convert_tokens_to_string(tokens)


if __name__ == "__main__":
    # Quick test of our model behaving like a Hugging Face model:
    config = GPTBConfig()
    model = GPTBForCausalLM(config)
    
    # Encode text using our custom method.
    input_text = "My name is"
    input_ids = model.encode_text(input_text)
    print(f"Input IDs: {input_ids}")

    # Get model outputs (loss will be computed if labels are provided).
    outputs = model(input_ids=input_ids, labels=input_ids)
    print("Loss:", outputs.loss.item())

    # Use generate
    generated_ids = model.generate(input_ids, max_length=20)
    # print the generated ids
    print(f"Generated IDs: {generated_ids}")
    generated_text = model.decode_tokens(generated_ids[0].tolist())
    print(f"Generated text: {generated_text}")