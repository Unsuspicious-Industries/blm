import torch
import torch.functional as F
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
    
    def from_pretrained(self, *args, **kwargs):
        # Load the underlying GPT2 model from pre-trained weights.
        self.model = self.model.from_pretrained(*args, **kwargs)
        return self

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

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    # Load the model and tokenizer
    model = GPTBForCausalLM(GPTBConfig())
    model = model.from_pretrained("XXXX")
    tokenizer = model.tokenizer

def next_distribution(text):

    global model, tokenizer

    if len(text) == 0:
        # return uniform distribution
        return {i: 1/config.vocab_size for i in range(config.vocab_size)}

    tokens = tokenizer(text)["input_ids"]
    input_tensor = torch.tensor(tokens,requires_grad=False).unsqueeze(0)  
    byte_probs = model(input_ids=input_tensor)
    normalized_probs = F.softmax(byte_probs.logits[0, -1, :], dim=-1).detach().cpu().numpy()
    return {i: p for i, p in enumerate(normalized_probs) if p > 0}




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