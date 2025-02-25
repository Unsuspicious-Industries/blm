import logging
from math import *

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import torch

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

model = None
tokenizer = None

def next_word_distribution(tokenized_text):
    """
    Returns the probability distribution over the next word given some text.
    """
    global model, tokenizer

    # Tokenized text logging
    logging.debug(f"Tokenized context length: {len(tokenized_text)}")

    # Get the model's output (we don't need gradients here)
    with torch.no_grad():
        input_tensor = torch.tensor(tokenized_text).unsqueeze(0)
        logging.debug(f"Input tensor shape: {input_tensor.shape}")
        outputs = model(input_ids=input_tensor)
        logits = outputs.logits
        logging.debug(f"Logits shape: {logits.shape}")

    # Get the logits for the next token (last position)
    next_token_logits = logits[0, -1, :]

    # Apply temperature (if needed, add temperature scaling here)
    next_token_logits = next_token_logits

    # Convert to probabilities using softmax
    probs = F.softmax(next_token_logits, dim=-1)

    # Convert to numpy for easier handling
    probs = probs.numpy()

    # Debug: Log the maximum probability and its index
    max_prob = probs.max()
    max_idx = probs.argmax()
    logging.debug(f"Max probability: {max_prob} at index: {max_idx}")



    return probs

# Example usage:
def load_model(model_name="gpt2"):
    """Loads model and tokenizer"""
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    logging.debug(f"Loaded model: {model_name}")


def estimate_sequence_entropy(tokenized_text):
    total_entropy = 0
    context = []
    for wt in tokenized_text:
        context.append(wt)
        probs = next_word_distribution(context)

        top_k = 50
        top_k_indices = probs.argsort()[-top_k:][::-1]
        probs = probs[top_k_indices]

        entropy_i = -sum(p * log2(p) for p in probs if p > 0)
        total_entropy += entropy_i
        logging.debug(f"Current context length: {len(context)}, entropy: {entropy_i}, total_entropy: {total_entropy}")

    return total_entropy

def estimate_token_entropy(tokenized_text, position):
    context = tokenized_text[:position]
    probs = next_word_distribution(context)
    token_prob = probs[tokenized_text[position]]  # May need to adjust indexing
    logging.debug(f"Entropy for token at position {position}: probability {token_prob}")
    return -log2(token_prob)


def cut_sentence_entropy(text, entropy_budget=1.0, entropy_chunks_num=None):

    tokenized_text = tokenizer(text, return_tensors="pt")["input_ids"].squeeze().tolist()
    logging.debug(f"Tokenized text: {tokenized_text}")
    current_entropy = estimate_sequence_entropy(tokenized_text)
    logging.debug(f"Total estimated sequence entropy: {current_entropy}")

    if entropy_chunks_num:
        entropy_budget = current_entropy / entropy_chunks_num

    if current_entropy <= entropy_budget:
        return text

    # get the varentropy of each word
    varentropies = []
    for i in range(1,len(tokenized_text)):
        logging.debug(f"Estimating entropy for token at position {i}")
        varentropy = estimate_token_entropy(tokenized_text, i)
        varentropies.append(varentropy)
        logging.debug(f"Varentropy at position {i}: {varentropy}")

    # make word groups that fits the best our budget
    groups = []
    group = []
    group_entropy = 0
    for i, varentropy in enumerate(varentropies):
        if group_entropy + varentropy <= entropy_budget:
            group.append(tokenized_text[i])
            group_entropy += varentropy
        else:
            groups.append(group)
            group = [tokenized_text[i]]
            group_entropy = varentropy
    if group:  # add any remaining group
        groups.append(group)

    # map the groups to text
    text_groups = []
    for group in groups:
        text_group = tokenizer.decode(group)
        text_groups.append(text_group)


    return text_groups

if __name__ == "__main__":
    load_model("gpt2")
    text = "The fundamental theorem of calculus elegantly connects differentiation and integration, yet xylophone narwhals quantum-fluctuate beneath cerulean crystalline structures while drinking coffee at Starbucks and checking their Instagram feed for the latest viral cat videos."
    
    print(cut_sentence_entropy(text, entropy_chunks_num=3))