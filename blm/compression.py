from decimal import Decimal, getcontext
getcontext().prec = 200  # Increase precision as needed

from probs.gpt2 import next_distribution as dist_gpt2
from probs.gptb import next_distribution as dist_gptb


next_distribution = dist_gpt2  # or dist_gptb

def set_distribution(distribution):

    global next_distribution

    if distribution == "gpt2":
        next_distribution = dist_gpt2
    elif distribution == "gptb":
        next_distribution = dist_gptb

def adjust_precision(text_length):
    required = max(200, text_length * 10)
    getcontext().prec = required

def _distribution(context):
    if len(context) == 0:
        # return uniform probability for all bytes as Decimals (redundant but consistent)
        probs = [Decimal(1) / Decimal(256)] * 256
        return dict(zip(range(256), probs))
    # convert context to string
    context = "".join(chr(c) for c in context)
    probs = next_distribution(context)
    # Convert probability floats to Decimal
    return {k: Decimal(str(v)) for k, v in probs.items()}

def _sorted_list_dist(dist):
    return sorted(dist.items(), key=lambda x: x[0], reverse=True)

def _cumsum(id, dist):
    P = _sorted_list_dist(dist)
    cum = Decimal(0)
    i = 0
    while P[i][0] != id:
        cum += P[i][1]
        i += 1
    return cum

def _encode(text, pos=0, space=(Decimal(0), Decimal(1))):
    if pos >= len(text):
        low, high = space
        result = (low, high)
        return result

    low, high = space
    range_size = high - low  # now a Decimal between 0 and 1

    dist = _distribution(text[:pos])
    cum = _cumsum(text[pos], dist)
    p = dist[text[pos]]

    # Update the new space without the integer conversion.
    new_low = low + cum * range_size
    new_high = low + (cum + p) * range_size

    if new_high <= new_low:
        print("WARNING: Insufficient precision in the range! Consider increasing getcontext().prec.")

    print(f"Position {pos}: '{chr(text[pos])}' -> Range: ({new_low:.10f}, {new_high:.10f})")

    return _encode(text, pos + 1, (new_low, new_high))

def _decode(encoded_value, length, text=[], int_range=(Decimal(0), Decimal(1))):
    if length <= len(text):
        return text

    low, high = int_range
    range_size = high - low

    dist = _distribution(text)

    # With full Decimal precision, we can work directly in (0,1)
    value_scaled = Decimal(encoded_value) - low
    cum = Decimal(0)
    c = None
    p = Decimal(0)

    sorted_dist = _sorted_list_dist(dist)
    for symbol, prob in sorted_dist:
        next_cum = cum + prob
        # Check whether the scaled value falls in the current cumulative interval.
        if value_scaled >= cum and value_scaled < next_cum:
            c = symbol
            p = prob
            break
        cum = next_cum

    if p == 0 or c is None:
        print("Error: Could not find matching symbol!")
        return text

    symbol_low = low + cum * range_size
    symbol_high = low + (cum + p) * range_size

    new_range_size = symbol_high - symbol_low
    new_value = low + ((Decimal(encoded_value) - symbol_low) / new_range_size) * range_size
    # Clamp new_value in [low, high]
    new_value = max(low, min(high, new_value))

    print(f"Position {len(text) + 1}: '{chr(c)}' -> Range: ({symbol_low:.10f}, {symbol_high:.10f})")

    return _decode(new_value, length, text=text + [c], int_range=(low, high))

buffer_size = 1  # remains unchanged
# 'k' is no longer needed since we use (0, 1) directly.

def encode(text):

    # Adjust precision based on the length of the text. 
    adjust_precision(len(text))

    text = text + " " * buffer_size
    text = [ord(c) for c in text]
    # Use the (0,1) range directly.
    encoded = _encode(text, space=(Decimal(0), Decimal(1)))
    return encoded

def decode(encoded_value, length):

    # Adjust precision based on the length of the text.
    adjust_precision(length)

    padded_length = length + buffer_size
    text = _decode(encoded_value, padded_length, int_range=(Decimal(0), Decimal(1)))
    text = "".join(chr(c) for c in text)
    result = text[:-buffer_size]
    return result

if __name__ == "__main__":
    text = "My name is John Doe"
    encoded_value = encode(text)
    print(f"Encoded value: {encoded_value}")
    decoded_text = decode(encoded_value, len(text))
    print(f"Decoded text: {decoded_text}")
    if decoded_text == text:
        print("Test Passed!! ✨")
    else:
        print(f"Test Failed! Expected '{text}' but got '{decoded_text}'")