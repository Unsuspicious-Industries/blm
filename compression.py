from probs.gpt2 import next_distribution

def _distribution(context):
    if context == "":
        # return uniform probability for all bytes
        probs = [1/256] * 256
        # convert to dictionary
        probs = dict(zip([chr(i) for i in range(256)],probs))
        return probs

    probs = next_distribution(context)
    # prob is a dictionary of byte:probability
    # convert to dictionary of 'char':probability
    probs = {chr(k):v for k,v in probs.items()}
    return probs

def _cumsum(id,dist):
    P = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    cum = 0
    i = 0
    while P[i][0] != id:
        cum += P[i][1]
        i+=1
    return cum

def _encode(text, pos=0, space=(0, 2**128)):
    """
    Arithmetic encoding using integer arithmetic to avoid floating-point precision issues.
    """
    if pos >= len(text):
        # Return a specific value from the final range (e.g., midpoint)
        low, high = space
        return low + (high - low) // 2  # Return midpoint as a single value

    low, high = space
    range_size = high - low + 1  # +1 because we're using inclusive ranges

    dist = _distribution(text[:pos])
    cum = _cumsum(text[pos], dist)
    p = dist[text[pos]]

    # Convert floating-point probabilities to integer ranges
    new_low = low + int(cum * range_size)
    new_high = low + int((cum + p) * range_size) - 1  # -1 to keep ranges inclusive

    print(f'symbol: {text[pos]}, range: [{new_low}, {new_high}]')

    # Detect potential underflow (when range gets too small)
    if new_high <= new_low:
        print("WARNING: Integer precision limit reached! Increase bit width.")

    return _encode(text, pos+1, (new_low, new_high))

def _decode(encoded_value, length, text="", int_range=(0, 2**128)):
    """
    Arithmetic decoding using integer arithmetic to avoid floating-point precision issues.
    """
    if length <= len(text):
        print("returning size", length)
        return text

    low, high = int_range
    range_size = high - low + 1

    dist = _distribution(text)

    # Use integer arithmetic for scale calculations
    # Instead of floating-point division, use scaled integers
    PRECISION_SCALE = 10**100  # Large scaling factor to avoid precision loss

    value_scaled = (encoded_value - low) * PRECISION_SCALE
    range_scaled = range_size * PRECISION_SCALE

    # Track cumulative probability
    cum = 0
    c = None
    p = 0

    sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    for symbol, prob in sorted_dist:
        next_cum = cum + prob

        # Convert probability ranges to scaled integer bounds
        lower_bound = int(cum * range_scaled) // PRECISION_SCALE
        upper_bound = int(next_cum * range_scaled) // PRECISION_SCALE

        if value_scaled >= lower_bound * PRECISION_SCALE and value_scaled < upper_bound * PRECISION_SCALE:
            scale_value = value_scaled / range_scaled  # Only for display
            print(f"{scale_value:.6f} in [{cum:.6f}, {next_cum:.6f})")
            c = symbol
            p = prob
            break
        cum = next_cum

    if p == 0 or c is None:
        print("Error: Could not find matching symbol!")
        return text

    # Calculate bounds for this symbol in integer space
    symbol_low = low + int(cum * range_size)
    symbol_high = low + int((cum + p) * range_size) - 1

    # Calculate the new value scaled back to the full range using INTEGER division
    new_range_size = symbol_high - symbol_low + 1
    # Use integer division to prevent floating-point errors
    new_value = low + ((encoded_value - symbol_low) * range_size) // new_range_size

    # Make sure the new value stays within our integer range
    new_value = max(low, min(high, new_value))

    print(f"Symbol: {c}, New value: {new_value}")

    return _decode(new_value, length, text=text+c, int_range=int_range)

buffer_size = 4

# Interface functions
def encode(text):
    # Add buffer padding to the end
    text = text + " "*buffer_size
    # Return a single value (not a range)
    return _encode(text, space=(0, 2**(len(text)*8)))

def decode(encoded_value, length):
    # Add buffer to expected length
    padded_length = length + buffer_size
    # Decode with buffer
    text = _decode(encoded_value, padded_length, int_range=(0, 2**(padded_length*8)))
    # Remove buffer padding
    return text[:-buffer_size]


if __name__ == "__main__":
    text = "My name is"

    # Test through the interface functions
    encoded_value = encode(text)
    print(f"Encoded value: {encoded_value}")

    decoded_text = decode(encoded_value, len(text))
    print(f"Decoded text: {decoded_text}")

    if decoded_text == text:
        print("Test Passed!! ✨")
    else:
        print(f"Test Failed! Expected '{text}' but got '{decoded_text}'")