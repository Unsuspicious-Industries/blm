import matplotlib.pyplot as plt
from math import log2

def plot_probs(probs, top_k=10):
    # Sort tokens by probability descending
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    sorted_items = sorted_items[:top_k]
    tokens, probabilities = zip(*sorted_items)

    plt.figure(figsize=(12, 8))
    plt.bar(range(len(probabilities)), probabilities, align='center')
    plt.xticks(range(len(probabilities)), tokens, rotation='vertical')
    plt.xlabel("Token (decoded)")
    plt.ylabel("Probability")
    plt.title("Next Token Probability Distribution")
    plt.tight_layout()
    plt.show()

def autoregressive_byte_analysis(base_text, distribution_func, target_word=" dog",):
    """
    Analyze byte-level probabilities and entropy for each position when generating a word
    Args:
        base_text: The context text
        target_word: The word to analyze character by character
    """
    results = []
    current_text = base_text

    print(f"Starting with: '{base_text}'")
    print(f"Analyzing generation of: '{target_word}'")
    print("-" * 50)

    # For each character in the target word
    for i, char in enumerate(target_word):
        print(f"\nPosition {i+1}: Generating '{char}'")

        # Get byte value of the character
        char_byte = ord(char)

        # Get the distribution for the next byte
        probs = distribution_func(current_text)

        # Calculate entropy of this position
        entropy = -sum([p * log2(p) for t, p in probs.items()])

        # Get actual probability of the correct byte
        char_prob = probs.get(char_byte, 0)
        char_suprise = -log2(char_prob) if char_prob > 0 else float('inf')

        # Top predicted bytes
        top_5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        top_5_formatted = [(chr(t) if 32 <= t <= 126 else f"\\x{t:02x}", f"{p:.6f}")
                          for t, p in top_5]

        # Print information
        print(f"Entropy at this position: {entropy:.4f} bits")
        print(f"Probability of '{char}': {char_prob:.6f}")
        print(f"Surprise value: {char_suprise:.4f} bits")
        print("Top 5 predicted bytes:")
        for byte_repr, prob in top_5_formatted:
            print(f"  {byte_repr}: {prob}")

        # Store results
        results.append({
            'position': i+1,
            'char': char,
            'entropy': entropy,
            'probability': char_prob,
            'surprise': char_suprise,
            'top_predictions': top_5
        })

        # Update for next iteration
        current_text += char

    # Plot entropy by position
    positions = [r['position'] for r in results]
    entropies = [r['entropy'] for r in results]
    surprises = [r['surprise'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(positions, entropies, 'b-', marker='o', label='Position Entropy')
    plt.plot(positions, surprises, 'r--', marker='x', label='Character Surprise')

    # Add character labels
    for i, char in enumerate([r['char'] for r in results]):
        plt.annotate(char, (positions[i], entropies[i]),
                    textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(f"Byte-level Entropy Analysis of '{target_word}'")
    plt.xlabel("Position in Word")
    plt.ylabel("Bits")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return results