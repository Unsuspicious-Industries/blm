import matplotlib.pyplot as plt
from math import log2

import numpy as np

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

def autoregressive_byte_analysis(distribution_func, text):
    """
    Analyze byte-level probabilities and entropy for each position when generating a word
    Args:
        base_text: The context text
        target_word: The word to analyze character by character
    """
    results = []
    current_text = ""

    print(f"Starting with: '{text}'")
    print(f"Analyzing generation of: '{text}'")
    print("-" * 50)

    # For each character in the target word
    for i, char in enumerate(text):
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

    plt.title(f"Byte-level Entropy Analysis of '{text}'")
    plt.xlabel("Position in Word")
    plt.ylabel("Bits")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return results


import matplotlib.pyplot as plt
import numpy as np

# analyse the results of an autoregressive byte analysis
def sentence_information_visualization(text,results):

    # Data from results (assumes results is defined)
    positions = [r['position'] for r in results]
    entropies = [r['entropy'] for r in results]
    surprises = [r['surprise'] for r in results]



    # Create 2D arrays of shape (1, N) for imshow for entropy and surprise
    entropy_array = np.array(entropies).reshape(1, -1)
    surprise_array = np.array(surprises).reshape(1, -1)

    # Create labels for each byte position (assumes text is defined)
    labels = [text[i-1] for i in positions]

    # Set up the subplots: one for entropy, one for surprise, and one for the combined derivative/surprise plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Plot entropy as a colorbar-like image
    im1 = ax1.imshow(entropy_array, aspect="auto", cmap="RdYlGn_r")
    ax1.set_title("Entropy per Byte")
    ax1.set_yticks([])  # Remove y-axis ticks
    ax1.set_xticks(np.arange(len(positions)))
    ax1.set_xticklabels(labels)

    # Plot surprise as a colorbar-like image
    im2 = ax2.imshow(surprise_array, aspect="auto", cmap="RdYlGn_r")
    ax2.set_title("Surprise per Byte")
    ax2.set_yticks([])
    ax2.set_xticks(np.arange(len(positions)))
    ax2.set_xticklabels(labels)

    # Plot the ration of entropy over surprise as a curve
    derivative = np.array(entropies) / np.array(surprises)
    ax3.plot(positions, derivative, 'b-', marker='o')
    ax3.set_title("Entropy / Surprise Ratio")
    ax3.set_xlabel("Position in Word")
    ax3.set_ylabel("Ratio")
    ax3.set_xticks(positions)
    ax3.set_xticklabels(labels)


    plt.tight_layout()
    plt.show()

