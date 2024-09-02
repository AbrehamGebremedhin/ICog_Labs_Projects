import numpy as np
from collections import defaultdict


def build_ngram_probabilities(ngrams):
    """
    Builds a probabilistic n-gram model from a list of n-grams.

    The function counts the occurrences of each suffix (next word) given a prefix (previous words) in the n-grams.
    It then converts these counts into probabilities by normalizing them over the total occurrences of the prefix.

    Args:
        ngrams (list of tuples): A list of n-grams, where each n-gram is a tuple of words. For example,
                                 with n=3, each n-gram could be ('I', 'love', 'coding').

    Returns:
        dict: A dictionary where the keys are prefixes (tuples of words of length n-1) and the values are dictionaries 
              of suffixes (next words) and their probabilities. The suffix probabilities are normalized to sum to 1.
              Example: {('I', 'love'): {'coding': 0.5, 'reading': 0.5}}
    """
    model = defaultdict(lambda: defaultdict(int))

    # Count occurrences of each prefix-suffix pair
    for ngram in ngrams:
        prefix = ngram[:-1]
        suffix = ngram[-1]
        model[prefix][suffix] += 1

    # Convert counts to probabilities
    for prefix in model:
        total_count = float(sum(model[prefix].values()))
        for suffix in model[prefix]:
            # Normalize to create probabilities
            model[prefix][suffix] /= total_count

    return model


def generate_text_with_probabilities(model, n, length=50):
    """
    Generates text based on a trained n-gram model with probabilities.

    The function starts with a random prefix from the model and generates the next word based on the suffix 
    probabilities. It appends the chosen word to the generated text and shifts the prefix to include this new word.
    The process repeats until the desired text length is reached.

    Args:
        model (dict): The trained n-gram model with probabilities. The keys are prefixes (tuples of words), 
                      and the values are dictionaries of suffixes and their probabilities.
        n (int): The number of words in each n-gram (e.g., 3 for trigrams, 2 for bigrams).
        length (int): The length of the generated text in words. Default is 50.

    Returns:
        str: The generated text, as a single string of words.
    """
    # Start with a random prefix from the model
    prefix = list(model.keys())[np.random.choice(len(model))]
    generated_text = list(prefix)

    # Generate text by sampling from suffix probabilities
    for _ in range(length - n + 1):
        suffixes = model.get(tuple(prefix), None)
        if not suffixes:  # If no suffixes are found for the prefix, stop generation
            break
        suffix_choices = list(suffixes.keys())
        suffix_probs = list(suffixes.values())
        # Choose suffix based on probabilities
        suffix = np.random.choice(suffix_choices, p=suffix_probs)
        generated_text.append(suffix)
        # Update the prefix with the last n-1 words
        prefix = generated_text[-(n-1):]

    return ' '.join(generated_text)


def backoff_generate_text(model, n, length=50):
    """
    Generates text with a backoff strategy, reducing the n-gram size when no suitable n-gram is found.

    The function attempts to generate text using the highest-order n-grams first (e.g., trigrams for n=3).
    If no suitable n-gram is found for the current prefix, it "backs off" to a smaller n-gram (e.g., bigram, then unigram).
    This helps in cases where higher-order n-grams are sparse in the training data.

    Args:
        model (dict): The trained n-gram model with probabilities. The keys are prefixes (tuples of words), 
                      and the values are dictionaries of suffixes and their probabilities.
        n (int): The number of words in the initial n-gram (e.g., 3 for trigrams).
        length (int): The length of the generated text in words. Default is 50.

    Returns:
        str: The generated text, as a single string of words.
    """
    generated_text = []

    # Start with a random prefix from the model
    prefix = list(model.keys())[np.random.choice(len(model))]
    generated_text.extend(prefix)

    # Generate text with backoff strategy
    for _ in range(length - n + 1):
        found_suffix = False

        # Backoff loop: Start from n-grams and back off to unigrams if necessary
        for k in range(n, 0, -1):  # k decreases from n down to 1 (backing off)
            # Get the last (k-1) words as the prefix
            current_prefix = tuple(generated_text[-(k-1):])
            suffixes = model.get(current_prefix, None)
            if suffixes:
                suffix_choices = list(suffixes.keys())
                suffix_probs = list(suffixes.values())
                suffix = np.random.choice(suffix_choices, p=suffix_probs)
                generated_text.append(suffix)
                found_suffix = True
                break

        # If no suitable suffix is found, stop generating text
        if not found_suffix:
            break

    return ' '.join(generated_text)


# Main execution with enhancements
file_path = 'sample_train.txt'
text = read_file(file_path)
tokens = tokenize_text(text)
n = 3  # You can change this to any 'n' value
ngrams = build_ngrams(tokens, n)
model = build_ngram_probabilities(ngrams)

# Generate some text with probabilities
generated_text = generate_text_with_probabilities(model, n)
print(generated_text)

# Generate some text with backoff strategy
generated_text_backoff = backoff_generate_text(model, n)
print(generated_text_backoff)
