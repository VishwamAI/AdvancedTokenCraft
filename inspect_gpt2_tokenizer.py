from transformers import GPT2Tokenizer

# Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Print the tokenizer's attributes and methods
print(dir(tokenizer))

# Print the tokenizer's vocabulary size
print("Vocabulary size:", tokenizer.vocab_size)

# Print a sample tokenization
sample_text = "Hello, this is a test sentence."
encoded_input = tokenizer(sample_text)
print("Encoded input:", encoded_input)
