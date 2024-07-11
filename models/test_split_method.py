from models.custom_tokenizer import CustomTokenizer

# Initialize the tokenizer with a placeholder vocabulary file path
tokenizer = CustomTokenizer("vocab_file_path_placeholder")

# Define test strings
test_strings = [
    "This is a test string.",
    "Averylongwordthatexceedsthemaxlength",
    "This string contains multiple sentences. Each one should be split correctly.",
    "Short words",
    ""
]

# Define the maximum length for splitting
max_len = 10

# Test the _split_whitespaces_or_nonwhitespaces method with the test strings
for s in test_strings:
    output = tokenizer._split_whitespaces_or_nonwhitespaces(s, max_len)
    print(f"Input: {s}, Output: {output}")
