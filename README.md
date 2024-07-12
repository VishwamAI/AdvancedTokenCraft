# AdvancedTokenCraft

## Overview
AdvancedTokenCraft is a custom tokenization model that incorporates elements from existing models like GPT-2, BERT, and Meta Llama. The goal is to create a next-generation tokenization tool with high accuracy and efficiency.

## Custom Tokenizer
The `CustomTokenizer` class is designed to tokenize input text while respecting a maximum token length (`max_len`). It uses a regular expression pattern to identify tokens and handles spaces by inserting a special `<|space|>` token.

### Regular Expression Pattern
The regular expression pattern `pat_str` used for tokenization is as follows:
```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|\w+|[^\s\w]|<\|space\|>|\s+
```
This pattern matches contractions, words, non-word characters, the special `<|space|>` token, and various whitespace characters.

### Tokenization Logic
The `_split_whitespaces_or_nonwhitespaces` method tokenizes input strings while respecting the `max_len` parameter. It ensures that tokens are merged without exceeding `max_len` and handles spaces by inserting a special `<|space|>` token.

### Usage
To use the `CustomTokenizer` class, follow these steps:

1. Import the `CustomTokenizer` class:
```python
from models.custom_tokenizer_revision_v3 import CustomTokenizer
```

2. Create an instance of the tokenizer:
```python
tokenizer = CustomTokenizer()
```

3. Tokenize input text:
```python
text = "Your input text here"
tokens = tokenizer._split_whitespaces_or_nonwhitespaces(text, max_len=10)
print(tokens)
```

### Example
Here is an example of how to use the `CustomTokenizer` class:
```python
from models.custom_tokenizer_revision_v3 import CustomTokenizer

tokenizer = CustomTokenizer()
text = "What are the key considerations for scheduling and logistics when hosting a multi-show festival at a performing arts center like the Broward Center?"
tokens = tokenizer._split_whitespaces_or_nonwhitespaces(text, max_len=10)
print(tokens)
```

## Testing
The `dataset_tokenizer_test.py` script can be used to test the tokenizer with a dataset. The script tokenizes a subset of the dataset and outputs the tokens for evaluation.

### Running the Test Script
To run the test script, use the following command:
```bash
python3 models/dataset_tokenizer_test.py
```

## Development Process
The development process involved the following steps:
- Setting up the repository and developing initial tokenization logic.
- Refining tokenizer logic to pass initial unit tests and implementing tight testing loops for debugging.
- Resolving Python module import errors by adjusting import statements and the Python path in the dataset tokenizer test script.
- Testing the prototype with various datasets to ensure robustness and accuracy.
- Optimizing the model for performance and efficiency.
- Documenting the development process and model usage guidelines.

## Conclusion
AdvancedTokenCraft aims to be a state-of-the-art tokenization model that leverages the strengths of existing models while introducing its own unique features. The focus on accuracy, performance, and efficiency ensures that it meets the needs of various text processing tasks.
