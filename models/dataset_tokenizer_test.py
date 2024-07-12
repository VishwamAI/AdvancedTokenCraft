import json
import sys
import os
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from custom_tokenizer_revision_v3 import CustomTokenizer

def load_dataset(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)

def test_tokenizer_on_dataset(tokenizer, dataset_path, max_len=10, subset_size=10):
    for i, entry in enumerate(load_dataset(dataset_path)):
        if i >= subset_size:
            break
        text = entry.get('synthesized text', '')  # Use 'synthesized text' field for tokenization
        print(f"Original: {text}")
        tokens = tokenizer._split_whitespaces_or_nonwhitespaces(text, max_len)
        print(f"Tokens: {tokens}")
        print()

        # Debugging: Print intermediate steps
        print("Debugging Information:")
        for match in re.finditer(tokenizer.pat_str, text):
            token = match.group()
            print(f"Matched Token: {token}")
        print()

if __name__ == "__main__":
    tokenizer = CustomTokenizer()
    dataset_path = "instruction.jsonl"
    test_tokenizer_on_dataset(tokenizer, dataset_path, subset_size=10)
