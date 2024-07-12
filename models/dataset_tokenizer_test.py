import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from custom_tokenizer_revision_v3 import CustomTokenizer

def load_dataset(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)

def test_tokenizer_on_dataset(tokenizer, dataset_path, max_len=10):
    for entry in load_dataset(dataset_path):
        text = entry.get('text', '')
        tokens = tokenizer._split_whitespaces_or_nonwhitespaces(text, max_len)
        print(f"Original: {text}")
        print(f"Tokens: {tokens}")
        print()

if __name__ == "__main__":
    tokenizer = CustomTokenizer()
    dataset_path = "instruction.jsonl"
    test_tokenizer_on_dataset(tokenizer, dataset_path)
