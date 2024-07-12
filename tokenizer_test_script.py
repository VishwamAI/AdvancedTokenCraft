import re
from typing import List

class CustomTokenizer:
    def __init__(self):
        # Define a regex pattern that matches words, spaces, and special characters
        self.pat_str = r"(\s+|[^\s]+)"

    def _split_whitespaces_or_nonwhitespaces(self, s: str, max_len: int) -> List[str]:
        # Initialize variables
        tokens = []
        current_token = ""
        space_encountered = False

        # Iterate over matches
        for match in re.finditer(self.pat_str, s):
            token = match.group()
            if token.isspace():
                if not space_encountered:
                    if current_token:
                        # If the current token plus a space exceeds max_len, append the current token
                        if len(current_token) + 1 > max_len:
                            tokens.append(current_token)
                            current_token = ""
                        # Otherwise, add a space to the current token
                        else:
                            current_token += " "
                    # Append a space token
                    tokens.append("<|space|>")
                space_encountered = True
            else:
                space_encountered = False
                # If adding the new token exceeds max_len, append the current token and reset it
                if len(current_token) + len(token) > max_len:
                    if current_token:
                        tokens.append(current_token)
                    current_token = token
                # Otherwise, add the token to the current token
                else:
                    current_token += token

        # Append the last token if it exists
        if current_token:
            tokens.append(current_token)

        # Split very long words that exceed max_len
        tokens = [subtoken for token in tokens for subtoken in [token[i:i+max_len] for i in range(0, len(token), max_len)]]

        return tokens

# Test the refined logic
tokenizer = CustomTokenizer()
test_string = "This is a test string with multiple   spaces and a veryverylongword."
max_len = 10  # Example max_len
print(tokenizer._split_whitespaces_or_nonwhitespaces(test_string, max_len))
