import re
from typing import List, Dict

class CustomTokenizer:
    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 256
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|\w+|\d+|[^\s\w\d]|<\|space\|>|\s+|\n|\t"

    def _split_whitespaces_or_nonwhitespaces(self, s: str, max_len: int) -> List[str]:
        if not isinstance(max_len, int) or max_len <= 0:
            raise ValueError('max_len must be a positive integer.')

        tokens = []
        current_token = ''
        space_encountered = False

        for match in re.finditer(self.pat_str, s):
            token = match.group()
            if token.isspace() or token == '\n' or token == '\t':
                if not space_encountered:
                    if current_token:
                        tokens.append(current_token)
                        current_token = ''
                    tokens.append('<|space|>')
                space_encountered = True
            else:
                space_encountered = False
                if len(token) > max_len:
                    start = 0
                    while start < len(token):
                        end = min(start + max_len, len(token))
                        tokens.append(token[start:end])
                        start = end
                else:
                    if current_token:
                        if len(current_token) + len(token) > max_len:  # Do not consider space when merging
                            tokens.append(current_token)
                            current_token = token
                        else:
                            current_token += token  # Do not add space when merging
                    else:
                        current_token = token

        if current_token:
            tokens.append(current_token)

        # Merge tokens to respect max_len
        merged_tokens = []
        current_token = ''
        for token in tokens:
            if token == '<|space|>':
                if current_token:
                    merged_tokens.append(current_token)
                    current_token = ''
                if not merged_tokens or merged_tokens[-1] != '<|space|>':
                    merged_tokens.append(token)
            else:
                if len(current_token) + len(token) > max_len:
                    if current_token:
                        merged_tokens.append(current_token)
                    current_token = token
                else:
                    if current_token:
                        current_token += token  # Do not add space when merging
                    else:
                        current_token = token

        if current_token:
            merged_tokens.append(current_token)

        # Remove leading <|space|> tokens
        while merged_tokens and merged_tokens[0] == '<|space|>':
            merged_tokens.pop(0)

        # Remove trailing <|space|> tokens
        while merged_tokens and merged_tokens[-1] == '<|space|>':
            merged_tokens.pop()

        return merged_tokens
