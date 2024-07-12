import re
from typing import Dict, List, Union, Literal, AbstractSet, Collection

class CustomTokenizer:
    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 256
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|\w+|\d+|[^\s\w\d]|<\|space\|>|\s+"

    def __init__(self, vocab_file: str):
        # Load vocabulary from file
        self.special_tokens = self._load_vocab(vocab_file)

    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        # Load vocabulary from the given file
        # ...
        pass

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        max_len: int = 10,  # Add max_len parameter with a default value
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        # ...
        pass

    def _split_whitespaces_or_nonwhitespaces(self, s: str, max_len: int) -> List[str]:
        if not isinstance(max_len, int) or max_len <= 0:
            raise ValueError("max_len must be a positive integer.")

        tokens = []
        current_token = ""
        space_encountered = False

        for match in re.finditer(self.pat_str, s):
            token = match.group()
            if token.isspace():
                if not space_encountered:
                    if current_token:
                        # If the current token plus a space exceeds max_len, append the current token
                        if len(current_token) + 1 > max_len:
                            tokens.append(current_token)
                            current_token = ""
                        # Append the current token and reset it
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
