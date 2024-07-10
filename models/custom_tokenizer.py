import re
from typing import List, Dict, Union, Sequence, Literal, AbstractSet, Collection

class Model:
    def __init__(self, vocab_file: str):
        self.token_to_id_map = self._load_vocab(vocab_file)
        self.id_to_token_map = {v: k for k, v in self.token_to_id_map.items()}
        self.n_vocab = len(self.token_to_id_map)

    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        token_to_id_map = {}
        try:
            with open(vocab_file, 'r') as f:
                for line in f:
                    token, token_id = line.strip().split()
                    token_to_id_map[token] = int(token_id)
        except FileNotFoundError:
            raise FileNotFoundError(f"Vocabulary file {vocab_file} not found.")
        except ValueError:
            raise ValueError(f"Invalid format in vocabulary file {vocab_file}. Each line should contain a token and its ID separated by a space.")
        return token_to_id_map

    def token_to_id(self, token: str) -> int:
        return self.token_to_id_map.get(token, -1)  # Return -1 for unknown tokens

class CustomTokenizer:
    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 256
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def __init__(self, model_path: str):
        """
        Initializes the CustomTokenizer with a model file.

        Args:
        model_path (str): The path to the model file.
        """
        self.model_path = model_path
        self.special_tokens = self._initialize_special_tokens()
        self.bos_id = self.special_tokens["<|begin_of_text|>"]
        self.eos_id = self.special_tokens["<|end_of_text|>"]
        if (model_path != "vocab_file_path_placeholder"):
            try:
                self.model = self._load_model(model_path)
                self.n_words = self.model.n_vocab
            except (FileNotFoundError, ValueError) as e:
                raise RuntimeError(f"Failed to load model: {e}")
        else:
            self.model = None
            self.n_words = 0

    def _initialize_special_tokens(self) -> Dict[str, int]:
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        return {token: i for i, token in enumerate(special_tokens)}

    def _load_model(self, model_path: str):
        """
        Loads the tokenizer model from the specified path.

        Args:
        model_path (str): The path to the model file.

        Returns:
        model: The loaded tokenizer model.
        """
        return Model(model_path)

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
        s (str): The input string to be encoded.
        bos (bool): Whether to prepend the beginning-of-sequence token.
        eos (bool): Whether to append the end-of-sequence token.
        allowed_special ("all"|set[str]): allowed special tokens in string
        disallowed_special ("all"|set[str]): special tokens that raise an error when in string

        Returns:
        list[int]: A list of token IDs.
        """
        assert type(s) is str

        allowed_special_set = set(allowed_special) if allowed_special != "all" else set(self.special_tokens.keys())
        disallowed_special_set = set(disallowed_special) if disallowed_special != "all" else set()

        # Tokenize the input string using the regular expression pattern
        tokens = re.findall(self.pat_str, s)

        # Check for disallowed special tokens
        for token in tokens:
            if token in disallowed_special_set:
                raise ValueError(f"Disallowed special token found in input: {token}")

        # Convert tokens to token IDs, filtering out unknown tokens
        if self.model:
            token_ids = [self.model.token_to_id(token) for token in tokens if self.model.token_to_id(token) != -1 or token in allowed_special_set]
        else:
            token_ids = []

        # Handle special tokens
        if bos:
            token_ids.insert(0, self.bos_id)
        if eos:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
        t (Sequence[int]): The list of token IDs to be decoded.

        Returns:
        str: The decoded string.
        """
        tokens = []
        if self.model:
            for token_id in t:
                if token_id in self.model.id_to_token_map:
                    tokens.append(self.model.id_to_token_map[token_id])
                else:
                    raise ValueError(f"Unknown token ID: {token_id}")
        return ''.join(tokens)

    def _split_whitespaces_or_nonwhitespaces(self, s: str, max_len: int) -> List[str]:
        """
        Splits a string into substrings based on whitespace or non-whitespace characters,
        ensuring that each substring does not exceed the specified maximum length and
        respects word boundaries.

        Args:
        s (str): The input string to be split.
        max_len (int): The maximum length of each substring. Must be a positive integer.

        Returns:
        list[str]: A list of substrings.
        """
        if not isinstance(max_len, int) or max_len <= 0:
            raise ValueError("max_len must be a positive integer.")

        substrings = []
        current_substring = ""
        for char in s:
            if len(current_substring) + len(char) > max_len:
                # Find the last whitespace in the current substring
                last_whitespace = current_substring.rfind(' ')
                if last_whitespace != -1:
                    # Split at the last whitespace
                    substrings.append(current_substring[:last_whitespace])
                    current_substring = current_substring[last_whitespace + 1:] + char
                else:
                    # No whitespace found, split at max_len
                    substrings.append(current_substring)
                    current_substring = char
            else:
                current_substring += char
        if current_substring:
            substrings.append(current_substring)
        return substrings

class ChatFormat:
    def __init__(self, tokenizer: CustomTokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message: Dict[str, str]) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Dict[str, str]) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialog_prompt(self, dialog: List[Dict[str, str]]) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(self.encode_message(message))
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens