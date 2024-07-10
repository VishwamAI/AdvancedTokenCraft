from transformers import GPT2Tokenizer, BertTokenizer

class CustomTokenizer:
    def __init__(self):
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.special_tokens = {
            'pad_token': '[PAD]',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'mask_token': '[MASK]'
        }
        self._add_special_tokens()

    def _add_special_tokens(self):
        self.gpt2_tokenizer.add_special_tokens(self.special_tokens)
        self.bert_tokenizer.add_special_tokens(self.special_tokens)

    def tokenize(self, text):
        gpt2_tokens = self.gpt2_tokenizer.tokenize(text)
        bert_tokens = self.bert_tokenizer.tokenize(text)
        combined_tokens = self._combine_tokens(gpt2_tokens, bert_tokens)
        return combined_tokens

    def _combine_tokens(self, gpt2_tokens, bert_tokens):
        # Custom logic to combine GPT-2 and BERT tokens
        combined_tokens = []
        max_len = max(len(gpt2_tokens), len(bert_tokens))
        for i in range(max_len):
            if i < len(gpt2_tokens):
                combined_tokens.append(gpt2_tokens[i])
            if i < len(bert_tokens):
                combined_tokens.append(bert_tokens[i])
        return combined_tokens

    def encode(self, text):
        gpt2_encoded = self.gpt2_tokenizer.encode(text, add_special_tokens=True)
        bert_encoded = self.bert_tokenizer.encode(text, add_special_tokens=True)
        combined_encoded = self._combine_encoded(gpt2_encoded, bert_encoded)
        return combined_encoded

    def _combine_encoded(self, gpt2_encoded, bert_encoded):
        # Custom logic to combine GPT-2 and BERT encoded tokens
        combined_encoded = []
        max_len = max(len(gpt2_encoded), len(bert_encoded))
        for i in range(max_len):
            if i < len(gpt2_encoded):
                combined_encoded.append(gpt2_encoded[i])
            if i < len(bert_encoded):
                combined_encoded.append(bert_encoded[i])
        return combined_encoded

    def decode(self, token_ids):
        # Custom decoding logic for combined tokens
        gpt2_token_ids = token_ids[::2]
        bert_token_ids = token_ids[1::2]
        gpt2_decoded = self.gpt2_tokenizer.decode(gpt2_token_ids)
        bert_decoded = self.bert_tokenizer.decode(bert_token_ids)
        return gpt2_decoded + bert_decoded

    def create_attention_mask(self, token_ids):
        # Custom attention mask for combined tokens
        gpt2_token_ids = token_ids[::2]
        bert_token_ids = token_ids[1::2]
        gpt2_attention_mask = [1 if token != self.gpt2_tokenizer.pad_token_id else 0 for token in gpt2_token_ids]
        bert_attention_mask = [1 if token != self.bert_tokenizer.pad_token_id else 0 for token in bert_token_ids]
        return gpt2_attention_mask + bert_attention_mask

    def create_token_type_ids(self, token_ids):
        # Custom token type IDs for combined tokens
        gpt2_token_ids = token_ids[::2]
        bert_token_ids = token_ids[1::2]
        gpt2_token_type_ids = [0] * len(gpt2_token_ids)
        bert_token_type_ids = [0] * len(bert_token_ids)
        return gpt2_token_type_ids + bert_token_type_ids

# Example usage
if __name__ == "__main__":
    tokenizer = CustomTokenizer()
    text = "Hello, this is a test sentence."
    combined_tokens = tokenizer.tokenize(text)
    print("Combined Tokens:", combined_tokens)

    combined_encoded = tokenizer.encode(text)
    print("Combined Encoded:", combined_encoded)

    decoded_text = tokenizer.decode(combined_encoded)
    print("Decoded Text:", decoded_text)

    attention_mask = tokenizer.create_attention_mask(combined_encoded)
    print("Attention Mask:", attention_mask)

    token_type_ids = tokenizer.create_token_type_ids(combined_encoded)
    print("Token Type IDs:", token_type_ids)
