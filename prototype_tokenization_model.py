from transformers import GPT2Tokenizer, BertTokenizer

class HybridTokenizer:
    def __init__(self, gpt2_model_name='gpt2', bert_model_name='bert-base-uncased'):
        # Initialize GPT-2 and BERT tokenizers
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # Combine vocabularies and limit to 40,000 tokens
        self.vocab = self.combine_vocabularies(self.gpt2_tokenizer.get_vocab(), self.bert_tokenizer.get_vocab(), 40000)
        self.vocab_size = len(self.vocab)

        # Define special tokens from both GPT-2 and BERT
        self.special_tokens = {
            'cls_token': self.bert_tokenizer.cls_token_id,
            'sep_token': self.bert_tokenizer.sep_token_id,
            'pad_token': self.bert_tokenizer.pad_token_id,
            'unk_token': self.bert_tokenizer.unk_token_id,
            'mask_token': self.bert_tokenizer.mask_token_id,
            'gpt2_cls_token': self.gpt2_tokenizer.cls_token_id,
            'gpt2_sep_token': self.gpt2_tokenizer.sep_token_id,
            'gpt2_pad_token': self.gpt2_tokenizer.pad_token_id,
            'gpt2_unk_token': self.gpt2_tokenizer.unk_token_id,
            'gpt2_mask_token': self.gpt2_tokenizer.mask_token_id
        }

    def combine_vocabularies(self, gpt2_vocab, bert_vocab, target_size):
        # Combine vocabularies and limit to target size based on frequency
        combined_vocab = {**gpt2_vocab, **bert_vocab}
        if len(combined_vocab) > target_size:
            # Sort tokens by frequency and select the top tokens
            sorted_vocab = sorted(combined_vocab.items(), key=lambda item: item[1], reverse=True)
            combined_vocab = dict(sorted_vocab[:target_size])
        return combined_vocab

    def tokenize(self, text):
        # Tokenize using the combined vocabulary with subword tokenization
        tokens = []
        for word in text.split():
            gpt2_subwords = self.gpt2_tokenizer.tokenize(word)
            bert_subwords = self.bert_tokenizer.tokenize(word)
            subwords = gpt2_subwords if len(gpt2_subwords) > len(bert_subwords) else bert_subwords
            for subword in subwords:
                if subword in self.vocab:
                    tokens.append(subword)
                else:
                    tokens.append(self.special_tokens['unk_token'])
        return tokens

    def encode(self, text):
        # Encode text using the combined vocabulary
        tokens = self.tokenize(text)
        token_ids = [self.vocab.get(token, self.special_tokens['unk_token']) for token in tokens]
        token_ids = self.add_special_tokens(token_ids)
        return token_ids

    def decode(self, token_ids):
        # Decode tokens using the combined vocabulary
        tokens = []
        for token_id in token_ids:
            if token_id in self.gpt2_tokenizer.get_vocab().values():
                tokens.append(self.gpt2_tokenizer.decode([token_id], skip_special_tokens=True))
            elif token_id in self.bert_tokenizer.get_vocab().values():
                tokens.append(self.bert_tokenizer.decode([token_id], skip_special_tokens=True))
            else:
                tokens.append(self.special_tokens['unk_token'])
        decoded_text = ' '.join(tokens)
        return decoded_text

    def add_special_tokens(self, token_ids):
        # Add special tokens to the token list
        token_ids = [self.special_tokens['cls_token']] + token_ids + [self.special_tokens['sep_token']]
        return token_ids

    def create_attention_mask(self, tokens):
        # Create attention mask for the tokens
        attention_mask = [1 if token != self.special_tokens['pad_token'] else 0 for token in tokens]
        return attention_mask

    def create_token_type_ids(self, tokens):
        # Create token type IDs for the tokens
        token_type_ids = []
        current_type = 0
        for token in tokens:
            token_type_ids.append(current_type)
            if token == self.special_tokens['sep_token']:
                current_type = 1 - current_type
        return token_type_ids

# Example usage
if __name__ == "__main__":
    tokenizer = HybridTokenizer()
    sample_text = "Hello, this is a test sentence."

    tokens = tokenizer.tokenize(sample_text)
    tokens_with_special = tokenizer.add_special_tokens(tokens)
    print("Tokens with special tokens:", tokens_with_special)

    encoded = tokenizer.encode(sample_text)
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

    attention_mask = tokenizer.create_attention_mask(tokens_with_special)
    print("Attention mask:", attention_mask)

    token_type_ids = tokenizer.create_token_type_ids(tokens_with_special)
    print("Token type IDs:", token_type_ids)
