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
        return gpt2_tokens, bert_tokens

    def encode(self, text):
        gpt2_encoded = self.gpt2_tokenizer.encode(text, add_special_tokens=True)
        bert_encoded = self.bert_tokenizer.encode(text, add_special_tokens=True)
        return gpt2_encoded, bert_encoded

    def decode(self, token_ids, model='gpt2'):
        if model == 'gpt2':
            return self.gpt2_tokenizer.decode(token_ids)
        elif model == 'bert':
            return self.bert_tokenizer.decode(token_ids)
        else:
            raise ValueError("Model must be 'gpt2' or 'bert'")

    def create_attention_mask(self, token_ids, model='gpt2'):
        if model == 'gpt2':
            return [1 if token != self.gpt2_tokenizer.pad_token_id else 0 for token in token_ids]
        elif model == 'bert':
            return [1 if token != self.bert_tokenizer.pad_token_id else 0 for token in token_ids]
        else:
            raise ValueError("Model must be 'gpt2' or 'bert'")

    def create_token_type_ids(self, token_ids, model='gpt2'):
        if model == 'gpt2':
            return [0] * len(token_ids)
        elif model == 'bert':
            return [0] * len(token_ids)
        else:
            raise ValueError("Model must be 'gpt2' or 'bert'")

# Example usage
if __name__ == "__main__":
    tokenizer = CustomTokenizer()
    text = "Hello, this is a test sentence."
    gpt2_tokens, bert_tokens = tokenizer.tokenize(text)
    print("GPT-2 Tokens:", gpt2_tokens)
    print("BERT Tokens:", bert_tokens)

    gpt2_encoded, bert_encoded = tokenizer.encode(text)
    print("GPT-2 Encoded:", gpt2_encoded)
    print("BERT Encoded:", bert_encoded)

    gpt2_decoded = tokenizer.decode(gpt2_encoded, model='gpt2')
    bert_decoded = tokenizer.decode(bert_encoded, model='bert')
    print("GPT-2 Decoded:", gpt2_decoded)
    print("BERT Decoded:", bert_decoded)

    gpt2_attention_mask = tokenizer.create_attention_mask(gpt2_encoded, model='gpt2')
    bert_attention_mask = tokenizer.create_attention_mask(bert_encoded, model='bert')
    print("GPT-2 Attention Mask:", gpt2_attention_mask)
    print("BERT Attention Mask:", bert_attention_mask)

    gpt2_token_type_ids = tokenizer.create_token_type_ids(gpt2_encoded, model='gpt2')
    bert_token_type_ids = tokenizer.create_token_type_ids(bert_encoded, model='bert')
    print("GPT-2 Token Type IDs:", gpt2_token_type_ids)
    print("BERT Token Type IDs:", bert_token_type_ids)
