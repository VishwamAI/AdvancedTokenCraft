from transformers import GPT2Tokenizer, BertTokenizer, AutoModel, AutoTokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CustomTokenizer:
    def __init__(self):
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
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
        # New strategy to combine GPT-2 and BERT tokens based on semantic similarity and context
        combined_tokens = []
        gpt2_index, bert_index = 0, 0
        while gpt2_index < len(gpt2_tokens) and bert_index < len(bert_tokens):
            gpt2_embedding = self._get_embedding(gpt2_tokens[gpt2_index])
            bert_embedding = self._get_embedding(bert_tokens[bert_index])
            similarity = cosine_similarity([gpt2_embedding], [bert_embedding])[0][0]
            if similarity > 0.5:  # Threshold for combining tokens
                combined_tokens.append(gpt2_tokens[gpt2_index])
                combined_tokens.append(bert_tokens[bert_index])
            gpt2_index += 1
            bert_index += 1
        combined_tokens.extend(gpt2_tokens[gpt2_index:])
        combined_tokens.extend(bert_tokens[bert_index:])
        return combined_tokens

    def _get_embedding(self, token):
        inputs = self.embedding_tokenizer(token, return_tensors='pt')
        outputs = self.embedding_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

    def encode(self, text):
        gpt2_encoded = self.gpt2_tokenizer.encode(text, add_special_tokens=True)
        bert_encoded = self.bert_tokenizer.encode(text, add_special_tokens=True)
        combined_encoded = self._combine_encoded(gpt2_encoded, bert_encoded)
        return combined_encoded

    def _combine_encoded(self, gpt2_encoded, bert_encoded):
        # Improved logic to combine GPT-2 and BERT encoded tokens
        combined_encoded = []
        gpt2_index, bert_index = 0, 0
        while gpt2_index < len(gpt2_encoded) and bert_index < len(bert_encoded):
            gpt2_embedding = self._get_embedding(self.gpt2_tokenizer.decode([gpt2_encoded[gpt2_index]]))
            bert_embedding = self._get_embedding(self.bert_tokenizer.decode([bert_encoded[bert_index]]))
            similarity = cosine_similarity([gpt2_embedding], [bert_embedding])[0][0]
            if similarity > 0.5:  # Threshold for combining tokens
                combined_encoded.append(gpt2_encoded[gpt2_index])
                combined_encoded.append(bert_encoded[bert_index])
            gpt2_index += 1
            bert_index += 1
        combined_encoded.extend(gpt2_encoded[gpt2_index:])
        combined_encoded.extend(bert_encoded[bert_index:])
        return combined_encoded

    def decode(self, token_ids):
        # Improved decoding logic for combined tokens
        gpt2_token_ids = token_ids[::2]
        bert_token_ids = token_ids[1::2]
        gpt2_decoded = self.gpt2_tokenizer.decode(gpt2_token_ids)
        bert_decoded = self.bert_tokenizer.decode(bert_token_ids)
        return gpt2_decoded + bert_decoded

    def create_attention_mask(self, token_ids):
        # Improved attention mask for combined tokens
        gpt2_token_ids = token_ids[::2]
        bert_token_ids = token_ids[1::2]
        gpt2_attention_mask = [1 if token != self.gpt2_tokenizer.pad_token_id else 0 for token in gpt2_token_ids]
        bert_attention_mask = [1 if token != self.bert_tokenizer.pad_token_id else 0 for token in bert_token_ids]
        return gpt2_attention_mask + bert_attention_mask

    def create_token_type_ids(self, token_ids):
        # Improved token type IDs for combined tokens
        gpt2_token_ids = token_ids[::2]
        bert_token_ids = token_ids[1::2]
        gpt2_token_type_ids = [0] * len(gpt2_token_ids)
        bert_token_type_ids = [1] * len(bert_token_ids)
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