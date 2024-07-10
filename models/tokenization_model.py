from transformers import GPT2Tokenizer, BertTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

class CustomTokenizer:
    def __init__(self, gpt2_model_name='gpt2', bert_model_name='bert-base-uncased', embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.special_tokens = {
            'pad_token': '[PAD]',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'mask_token': '[MASK]'
        }

        self.gpt2_tokenizer.add_special_tokens(self.special_tokens)
        self.bert_tokenizer.add_special_tokens(self.special_tokens)

    def tokenize(self, text):
        gpt2_tokens = self.gpt2_tokenizer.tokenize(text)
        bert_tokens = self.bert_tokenizer.tokenize(text)
        return self._combine_tokens(gpt2_tokens, bert_tokens)

    def encode(self, text):
        gpt2_encoded = self.gpt2_tokenizer.encode(text, add_special_tokens=True)
        bert_encoded = self.bert_tokenizer.encode(text, add_special_tokens=True)
        return self._combine_encoded(gpt2_encoded, bert_encoded)

    def decode(self, token_ids):
        gpt2_decoded = self.gpt2_tokenizer.decode(token_ids, skip_special_tokens=True)
        bert_decoded = self.bert_tokenizer.decode(token_ids, skip_special_tokens=True)
        # Use a more sophisticated selection approach
        gpt2_score = self._evaluate_text(gpt2_decoded)
        bert_score = self._evaluate_text(bert_decoded)
        return gpt2_decoded if gpt2_score > bert_score else bert_decoded

    def create_attention_mask(self, token_ids):
        return [1 if token != self.gpt2_tokenizer.pad_token_id else 0 for token in token_ids]

    def create_token_type_ids(self, token_ids):
        # Handle multi-sentence inputs
        token_type_ids = []
        current_type = 0
        for token in token_ids:
            token_type_ids.append(current_type)
            if token == self.gpt2_tokenizer.sep_token_id:
                current_type = 1 - current_type
        return token_type_ids

    def _combine_tokens(self, gpt2_tokens, bert_tokens):
        combined_tokens = []
        gpt2_embeddings = self.embedding_model.encode(gpt2_tokens, convert_to_tensor=True)
        bert_embeddings = self.embedding_model.encode(bert_tokens, convert_to_tensor=True)
        similarities = np.dot(gpt2_embeddings, bert_embeddings.T) / (np.linalg.norm(gpt2_embeddings, axis=1)[:, None] * np.linalg.norm(bert_embeddings, axis=1))
        for i, (gpt2_token, bert_token) in enumerate(zip(gpt2_tokens, bert_tokens)):
            similarity = similarities[i, i]
            combined_tokens.append(gpt2_token if similarity > 0.5 else bert_token)
        return combined_tokens

    def _combine_encoded(self, gpt2_encoded, bert_encoded):
        combined_encoded = []
        gpt2_decoded = [self.gpt2_tokenizer.decode([gpt2_id]) for gpt2_id in gpt2_encoded]
        bert_decoded = [self.bert_tokenizer.decode([bert_id]) for bert_id in bert_encoded]
        gpt2_embeddings = self.embedding_model.encode(gpt2_decoded, convert_to_tensor=True)
        bert_embeddings = self.embedding_model.encode(bert_decoded, convert_to_tensor=True)
        similarities = np.dot(gpt2_embeddings, bert_embeddings.T) / (np.linalg.norm(gpt2_embeddings, axis=1)[:, None] * np.linalg.norm(bert_embeddings, axis=1))
        for i, (gpt2_id, bert_id) in enumerate(zip(gpt2_encoded, bert_encoded)):
            similarity = similarities[i, i]
            combined_encoded.append(gpt2_id if similarity > 0.5 else bert_id)
        return combined_encoded

    def _evaluate_text(self, text):
        # Placeholder method for evaluating text coherence
        # This method should be implemented to use a language model to score the coherence of the text
        return len(text)  # Example: using length as a simple heuristic
