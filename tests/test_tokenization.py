import unittest
from AdvancedTokenCraft.models.custom_tokenizer import CustomTokenizer

class TestCustomTokenizer(unittest.TestCase):

    def setUp(self):
        # Initialize the tokenizer with a placeholder vocabulary file path
        self.tokenizer = CustomTokenizer("vocab_file_path_placeholder")

    def test_tokenize_simple_sentence(self):
        input_str = "This is a test string."
        expected_output = ['This is a', 'test', 'string.']
        self.assertEqual(self.tokenizer._split_whitespaces_or_nonwhitespaces(input_str, 10), expected_output)

    def test_tokenize_sentence_with_special_characters(self):
        input_str = "Hello, world! This is a test."
        expected_output = ['Hello,', 'world!', 'This is a', 'test.']
        self.assertEqual(self.tokenizer._split_whitespaces_or_nonwhitespaces(input_str, 10), expected_output)

    def test_tokenize_sentence_with_multiple_spaces(self):
        input_str = "This  is   a    test."
        expected_output = ['This', '<|space|>', 'is', '<|space|>', 'a', '<|space|>', 'test.']
        self.assertEqual(self.tokenizer._split_whitespaces_or_nonwhitespaces(input_str, 10), expected_output)

    def test_tokenize_empty_string(self):
        input_str = ""
        expected_output = []
        self.assertEqual(self.tokenizer._split_whitespaces_or_nonwhitespaces(input_str, 10), expected_output)

if __name__ == '__main__':
    unittest.main()
