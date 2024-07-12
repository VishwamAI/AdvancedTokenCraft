import unittest
from models.custom_tokenizer_revision_v3 import CustomTokenizer

class TestCustomTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CustomTokenizer()

    def test_tokenize_various_sentences(self):
        test_cases = [
            ("This is a test string.", ['This', '<|space|>', 'is', '<|space|>', 'a', '<|space|>', 'test', '<|space|>', 'string.']),
            ("Hello, world! This is a test.", ['Hello,', '<|space|>', 'world!', '<|space|>', 'This', '<|space|>', 'is', '<|space|>', 'a', '<|space|>', 'test.']),
            ("Multiple     spaces.", ['Multiple', '<|space|>', '<|space|>', '<|space|>', '<|space|>', '<|space|>', 'spaces.']),
            ("Special characters: @#&*()!", ['Special', '<|space|>', 'characters:', '<|space|>', '@', '#', '&', '*', '(', ')', '!']),
            ("Newline\ncharacters.", ['Newline', '<|space|>', 'characters.']),
            ("Tabs\tcharacters.", ['Tabs', '<|space|>', 'characters.']),
            ("Mixed: spaces, tabs\t, and\nnewlines.", ['Mixed:', '<|space|>', 'spaces,', '<|space|>', 'tabs', '<|space|>', ',', '<|space|>', 'and', '<|space|>', 'newlines.']),
            ("Different languages: こんにちは, 你好, 안녕하세요.", ['Different', '<|space|>', 'languages:', '<|space|>', 'こんにちは,', '<|space|>', '你好,', '<|space|>', '안녕하세요.']),
            ("Edge case: !@#$%^&*()_+{}|:\"<>?", ['Edge', '<|space|>', 'case:', '<|space|>', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '{', '}', '|', ':', '"', '<', '>', '?']),
        ]

        for input_str, expected_output in test_cases:
            with self.subTest(input_str=input_str):
                self.assertEqual(self.tokenizer._split_whitespaces_or_nonwhitespaces(input_str, 10), expected_output)

if __name__ == '__main__':
    unittest.main()
