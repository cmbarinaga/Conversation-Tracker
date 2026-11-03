import re
import unittest

from semantic_context import checked_encode, context_windows


class WordTokenizer:
    """Small deterministic tokenizer for window mechanics; real-token audits run on AMI."""
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, add_special_tokens=True, truncation=False, return_offsets_mapping=False):
        assert truncation is False
        if isinstance(text, list):
            return {"input_ids": [self(t, add_special_tokens=add_special_tokens)["input_ids"] for t in text]}
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches) + (2 if add_special_tokens else 0)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [m.span() for m in matches]
        return result


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer()

    def test_past_keeps_chronological_order_and_substance_through_short_replies(self):
        texts = ['the remote costs twenty five euros', 'yes', 'okay', 'is it worth that']
        windows = context_windows(texts, self.tokenizer, 10, 'past')
        self.assertEqual(windows[0], dict(text='', tokens=0, first=None, last=None))
        self.assertEqual(windows[3]['text'], 'the remote costs twenty five euros yes okay')
        self.assertEqual(windows[3]['tokens'], 10)
        self.assertEqual((windows[3]['first'], windows[3]['last']), (0, 2))
        self.assertNotIn('worth', windows[3]['text'])

    def test_far_boundary_is_trimmed_without_discarding_nearest_utterances(self):
        texts = ['one two three four five six seven', 'yes okay', 'current']
        window = context_windows(texts, self.tokenizer, 7, 'past')[2]
        self.assertEqual(window['text'], 'five six seven yes okay')
        self.assertEqual(window['tokens'], 7)
        future = context_windows(list(reversed(texts)), self.tokenizer, 7, 'future')[0]
        self.assertEqual(future['text'], 'yes okay one two three')

    def test_future_excludes_current_and_has_available_context_fallback(self):
        texts = ['current', 'new topic', 'continues here', 'end']
        windows = context_windows(texts, self.tokenizer, 20, 'future')
        self.assertEqual(windows[0]['text'], 'new topic continues here end')
        self.assertEqual(windows[0]['last'], 3)
        self.assertEqual(windows[2]['text'], 'end')
        self.assertEqual(windows[-1], dict(text='', tokens=0, first=None, last=None))

    def test_finalized_context_prefix_is_unchanged_when_more_speech_arrives(self):
        texts = ['a b', 'c d', 'e f', 'g h', 'i j']
        prefix = context_windows(texts[:4], self.tokenizer, 6, 'future')
        full = context_windows(texts, self.tokenizer, 6, 'future')
        self.assertEqual(prefix[0], full[0])
        self.assertEqual(prefix[1], full[1])

    def test_budget_includes_special_tokens_and_rejects_empty_content_capacity(self):
        with self.assertRaises(ValueError):
            context_windows(['a'], self.tokenizer, 2, 'past')

    def test_overlength_inputs_fail_before_model_encode(self):
        class Model:
            tokenizer = WordTokenizer()
            max_seq_length = 5
            def encode(self, *args, **kwargs):
                raise AssertionError('encode must not be reached')
        with self.assertRaisesRegex(ValueError, 'nothing was silently truncated'):
            checked_encode(Model(), ['one two three four'])


if __name__ == '__main__':
    unittest.main()
