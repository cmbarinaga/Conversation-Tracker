import io
import base64
import json
import re
import unittest
import zipfile

import numpy as np

from conversation_tracker import (
    Sentence, build_figure, hover_text, order_sentences, parse_speaker,
    project_embeddings, read_meeting, reconstruct_text, build_document,
)


def transcript(items, ranges):
    """Tiny NITE fixtures: text, time, punctuation flag, or a nonverbal event."""
    words = ['<nite:root xmlns:nite="http://nite.sourceforge.net/">']
    for i, (text, time, punc) in enumerate(items):
        if text is None:
            words.append(f'<vocalsound nite:id="w{i}" type="laugh"/>')
        else:
            flag = ' punc="true"' if punc else ''
            words.append(f'<w nite:id="w{i}" starttime="{time}" endtime="{time}"{flag}>{text}</w>')
    words.append('</nite:root>')
    segments = ['<nite:root xmlns:nite="http://nite.sourceforge.net/">']
    for start, end in ranges:
        reference = f'id(w{start})' + (f'..id(w{end})' if end != start else '')
        segments.append(f'<segment transcriber_start="0" transcriber_end="100"><nite:child href="test.xml#{reference}"/></segment>')
    segments.append('</nite:root>')
    return ''.join(words).encode(), ''.join(segments).encode()


class TranscriptTests(unittest.TestCase):
    def test_punctuation_and_original_spelling(self):
        self.assertEqual(reconstruct_text(['Hi', ',', "I'm", 'David', '.']), "Hi, I'm David.")
        self.assertEqual(reconstruct_text(['(', 'Yes', ')', '?']), '(Yes)?')

    def test_terminal_punctuation_and_fragment(self):
        words, segments = transcript([
            ('Hi', 1, False), (',', 1, True), ("I'm", 2, False), ('David', 3, False),
            ('.', 3, True), ('Really', 4, False), ('?', 4, True), ('Um', 5, False),
        ], [(0, 7)])
        actual = parse_speaker(words, segments, 'M', 'A')
        self.assertEqual([s.text for s in actual], ["Hi, I'm David.", 'Really?', 'Um'])
        self.assertEqual([(s.start, s.end) for s in actual], [(1, 3), (4, 4), (5, 5)])

    def test_fragments_do_not_cross_segment_boundaries(self):
        words, segments = transcript([('Maybe', 1, False), ('later', 9, False)], [(0, 0), (1, 1)])
        self.assertEqual([s.text for s in parse_speaker(words, segments, 'M', 'A')], ['Maybe', 'later'])

    def test_nonverbal_and_punctuation_only_segments_are_omitted(self):
        words, segments = transcript([
            ('Yes', 1, False), (None, 2, False), ('!', 3, True),
            (None, 4, False), ('.', 5, True),
        ], [(0, 2), (3, 4)])
        self.assertEqual([s.text for s in parse_speaker(words, segments, 'M', 'A')], ['Yes!'])

    def test_overlapping_sentences_order_by_onset_and_stable_ties(self):
        sentences = [
            Sentence('M', 'B', 3, 2, 3, 'B'), Sentence('M', 'A', 9, 2, 4, 'A9'),
            Sentence('M', 'A', 1, 1, 10, 'Long overlapping sentence'),
            Sentence('M', 'A', 4, 2, 5, 'A4'),
        ]
        ordered = order_sentences(sentences)
        self.assertEqual([s.text for s in ordered], ['Long overlapping sentence', 'A4', 'A9', 'B'])
        self.assertEqual([s.index for s in ordered], [1, 2, 3, 4])

    def test_meeting_merges_speaker_files(self):
        contents = io.BytesIO()
        with zipfile.ZipFile(contents, 'w') as archive:
            for speaker, time in [('A', 2), ('B', 1)]:
                words, segments = transcript([(speaker, time, False)], [(0, 0)])
                archive.writestr(f'words/M.{speaker}.words.xml', words)
                archive.writestr(f'segments/M.{speaker}.segments.xml', segments)
        with zipfile.ZipFile(contents) as archive:
            self.assertEqual([s.text for s in read_meeting(archive, 'M')], ['B', 'A'])


class ProjectionAndFigureTests(unittest.TestCase):
    def test_offline_document_embeds_aligned_data_and_escapes_transcript_html(self):
        text = '</script><script>alert("test")</script> @@VIEWER@@'
        sentences = [Sentence('M', 'A', i, i, i + 1, text, i + 1) for i in range(2)]
        embeddings = np.zeros((2, 384), dtype=np.float32)
        embeddings[:, 0] = 1
        coordinates = np.array([[0., 1.], [1., 2.]])
        document = build_document([('M', sentences, coordinates)], [embeddings])
        self.assertNotIn('</script><script>alert', document)
        encoded = re.search(r'<script id="meeting-data" type="application/json">(.*?)</script>', document).group(1)
        payload = json.loads(encoded)
        self.assertFalse(payload['pca']['layout'].get('updatemenus'))
        self.assertEqual(payload['meetings'][0]['sentences'], [text, text])
        vectors = np.frombuffer(base64.b64decode(payload['meetings'][0]['embeddings']), dtype='<f4').reshape(2, 384)
        np.testing.assert_array_equal(vectors, embeddings)
        self.assertIn('requestAnimationFrame(animate)', document)
        self.assertNotRegex(document, r'<script[^>]+src=')

    def test_pca_is_finite_and_repeatable(self):
        embeddings = np.random.default_rng(17).normal(size=(8, 384))
        coordinates = project_embeddings(embeddings)
        self.assertEqual(coordinates.shape, (8, 2))
        self.assertTrue(np.isfinite(coordinates).all())
        np.testing.assert_allclose(coordinates, project_embeddings(embeddings))

    def test_invalid_embeddings_are_rejected(self):
        for embeddings in [np.zeros((1, 384)), np.zeros((3, 10)), np.full((3, 384), np.nan)]:
            with self.subTest(shape=embeddings.shape), self.assertRaises(ValueError):
                project_embeddings(embeddings)

    def test_figure_order_hover_endpoints_and_meeting_selector(self):
        sentences = [Sentence('M', 'A', i, i, i + 1, f'Sentence {i}', i + 1) for i in range(3)]
        coordinates = np.array([[9., 1.], [-2., 4.], [3., 2.]])
        figure = build_figure([('M', sentences, coordinates), ('N', sentences, coordinates)])
        np.testing.assert_array_equal(figure.data[0].x, [9, -2, 3])
        self.assertEqual([row[0] for row in figure.data[0].customdata], [1, 2, 3])
        self.assertEqual(figure.data[0].mode, 'lines+markers')
        self.assertEqual(figure.data[1].x[0], 9)
        self.assertEqual(figure.data[2].x[0], 3)
        self.assertEqual([trace.visible for trace in figure.data], [True, True, True, False, False, False])
        self.assertEqual(list(figure.layout.updatemenus[0].buttons[1].args[0]['visible']), [False, False, False, True, True, True])
        self.assertIn('%{customdata[1]}', figure.data[0].hovertemplate)

    def test_hover_preserves_full_text_and_escapes_html(self):
        text = 'A <test> & sentence ' * 20
        wrapped = hover_text(Sentence('M', 'A', 0, 0, 1, text))
        self.assertIn('&lt;test&gt;', wrapped)
        self.assertNotIn('<test>', wrapped)
        self.assertEqual(wrapped.count('&lt;test&gt;'), 20)
        self.assertIn('<br>', wrapped)


if __name__ == '__main__':
    unittest.main()
