"""Synthetic JSON fixtures; no corpus access."""
import json
from pathlib import Path
import tempfile
import unittest

from music_critic.experiments.supervision_stream import iter_top_object


class SupervisionStreamTests(unittest.TestCase):
    def read(self, payload, chunk_size=1, streamed_arrays=frozenset({'rows'})):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'synthetic.json'
            path.write_text(payload, encoding='utf-8')
            return list(iter_top_object(path, streamed_arrays=streamed_arrays,
                                        chunk_size=chunk_size))

    def test_arrays_are_streamed_and_other_values_preserved(self):
        self.assertEqual(self.read('{"meta":{"n":2},"rows":[1,{"x":[2,3]}],"end":true}'),
                         [('meta', {'n': 2}), ('rows', 1), ('rows', {'x': [2, 3]}), ('end', True)])

    def test_tiny_chunks_match_json_semantics(self):
        data = {'ключ': {'nested': [[], {}, {'x': 'кавычка " \\ \n 🎼'}]},
                'rows': [None, False, True, 0, -12, 345678901234567890,
                         1.25, -3.125e-17, 9e35, {'rows': [1, 2]}, [1, {'z': 2}]],
                'tail': ['ordinary', 'array']}
        payload = json.dumps(data, ensure_ascii=False)
        expected = [('ключ', data['ключ'])] + [('rows', item) for item in data['rows']] + [('tail', data['tail'])]
        for chunk in (1, 2, 3, 7, 16, 65536):
            with self.subTest(chunk=chunk):
                self.assertEqual(self.read(payload, chunk), expected)
        self.assertEqual(self.read(json.dumps(data, ensure_ascii=True)), expected)

    def test_every_number_chunk_boundary(self):
        payload = '{"rows":[123456789,-1234.5678e-12,1e+30,-0.25,0]}'
        expected = [('rows', value) for value in json.loads(payload)['rows']]
        for chunk in range(1, len(payload) + 1):
            with self.subTest(chunk=chunk):
                self.assertEqual(self.read(payload, chunk), expected)

    def test_empty_arrays_and_objects(self):
        self.assertEqual(self.read('{}'), [])
        self.assertEqual(self.read('{"rows":[],"other":[],"end":{}}'),
                         [('other', []), ('end', {})])
        self.assertEqual(self.read('{"a":[1],"b":[2]}', streamed_arrays={'a', 'b'}),
                         [('a', 1), ('b', 2)])
        self.assertEqual(self.read('{"rows":[1]}', streamed_arrays=set()), [('rows', [1])])

    def test_whitespace_does_not_change_tokens(self):
        self.assertEqual(self.read(' \t\r\n{"rows" : [ 1 , 2 ] , "x" : null }\n '),
                         [('rows', 1), ('rows', 2), ('x', None)])

    def test_rejects_duplicate_keys_at_every_level(self):
        for payload in ('{"x":1,"x":2}', '{"rows":[],"rows":[]}',
                        '{"rows":[{"x":1,"x":2}]}',
                        '{"x":{"a":[{"y":1,"y":2}]}}',
                        '{"x":1,"\\u0078":2}'):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    self.read(payload)

    def test_rejects_requested_non_array(self):
        for value in ('null', 'true', '1', '"array"', '{}'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.read('{"rows":' + value + '}')

    def test_rejects_malformed_truncated_and_trailing(self):
        invalid = ('', '[]', 'null', '{', '{"rows":', '{"rows":[',
                   '{"rows":[1', '{"rows":["unterminated', '{"rows":[1,]}',
                   '{"x":1,}', '{"rows":[1 2]}', '{"x" 1}',
                   '{"x":[1,}', '{"x":"bad\\q"}', '{"x":"line\nbreak"}',
                   '{"x":01}', '{"x":1.}', '{"x":1e}', '{"x":NaN}',
                   '{"x":Infinity}', '{"x":-Infinity}', '{"x":truefalse}',
                   '{}{}', '{} false', '{} /*comment*/')
        for payload in invalid:
            for chunk in (1, 7, 65536):
                with self.subTest(payload=payload, chunk=chunk):
                    with self.assertRaises(ValueError):
                        self.read(payload, chunk)

    def test_chunk_size_contract(self):
        for chunk in (0, -1, 1.5, True):
            with self.subTest(chunk=chunk):
                with self.assertRaises(ValueError):
                    self.read('{}', chunk)

    def test_lazy_consumption_and_bounded_memory(self):
        from io import StringIO
        import tracemalloc
        from unittest.mock import patch

        class CountingText(StringIO):
            consumed = 0
            def read(self, size=-1):
                self.assert_bounded(size)
                result = super().read(size)
                self.consumed += len(result)
                return result

            @staticmethod
            def assert_bounded(size):
                if size != 64:
                    raise AssertionError('reader must use configured chunks')

        count = 5000
        payload = '{"rows":[' + ','.join(json.dumps({'i': i, 'text': 'x' * 80}) for i in range(count)) + ']}'
        handle = CountingText(payload)
        with patch('music_critic.experiments.supervision_stream.Path.open', return_value=handle):
            iterator = iter_top_object('synthetic-only.json', streamed_arrays={'rows'}, chunk_size=64)
            self.assertEqual(handle.consumed, 0)
            self.assertEqual(next(iterator), ('rows', {'i': 0, 'text': 'x' * 80}))
            self.assertLess(handle.consumed, 256)
            tracemalloc.start()
            try:
                observed = 1
                for key, value in iterator:
                    self.assertEqual(key, 'rows')
                    self.assertEqual(value['i'], observed)
                    observed += 1
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertEqual(observed, count)
            self.assertLess(peak, len(payload) // 4)
            self.assertTrue(handle.closed)

    def test_early_close_and_delayed_document_validation(self):
        from io import StringIO
        from unittest.mock import patch
        for close_early in (True, False):
            handle = StringIO('{"rows":[1, invalid]}')
            with patch('music_critic.experiments.supervision_stream.Path.open', return_value=handle):
                iterator = iter_top_object('synthetic-only.json', streamed_arrays={'rows'}, chunk_size=1)
                self.assertEqual(next(iterator), ('rows', 1))
                if close_early:
                    iterator.close()
                else:
                    with self.assertRaises(ValueError):
                        list(iterator)
                self.assertTrue(handle.closed)


if __name__ == '__main__':
    unittest.main()
