import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import infer
from verify_weights import verify


class CLI(unittest.TestCase):
    def test_readonly_mount_and_spaces(self):
        self.assertEqual(infer.mount(Path('/test with spaces'), '/inputs/0'),
                         ['--mount', 'type=bind,src=/test with spaces,dst=/inputs/0,readonly'])

    def test_invalid_mount(self):
        for path in ('/a,b', '/a\nb'):
            with self.assertRaises(ValueError):
                infer.mount(path, '/input')

    def test_bad_weights(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'BSP_AE_final.pth').write_bytes(b'not weights')
            with self.assertRaises(ValueError):
                verify(d)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'mesh.off').touch()
            (root/'out').mkdir()
            (root/'out'/'keep.txt').write_text('keep')
            with patch('infer.verify'), self.assertRaises(SystemExit):
                infer.main([str(root/'mesh.off'), '--weights-dir', d, '--output-dir', str(root/'out')])
            self.assertEqual((root/'out'/'keep.txt').read_text(), 'keep')

    def test_unique_names(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d)/'same.off'
            b = Path(d)/'same.ply'
            a.touch()
            b.touch()
            with self.assertRaises(SystemExit):
                infer.main([str(a), str(b), '--output-dir', str(Path(d)/'out')])


if __name__ == '__main__':
    unittest.main()
