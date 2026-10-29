"""Synthetic packaging excludes annotation and projection payloads."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
import test_hiergraph_data as data_fixture
from music_critic.experiments.hiergraph_export import write_json,file_binding,_json_bytes

spec=importlib.util.spec_from_file_location('package_inputs_script',Path(__file__).resolve().parents[2]/'scripts/package_hiergraph_inputs.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PackagingTests(unittest.TestCase):
    def test_exact_byte_portable_subset_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);ex=root/'export';ex.mkdir();src=root/'raw';src.mkdir()
            data_fixture.AuthenticatedDatasetTests().fixture(ex)
            piece=write_json(src/'piece.json',{'synthetic':True})
            manifest=_json_bytes((ex/'manifest.json').read_bytes())
            manifest['works'][0]['source_bindings']=dict(canonical_piece=piece,bundle=dict(path='excluded_bundle.json',sha256='0'*64,size_bytes=1))
            manifest['works'][0]['projection']=dict(path='excluded_projection.gz',sha256='0'*64,size_bytes=1)
            from music_critic.experiments.hiergraph_export import fingerprint,canonical
            manifest['fingerprint']=fingerprint(manifest);(ex/'manifest.json').write_bytes(canonical(manifest)+b'\n')
            sha=file_binding(ex/'manifest.json')['sha256'];out=root/'inputs'
            r=m.package_inputs(export_root=ex,manifest_sha256=sha,source_root=src,output=out)
            self.assertEqual(r['works'],1);self.assertEqual(len(r['files']),5)
            self.assertEqual((out/'targets/manifest.json').read_bytes(),(ex/'manifest.json').read_bytes())
            self.assertEqual((out/'source/piece.json').read_bytes(),(src/'piece.json').read_bytes())
            self.assertFalse((out/'targets/excluded_projection.gz').exists())
            with self.assertRaises(ValueError):m.package_inputs(export_root=ex,manifest_sha256=sha,source_root=src,output=out)
