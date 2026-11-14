"""Synthetic regression tests, no corpus/weights required: python -m unittest discover."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("archive_provenance", Path(__file__).with_name("verify_provenance.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.names = ["source/predict.py", "source/common.py", "source/prompt.txt", "notebooks/MTS_02_Qwen35_Check.ipynb"]
        files = {}
        for name in self.names:
            raw = ("fixture " + name + "\nSecond line\n").encode()
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(raw)
            eol = "LF" if name.endswith("prompt.txt") else "CRLF"
            original = raw if eol == "LF" else raw.replace(b"\n", b"\r\n")
            files[name] = {"git_lf_sha256": hashlib.sha256(raw).hexdigest(), "executed_sha256": hashlib.sha256(original).hexdigest(), "executed_eol": eol}
        (self.root / "results").mkdir()
        (self.root / "provenance_eol.json").write_text(json.dumps({"files": files}))
        (self.root / "results/run_manifest.json").write_text(json.dumps({"fingerprint": dict(zip(("script_sha256", "common_sha256", "prompt_sha256"), [files[n]["executed_sha256"] for n in self.names[:3]]))}))
        (self.root / "provenance.json").write_text(json.dumps({"source_artifact_sha256": {"MTS_02_Qwen35_Check.ipynb": files[self.names[3]]["executed_sha256"]}}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_lf_and_executed_hashes_are_separate(self):
        r = module.verify_archive(self.root)
        self.assertEqual(r["status"], "content_consistent")
        self.assertEqual(sum(not x["raw_worktree_equals_executed"] for x in r["checks"]), 3)
        self.assertFalse(r["original_artifact_bytes_checked"])
        self.assertFalse(r["gpu_run_authenticated"])

    def test_every_source_mutation_fails(self):
        for name in self.names:
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b"changed substantive content\n")
                with self.assertRaises(ValueError):
                    module.verify_archive(self.root)
                path.write_bytes(original)

    def test_exact_executed_crlf_worktree(self):
        for name in self.names[:2] + self.names[3:]:
            path = self.root / name
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        self.assertTrue(all(x["raw_worktree_equals_executed"] for x in module.verify_archive(self.root)["checks"]))

    def test_manifest_mutation_fails(self):
        path = self.root / "results/run_manifest.json"
        v = json.loads(path.read_text())
        v["fingerprint"]["script_sha256"] = "0" * 64
        path.write_text(json.dumps(v))
        with self.assertRaises(ValueError):
            module.verify_archive(self.root)

    def test_bare_cr_fails(self):
        path = self.root / self.names[0]
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r"))
        with self.assertRaises(ValueError):
            module.verify_archive(self.root)

    def test_missing_originals_are_not_claimed_verified(self):
        with self.assertRaises(FileNotFoundError):
            module.verify_archive(self.root, self.root / "missing")


if __name__ == "__main__":
    unittest.main()
