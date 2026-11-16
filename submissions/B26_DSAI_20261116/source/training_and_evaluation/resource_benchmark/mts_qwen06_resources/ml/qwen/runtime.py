"""Load-once Transformers engine. No HTTP server and no implicit network in serving."""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
from ml.qwen.bridge import QwenBridge, messages
from ml.qwen.prompting import effective_instruction
from src.contracts.models import CallInput
from src.pipeline.adapters import ModelDescriptor, ModelResources, PipelineError
from src.pipeline.lifecycle import LoadedModel, ManagedAdapter
from src.pipeline.service import Pipeline

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def load_base(device="cuda", *, revision=REVISION, local_only=False, base_path=MODEL):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    if device == "cuda" and not torch.cuda.is_available():
        raise PipelineError("QWEN_GPU_UNAVAILABLE", 503, "qwen.loader")
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_path, revision=revision, local_files_only=local_only)
    model = AutoModelForImageTextToText.from_pretrained(base_path, revision=revision,
        dtype=dtype, attn_implementation="sdpa", local_files_only=local_only).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class HFTextEngine:
    def __init__(self, model, tokenizer, instruction, catalog, *, context=16384, max_new_tokens=1536, device="cuda"):
        self.model, self.tokenizer = model, tokenizer
        self.instruction, self.catalog = effective_instruction(instruction), catalog
        self.context, self.max_new_tokens, self.device = context, max_new_tokens, device
        self.last_diagnostics = None
        self.last_raw = None

    def generate(self, call: CallInput):
        import torch
        self.last_diagnostics = None
        self.last_raw = None
        self.model.eval()
        inputs = self.tokenizer.apply_chat_template(messages(call, self.instruction, self.catalog),
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors="pt")
        n = int(inputs["input_ids"].shape[-1])
        if n + self.max_new_tokens > self.context:
            raise PipelineError("QWEN_CONTEXT_OVERFLOW", 422, "qwen.context")
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos]) | {self.tokenizer.eos_token_id}
        eos_ids.discard(None)
        try:
            with torch.inference_mode():
                outputs = self.model.generate(**inputs.to(self.device), do_sample=False, use_cache=True,
                    max_new_tokens=self.max_new_tokens, eos_token_id=sorted(eos_ids),
                    pad_token_id=self.tokenizer.pad_token_id)
        except torch.cuda.OutOfMemoryError:
            raise PipelineError("QWEN_OUT_OF_MEMORY", 503, "qwen.generate") from None
        suffix = outputs[0, n:]
        self.last_diagnostics = {"input_tokens": n, "output_tokens": len(suffix),
                                 "ended_with_eos": bool(len(suffix) and int(suffix[-1]) in eos_ids)}
        raw = self.tokenizer.decode(suffix, skip_special_tokens=True)
        self.last_raw = raw
        if len(suffix) >= self.max_new_tokens and int(suffix[-1]) not in eos_ids:
            raise PipelineError("QWEN_OUTPUT_LIMIT", 503, "qwen.generate")
        return raw


def artifact_manifest(path):
    root = Path(path).resolve()
    manifest = json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_state") != "trained_adapter_reload_verified" or not manifest.get("training_quality_gates_passed"):
        raise PipelineError("QWEN_ARTIFACT_NOT_VERIFIED", 503, "qwen.loader")
    if manifest["base_model"] != MODEL or manifest["base_revision"] != REVISION:
        raise PipelineError("QWEN_UNEXPECTED_BASE_REVISION", 503, "qwen.loader")
    for rel, expected in manifest["sha256"].items():
        p = (root / rel).resolve()
        if not p.is_relative_to(root) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise PipelineError("QWEN_ARTIFACT_HASH_MISMATCH", 503, "qwen.loader")
    return root, manifest


def factory():
    """MWS_PIPELINE_FACTORY=ml.qwen.runtime:factory; normalizers supplied by DE/FS."""
    artifact = os.environ["MTS_MODEL_DIR"]
    root, manifest = artifact_manifest(artifact)
    device = os.getenv("MTS_MODEL_DEVICE", "cpu")
    def loader(resources):
        from peft import PeftModel
        from transformers import AutoTokenizer
        model, _ = load_base(device, local_only=True)
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
        model = PeftModel.from_pretrained(model, root / "adapter").eval()
        session = HFTextEngine(model, tokenizer, (root / "prompt.txt").read_text(encoding="utf-8"),
            json.loads((root / "catalog.json").read_text(encoding="utf-8")), context=manifest["context"],
            max_new_tokens=manifest["max_new_tokens"], device=device)
        bridge = QwenBridge(session)
        # Structural startup smoke; not a benchmark or independent model-quality test.
        smoke = CallInput.model_validate({"call_id": "startup-independent-fixture", "started_at": "2026-11-14T10:00:00",
            "turns": [{"idx": 0, "speaker": "client", "text": "Здравствуйте."}]})
        bridge(smoke)
        def close():
            session.model = None
            gc.collect()
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()
        return LoadedModel(inference=bridge, close=close)
    adapter = ManagedAdapter(ModelResources(str(root), None, {"device": device}), loader,
        descriptor=ModelDescriptor(manifest["model_id"], "real", "transformers-peft"),
        revision=REVISION, artifact_sha256=hashlib.sha256((root / "model_manifest.json").read_bytes()).hexdigest(),
        dtype="float16" if device == "cuda" else "float32", context=manifest["context"],
        parameters=manifest["total_parameters"], manifest_id=manifest["manifest_id"])
    return Pipeline(adapter)
