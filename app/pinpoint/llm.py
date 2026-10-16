"""Model backends. Default: an open-weights instruct model held on the AMD GPU via PyTorch ROCm."""
from __future__ import annotations

import json
import os
import re
import time

DEFAULT_MODEL = os.environ.get("PINPOINT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MODEL_DIR = os.environ.get("PINPOINT_MODEL_DIR", "/models")


class TransformersLLM:
    """Loads once, stays resident. torch here is the ROCm build from the base image (HIP shows up as cuda)."""

    def __init__(self, model_id: str = DEFAULT_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        path = self._resolve(model_id)
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(path)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(dev).eval()
        self.device = dev
        self.load_s = round(time.time() - t0, 1)
        self.name = model_id
        self.gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
        self.hip = getattr(torch.version, "hip", None)

    @staticmethod
    def _resolve(model_id: str) -> str:
        local = os.path.join(MODEL_DIR, model_id.split("/")[-1])
        if os.path.isdir(local) and any(f.endswith(".safetensors") for f in os.listdir(local)):
            return local
        from huggingface_hub import snapshot_download
        os.makedirs(MODEL_DIR, exist_ok=True)
        return snapshot_download(model_id, local_dir=local, allow_patterns=["*.json", "*.safetensors", "*.txt",
                                                                             "tokenizer*", "*.model"])

    def chat(self, system: str, user: str, max_new_tokens: int = 160) -> str:
        torch = self.torch
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = {k: v.to(self.device) for k, v in ids.items()}
        with torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def vram_gib(self) -> float:
        if self.device != "cuda":
            return 0.0
        return round(self.torch.cuda.max_memory_allocated() / 2**30, 2)


class OpenAICompatLLM:
    """For development against a vLLM server (e.g. on an AMD notebook). Not used in the graded container."""

    def __init__(self, base_url: str, model: str, api_key: str = "x"):
        self.base, self.model, self.key = base_url.rstrip("/"), model, api_key
        self.name, self.gpu, self.hip, self.load_s = model, "remote", None, 0.0

    def chat(self, system: str, user: str, max_new_tokens: int = 160) -> str:
        import urllib.request
        body = json.dumps({"model": self.model, "temperature": 0, "max_tokens": max_new_tokens,
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        req = urllib.request.Request(self.base + "/chat/completions", data=body.encode(), headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def vram_gib(self) -> float:
        return 0.0


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    blob = m.group(0)
    for candidate in (blob, re.sub(r",\s*}", "}", blob)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {}


def make_llm():
    base = os.environ.get("PINPOINT_OPENAI_BASE")
    if base:
        return OpenAICompatLLM(base, os.environ.get("PINPOINT_OPENAI_MODEL", DEFAULT_MODEL),
                               os.environ.get("PINPOINT_OPENAI_KEY", "x"))
    return TransformersLLM()
