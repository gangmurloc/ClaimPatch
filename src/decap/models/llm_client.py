import json
from typing import Any, Dict, Optional


class MockLLMClient:
    """Deterministic fallback for P0/P1 tests when no model is available."""

    model_name = "mock"

    def generate_text(self, prompt: str, task: str = "generic", payload: Optional[Dict[str, Any]] = None) -> str:
        result = {"status": "mock", "task": task, "prompt_chars": len(prompt)}
        if payload:
            result.update(payload)
        return json.dumps(result, ensure_ascii=False)

    def generate_json(self, prompt: str, task: str = "generic", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        text = self.generate_text(prompt, task=task, payload=payload)
        return json.loads(text)


class LocalTransformersJSONClient:
    """Local Hugging Face causal-LM client for JSON-only prompted modules."""

    def __init__(
        self,
        model_name: str,
        revision: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        do_sample: bool = False,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        local_files_only: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("transformers/torch are required for local_transformers backend") from exc

        self.model_name = model_name
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.local_files_only = local_files_only
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = None
        if torch_dtype == "bf16":
            dtype = torch.bfloat16
        elif torch_dtype == "fp16":
            dtype = torch.float16
        elif torch_dtype == "fp32":
            dtype = torch.float32

        kwargs: Dict[str, Any] = {
            "revision": revision,
            "device_map": device_map,
            "local_files_only": local_files_only,
            "trust_remote_code": True,
        }
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        elif torch_dtype == "auto":
            kwargs["torch_dtype"] = "auto"

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("bitsandbytes/transformers BitsAndBytesConfig required for 4-bit loading") from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.generation_config.do_sample = do_sample
        if not do_sample:
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
        self.model.eval()

    def _format_prompt(self, prompt: str, task: str) -> str:
        system = (
            "You are a structured DECAP module. Return exactly one valid JSON object. "
            "Do not include markdown fences or prose."
        )
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Task: {task}\n\n{prompt}"},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"{system}\n\nTask: {task}\n\n{prompt}\n\nJSON:"

    def generate_text(self, prompt: str, task: str = "generic", payload: Optional[Dict[str, Any]] = None) -> str:
        import torch

        formatted = self._format_prompt(prompt, task)
        encoded = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=12000)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            output = self.model.generate(**encoded, **generation_kwargs)
        new_tokens = output[0, encoded["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_json(self, prompt: str, task: str = "generic", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from decap.models.structured_generation import extract_json_object

        return extract_json_object(self.generate_text(prompt, task=task, payload=payload))


def build_llm_client(config: Optional[Dict[str, Any]] = None):
    config = config or {}
    backend = config.get("backend", "mock")
    if backend == "mock":
        return MockLLMClient()
    if backend == "local_transformers":
        return LocalTransformersJSONClient(
            model_name=config.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
            revision=config.get("revision"),
            max_new_tokens=int(config.get("max_new_tokens", 1024)),
            temperature=float(config.get("temperature", 0.0)),
            do_sample=bool(config.get("do_sample", False)),
            torch_dtype=config.get("torch_dtype", "auto"),
            device_map=config.get("device_map", "auto"),
            load_in_4bit=bool(config.get("load_in_4bit", False)),
            local_files_only=bool(config.get("local_files_only", False)),
        )
    raise ValueError(f"unknown LLM backend: {backend}")
