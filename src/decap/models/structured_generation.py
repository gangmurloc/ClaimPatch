import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar


def require_json_object(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("structured generation output must be a JSON object")
    return value


T = TypeVar("T")


@dataclass
class StructuredGenerationRecord:
    """Raw and parsed structured generation artifact."""

    task: str
    prompt_version: str
    model_name: str
    prompt: str
    raw_text: str
    parsed: Dict[str, Any]
    parse_error: str = ""


def load_prompt(prompt_name: str, version: str = "v1", root: Path = None) -> str:
    root = root or Path("prompts")
    path = root / prompt_name / f"{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from model text."""

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return require_json_object(json.loads(stripped))
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in model output")
    return require_json_object(json.loads(stripped[start : end + 1]))


def validate_pydantic(model_cls: Type[T], payload: Dict[str, Any]) -> T:
    return model_cls.model_validate(payload)  # type: ignore[attr-defined]


def render_prompt(template: str, **kwargs: Any) -> str:
    return template.format(**{k: json.dumps(v, ensure_ascii=False, indent=2) for k, v in kwargs.items()})
