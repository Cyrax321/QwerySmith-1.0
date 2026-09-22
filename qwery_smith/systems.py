"""System registry: eval matrix rows defined in dataset config, not code (plan §5.2).

Each system entry names a driver (local vLLM server / OpenAI-compatible API /
adapter dir) + decoding. The runner treats them identically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .exceptions import ConfigError


@dataclass
class SystemConfig:
    name: str
    role: str
    driver: str          # "openai_compatible" | "script"
    base_url: Optional[str] = None
    model_id: Optional[str] = None
    adapter_dir: Optional[Path] = None
    api_key_env: Optional[str] = None
    hardware: str = ""
    decoding: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, d: dict[str, Any]) -> "SystemConfig":
        try:
            return cls(
                name=d["name"],
                role=d["role"],
                driver=d["driver"],
                base_url=d.get("base_url"),
                model_id=d.get("model_id"),
                adapter_dir=Path(d["adapter_dir"]) if d.get("adapter_dir") else None,
                api_key_env=d.get("api_key_env"),
                hardware=d.get("hardware", ""),
                decoding=d.get("decoding"),
            )
        except KeyError as e:
            raise ConfigError(f"system config missing key {e}") from e


def openai_compatible_call(
    base_url: str,
    model_id: str,
    api_key_env: Optional[str] = None,
    decoding: dict[str, Any] | None = None,
) -> Callable[[str, int], str]:
    """Driver for vLLM/SGLang/Ollama/frontier APIs — all OpenAI-compatible.

    The callable takes (prompt, seed): the runner passes a distinct seed per
    consistency run (plan §0 — 5 samples must actually vary) and a fixed
    seed for the headline run.
    """
    import os
    import urllib.request

    dec = {
        "temperature": 0.7, "top_p": 0.8, "top_k": 20,
        "max_tokens": 1024,
    }
    if decoding:
        dec.update(decoding)

    def call(prompt: str, seed: int = 42) -> str:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": dec["temperature"],
            "top_p": dec["top_p"],
            "max_tokens": dec["max_tokens"],
            "seed": seed,
        }
        if dec.get("top_k"):
            payload["top_k"] = dec["top_k"]
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {os.environ[api_key_env]}"}
                   if api_key_env and os.environ.get(api_key_env) else {}),
            },
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read())
        return body["choices"][0]["message"]["content"]

    return call


def build_system_fn(spec: SystemConfig) -> Callable[[str, int], str]:
    if spec.driver == "openai_compatible":
        if not spec.base_url or not spec.model_id:
            raise ConfigError(f"system {spec.name}: openai_compatible needs base_url + model_id")
        return openai_compatible_call(spec.base_url, spec.model_id, spec.api_key_env,
                                      spec.decoding)
    if spec.driver == "script":
        # external script reads prompt on stdin, writes output to stdout.
        # Deterministic by construction: same prompt -> same output, so the
        # runner's distinct seeds are irrelevant here (flip=0 is honest).
        if not spec.model_id:
            raise ConfigError(f"system {spec.name}: script driver needs model_id = command")
        def call(prompt: str, seed: int = 42) -> str:
            import subprocess

            out = subprocess.run(
                spec.model_id, shell=True, input=prompt.encode(),
                capture_output=True, timeout=300,
            )
            if out.returncode != 0:
                raise RuntimeError(f"system script failed: {out.stderr.decode()[:200]}")
            return out.stdout.decode()

        return call
    raise ConfigError(f"unknown driver: {spec.driver}")