"""
Local policy: Qwen3-4B with optional LoRA adapters.
Used for the trained policy in Phase 7 eval and during GRPO training rollouts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "search_theorems",
        "description": "Search the theorem corpus by natural language query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "k": {"type": "integer", "description": "Number of results to return", "default": 10},
            },
            "required": ["query"],
        },
    },
}]


class LocalAgent:
    def __init__(
        self,
        model_id: str,
        checkpoint_dir: str | Path | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        use_vllm: bool = False,
    ):
        self.model_id = model_id
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.use_vllm = use_vllm
        self._model = None
        self._tokenizer = None
        self._vllm_client = None

    def _load(self) -> None:
        if self._model is not None:
            return

        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        logger.info("Loading %s with transformers...", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if self.checkpoint_dir and self.checkpoint_dir.exists():
            logger.info("Loading LoRA from %s", self.checkpoint_dir)
            self._model = PeftModel.from_pretrained(self._model, str(self.checkpoint_dir))
        self._model.eval()

    def _load_vllm(self) -> None:
        from vllm import LLM, SamplingParams
        logger.info("Loading %s with vLLM...", self.model_id)
        self._vllm_client = LLM(model=self.model_id, dtype="bfloat16")

    def _build_prompt(self, messages: list[dict]) -> str:
        assert self._tokenizer is not None
        return self._tokenizer.apply_chat_template(
            messages, tools=SEARCH_TOOL, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    async def chat(self, messages: list[dict]) -> tuple[str | None, int | None]:
        import asyncio
        self._load()

        if self.use_vllm:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._chat_vllm, messages
            )
        else:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._chat_transformers, messages
            )

    def _chat_transformers(self, messages: list[dict]) -> tuple[str | None, int | None]:
        import torch
        prompt = self._build_prompt(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        do_sample = self.temperature > 0
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._parse_tool_call(text)

    def _chat_vllm(self, messages: list[dict]) -> tuple[str | None, int | None]:
        from vllm import SamplingParams
        prompt = self._build_prompt(messages)
        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        outputs = self._vllm_client.generate([prompt], params)
        text = outputs[0].outputs[0].text
        return self._parse_tool_call(text)

    def _parse_tool_call(self, text: str) -> tuple[str | None, int | None]:
        # Qwen3's chat template emits tool calls as JSON inside <tool_call> tags
        import re
        m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                if data.get("name") == "search_theorems":
                    args = data.get("arguments", data.get("parameters", {}))
                    query = args.get("query", "")
                    k = int(args.get("k", 10))
                    return query, k
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: look for JSON blob with a "query" key
        m2 = re.search(r'\{"query":\s*"([^"]+)"', text)
        if m2:
            return m2.group(1), 10

        return None, None


def make_local_agent(config: dict, checkpoint_dir: str | None = None) -> LocalAgent:
    return LocalAgent(
        model_id=config["model"],
        checkpoint_dir=checkpoint_dir or config.get("checkpoint_dir"),
        temperature=config.get("temperature", 0.0),
        max_new_tokens=config.get("max_tokens", 512),
        use_vllm=config.get("use_vllm", False),
    )
